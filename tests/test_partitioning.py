"""Mesh partitioning: the NODAL default and the dual-layout scheme.

`distribute_mesh` balances the nodal graph (or the element graph on request). That
suits the Bloch solve, which costs O(local nodes), but not the quadrature signal
path, which costs O(local quadrature points) -- and `set_assembler` gives large
elements a far more expensive rule (24 points at order 6 against 1 at order 1).

`enable_dual_partition` therefore keeps **two** layouts: node-balanced for the solve,
quadrature-balanced for k-space, with nodal data moved between them by
`redistribute_nodal`.

These tests pin the objective and the machinery, not the balance ratios: a partition
can have worse node *and* quadrature ratios and still be better, because what sets
the wall time is `max_r (N_r + Q_r / cost_ratio)`.
"""
from __future__ import annotations

import numpy as np
import pytest

from feelmri.MRIAssemble import quadrature_npoints

from _phantom_fixtures import make_cube_mesh


def _graded_phantom(tmp_path):
  """Cube meshed as tetrahedra, with an artificially graded element-size field.

  The mesh itself is uniform; the sizes are overwritten so a `voxel_size`
  threshold produces a *spatially clustered* expensive group, which is the
  situation that defeats node-balanced partitioning.
  """
  from feelmri import FEMPhantom
  path, _ = make_cube_mesh(tmp_path / 'cube.vtu', 'tetra', n=6, scale=1e-3)
  phantom = FEMPhantom(path=str(path))
  centroids = phantom.global_nodes[phantom.global_elements].mean(axis=1)
  # One corner octant is "large"; everything else is "small".
  big = np.all(centroids > centroids.mean(axis=0), axis=1)
  sizes = np.where(big, 10.0, 1.0).astype(np.float32)
  phantom.global_elem_size = sizes
  return phantom


def _cost(phantom, membership, weights, nparts, ratio=47.3):
  m = np.asarray(membership)
  N = np.array([len(np.unique(phantom.global_elements[m == r].ravel()))
                for r in range(nparts)], float)
  Q = np.array([weights[m == r].sum() for r in range(nparts)], float)
  return (N + Q).max(), N, Q


def test_quadrature_npoints_matches_basix_rules():
  """The weights must come from basix, not a hardcoded table that can drift."""
  assert quadrature_npoints('tetra', 1) == 1
  assert quadrature_npoints('tetra', 2) == 4
  assert quadrature_npoints('tetra', 6) == 24     # Keast
  assert quadrature_npoints('hexahedron', 2) == 8  # 2x2x2 Gauss
  # Cost really is dominated by the high-order group.
  assert quadrature_npoints('tetra', 6) > 20 * quadrature_npoints('tetra', 1)


def test_weights_track_the_set_assembler_split(tmp_path):
  """Weights must mirror the `local_elem_size < voxel_size` split exactly."""
  phantom = _graded_phantom(tmp_path)
  w = phantom.quadrature_cost_weights(5.0, lorder=1, horder=6, cost_ratio=1.0)
  big = phantom.global_elem_size >= 5.0
  assert np.all(w[big] == quadrature_npoints('tetra', 6))
  assert np.all(w[~big] == quadrature_npoints('tetra', 1))


def test_nodal_approximation_zeroes_the_small_group(tmp_path):
  """With nodal_approximation the small group runs through signal_nodal, whose
  cost is O(nodes) and already counted by the node term. Charging it again
  double-counts and measurably degrades the partition."""
  phantom = _graded_phantom(tmp_path)
  w = phantom.quadrature_cost_weights(5.0, lorder=1, horder=6, cost_ratio=1.0,
                                      nodal_approximation=True)
  assert np.all(w[phantom.global_elem_size < 5.0] == 0.0)
  assert np.all(w[phantom.global_elem_size >= 5.0] > 0.0)


@pytest.mark.parametrize('nparts', [4, 8])
def test_cost_weighted_partition_beats_node_balancing_on_the_real_objective(tmp_path, nparts):
  """The whole point: lower `max_r (N_r + Q_r/ratio)` than plain NODAL."""
  pytest.importorskip('pymetis')
  import pymetis

  phantom = _graded_phantom(tmp_path)
  weights = phantom.quadrature_cost_weights(5.0, lorder=1, horder=6,
                                            cost_ratio=47.3)
  _, nodal, _ = pymetis.part_mesh(nparts, phantom.global_elements, None, None,
                                  pymetis.GType.NODAL)
  weighted = phantom._joint_cost_partition(nparts, weights, 2)

  cost_nodal, _, _ = _cost(phantom, nodal, weights, nparts)
  cost_weighted, _, _ = _cost(phantom, weighted, weights, nparts)
  assert cost_weighted <= cost_nodal, (
    f'cost-weighted partition should not be worse: {cost_weighted} vs {cost_nodal}')


def test_partition_is_a_valid_total_assignment(tmp_path):
  """Every element assigned exactly once, to a real rank."""
  pytest.importorskip('pymetis')
  phantom = _graded_phantom(tmp_path)
  weights = phantom.quadrature_cost_weights(5.0, lorder=1, horder=6)
  m = phantom._joint_cost_partition(8, weights, 2)
  assert m.shape == (phantom.global_elements.shape[0],)
  assert m.min() >= 0 and m.max() < 8
  assert len(np.unique(m)) > 1, 'partition collapsed onto one rank'


def test_chunk_weights_survive_integer_rounding(tmp_path):
  """METIS takes integer weights, and the blended cost is O(1) in
  node-equivalents. Rounding it directly collapses the cheap and expensive
  element classes onto the same integer and silently discards the cost
  distinction.

  Asserted on the integer weights themselves: the bin-packing step still uses the
  float costs, so it masks the problem at the partition level.
  """
  phantom = _graded_phantom(tmp_path)
  w = phantom.quadrature_cost_weights(5.0, lorder=1, horder=6, cost_ratio=47.3)
  iw = phantom._chunk_weights(w)
  cheap = iw[phantom.global_elem_size < 5.0]
  pricey = iw[phantom.global_elem_size >= 5.0]
  assert cheap.min() >= 1, 'weights must stay strictly positive for METIS'
  # The float ratio must survive quantisation, not collapse to 1.0.
  float_ratio = (w.max() + 1.0) / (w.min() + 1.0)
  assert pricey.min() / cheap.max() > 1.0 + 0.5 * (float_ratio - 1.0), (
    f'cost distinction lost in rounding: {cheap.max()} vs {pricey.min()}')


def test_enable_dual_partition_refuses_after_the_partition_is_in_use(tmp_path):
  """Building a second layout after a POD or solver has bound to the partition
  would silently invalidate it, so it must raise instead."""
  phantom = _graded_phantom(tmp_path)
  _ = phantom.local_to_global_nodes          # what POD(global_to_local=...) does
  with pytest.raises(RuntimeError, match='already in use'):
    phantom.enable_dual_partition(voxel_size=5.0, lorder=1, horder=6,
                                  nodal_approximation=False, lumped=False)


def test_set_assembler_does_not_repartition(tmp_path):
  """set_assembler builds groups on the live partition and must not move it.

  Partitioning belongs to distribute_mesh and enable_dual_partition.
  """
  phantom = _graded_phantom(tmp_path)
  before = np.array(phantom.partitioning, copy=True)
  phantom.set_assembler(voxel_size=5.0, lorder=1, horder=6,
                        nodal_approximation=False, lumped=False)
  assert np.array_equal(np.asarray(phantom.partitioning), before)


# --------------------------------------------------------------------------
# Dual partitioning: two layouts, one for the Bloch solve and one for k-space
# --------------------------------------------------------------------------

def test_partition_swap_restores_state(tmp_path):
  """`activate` must exchange the whole partition, not part of it.

  The mesh state is eight flat attributes rather than an object, so a partial
  swap would leave the phantom internally inconsistent with no error.
  """
  phantom = _graded_phantom(tmp_path)
  phantom.enable_dual_partition(voxel_size=5.0, lorder=1, horder=6,
                                nodal_approximation=False, lumped=False)
  assert set(phantom._partitions) == {'bloch', 'signal'}
  snap = {a: getattr(phantom, a) for a in phantom._PARTITION_ATTRS}
  phantom.activate('signal')
  phantom.activate('bloch')
  for a in phantom._PARTITION_ATTRS:
    now, before = getattr(phantom, a), snap[a]
    if isinstance(before, np.ndarray):
      assert np.array_equal(now, before), f'{a} not restored'
    else:
      assert now == before, f'{a} not restored'


def test_using_restores_partition_even_on_error(tmp_path):
  """A failure inside a signal call must not leave the wrong partition live."""
  phantom = _graded_phantom(tmp_path)
  phantom.enable_dual_partition(voxel_size=5.0, lorder=1, horder=6,
                                nodal_approximation=False, lumped=False)
  assert phantom._active_partition == 'bloch'
  try:
    with phantom._using('signal'):
      raise ValueError('boom')
  except ValueError:
    pass
  assert phantom._active_partition == 'bloch'


def test_redistribution_is_exact_and_total(tmp_path):
  """Moving nodal data between layouts must be lossless and cover every node.

  Values are keyed to the *global* node index, so a correct redistribution
  reproduces them exactly.
  """
  phantom = _graded_phantom(tmp_path)
  phantom.enable_dual_partition(voxel_size=5.0, lorder=1, horder=6,
                                nodal_approximation=False, lumped=False)
  g_src = phantom._partitions['bloch']['_local_to_global_nodes']
  g_dst = phantom._partitions['signal']['_local_to_global_nodes']
  src = (np.cos(g_src * 0.01) + 1j * np.sin(g_src * 0.02)).astype(np.complex64)
  out = phantom.redistribute_nodal(src.reshape(-1, 1), 'bloch', 'signal')
  want = (np.cos(g_dst * 0.01) + 1j * np.sin(g_dst * 0.02)).astype(np.complex64)
  assert out.shape == (g_dst.size, 1)
  assert np.array_equal(out.reshape(-1), want), 'redistribution is not exact'


def test_redistribution_round_trip(tmp_path):
  """signal -> bloch -> signal must return the original values."""
  phantom = _graded_phantom(tmp_path)
  phantom.enable_dual_partition(voxel_size=5.0, lorder=1, horder=6,
                                nodal_approximation=False, lumped=False)
  g = phantom._partitions['signal']['_local_to_global_nodes']
  a = (np.cos(g * 0.03)).astype(np.float32).reshape(-1, 1)
  back = phantom.redistribute_nodal(
    phantom.redistribute_nodal(a, 'signal', 'bloch'), 'bloch', 'signal')
  assert np.array_equal(back, a)


def test_mode_arrays_are_address_stable(tmp_path):
  """The assembler caches `S_global_ * modes` keyed on `modes_x.data()`.

  A freshly allocated array can reuse a previously freed address, which either
  invalidates the cache on every call or matches it against different modes.
  Holding the buffers pins the address.
  """
  pytest.importorskip('mpi4py')
  from feelmri.Motion import POD
  phantom = _graded_phantom(tmp_path)
  phantom.set_assembler(voxel_size=5.0, lorder=1, horder=6,
                        nodal_approximation=False, lumped=False)
  n_global = phantom.global_nodes.shape[0]
  rng = np.random.default_rng(0)
  data = rng.normal(0, 1e-3, (n_global, 3, 6)).astype(np.float32)
  pod = POD(times=np.linspace(0, 100, 6, dtype=np.float32), data=data,
            global_to_local=phantom.local_to_global_nodes, n_modes=3,
            is_periodic=True, interpolation_method='Pchip')
  addrs = {phantom._cached_mode_arrays(pod)[0].__array_interface__['data'][0]
           for _ in range(6)}
  assert len(addrs) == 1, f'mode array address is not stable: {addrs}'


# --------------------------------------------------------------------------
# signal_sum must not count interface nodes once per owning rank
# --------------------------------------------------------------------------

def test_node_ownership_mask_drops_masked_nodes(tmp_path):
  """`set_node_ownership` must remove a node's contribution entirely.

  A rank owns every node its elements touch, so interface nodes live on several
  ranks. `signal_sum` is a bare sum over local nodes and the caller reduces it
  with MPI_SUM, so without a canonical-owner mask every interface node is counted
  once per owning rank and the result does not converge with rank count.
  """
  pytest.importorskip('mpi4py')
  from feelmri import FEMPhantom
  from _phantom_fixtures import make_cube_mesh

  path, _ = make_cube_mesh(tmp_path / 'cube.vtu', 'tetra', n=3, scale=2e-3)
  phantom = FEMPhantom(path=str(path))
  phantom.set_assembler(voxel_size=0.0, lorder=2, horder=2,
                        nodal_approximation=False, lumped=False)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 1e9, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))
  rng = np.random.default_rng(0)
  Mxy = (rng.uniform(-1, 1, (n, 1)) + 1j * rng.uniform(-1, 1, (n, 1))).astype(np.complex64)
  S = 8
  pts = [np.ascontiguousarray(rng.uniform(-60, 60, (S, 1, 1)).astype(np.float32))
         for _ in range(3)]
  ts = np.zeros((S, 1, 1), dtype=np.float32)
  keep = np.zeros(n, dtype=np.float32)
  keep[::2] = 1.0

  phantom.update_magnetization(Mxy)
  full = np.asarray(phantom.signal_sum(pts, ts, None)).reshape(-1)
  phantom.assembler[0].set_node_ownership(keep)
  masked = np.asarray(phantom.signal_sum(pts, ts, None)).reshape(-1)
  phantom.assembler[0].set_node_ownership(np.ones(n, dtype=np.float32))
  phantom.update_magnetization(Mxy * keep.reshape(-1, 1))
  reference = np.asarray(phantom.signal_sum(pts, ts, None)).reshape(-1)

  assert np.allclose(masked, reference, rtol=1e-5, atol=1e-12)
  assert not np.allclose(masked, full, rtol=1e-3), 'mask had no effect'


def test_ownership_mask_is_a_partition_of_the_global_nodes(tmp_path):
  """Serially every node is owned exactly once, so nothing is dropped or doubled."""
  phantom = _graded_phantom(tmp_path)
  mask = phantom._node_ownership_mask()
  assert mask.shape == (phantom.local_nodes.shape[0],)
  assert mask.sum() == phantom.local_nodes.shape[0], 'serial run must own every node'


def test_wrong_length_ownership_mask_raises(tmp_path):
  """A mask sized for another partition must fail loudly, not silently mis-weight."""
  phantom = _graded_phantom(tmp_path)
  phantom.set_assembler(voxel_size=5.0, lorder=1, horder=6,
                        nodal_approximation=False, lumped=False)
  n = phantom.local_nodes.shape[0]
  with pytest.raises(RuntimeError, match='set_node_ownership'):
    phantom.assembler[0].set_node_ownership(np.ones(n + 7, dtype=np.float32))
