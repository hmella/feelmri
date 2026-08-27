"""meshio -> Basix DOF ordering in the finite-element quadrature cache.

meshio and Basix agree that vertex DOFs come before edge/face DOFs, but not on the
order *within* those blocks: VTK walks the bottom face of a hexahedron cyclically
while Basix uses a tensor-product lattice, and the two number tetrahedron edges
differently. Handing meshio-ordered connectivity to a Basix element scrambles the
geometry map without raising, since the Jacobian stays computable.

P1 ``tetra`` is one of two cell types where the conventions coincide, so it cannot
detect the problem. These tests exercise all four cell types the 3-D assembler
supports.

``triangle`` is declared in ``fe_from_meshio`` but cannot be used here: the
quadrature cache always tabulates with a 3-D point dimension, so Basix rejects a
2-D cell. That limitation is unrelated to DOF ordering.
"""
from __future__ import annotations

import numpy as np
import pytest

from feelmri.MRIAssemble import SignalAssembler

# Reference-cell coordinates in meshio/VTK order, and the reference cell volume.
# Stated independently of the C++ table on purpose: if the two ever disagree,
# one of them is wrong about the VTK convention and these tests should say so.
REFERENCE_CELLS = {
  'tetra': (
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    1.0 / 6.0,
  ),
  'tetra10': (
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
     [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
     [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    1.0 / 6.0,
  ),
  'hexahedron': (
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    1.0,
  ),
  'wedge': (
    [[0, 0, 0], [1, 0, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [0, 1, 1]],
    0.5,
  ),
}

CELL_TYPES = sorted(REFERENCE_CELLS)


def _measured_volume(elems, nodes, cell_type, degree):
  """Total volume the assembler attributes to the mesh.

  ``estimate_element_sizes`` returns the cube root of each element's
  Jacobian volume, so cubing and summing recovers the mesh volume.
  """
  a = SignalAssembler(np.ascontiguousarray(elems, dtype=np.int32),
                      np.ascontiguousarray(nodes, dtype=np.float32),
                      cell_type, degree)
  sizes = np.asarray(a.estimate_element_sizes(), dtype=np.float64)
  return float(np.sum(sizes ** 3))


# --------------------------------------------------------------------------
# 1. Single reference element -- needs no mesh file and no external geometry
# --------------------------------------------------------------------------

@pytest.mark.parametrize('cell_type', CELL_TYPES)
def test_reference_element_has_reference_volume(cell_type):
  """An element whose physical nodes *are* the reference coordinates must
  measure exactly the reference cell volume.

  This is the tightest possible statement of the ordering contract: any
  permutation error distorts the geometry map and changes the volume.
  """
  coords, expected = REFERENCE_CELLS[cell_type]
  nodes = np.asarray(coords, dtype=np.float64)
  elems = np.arange(len(coords), dtype=np.int32).reshape(1, -1)
  # Degree 2 so the trilinear hexahedron Jacobian is integrated exactly;
  # a 1-point rule is not exact for it even when the ordering is correct.
  measured = _measured_volume(elems, nodes, cell_type, 2)
  assert measured == pytest.approx(expected, rel=1e-5)


# Every vertex permutation of a simplex only flips the sign of det J, and the
# assembler takes its absolute value -- so a P1 tetrahedron cannot detect an
# ordering error by volume. That is exactly why the meshio/Basix mismatch was
# invisible on ``tetra`` meshes, and why the rest of the suite never caught it.
ORDER_SENSITIVE = [c for c in CELL_TYPES if c != 'tetra']


def test_linear_tetra_volume_is_permutation_invariant():
  """Documents the blind spot: no vertex permutation changes a P1 tet's volume."""
  coords, expected = REFERENCE_CELLS['tetra']
  nodes = np.asarray(coords, dtype=np.float64)
  for order in ([0, 1, 2, 3], [1, 0, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
    elems = np.asarray(order, dtype=np.int32).reshape(1, -1)
    assert _measured_volume(elems, nodes, 'tetra', 2) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize('cell_type', ORDER_SENSITIVE)
def test_scrambled_ordering_is_detectable(cell_type):
  """Guard against the reference-volume test passing vacuously.

  Swapping two nodes must change the measured volume; if it did not, that
  test could not detect a bad permutation either.
  """
  coords, expected = REFERENCE_CELLS[cell_type]
  nodes = np.asarray(coords, dtype=np.float64)
  order = list(range(len(coords)))
  # Swap the last two DOFs: an edge/face pair for the higher-order cells,
  # a vertex pair for the linear ones. Either distorts the map.
  order[-1], order[-2] = order[-2], order[-1]
  elems = np.asarray(order, dtype=np.int32).reshape(1, -1)
  measured = _measured_volume(elems, nodes, cell_type, 2)
  assert measured != pytest.approx(expected, rel=1e-3)


# --------------------------------------------------------------------------
# 2. Structured multi-element meshes with an analytically known volume
# --------------------------------------------------------------------------

def _unit_cube_grid(n=2, scale=1e-2):
  """(n+1)^3 lattice of points spanning a cube of side ``n * scale``."""
  c = np.arange(n + 1, dtype=np.float64) * scale
  pts = np.array([[x, y, z] for z in c for y in c for x in c])
  idx = lambda i, j, k: i + (n + 1) * (j + (n + 1) * k)
  # VTK hexahedron order: bottom face walked cyclically, then the top face.
  hexes = [[idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
            idx(i, j, k + 1), idx(i + 1, j, k + 1),
            idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)]
           for k in range(n) for j in range(n) for i in range(n)]
  return pts, np.asarray(hexes, dtype=np.int64), float((n * scale) ** 3)


def _build_mesh(cell_type, n=2, scale=1e-2):
  """Structured mesh of ``cell_type`` filling a cube, plus its exact volume."""
  pts, hexes, volume = _unit_cube_grid(n, scale)

  if cell_type == 'hexahedron':
    return pts, hexes, volume

  if cell_type == 'wedge':
    # Cut each cube along the bottom-face diagonal into two prisms.
    # VTK wedge: bottom triangle (0,1,2), top triangle (3,4,5) directly above.
    cells = []
    for h in hexes:
      cells.append([h[0], h[1], h[2], h[4], h[5], h[6]])
      cells.append([h[0], h[2], h[3], h[4], h[6], h[7]])
    return pts, np.asarray(cells, dtype=np.int64), volume

  # Six-tetrahedron decomposition of each cube.
  tets = []
  for h in hexes:
    for a, b, c, d in ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
                       (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)):
      tets.append([h[a], h[b], h[c], h[d]])
  tets = np.asarray(tets, dtype=np.int64)

  if cell_type == 'tetra':
    return pts, tets, volume

  # tetra10: append edge midpoints in VTK edge order.
  pts = list(map(list, pts))
  midpoints = {}

  def mid(u, v):
    key = (min(u, v), max(u, v))
    if key not in midpoints:
      midpoints[key] = len(pts)
      pts.append(list(0.5 * (np.asarray(pts[u]) + np.asarray(pts[v]))))
    return midpoints[key]

  cells = [list(t) + [mid(t[0], t[1]), mid(t[1], t[2]), mid(t[0], t[2]),
                      mid(t[0], t[3]), mid(t[1], t[3]), mid(t[2], t[3])]
           for t in tets]
  return np.asarray(pts, dtype=np.float64), np.asarray(cells, dtype=np.int64), volume


@pytest.mark.parametrize('cell_type', CELL_TYPES)
def test_structured_mesh_volume_matches_analytic(cell_type):
  """A structured mesh filling a cube must measure the cube's volume."""
  nodes, elems, volume = _build_mesh(cell_type)
  measured = _measured_volume(elems, nodes, cell_type, 2)
  assert measured == pytest.approx(volume, rel=1e-4)


@pytest.mark.parametrize('cell_type', CELL_TYPES)
def test_volume_is_independent_of_quadrature_degree(cell_type):
  """Refining the quadrature must not move the measured volume.

  A scrambled geometry map gives a self-intersecting element, for which this
  does not converge.
  """
  nodes, elems, volume = _build_mesh(cell_type)
  measured = [_measured_volume(elems, nodes, cell_type, d) for d in (2, 4, 6)]
  for m in measured:
    assert m == pytest.approx(volume, rel=1e-4)


def test_unregistered_cell_type_raises():
  """An unsupported type must fail loudly rather than silently mis-map."""
  nodes = np.zeros((4, 3), dtype=np.float32)
  elems = np.arange(4, dtype=np.int32).reshape(1, -1)
  with pytest.raises(RuntimeError):
    SignalAssembler(elems, nodes, 'pyramid', 2)


# --------------------------------------------------------------------------
# 3. End-to-end: the same domain, meshed four ways, must integrate the same
# --------------------------------------------------------------------------

def _cube_signal(tmp_path, cell_type, kx):
  """Quadrature-path k-space signal over a cube meshed as ``cell_type``."""
  from feelmri import FEMPhantom
  from _phantom_fixtures import make_cube_mesh

  mesh_path = tmp_path / f'cube_{cell_type}.vtu'
  _, volume = make_cube_mesh(mesh_path, cell_type)
  phantom = FEMPhantom(path=str(mesh_path))
  # voxel_size=0 sends every element to the high-order group, so this
  # exercises the quadrature path (signal) rather than the nodal one.
  phantom.set_assembler(voxel_size=0.0, lorder=2, horder=4,
                        nodal_approximation=False, lumped=False)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 1e9, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))
  phantom.update_magnetization(np.ones((n, 1), dtype=np.complex64))

  shape = (kx.size, 1, 1)
  zeros = np.zeros(shape, dtype=np.float32)
  pts = [np.ascontiguousarray(kx.reshape(shape), dtype=np.float32), zeros, zeros]
  t3 = np.zeros(shape, dtype=np.float32)
  return np.asarray(phantom.signal(pts, t3, None)).reshape(-1), volume


@pytest.mark.parametrize('cell_type', CELL_TYPES)
def test_cube_signal_at_k0_equals_volume(tmp_path, cell_type):
  """With constant M_xy = 1 and no relaxation, S(0) is the domain volume.

  This is the ordering contract seen through the full quadrature path --
  ``wq``, ``xq`` and the ``S_global_`` projection -- not just element sizing.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')

  S, volume = _cube_signal(tmp_path, cell_type, np.array([0.0], dtype=np.float32))
  assert complex(S[0]).real == pytest.approx(volume, rel=1e-4)
  assert abs(complex(S[0]).imag) < 1e-6 * volume


def test_cube_signal_agrees_across_cell_types(tmp_path):
  """All four discretisations of the same cube must give the same S(k).

  The four cell types use four different meshio/Basix DOF conventions, so a
  permutation error in any one of them shows up as a disagreement here.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')

  # Well inside the first lobe, where the quadrature is comfortably resolved.
  kx = np.array([0.0, 40.0, 80.0], dtype=np.float32)
  results = {c: _cube_signal(tmp_path, c, kx)[0] for c in CELL_TYPES}

  reference = results['tetra']
  for cell_type, S in results.items():
    assert np.allclose(S, reference, rtol=2e-3, atol=1e-12), (
      f'{cell_type} disagrees with tetra: {S} vs {reference}'
    )
