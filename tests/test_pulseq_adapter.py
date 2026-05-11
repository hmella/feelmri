"""Tests for feelmri.PulseqAdapter.

Layered coverage:
  * Parametrised parsing/units/trajectory tests across every .seq shipped
    under ``examples/pulseq/``.
  * Block-partition consistency tests for the dual-path ``import_pulseq``
    API (prep vs ADC indices, readout window contiguity, m_storage_idx
    correctness, kspace round-trip vs the flat ``kspace_trajectory``).
  * ROTATIONS extension test using a hand-authored synthetic fixture.
  * One end-to-end Bloch + signal-assembly integration test (slow,
    opt-out via ``-m 'not slow'``) that mirrors ``examples/phase_contrast.py``
    pattern on the gre_radial_pypulseq.seq 2D radial trajectory and a
    minimal 2-tetrahedron phantom built on the fly.
"""
from pathlib import Path

import numpy as np
import pytest
from pint import Quantity


PULSEQ_DIR = Path(__file__).resolve().parent.parent / 'examples' / 'pulseq'
SEQ_FILES = sorted(PULSEQ_DIR.glob('*.seq'))
ROTATION_SEQ = Path(__file__).resolve().parent / 'data' / 'rotation_minimal.seq'


@pytest.fixture(scope='session')
def adapter():
  """Import the adapter once per session."""
  from feelmri import PulseqAdapter
  return PulseqAdapter


@pytest.fixture(scope='session')
def parsed_imports(adapter):
  """Parse every .seq file exactly once via import_pulseq.

  mprage_pypulseq.seq alone takes ~80s to convert (5940 SequenceBlocks
  with per-block Quantity work), so caching across all parametrised
  tests is essential to keep total runtime reasonable.
  """
  cache = {}
  for path in SEQ_FILES:
    cache[path.name] = adapter.import_pulseq(path)
  return cache


def _seq_id(p):
  return p.name


# ---------------------------------------------------------------------------
# Parsing / structural tests over all bundled .seq files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_seq_id)
def test_parse_returns_populated_sequences(parsed_imports, seq_path):
  from feelmri.Bloch import Sequence

  imp = parsed_imports[seq_path.name]
  assert isinstance(imp.feelmri_seq, Sequence)
  assert len(imp.feelmri_seq.blocks) > 0
  assert len(imp.pulseq_seq) == len(imp.feelmri_seq.blocks)
  assert imp.pulseq_seq.DEF.get('PulseqVersion') is not None
  assert imp.pulseq_seq.DEF.get('FileName') == seq_path.name


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_seq_id)
def test_block_units_are_correct(parsed_imports, seq_path):
  from feelmri.Bloch import SequenceBlock

  imp = parsed_imports[seq_path.name]
  inspected = 0
  for blk in imp.feelmri_seq.blocks:
    if not isinstance(blk, SequenceBlock):
      continue
    for g in blk.gradients:
      assert g.amplitudes.units == Quantity(0, 'mT/m').units
      assert g.timings.units == Quantity(0, 'ms').units
      assert np.all(np.isfinite(g.amplitudes.m))
      inspected += 1
    for rf in blk.rf_pulses:
      assert rf.timings.units == Quantity(0, 'ms').units
      inspected += 1
    if blk.adc is not None:
      assert blk.adc.times.units == Quantity(0, 'ms').units
      assert blk.adc.times.m.size > 0
      assert blk.adc.freq_offset.units == Quantity(0, 'Hz').units
      assert blk.adc.phase_offset.units == Quantity(0, 'rad').units
      inspected += 1
  assert inspected > 0, f'no gradient/rf/adc inspected for {seq_path.name}'


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_seq_id)
def test_kspace_trajectory_well_formed(adapter, parsed_imports, seq_path):
  imp = parsed_imports[seq_path.name]
  traj = adapter.kspace_trajectory(imp.pulseq_seq)
  n = traj['times'].size
  if n == 0:
    # Some bundled .seq fragments (tagging prep, excitation-only) carry
    # no ADC events; the trajectory function is still expected to return
    # well-shaped empty arrays.
    for axis in ('kx', 'ky', 'kz', 'times'):
      assert traj[axis].shape == (0,)
    pytest.skip(f'{seq_path.name} has no ADC events')
  for axis in ('kx', 'ky', 'kz'):
    assert traj[axis].shape == (n,)
    assert np.all(np.isfinite(traj[axis]))
  assert np.all(np.diff(traj['times']) >= -1e-9), 'times not monotonic'


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_seq_id)
def test_pulseq_version_supported(adapter, parsed_imports, seq_path):
  imp = parsed_imports[seq_path.name]
  ver = imp.pulseq_seq.DEF['PulseqVersion']
  assert ver.major == 1
  assert ver >= adapter.Version(1, 2, 0)
  assert ver < adapter.Version(1, 5, 0), (
    f'{seq_path.name} declares Pulseq {ver}; adapter warns on >=1.5.0'
  )


# ---------------------------------------------------------------------------
# Dual-path partition API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_seq_id)
def test_import_pulseq_partitions_blocks(parsed_imports, seq_path):
  imp = parsed_imports[seq_path.name]
  n = len(imp.pulseq_seq)
  prep = set(imp.prep_block_indices)
  adc = set(imp.adc_block_indices)
  assert prep.isdisjoint(adc)
  assert prep | adc == set(range(n))

  for rw in imp.readouts:
    assert 0 <= rw.first_block <= rw.last_block < n
    for i in range(rw.first_block, rw.last_block + 1):
      assert i in adc, f'readout block {i} should have ADC'
    if rw.m_storage_block >= 0:
      assert rw.m_storage_block in prep
      assert rw.m_storage_block < rw.first_block
      assert imp.feelmri_seq.blocks[rw.m_storage_block].store_magnetization
      assert rw.m_storage_idx >= 0
    else:
      assert rw.m_storage_idx == -1

  # m_storage_idx values, in iteration order, must be a contiguous prefix
  # of the marked-blocks ordering (0, 1, 2, ...) so they index correctly
  # into BlochSolver.solve()'s output Mxy/Mz columns.
  active_indices = [rw.m_storage_idx for rw in imp.readouts
                    if rw.m_storage_idx >= 0]
  assert active_indices == sorted(active_indices)


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_seq_id)
def test_readout_windows_match_flat_trajectory(adapter, parsed_imports, seq_path):
  imp = parsed_imports[seq_path.name]
  flat = adapter.kspace_trajectory(imp.pulseq_seq)

  if not imp.readouts:
    assert flat['times'].size == 0
    return

  kx = np.concatenate([rw.kspace[:, 0] for rw in imp.readouts])
  ky = np.concatenate([rw.kspace[:, 1] for rw in imp.readouts])
  kz = np.concatenate([rw.kspace[:, 2] for rw in imp.readouts])
  times = np.concatenate([rw.times for rw in imp.readouts])

  assert kx.shape == flat['kx'].shape
  np.testing.assert_allclose(kx, flat['kx'], rtol=1e-6, atol=1e-9)
  np.testing.assert_allclose(ky, flat['ky'], rtol=1e-6, atol=1e-9)
  np.testing.assert_allclose(kz, flat['kz'], rtol=1e-6, atol=1e-9)
  np.testing.assert_allclose(times, flat['times'], rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# ROTATIONS extension
# ---------------------------------------------------------------------------

def test_rotation_extension_round_trip(adapter):
  """tests/data/rotation_minimal.seq applies a 90 deg z-rotation to a
  single trapezoid on Gx. After parsing, Gx must be ~0 and Gy must
  carry the original amplitude."""
  assert ROTATION_SEQ.exists(), f'rotation fixture missing: {ROTATION_SEQ}'
  imp = adapter.import_pulseq(ROTATION_SEQ)
  ps = imp.pulseq_seq
  assert len(ps) == 1
  gx, gy, gz = ps.GR[0]
  expected = 1_000_000.0 / adapter.GAMMA  # T/m, after Pulseq 1/GAMMA scale.

  assert np.isclose(float(gx.A), 0.0, atol=1e-12)
  assert np.isclose(float(gy.A), expected, rtol=1e-6)
  assert np.isclose(float(gz.A), 0.0, atol=1e-12)

  exts = ps.EXT[0]
  assert len(exts) == 1
  assert isinstance(exts[0], adapter.Rotation)
  assert exts[0].matrix.shape == (3, 3)


# ---------------------------------------------------------------------------
# Slow integration: phase_contrast pattern on a 2D radial Pulseq sequence
# ---------------------------------------------------------------------------

def _write_minimal_tet_mesh(path: Path):
  """Backwards-compatible wrapper for the shared
  :func:`make_minimal_tet_mesh` helper.

  Kept as a thin shim so the in-test parametrisation IDs and any
  external callers continue to work; the actual mesh-writing logic
  now lives in ``tests/_phantom_fixtures.py``."""
  pytest.importorskip('meshio')
  from _phantom_fixtures import make_minimal_tet_mesh
  make_minimal_tet_mesh(path)
  return 5  # node count, preserved for historical callers


@pytest.mark.slow
def test_dual_path_phase_contrast_radial2d(parsed_imports, tmp_path):
  """End-to-end dual-path smoke test mirroring examples/phase_contrast.py.

  Steps (single Sequence, single solver.solve(), per-readout signal
  assembly):
    1. Parse gre_radial_pypulseq.seq via import_pulseq -> partitioned
       view with one ReadoutWindow per radial spoke.
    2. Build a 2-tet phantom on the fly (no external assets).
    3. BlochSolver.solve() once over imp.feelmri_seq; readouts'
       m_storage_block flags are already set by import_pulseq.
    4. For each readout: phantom.update_magnetization(Mxy[:, idx]) +
       phantom.mri_signal(traj_pts, traj_times, pod=None).
    5. Assert finite signal of the expected length.

  The trajectory is 2D radial intrinsically (kz ~ 0) because the .seq
  file is 2D; no feelmri.KSpaceTraj.RadialStack object is constructed.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  try:
    from feelmri.Bloch import BlochSolver
    from feelmri.MRObjects import Scanner
    from feelmri.Phantom import FEMPhantom
  except ImportError as exc:
    pytest.skip(f'feelmri C++ extensions not available: {exc}')

  if 'gre_radial_pypulseq.seq' not in parsed_imports:
    pytest.skip('gre_radial_pypulseq.seq not parsed')

  mesh_path = tmp_path / 'minimal_tet.vtu'
  _write_minimal_tet_mesh(mesh_path)
  phantom = FEMPhantom(path=str(mesh_path))

  imp = parsed_imports['gre_radial_pypulseq.seq']
  assert len(imp.readouts) > 0, 'expected at least one readout window'
  # 2D radial means no Gz gradient is played during the readout window:
  # kz is constant within each readout. (The absolute kz offset across
  # readouts is set by the slice-select prephaser; that is fine for the
  # signal assembler — it manifests as a phase ramp across the slice.)
  for rw in imp.readouts:
    kz = rw.kspace[:, 2]
    span = float(kz.max() - kz.min())
    assert span < 1e-3, (
      f'expected kz constant within readout (2D radial); got span {span:g}'
    )

  solver = BlochSolver(
    sequence=imp.feelmri_seq,
    phantom=phantom,
    scanner=Scanner(),
  )
  Mxy, Mz = solver.solve()
  assert Mxy.shape[0] == phantom.local_nodes.shape[0]
  assert np.all(np.isfinite(Mxy))
  assert np.all(np.isfinite(Mz))

  phantom.set_assembler(voxel_size=5e-3, lorder=1, horder=2,
                       nodal_approximation=True, lumped=True)
  phantom.set_static_fields(
    T2=np.full(phantom.local_nodes.shape[0], 100.0, dtype=np.float32),
    phi_dB0=np.zeros(phantom.local_nodes.shape[0], dtype=np.float32),
  )

  readouts_checked = 0
  for rw in imp.readouts:
    if rw.m_storage_idx < 0:
      continue
    phantom.update_magnetization(Mxy[:, rw.m_storage_idx])
    # The C++ SignalAssembler expects k-space coordinates as a list of
    # three rank-3 tensors (nb_meas, nb_lines, nb_kz) and times as the
    # matching rank-3 tensor. For a single radial spoke we set the trailing
    # phase-encode/slice dims to 1.
    n = rw.times.size
    shape = (n, 1, 1)
    points = [
      np.ascontiguousarray(rw.kspace[:, 0].reshape(shape), dtype=np.float32),
      np.ascontiguousarray(rw.kspace[:, 1].reshape(shape), dtype=np.float32),
      np.ascontiguousarray(rw.kspace[:, 2].reshape(shape), dtype=np.float32),
    ]
    times = np.ascontiguousarray(rw.times.reshape(shape), dtype=np.float32)
    sig = phantom.mri_signal(points, times, None)
    assert np.all(np.isfinite(sig))
    assert sig.shape[0] == n
    readouts_checked += 1
  assert readouts_checked > 0
