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
  assert ver < adapter.Version(1, 6, 0), (
    f'{seq_path.name} declares Pulseq {ver}; adapter warns on >=1.6.0'
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
    # At minimum, first_block and last_block must themselves be ADC
    # blocks (anchor groups start and end on an ADC). Intervening
    # blocks may be phase-encode blips or spoilers when an EPI echo
    # train shares one coherence anchor.
    assert rw.first_block in adc, (
      f'window first_block {rw.first_block} should have ADC'
    )
    assert rw.last_block in adc, (
      f'window last_block {rw.last_block} should have ADC'
    )
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
# v1.5 column-layout dispatch (RF / ADC)
# ---------------------------------------------------------------------------

def test_v1_5_column_layout(adapter):
  """v1.5 RF rows are 11 columns; freq / phase sit at indices 8 / 9,
  not 5 / 6 as in v1.4. The string ``use`` column lives at index 10.
  Same column-drift story for ADC: v1.5 puts freq / phase at 5 / 6,
  not 3 / 4."""
  shape_id_mag = 1
  shape_id_phase = 2
  shape_library = {
    shape_id_mag: (2, np.array([1.0, 1.0])),
    shape_id_phase: (2, np.array([0.0, 0.0])),
  }
  rf_library = {
    1: {'data': [
      100.0,           # amplitude (T)
      shape_id_mag,    # mag_id
      shape_id_phase,  # phase_id
      0.0,             # time_shape_id
      0.5e-3,          # center (s)
      1e-4,            # delay (s)
      0.01,            # freq_ppm
      0.02,            # phase_ppm
      500.0,           # freq (Hz)   <- v1.5 picks this up at index 8
      0.5,             # phase (rad) <- v1.5 picks this up at index 9
      'e',             # use (single-char enum)
    ]},
  }
  rf = adapter.read_RF(rf_library, shape_library, dt_rf=1e-6,
                       idx=1, pulseq_version=adapter.Version(1, 5, 0))
  assert rf.df == pytest.approx(500.0)
  assert rf.use == 'excitation'
  assert rf.freq_ppm == pytest.approx(0.01)
  assert rf.phase_ppm == pytest.approx(0.02)

  adc_library = {
    1: {'data': [
      32.0, 1e-6, 1e-5,
      0.03, 0.04,
      700.0, 1.2,
      0.0,
    ]},
  }
  adc = adapter.read_ADC(adc_library, idx=1,
                         pulseq_version=adapter.Version(1, 5, 0))
  assert adc.df == pytest.approx(700.0)
  assert adc.phase == pytest.approx(1.2)
  assert adc.freq_ppm == pytest.approx(0.03)
  assert adc.phase_ppm == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# LABELSET-driven filter API
# ---------------------------------------------------------------------------

def _build_labelset_seq(tmp_path):
  """Build a tiny pp.Sequence with 4 blocks tagged via make_label SET.

  Layout:
    block 0: trapezoid + SET=0      (prep)
    block 1: trapezoid + SET=1      (excitation)
    block 2: trapezoid + SET=2      (readout)
    block 3: trapezoid (no label)   (carries SET=2 by inheritance)
  """
  pp = pytest.importorskip('pypulseq')
  system = pp.Opts(max_grad=10, grad_unit='mT/m',
                   max_slew=80, slew_unit='T/m/s')
  seq = pp.Sequence(system)
  g = pp.make_trapezoid(channel='x', system=system,
                        amplitude=5e-3, flat_time=200e-6)
  seq.add_block(g, pp.make_label(label='SET', type='SET', value=0))
  seq.add_block(g, pp.make_label(label='SET', type='SET', value=1))
  seq.add_block(g, pp.make_label(label='SET', type='SET', value=2))
  seq.add_block(g)
  out = tmp_path / 'labelset_minimal.seq'
  seq.write(str(out))
  return out


def test_labelset_filter(adapter, tmp_path):
  """SET labels are LABELSET counters; only the *first* block of each
  group needs the tag because the value carries forward. After import,
  ``block_labels`` shows the running state and ``filter_blocks`` returns
  matching block indices."""
  pytest.importorskip('pypulseq')
  seq_path = _build_labelset_seq(tmp_path)
  imp = adapter.import_pulseq(seq_path)

  assert len(imp.block_labels) == 4
  assert imp.block_labels[0] == {'SET': 0}
  assert imp.block_labels[1] == {'SET': 1}
  assert imp.block_labels[2] == {'SET': 2}
  # Block 3 inherits the running SET=2 from block 2 without a label
  # extension of its own.
  assert imp.block_labels[3] == {'SET': 2}

  assert imp.filter_blocks(SET=0) == [0]
  assert imp.filter_blocks(SET=1) == [1]
  assert imp.filter_blocks(SET=2) == [2, 3]
  assert imp.filter_blocks(SET=99) == []
  # No-kwargs returns every block.
  assert imp.filter_blocks() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Trajectory data bridge: kspace_to_signal_inputs(pp_seq)
# ---------------------------------------------------------------------------

def test_kspace_to_signal_inputs(adapter, tmp_path):
  """kspace_to_signal_inputs takes a pp.Sequence directly, calls
  calculate_kspace, and returns (pts, times) ready for mri_signal:
  3-tuple of C-contiguous rank-3 float32 arrays in 1/m for k-space and
  one matching rank-3 array in ms for times."""
  pp = pytest.importorskip('pypulseq')
  system = pp.Opts(max_grad=30, grad_unit='mT/m',
                   max_slew=150, slew_unit='T/m/s',
                   rf_dead_time=10e-6, rf_ringdown_time=10e-6)
  seq = pp.Sequence(system)
  rf, gz, _ = pp.make_sinc_pulse(
    flip_angle=np.deg2rad(15), system=system,
    duration=1e-3, slice_thickness=5e-3,
    apodization=0.5, time_bw_product=4, return_gz=True,
    delay=system.rf_dead_time,
  )
  # area + duration: pypulseq derives amplitude / rise_time within
  # the supplied window.
  gx = pp.make_trapezoid(channel='x', system=system,
                         area=200.0, duration=600e-6)
  adc = pp.make_adc(num_samples=16, duration=400e-6,
                    delay=gx.rise_time + 50e-6)
  seq.add_block(rf, gz)
  seq.add_block(gx, adc)
  out = tmp_path / 'bridge_minimal.seq'
  seq.write(str(out))

  pp_seq = pp.Sequence()
  pp_seq.read(str(out), detect_rf_use=False)
  pts, times = adapter.kspace_to_signal_inputs(pp_seq)

  assert len(pts) == 3
  for axis_arr in pts:
    assert axis_arr.dtype == np.float32
    assert axis_arr.flags['C_CONTIGUOUS']
    assert axis_arr.shape == (16, 1, 1)
    assert np.all(np.isfinite(axis_arr))
  assert times.dtype == np.float32
  assert times.flags['C_CONTIGUOUS']
  assert times.shape == (16, 1, 1)

  # Cross-check: the bridge multiplied seconds by 1e3 to get ms.
  k_traj_adc, _kf, _te, _tr, t_adc = pp_seq.calculate_kspace()
  np.testing.assert_allclose(times.reshape(-1), t_adc * 1e3,
                             rtol=1e-6, atol=1e-9)
  np.testing.assert_allclose(pts[0].reshape(-1), k_traj_adc[0],
                             rtol=1e-6, atol=1e-9)


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
