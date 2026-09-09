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


# The whole module exercises the PulseqAdapter, whose end-to-end paths
# (kspace_trajectory, calculate_kspace bridge) require the optional
# 'pypulseq' package. CI runs with '-m "not pulseq"' so the file is
# deselected when pypulseq is unavailable. Existing
# pytest.importorskip('pypulseq') calls inside individual tests remain
# as a second line of defense for developers running the file directly.
pytestmark = pytest.mark.pulseq


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
  # import_pulseq reads with the scanner's gammabar, not the module default,
  # so that gamma * B in the solver reproduces the file's Hz/m.
  from feelmri.MRObjects import Scanner
  expected = 1_000_000.0 / Scanner().gammabar.m_as('Hz/T')  # T/m

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
# Readout-placeholder substitution: feelmri_sim_seq
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_seq_id)
def test_feelmri_sim_seq_default_matches_seq(parsed_imports, seq_path):
  """With the default ``readout_set_values=(3,)`` the simulation
  sequence has identical block count and absolute duration as
  ``feelmri_seq``. Per-block durations match exactly so the global
  timing grid is preserved across the substitution."""
  imp = parsed_imports[seq_path.name]
  assert len(imp.feelmri_sim_seq.blocks) == len(imp.feelmri_seq.blocks)
  durs_o = np.array([b.dur.m_as('ms') for b in imp.feelmri_seq.blocks])
  durs_s = np.array([b.dur.m_as('ms') for b in imp.feelmri_sim_seq.blocks])
  np.testing.assert_allclose(durs_s, durs_o, rtol=0, atol=1e-9)
  np.testing.assert_allclose(
    imp.feelmri_sim_seq.dur.m_as('ms'),
    imp.feelmri_seq.dur.m_as('ms'),
    rtol=0, atol=1e-9,
  )


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_seq_id)
def test_feelmri_sim_seq_storage_flag_carryover(parsed_imports, seq_path):
  """``store_magnetization`` flags on non-readout blocks survive the
  substitution. Readout-tagged blocks themselves never carry storage
  flags by construction (the anchor block is the preceding RF)."""
  imp = parsed_imports[seq_path.name]
  flags_o = [b.store_magnetization for b in imp.feelmri_seq.blocks]
  flags_s = [b.store_magnetization for b in imp.feelmri_sim_seq.blocks]
  assert flags_o == flags_s


def test_feelmri_sim_seq_collapses_set3(adapter):
  """End-to-end: every SET=3 block on the bundled EPI tagging file is
  empty (no RF, no gradients, no ADC) on ``feelmri_sim_seq``; all
  non-readout blocks keep their event content via deep copy."""
  epi = PULSEQ_DIR / 'epi_pypulseq.seq'
  if not epi.exists():
    pytest.skip(f'{epi.name} not bundled')
  imp = adapter.import_pulseq(epi)
  set3 = set(imp.filter_blocks(SET=3))
  assert set(imp.readout_sim_block_indices) == set3
  assert len(set3) > 0
  for i in set3:
    b = imp.feelmri_sim_seq.blocks[i]
    assert b.empty is True
    assert b.rf_pulses == []
    assert b.gradients == []
    assert b.adc is None
  # Non-readout SET=2 blocks must still carry their RF / gradient events.
  for i in imp.filter_blocks(SET=2):
    assert imp.feelmri_sim_seq.blocks[i].empty is False


def test_feelmri_sim_seq_optout(adapter):
  """``readout_set_values=()`` disables the substitution; sim_seq
  becomes an event-for-event mirror of feelmri_seq."""
  epi = PULSEQ_DIR / 'epi_pypulseq.seq'
  if not epi.exists():
    pytest.skip(f'{epi.name} not bundled')
  imp = adapter.import_pulseq(epi, readout_set_values=())
  assert imp.readout_sim_block_indices == []
  durs_o = np.array([b.dur.m_as('ms') for b in imp.feelmri_seq.blocks])
  durs_s = np.array([b.dur.m_as('ms') for b in imp.feelmri_sim_seq.blocks])
  np.testing.assert_allclose(durs_s, durs_o, rtol=0, atol=1e-9)
  # No block should have been collapsed to empty when opt-out is requested.
  for b_o, b_s in zip(imp.feelmri_seq.blocks, imp.feelmri_sim_seq.blocks):
    assert b_o.empty == b_s.empty


def test_feelmri_sim_seq_custom_set_values(adapter, tmp_path):
  """The ``readout_set_values`` knob accepts arbitrary SET integer
  values; the substitution targets only the requested values on a
  synthetic fixture."""
  seq_path = _build_labelset_seq(tmp_path)
  imp = adapter.import_pulseq(seq_path, readout_set_values=(2,))
  # Blocks 2 and 3 both carry running SET=2; both should be collapsed.
  assert imp.readout_sim_block_indices == [2, 3]
  for i in (2, 3):
    assert imp.feelmri_sim_seq.blocks[i].empty is True
    assert imp.feelmri_sim_seq.blocks[i].gradients == []
  # Block 0 (SET=0) and block 1 (SET=1) keep their gradient events.
  for i in (0, 1):
    assert imp.feelmri_sim_seq.blocks[i].empty is False
    assert len(imp.feelmri_sim_seq.blocks[i].gradients) > 0


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


# ---------------------------------------------------------------------------
# v1.5 event fields the solver consumes
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / 'data'
PPM_SEQ = DATA_DIR / 'ppm_v15.seq'
ARB_SEQ = DATA_DIR / 'arb_v15.seq'


def _ppm_to_hz():
  from feelmri.MRObjects import Scanner
  s = Scanner()
  return s.gammabar.m_as('Hz/T') * s.field_strength.m_as('T') * 1e-6


def test_rf_ppm_offsets_are_consumed(adapter):
  """freq_ppm and phase_ppm reach the RF's Hz / rad offsets, scaled by the
  Larmor frequency. The fixture's saturation pulse carries -3.3 ppm and
  0.25 rad/MHz."""
  if not PPM_SEQ.exists():
    pytest.skip('run tests/data/generate_seq_fixtures.py to build ppm_v15.seq')
  scale = _ppm_to_hz()
  imp = adapter.import_pulseq(PPM_SEQ)
  sat = [b.rf_pulses[0] for b in imp.feelmri_seq.blocks
         if b.rf_pulses and b.rf_pulses[0].use == 'saturation']
  assert sat, 'fixture has no saturation pulse'
  for rf in sat:
    assert rf.frequency_offset.m_as('Hz') == pytest.approx(-3.3 * scale)
    assert rf.phase_offset.m_as('rad') == pytest.approx(0.25 * scale)


def test_adc_ppm_offsets_and_phase_modulation_are_consumed(adapter):
  """The ADC's ppm offsets are folded in the same way, and the v1.5
  phase_id column resolves to a per-sample phase shape."""
  if not PPM_SEQ.exists():
    pytest.skip('run tests/data/generate_seq_fixtures.py to build ppm_v15.seq')
  scale = _ppm_to_hz()
  imp = adapter.import_pulseq(PPM_SEQ)
  adcs = [b.adc for b in imp.feelmri_seq.blocks if b.adc is not None]
  assert adcs
  for adc in adcs:
    assert adc.freq_offset.m_as('Hz') == pytest.approx(1.5 * scale)
    assert adc.phase_offset.m_as('rad') == pytest.approx(-0.5 * scale)
    assert adc.phase_modulation is not None
    mod = adc.phase_modulation.m_as('rad')
    assert mod.size == adc.times.m.size
    assert mod == pytest.approx(np.linspace(0.0, np.pi, mod.size), abs=1e-5)

  for rw in imp.readouts:
    assert rw.adc_freq_offset == pytest.approx(1.5 * scale)
    assert rw.adc_phase_offset == pytest.approx(-0.5 * scale)
    assert rw.adc_phase_modulation is not None


def test_arbitrary_gradient_carries_boundary_samples(adapter):
  """A v1.5 arbitrary gradient on the regular raster has its samples at
  raster centres; the amplitudes at the block boundaries come from the
  file's first/last columns. Without them the waveform is shifted half a
  raster and starts and ends at the wrong value."""
  if not ARB_SEQ.exists():
    pytest.skip('run tests/data/generate_seq_fixtures.py to build arb_v15.seq')
  ps = adapter.read_seq(str(ARB_SEQ))
  shaped = [g for gr in ps.GR for g in gr
            if isinstance(g.A, np.ndarray) and not isinstance(g.T, np.ndarray)
            and np.any(np.abs(g.A) > 0)]
  assert shaped, 'fixture has no regular-raster arbitrary gradient'
  g = shaped[0]
  # A half raster of shoulder on each side, and boundary values that are the
  # waveform extrapolated back by half a sample.
  assert g.rise == pytest.approx(g.fall)
  assert g.first == pytest.approx(g.A[0] - 0.5 * (g.A[1] - g.A[0]))
  assert g.last == pytest.approx(g.A[-1] + 0.5 * (g.A[-1] - g.A[-2]))

  t_s, a = adapter._shaped_waveform_seconds(g)
  assert t_s.size == g.A.size + 2
  assert t_s[0] == pytest.approx(g.delay)
  assert t_s[-1] == pytest.approx(g.delay + g.rise + g.T + g.fall)
  assert a[0] == pytest.approx(g.first)
  assert a[-1] == pytest.approx(g.last)


def test_import_reads_with_the_scanner_gamma(adapter):
  """The Hz/m in the file must be divided by the same gamma the solver
  multiplies back, or every encoding phase is off by their ratio."""
  from feelmri.MRObjects import Scanner
  seq_path = SEQ_FILES[0]
  scanner = Scanner(field_strength=Quantity(3.0, 'T'))
  default = adapter.read_seq(str(seq_path))
  matched = adapter.read_seq(str(seq_path),
                             gamma=scanner.gammabar.m_as('Hz/T'))
  ratio = adapter.GAMMA / scanner.gammabar.m_as('Hz/T')
  for (gx_d, _, _), (gx_m, _, _) in zip(default.GR, matched.GR):
    if isinstance(gx_d.A, np.ndarray) or gx_d.A == 0.0:
      continue
    assert gx_m.A == pytest.approx(gx_d.A * ratio, rel=1e-12)
    break


def test_pulseq_import_disables_perfect_spoiling(adapter):
  """A .seq file spells out its own spoilers, so BlochSolver must not zero
  Mxy between blocks on top of them. The flag rides on the Sequence and is
  resolved when perfect_spoiling is left at its None default."""
  imp = adapter.import_pulseq(SEQ_FILES[0])
  assert imp.feelmri_seq.explicit_spoiling is True
  assert imp.feelmri_sim_seq.explicit_spoiling is True

  from feelmri.Bloch import Sequence
  assert Sequence().explicit_spoiling is False


# ---------------------------------------------------------------------------
# Sections and extension lists the reader must survive
# ---------------------------------------------------------------------------

def test_unhandled_extension_section_is_consumed(adapter, tmp_path):
  """A section the reader only warns about still has to have its body read.
  Otherwise the next line of that body is taken for a section header and the
  file dies with 'Unknown section code'."""
  src = DATA_DIR / 'gre_v15.seq'
  if not src.exists():
    pytest.skip('run tests/data/generate_seq_fixtures.py to build gre_v15.seq')
  txt = src.read_text()
  assert '# Sequence Shapes' in txt
  injected = 'extension DELAYS 7\n1 100\n2 200\n\n# Sequence Shapes'
  out = tmp_path / 'with_delays.seq'
  out.write_text(txt.replace('# Sequence Shapes', injected, 1))

  ps = adapter.read_seq(str(out))
  assert len(ps) == len(adapter.read_seq(str(src)))

  # Same for a section name the reader has never heard of.
  injected = 'extension NOSUCHTHING 8\n1 0 0\n\n# Sequence Shapes'
  out2 = tmp_path / 'with_unknown.seq'
  out2.write_text(txt.replace('# Sequence Shapes', injected, 1))
  assert len(adapter.read_seq(str(out2))) == len(ps)


def test_cyclic_extension_list_terminates(adapter):
  """The extension list is walked by next_id, which a file is free to make
  cyclic. The walk must stop instead of hanging."""
  extension_library = {
    1: {'data': [1, 1, 2]},
    2: {'data': [1, 2, 1]},   # points back at 1
  }
  extension_type = {1: {'data': 'LABELSET'}}
  labelset_library = {1: {'data': [3, 'SET']}, 2: {'data': [4, 'LIN']}}

  out = adapter.read_extension(extension_library, extension_type, {},
                               labelset_library, {}, idx=1)
  assert [(e.label, e.value) for e in out] == [('SET', 3), ('LIN', 4)]


def test_dangling_extension_reference_is_reported(adapter):
  """A next_id that is not in the library stops the walk rather than raising
  a KeyError out of the parser."""
  extension_library = {1: {'data': [1, 1, 99]}}
  extension_type = {1: {'data': 'LABELSET'}}
  labelset_library = {1: {'data': [3, 'SET']}}
  out = adapter.read_extension(extension_library, extension_type, {},
                               labelset_library, {}, idx=1)
  assert len(out) == 1
  assert adapter.read_extension(extension_library, extension_type, {},
                                labelset_library, {}, idx=42) == []


# ---------------------------------------------------------------------------
# The in-house parser against the pypulseq APIs that cover the same ground
# ---------------------------------------------------------------------------

def test_check_timing_is_reported(adapter, tmp_path):
  """import_pulseq runs check_timing and surfaces what it finds. A file that
  fails it still imports -- the violations are what a scanner would reject,
  not what the simulator cannot handle."""
  src = DATA_DIR / 'gre_v15.seq'
  if not src.exists():
    pytest.skip('run tests/data/generate_seq_fixtures.py to build gre_v15.seq')
  assert adapter.import_pulseq(src).timing_errors == ()

  # Halve one block's stored duration so its events no longer fit.
  lines = src.read_text().splitlines()
  i = lines.index('[BLOCKS]')
  row = lines[i + 3].split()
  row[1] = str(max(1, int(row[1]) // 2))
  lines[i + 3] = ' '.join(row)
  bad = tmp_path / 'short_block.seq'
  bad.write_text('\n'.join(lines) + '\n')

  imp = adapter.import_pulseq(bad)
  assert imp.timing_errors
  assert any('BLOCK_DURATION_MISMATCH' in e for e in imp.timing_errors)
  assert adapter.import_pulseq(bad, validate=False).timing_errors == ()


def test_block_labels_match_evaluate_labels(adapter, tmp_path):
  """_compute_block_labels reimplements what pypulseq's evaluate_labels does.
  They must agree, with one documented difference: pypulseq back-fills a
  label with 0 on the blocks before it first appears, while block_labels
  leaves the key out (filter_blocks never matches a missing label)."""
  pp = pytest.importorskip('pypulseq')
  seq_path = _build_labelset_seq(tmp_path)
  ref = pp.Sequence()
  ref.read(str(seq_path), detect_rf_use=False)
  expected = ref.evaluate_labels(evolution='blocks')
  assert expected, 'fixture carries no labels'

  imp = adapter.import_pulseq(seq_path)
  assert len(imp.block_labels) == len(ref.block_durations)
  for label, values in expected.items():
    values = np.atleast_1d(values)
    got = [state.get(label, 0) for state in imp.block_labels]
    assert got == list(values), f'label {label}: {got} != {list(values)}'


# The v1.5 fixtures alongside the examples: they carry the use labels the
# anchor logic keys on, which the v1.4 example does not.
ANCHOR_SEQ_FILES = sorted(DATA_DIR.glob('*_v15.seq')) + SEQ_FILES


@pytest.mark.parametrize('seq_path', ANCHOR_SEQ_FILES, ids=lambda p: p.stem)
def test_readout_anchors_agree_with_calculate_kspace(adapter, seq_path):
  """_identify_readout_groups picks a coherence anchor from the RF use
  labels; calculate_kspace resets or reflects k at the same pulses. Every
  window's anchor block must therefore hold one of those pulses."""
  pp = pytest.importorskip('pypulseq')
  ref = pp.Sequence()
  ref.read(str(seq_path), detect_rf_use=False)
  _k, _kf, t_exc, t_ref, _t = ref.calculate_kspace()
  anchors = np.sort(np.concatenate([np.atleast_1d(t_exc).ravel(),
                                    np.atleast_1d(t_ref).ravel()]))
  imp = adapter.import_pulseq(seq_path)
  if not imp.readouts:
    pytest.skip('no ADC in this sequence')
  assert anchors.size, 'sequence has readouts but no excitation or refocusing'
  for rw in imp.readouts:
    assert rw.m_storage_block >= 0
    block = imp.feelmri_seq.blocks[rw.m_storage_block]
    t0 = block.time_extent[0].m_as('ms') * 1e-3
    t1 = block.time_extent[1].m_as('ms') * 1e-3
    assert np.any((anchors >= t0 - 1e-12) & (anchors <= t1 + 1e-12)), (
        f'anchor block {rw.m_storage_block} [{t0:.6g}, {t1:.6g}] s holds no '
        f'pulse that calculate_kspace treats as an anchor')


# ---------------------------------------------------------------------------
# One-call simulation
# ---------------------------------------------------------------------------

def test_simulate_pulseq_end_to_end(adapter, tmp_path):
  """simulate_pulseq is the dual-path workflow in one call: import, one
  Bloch pass, then per-readout update_magnetization + mri_signal. Run it on
  a 2-tet phantom and check it lines up with the readout windows."""
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  try:
    from feelmri.MRObjects import Scanner
    from feelmri.Phantom import FEMPhantom
  except ImportError as exc:
    pytest.skip(f'feelmri C++ extensions not available: {exc}')

  seq_path = DATA_DIR / 'gre_v15.seq'
  if not seq_path.exists():
    pytest.skip('run tests/data/generate_seq_fixtures.py to build gre_v15.seq')

  mesh_path = tmp_path / 'minimal_tet.vtu'
  _write_minimal_tet_mesh(mesh_path)
  phantom = FEMPhantom(path=str(mesh_path))
  phantom.set_assembler(voxel_size=5e-3, lorder=1, horder=2,
                        nodal_approximation=True, lumped=True)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 100.0, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))

  sim = adapter.simulate_pulseq(seq_path, phantom, scanner=Scanner())

  assert len(sim.kspace) == len(sim.imp.readouts) > 0
  assert sim.Mxy.shape[0] == n
  for k, t, rw in zip(sim.kspace, sim.times, sim.imp.readouts):
    assert k.shape[:3] == (rw.times.size, 1, 1)
    assert np.all(np.isfinite(k))
    assert t.size == rw.times.size

  # The flattened views must cover exactly the file's ADC samples.
  pp = pytest.importorskip('pypulseq')
  ref = pp.Sequence()
  ref.read(str(seq_path), detect_rf_use=False)
  ref_times = np.sort(ref.adc_times()[0] * 1e3)
  assert sim.kspace_flat.shape[0] == ref_times.size
  assert np.abs(np.sort(sim.times_flat) - ref_times).max() < 1e-6
