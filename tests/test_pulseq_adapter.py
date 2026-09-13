"""Tests for feelmri.PulseqAdapter: the API, not the waveforms.

What the reader makes of a file is compared against pypulseq in
``test_pulseq_timing.py``, and the structural invariants of an import live in
``test_pulseq_invariants.py``. This file covers what neither does:

  * the dual-path partition API -- prep vs ADC indices, readout window
    contiguity, ``m_storage_idx`` ordering -- over every bundled fixture;
  * the placeholder-substitution sequence (``feelmri_sim_seq``);
  * parser paths no bundled file exercises: the ROTATIONS extension, the v1.5
    column layout, a cyclic or dangling extension chain, an unhandled section;
  * LABELSET / LABELINC retrieval on a synthetic four-block sequence;
  * the end-to-end ``simulate_pulseq`` path, including that ADC demodulation
    actually reaches the signal.
"""
from pathlib import Path

import numpy as np
import pytest
from pint import Quantity

from conftest import (DATA_DIR, EXAMPLES_SEQ_DIR, SEQ_FILES, seq_ids,
                      skip_if_pypulseq_too_old)


# The whole module exercises the PulseqAdapter, whose end-to-end paths
# (kspace_trajectory, calculate_kspace bridge) require the optional
# 'pypulseq' package. CI runs with '-m "not pulseq"' so the file is
# deselected when pypulseq is unavailable. Existing
# pytest.importorskip('pypulseq') calls inside individual tests remain
# as a second line of defense for developers running the file directly.
pytestmark = pytest.mark.pulseq


ROTATION_SEQ = Path(__file__).resolve().parent / 'data' / 'rotation_minimal.seq'


@pytest.fixture(scope='session')
def adapter():
  """Import the adapter once per session."""
  from feelmri import PulseqAdapter
  return PulseqAdapter


def _imp(pulseq_import, seq_path):
  """The parsed import for one fixture, skipping when the installed pypulseq
  cannot read that file's format.

  The fixture list is conftest's SEQ_FILES: all fourteen files under
  tests/data plus the one under examples/pulseq. This file used to glob
  examples/pulseq alone, which holds a single sequence, so every test here
  described as running "over every .seq" was in fact running over one.
  """
  try:
    return pulseq_import(seq_path)
  except RuntimeError:
    skip_if_pypulseq_too_old(seq_path)
    pytest.skip(f'{seq_path.name} could not be parsed')
# ---------------------------------------------------------------------------
# Dual-path partition API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=seq_ids(SEQ_FILES))
def test_import_pulseq_partitions_blocks(pulseq_import, seq_path):
  imp = _imp(pulseq_import, seq_path)
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
def test_kspace_trajectory_matches_the_windows(pulseq_import):
  """``kspace_trajectory`` is the flat back-compat view of the same samples
  the readout windows carry, and it short-circuits when there is no ADC.

  This is the one caller of that wrapper. test_readout_anchor_invariant
  compares ``rw.kspace_file`` against ``calculate_kspace`` directly and never
  goes through it, so deleting the old shape-only test left the function
  uncovered -- caught by the coverage gate, not by a failure.
  """
  from feelmri.PulseqAdapter import kspace_trajectory

  # Through _imp, not pulseq_import directly: gre_v15 is a v1.5 file with an
  # ADC, which pypulseq 1.4 cannot read at all, and the trajectory step raises
  # RuntimeError there. Every other test in this file is gated the same way.
  imp = _imp(pulseq_import, DATA_DIR / 'gre_v15.seq')
  traj = kspace_trajectory(imp.pulseq_seq)
  expected = np.concatenate([rw.kspace_file for rw in imp.readouts])
  assert traj['times'].shape == expected[:, 0].shape
  assert np.all(np.diff(traj['times']) >= -1e-9), 'times must not run backwards'
  for axis, key in enumerate(('kx', 'ky', 'kz')):
    np.testing.assert_allclose(traj[key], expected[:, axis], atol=1e-3)

  # A sequence with no ADC has no trajectory, and must say so rather than
  # calling into pypulseq.
  empty = kspace_trajectory(_imp(pulseq_import, DATA_DIR / 'flash_tr_v15.seq').pulseq_seq)
  for key in ('kx', 'ky', 'kz', 'times'):
    assert empty[key].size == 0


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


def test_rotation_preserves_moments_across_unequal_timings(adapter):
  """A rotation must transform the three gradient MOMENTS as a vector.

  Integration is linear and the matrix is time-independent, so
  `moment(rotated)[i] == (R @ moment(original))[i]` exactly, whatever the
  three waveforms look like. That is the whole contract, and it is checkable
  without a reference implementation.

  The case that matters is unequal timings, which no bundled fixture has:
  `rotation_minimal.seq` drives a single axis, so it cannot see this. The
  rotated amplitudes used to inherit one donor axis's geometry, and on an
  ordinary block -- in-plane prephasers 0.5 ms flat against a 2.0 ms
  slice-select -- an IDENTITY matrix inflated the x and y moments by 250%.
  """
  import numpy as np
  from feelmri.PulseqAdapter import _grad_corners_seconds

  def moment(g):
    t, a = _grad_corners_seconds(g)
    return float(np.trapezoid(a, t))

  theta = np.radians(30.0)
  c, s_ = np.cos(theta), np.sin(theta)
  rot = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

  trap = lambda a, flat, delay=0.0: adapter.Grad(
      A=a, T=flat, rise=0.1, fall=0.1, delay=delay, first=0.0, last=0.0)

  cases = {
    'three trapezoids, unequal flat tops':
        (trap(5.0, 0.5), trap(3.0, 0.5), trap(20.0, 2.0)),
    'a trapezoid mixed with a shaped gradient':
        (trap(5.0, 0.5),
         adapter.Grad(A=np.sin(np.linspace(0, np.pi, 64)) * 8.0, T=1.0,
                      rise=0.05, fall=0.05, delay=0.2, first=0.0, last=0.0),
         trap(0.0, 0.0)),
  }
  for name, grads in cases.items():
    before = np.array([moment(g) for g in grads])
    for matrix in (np.eye(3), rot):
      after = np.array([moment(g)
                        for g in adapter._apply_rotation_to_grads(matrix, *grads)])
      expected = matrix @ before
      assert np.allclose(after, expected, rtol=1e-9, atol=1e-12), (
        f'{name}: moments {after} != R @ {before} = {expected}')

  # Axes that already share timing keep the exact scalar path -- no
  # resampling, no shaped output.
  same = (trap(5.0, 0.5), trap(3.0, 0.5), trap(2.0, 0.5))
  out = adapter._apply_rotation_to_grads(rot, *same)
  assert all(not isinstance(g.A, np.ndarray) for g in out)


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

@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=seq_ids(SEQ_FILES))
def test_feelmri_sim_seq_default_matches_seq(pulseq_import, seq_path):
  """With the default ``readout_set_values=(3,)`` the simulation
  sequence has identical block count and absolute duration as
  ``feelmri_seq``. Per-block durations match exactly so the global
  timing grid is preserved across the substitution."""
  imp = _imp(pulseq_import, seq_path)
  assert len(imp.feelmri_sim_seq.blocks) == len(imp.feelmri_seq.blocks)
  durs_o = np.array([b.dur.m_as('ms') for b in imp.feelmri_seq.blocks])
  durs_s = np.array([b.dur.m_as('ms') for b in imp.feelmri_sim_seq.blocks])
  np.testing.assert_allclose(durs_s, durs_o, rtol=0, atol=1e-9)
  np.testing.assert_allclose(
    imp.feelmri_sim_seq.dur.m_as('ms'),
    imp.feelmri_seq.dur.m_as('ms'),
    rtol=0, atol=1e-9,
  )
def test_feelmri_sim_seq_collapses_set3(adapter):
  """End-to-end: every SET=3 block on the bundled EPI tagging file is
  empty (no RF, no gradients, no ADC) on ``feelmri_sim_seq``; all
  non-readout blocks keep their event content via deep copy."""
  epi = EXAMPLES_SEQ_DIR / 'epi_pypulseq.seq'
  if not epi.exists():
    pytest.skip(f'{epi.name} not bundled')
  skip_if_pypulseq_too_old(epi)
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
  epi = EXAMPLES_SEQ_DIR / 'epi_pypulseq.seq'
  if not epi.exists():
    pytest.skip(f'{epi.name} not bundled')
  skip_if_pypulseq_too_old(epi)
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


def test_dual_path_multi_window(adapter, tmp_path):
  """End-to-end dual-path smoke test over SEVERAL readout windows.

  Was written against `gre_radial_pypulseq.seq`, which is not in the repo, so
  it skipped on every run since it was added -- dead coverage. Repointed at
  `cpmg_v15.seq`, whose four echoes give four windows and therefore exercise
  the same thing the radial file was chosen for: the per-window
  update_magnetization + mri_signal loop, with one m_storage_idx per window.

  Steps (single Sequence, single solver.solve(), per-readout signal
  assembly):
    1. Parse the .seq via import_pulseq -> partitioned view with one
       ReadoutWindow per echo.
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

  # Not from `parsed_imports`: that fixture globs examples/pulseq, which holds
  # only the EPI file. This one lives in tests/data.
  seq_path = DATA_DIR / 'cpmg_v15.seq'
  if not seq_path.exists():
    pytest.skip('run tests/data/generate_seq_fixtures.py to build cpmg_v15.seq')
  skip_if_pypulseq_too_old(seq_path)

  mesh_path = tmp_path / 'minimal_tet.vtu'
  _write_minimal_tet_mesh(mesh_path)
  phantom = FEMPhantom(path=str(mesh_path))

  imp = adapter.import_pulseq(seq_path)
  assert len(imp.readouts) > 1, 'expected several readout windows'
  # The CPMG fixture plays no gradients at all, so k is constant within each
  # window and the windows differ only in time -- which is what makes the
  # per-window m_storage_idx bookkeeping the thing under test here.
  for rw in imp.readouts:
    span = float(rw.kspace.max() - rw.kspace.min())
    assert span < 1e-3, f'expected k constant within readout; got span {span:g}'
    assert rw.m_storage_idx >= 0, 'every window needs a coherence anchor'

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
  # Named, not SEQ_FILES[0]. The list moved from examples/pulseq to
  # tests/data, which silently changed [0] from epi_pypulseq (115 blocks with
  # a scalar gx) to arb_v15 (0 of 8 -- every gx is an array or zero), and the
  # loop below then skipped every block without asserting anything.
  seq_path = DATA_DIR / 'gre_v15.seq'
  skip_if_pypulseq_too_old(seq_path)
  scanner = Scanner(field_strength=Quantity(3.0, 'T'))
  default = adapter.read_seq(str(seq_path))
  matched = adapter.read_seq(str(seq_path),
                             gamma=scanner.gammabar.m_as('Hz/T'))
  ratio = adapter.GAMMA / scanner.gammabar.m_as('Hz/T')
  checked = 0
  for (gx_d, _, _), (gx_m, _, _) in zip(default.GR, matched.GR):
    if isinstance(gx_d.A, np.ndarray) or gx_d.A == 0.0:
      continue
    checked += 1
    assert gx_m.A == pytest.approx(gx_d.A * ratio, rel=1e-12)
    break

  assert checked > 0, (
    f'{seq_path.name} has no block with a scalar non-zero gx, so the '
    f'assertion above never ran and this test proved nothing')


def test_pulseq_import_disables_perfect_spoiling(adapter):
  """A .seq file spells out its own spoilers, so BlochSolver must not zero
  Mxy between blocks on top of them. The flag rides on the Sequence and is
  resolved when perfect_spoiling is left at its None default."""
  skip_if_pypulseq_too_old(SEQ_FILES[0])
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
  skip_if_pypulseq_too_old(src)
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
  skip_if_pypulseq_too_old(seq_path)

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


def test_read_seq_feelmri_round_trip_and_pass_through(adapter):
  """The back-compat wrapper returns the same objects import_pulseq builds, and
  forwards scanner= / validate=.

  It is in __all__ and had no coverage at all; it also could not reach either
  keyword, so a caller stuck on it had no way to say what field the file was
  written for -- and the gamma the file is read with must be the gamma the
  solver integrates.
  """
  from feelmri.MRObjects import Scanner

  seq_path = DATA_DIR / 'gre_v15.seq'
  skip_if_pypulseq_too_old(seq_path)

  feelmri_seq, pulseq_seq = adapter.read_seq_feelmri(seq_path)
  imp = adapter.import_pulseq(seq_path)
  assert len(feelmri_seq.blocks) == len(imp.feelmri_seq.blocks)
  assert feelmri_seq.dur.m_as('ms') == pytest.approx(imp.feelmri_seq.dur.m_as('ms'))
  assert len(pulseq_seq) == len(imp.pulseq_seq)

  # scanner= reaches the reader. ppm offsets are a fraction of the Larmor
  # frequency and the file records no B0, so they scale with field_strength --
  # which is exactly why the argument has to be reachable from here.
  ppm_path = DATA_DIR / 'ppm_v15.seq'
  skip_if_pypulseq_too_old(ppm_path)

  def first_rf_offset_hz(scanner):
    seq, _ = adapter.read_seq_feelmri(ppm_path, scanner=scanner)
    for block in seq.blocks:
      if block.rf_pulses:
        return float(block.rf_pulses[0].frequency_offset.m_as('Hz'))
    pytest.skip('no RF in the ppm fixture')

  at_1p5 = first_rf_offset_hz(Scanner(field_strength=Quantity(1.5, 'T')))
  at_3p0 = first_rf_offset_hz(Scanner(field_strength=Quantity(3.0, 'T')))
  assert abs(at_1p5) > 0.0, 'the ppm fixture carries no frequency offset'
  assert at_3p0 == pytest.approx(2.0 * at_1p5, rel=1e-9), (
      f'doubling B0 must double the ppm-derived offset: {at_1p5} -> {at_3p0} Hz')

  # validate= reaches check_timing.
  quiet, _ = adapter.read_seq_feelmri(seq_path, validate=False)
  assert len(quiet.blocks) == len(feelmri_seq.blocks)


def test_adc_demodulation_is_applied_by_simulate_pulseq(adapter, tmp_path):
  """The receiver's frequency/phase offsets must reach the signal.

  The solver never samples the ADC -- the readout is synthesized from the
  trajectory -- so if `simulate_pulseq` does not apply them, nothing does. They
  were parsed onto `ReadoutWindow` and read by no one until 2026-09-10; on
  `ppm_v15` that left 228 deg of phase unapplied, a worst-case per-sample error
  of |1 - e^(i phi)| = 1.99 against a maximum of 2.0.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  try:
    from feelmri.Phantom import FEMPhantom
  except ImportError as exc:
    pytest.skip(f'feelmri C++ extensions not available: {exc}')
  from _phantom_fixtures import make_cube_mesh

  seq_path = DATA_DIR / 'ppm_v15.seq'
  skip_if_pypulseq_too_old(seq_path)

  path, _vol = make_cube_mesh(tmp_path / 'cube.vtu', 'tetra', n=2, scale=2e-3)
  phantom = FEMPhantom(path=str(path))
  phantom.set_assembler(voxel_size=0.0, lorder=2, horder=2,
                        nodal_approximation=False, lumped=False)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 60.0, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))

  sim = adapter.simulate_pulseq(seq_path, phantom, M0=1.0,
                                T1=Quantity(1e9, 'ms'), T2=Quantity(60.0, 'ms'),
                                dtype='float64')
  rw = sim.imp.readouts[0]

  # The fixture must actually carry a non-trivial demodulation, or this test
  # would pass against a no-op implementation.
  phase = rw.demodulation_phase()
  assert phase.size == rw.times.size
  assert phase.max() - phase.min() > 1.0, (
      'ppm_v15 is expected to carry a per-sample ADC phase shape; '
      f'span is only {phase.max() - phase.min():.4f} rad')

  # Undoing the demodulation must recover the raw integral, so the signal the
  # caller gets is exactly the raw one times exp(-i phase).
  raw = np.asarray(sim.kspace[0]).reshape(-1) * np.exp(1j * phase)
  points, t = adapter._reshape_signal_inputs(
      rw.kspace[:, 0], rw.kspace[:, 1], rw.kspace[:, 2],
      rw.times - rw.t_anchor, None)
  phantom.update_magnetization(sim.Mxy[:, rw.m_storage_idx])
  expect = np.asarray(phantom.mri_signal(list(points), t, None)).reshape(-1)
  assert np.abs(raw - expect).max() <= 1e-6 * np.abs(expect).max()

  # A window with no offsets must be left untouched, bit for bit.
  plain = adapter.import_pulseq(DATA_DIR / 'gre_an_v15.seq').readouts[0]
  assert not np.any(plain.demodulation_phase())
  probe = np.arange(plain.times.size, dtype=np.complex128).reshape(-1, 1, 1, 1)
  assert plain.demodulate(probe) is probe


def test_a_lab_frame_b0_field_shifts_the_readout_but_not_the_trajectory(
        adapter, tmp_path):
  """`simulate_pulseq` must apply the shift and leave `rw.kspace` alone.

  The scanner field displaces where the signal comes from, and the
  reconstruction still grids on the nominal trajectory -- the difference
  between the two IS the geometric distortion, so writing the shift back into
  `rw.kspace` would cancel exactly the effect being modelled.

  The signal is compared against the same readout driven by hand at the shifted
  k, which fixes the shift's sign and scale independently of anything the
  adapter does, and against the unshifted one, which must differ.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  try:
    from feelmri.Phantom import FEMPhantom
  except ImportError as exc:
    pytest.skip(f'feelmri C++ extensions not available: {exc}')
  from feelmri import B0Field, b0_kspace_shift
  from feelmri.Bloch import apply_demodulation
  from feelmri.MRObjects import Scanner
  from _phantom_fixtures import make_cube_mesh

  seq_path = DATA_DIR / 'gre_v15.seq'
  skip_if_pypulseq_too_old(seq_path)

  scanner = Scanner()
  field = B0Field(offset=Quantity(1.5e-3, 'mT'),
                  gradient=Quantity(np.array([5.0e-3, -3.0e-3, 7.0e-3]),
                                    'mT/m'))

  path, _vol = make_cube_mesh(tmp_path / 'cube.vtu', 'tetra', n=2, scale=2e-2)
  phantom = FEMPhantom(path=str(path))
  phantom.set_assembler(voxel_size=0.0, lorder=2, horder=2,
                        nodal_approximation=False, lumped=False)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 1e9, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))

  imp = adapter.import_pulseq(seq_path, scanner=scanner)
  nominal = np.array(imp.readouts[0].kspace, copy=True)

  sim = adapter.simulate_pulseq(seq_path, phantom, scanner=scanner, M0=1.0,
                                T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
                                dtype='float64', b0_field=field)
  rw = sim.imp.readouts[0]
  assert np.array_equal(rw.kspace, nominal), (
      'the shift was written back into the trajectory, so the reconstruction '
      'would grid on the distorted k and see no distortion at all')

  points, t = adapter._reshape_signal_inputs(
      rw.kspace[:, 0], rw.kspace[:, 1], rw.kspace[:, 2],
      rw.times - rw.t_anchor, None)
  phantom.update_magnetization(sim.Mxy[:, rw.m_storage_idx])
  plain = np.asarray(phantom.mri_signal(list(points), t, None)).reshape(-1)

  dk, phase = b0_kspace_shift(field, t, scanner)
  shifted = [np.ascontiguousarray(points[i] + dk[..., i],
                                  dtype=points[i].dtype) for i in range(3)]
  expect = apply_demodulation(
      np.asarray(phantom.mri_signal(shifted, t, None)), phase.reshape(-1))
  expect = rw.demodulate(expect).reshape(-1)

  got = np.asarray(sim.kspace[0]).reshape(-1)
  scale = np.abs(expect).max()
  assert np.abs(got - expect).max() <= 1e-6 * scale, (
      'simulate_pulseq does not reproduce the hand-driven shifted readout')

  # The shift has to matter on this geometry, or the agreement above is vacuous.
  gap = np.abs(got - rw.demodulate(plain.reshape(-1, 1, 1, 1)).reshape(-1)).max()
  assert gap > 1e-2 * scale, (
      f'the field moves the readout by only {gap / scale:.3e} of peak, so this '
      f'sequence cannot show whether the shift was applied')


def test_a_per_node_b0_field_reaches_the_readout_and_leaves_the_phantom_clean(
        adapter, tmp_path):
  """A field no polynomial can carry rides the PHANTOM, not the trajectory.

  `simulate_pulseq` adds the per-node field to whatever `phi_dB0` the caller
  set and hands the gradient to `FEMPhantom.set_b0_gradient`, so the assembler
  evaluates it at the deformed position. Both are temporary: a stale per-node
  gradient left behind is a wrong image with no symptom, and the caller's own
  off-resonance map has to come back untouched.

  The static arm is EXACT -- with nothing moving, the nodal value IS the
  Eulerian answer -- so it is checked against the same field handed to the
  solver as `delta_B` and to the readout as `phi_dB0`, which is the Lagrangian
  spelling that agrees with it only because the phantom is still.

  The moving arm replays the SAME magnetization through the readout with the
  gradient channel dropped, so the solver is identical by construction and the
  difference is the readout channel alone.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  try:
    from feelmri.Phantom import FEMPhantom
  except ImportError as exc:
    pytest.skip(f'feelmri C++ extensions not available: {exc}')
  from feelmri import B0Field
  from feelmri.Motion import POD
  from feelmri.MRObjects import Scanner
  from _phantom_fixtures import make_cube_mesh

  seq_path = DATA_DIR / 'gre_v15.seq'
  skip_if_pypulseq_too_old(seq_path)

  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')
  path, _vol = make_cube_mesh(tmp_path / 'nodal_b0.vtu', 'tetra', n=2,
                              scale=6e-2)

  def fresh():
    ph = FEMPhantom(path=str(path))
    ph.set_assembler(voxel_size=0.0, lorder=2, horder=2,
                     nodal_approximation=False, lumped=False)
    return ph

  # Curved on the scale of the object, so `on_phantom` cannot fit it.
  expr = lambda p: 1.0e-3 * np.sin(p[:, 0] / 0.04) * np.cos(p[:, 1] / 0.05)

  phantom = fresh()
  n = phantom.local_nodes.shape[0]
  T2 = np.full(n, 1e9, dtype=np.float32)
  tissue = np.full(n, 0.013, dtype=np.float32)     # the caller's own map
  phantom.set_static_fields(T2=T2, phi_dB0=tissue)
  field = B0Field.on_phantom(expr, phantom, collective=False)
  assert field.kind == 'nodal'
  nodal_mT = B0Field._sample(expr, B0Field._scanner_nodes(phantom))

  kw = dict(scanner=scanner, M0=1.0, T1=Quantity(1e9, 'ms'),
            T2=Quantity(1e9, 'ms'), dtype='float64')
  sim = adapter.simulate_pulseq(seq_path, phantom, b0_field=field, **kw)

  # The caller's own off-resonance map comes back untouched.
  assert np.allclose(phantom._static_fields[1], tissue), (
      "the per-node field was left on the caller's phi_dB0")

  # And so does the gradient channel: re-running the readout by hand must give
  # the same answer as re-running it after an explicit clear.
  rw = sim.imp.readouts[0]
  pts, t = _readout_inputs(adapter, rw)
  phantom.update_magnetization(sim.Mxy[:, rw.m_storage_idx])
  after = np.asarray(phantom.mri_signal(pts, t, None)).reshape(-1)
  phantom.set_b0_gradient(None)
  assert np.array_equal(
      after, np.asarray(phantom.mri_signal(pts, t, None)).reshape(-1)), (
      'the per-node gradient was left on the assembler')

  # The static arm, against the Lagrangian spelling of the same field.
  ref = fresh()
  ref.set_static_fields(T2=T2,
                        phi_dB0=(tissue + gamma * nodal_mT).astype(np.float32))
  ref_sim = adapter.simulate_pulseq(
      seq_path, ref, delta_B=nodal_mT.reshape(-1, 1), **kw)
  got = np.asarray(sim.kspace[0]).reshape(-1)
  want = np.asarray(ref_sim.kspace[0]).reshape(-1)
  scale = np.abs(want).max()
  assert np.abs(got - want).max() <= 1e-5 * scale, (
      'a static phantom must sample the field exactly at its nodes')

  # Moving: the same field, now sampled where the spins go.
  data = np.zeros((n, 3, 4), dtype=np.float32)
  data[:, 0, :] = 0.012
  data[:, 1, :] = -0.009
  pod = POD(data=data, times=np.linspace(0.0, 60.0, 4), n_modes=1,
            is_periodic=True)

  moved = fresh()
  moved.set_static_fields(T2=T2, phi_dB0=tissue)
  moving = adapter.simulate_pulseq(seq_path, moved, b0_field=field, pod=pod,
                                   **kw)
  mrw = moving.imp.readouts[0]
  mpts, mt = _readout_inputs(adapter, mrw)
  # The frozen description: the same magnetization, the same trajectory, the
  # field written per node and the gradient channel absent.
  moved.set_static_fields(T2=T2,
                          phi_dB0=(tissue + gamma * nodal_mT).astype(np.float32))
  moved.set_b0_gradient(None)
  moved.update_magnetization(moving.Mxy[:, mrw.m_storage_idx])
  frozen = mrw.demodulate(
      np.asarray(moved.mri_signal(mpts, mt, pod))).reshape(-1)
  eulerian = np.asarray(moving.kspace[0]).reshape(-1)
  gap = np.abs(eulerian - frozen).max() / np.abs(frozen).max()
  assert gap > 1e-2, (
      f'following the spins changes the readout by only {gap:.3e}, so the '
      f'per-node gradient is not reaching the assembler')


def _readout_inputs(adapter, rw):
  points, t = adapter._reshape_signal_inputs(
      rw.kspace[:, 0], rw.kspace[:, 1], rw.kspace[:, 2],
      rw.times - rw.t_anchor, None)
  return list(points), t


def test_a_degree_two_b0_field_reaches_the_readout_through_simulate_pulseq(
        adapter, tmp_path):
  """`simulate_pulseq` could not carry a degree-2 field at all.

  The polynomial arm went to `b0_kspace_shift`, which reaches `in_frame` and
  refuses any non-zero quadratic by name -- so a field `B0Field.on_phantom`
  builds happily raised `NotImplementedError` out of the one-call API, with no
  message saying the API was the problem rather than the field. The quadratic
  part rides the six `maxwell` coefficients the same loop already assembles for
  the concomitant term, and the two ADD.

  Checked against the Lagrangian spelling of the same field, which on a
  phantom that does not move is the exact answer: the field on `delta_B` for
  the solver and on `phi_dB0` for the readout.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  try:
    from feelmri.Phantom import FEMPhantom
  except ImportError as exc:
    pytest.skip(f'feelmri C++ extensions not available: {exc}')
  from feelmri import B0Field
  from feelmri.MRObjects import Scanner
  from _phantom_fixtures import make_cube_mesh

  seq_path = DATA_DIR / 'gre_v15.seq'
  skip_if_pypulseq_too_old(seq_path)

  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')
  path, _vol = make_cube_mesh(tmp_path / 'quad_b0.vtu', 'tetra', n=2, scale=6e-2)

  def fresh():
    ph = FEMPhantom(path=str(path))
    # NODAL, because the reference writes the field per node on `phi_dB0`
    # while the channel under test evaluates the monomials where the assembler
    # samples them: on a quadrature path those are genuinely different
    # quantities and the comparison becomes a 1.3e-03 interpolation gap rather
    # than an identity.
    ph.set_assembler(voxel_size=1e3, lorder=2,
                     nodal_approximation=True, lumped=True)
    return ph

  # Degree 2 in every slot, so no coefficient is zero by luck.
  expr = lambda p: 1.0e-3 * (0.3 + 2.0 * p[:, 0] - 1.5 * p[:, 2]
                             + 5.0 * p[:, 0] ** 2 - 4.0 * p[:, 1] ** 2
                             + 2.5 * p[:, 2] ** 2 + 1.7 * p[:, 0] * p[:, 1]
                             - 3.1 * p[:, 0] * p[:, 2] + 2.2 * p[:, 1] * p[:, 2])

  phantom = fresh()
  n = phantom.local_nodes.shape[0]
  T2 = np.full(n, 1e9, dtype=np.float32)
  phantom.set_static_fields(T2=T2, phi_dB0=np.zeros(n, dtype=np.float32))
  field = B0Field.on_phantom(expr, phantom, collective=False)
  assert field.order == 2 and field.kind == 'polynomial'
  nodal_mT = B0Field._sample(expr, B0Field._scanner_nodes(phantom))

  kw = dict(scanner=scanner, M0=1.0, T1=Quantity(1e9, 'ms'),
            T2=Quantity(1e9, 'ms'), dtype='float64')
  got = np.asarray(adapter.simulate_pulseq(seq_path, phantom, b0_field=field,
                                           **kw).kspace[0]).reshape(-1)

  ref = fresh()
  ref.set_static_fields(T2=T2, phi_dB0=(gamma * nodal_mT).astype(np.float32))
  want = np.asarray(adapter.simulate_pulseq(
      seq_path, ref, delta_B=nodal_mT.reshape(-1, 1), **kw).kspace[0]).reshape(-1)

  scale = np.abs(want).max()
  assert np.abs(got - want).max() <= 1e-5 * scale, (
      'the split channels disagree with the per-node field on a static phantom')

  # Not vacuous: the field has to change the readout by far more than the
  # agreement above, or the two arms could both be ignoring it.
  plain = fresh()
  plain.set_static_fields(T2=T2, phi_dB0=np.zeros(n, dtype=np.float32))
  bare = np.asarray(adapter.simulate_pulseq(seq_path, plain, **kw).kspace[0]
                    ).reshape(-1)
  assert np.abs(bare - want).max() > 100.0 * np.abs(got - want).max(), (
      'this field barely changes the readout, so the test cannot see it')
