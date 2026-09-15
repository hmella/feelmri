import numpy as np
import pytest
from pathlib import Path
from pint import Quantity as Q_

# One physical specification, written twice: once with feelmri's own sequence
# primitives, once with pypulseq and then round-tripped through a .seq file.
# The two must simulate the same. That is the claim the whole adapter rests on,
# and nothing else in the suite tests it: tests/test_pulseq_timing.py checks
# the imported sequence against pypulseq's own reading of the same file, which
# cannot catch an error the two share.

pytestmark = pytest.mark.pulseq

FLIP_DEG = 30.0
RF_DUR_MS = 0.5
RISE_MS = 0.2
FLAT_MS = 1.0
GX_MTM, GY_MTM, GZ_MTM = 10.0, 4.0, 6.0
N_ADC = 16
DELAY_MS = 2.0
N_REP = 2

# The RF amplitude is written with six significant digits ('{:12g}'), which is
# the largest source of disagreement between the two constructions.
RF_WRITE_REL = 1e-5


def _scanner():
  from feelmri.MRObjects import Scanner
  return Scanner()


def _trapezoid_corners():
  return np.array([0.0, RISE_MS, RISE_MS + FLAT_MS, 2 * RISE_MS + FLAT_MS])


def _native_sequence(scanner):
  """The specification in feelmri primitives."""
  from feelmri.Bloch import Sequence, SequenceBlock
  from feelmri.MRObjects import RF, Gradient

  gamma = scanner.gamma.m_as('rad/ms/mT')
  # flip = gamma * B1 * duration for a constant-amplitude pulse.
  b1 = np.deg2rad(FLIP_DEG) / (gamma * RF_DUR_MS)
  corners = _trapezoid_corners()

  seq = Sequence()
  for rep in range(N_REP):
    rf = RF(scanner=scanner, shape='custom',
            flip_angle=Q_(np.deg2rad(FLIP_DEG), 'rad'),
            dur=Q_(RF_DUR_MS, 'ms'), ref=Q_(0.0, 'ms'), time=Q_(0.0, 'ms'),
            timings=Q_(np.array([0.0, RF_DUR_MS]), 'ms'),
            waveform=Q_(np.array([b1, b1], dtype=complex), 'mT'))
    seq.add_block(SequenceBlock(rf_pulses=[rf], dur=Q_(RF_DUR_MS, 'ms')))

    gradients = []
    for axis, amp in enumerate((GX_MTM, GY_MTM * (rep + 1), GZ_MTM)):
      gradients.append(Gradient(
          timings=Q_(corners, 'ms'),
          amplitudes=Q_(np.array([0.0, amp, amp, 0.0]), 'mT/m'),
          scanner=scanner, ref=Q_(0.0, 'ms'), time=Q_(0.0, 'ms'), axis=axis))
    seq.add_block(SequenceBlock(gradients=gradients,
                                dur=Q_(2 * RISE_MS + FLAT_MS, 'ms')))
    seq.add_block(Q_(DELAY_MS, 'ms'))
  return seq


def _write_pulseq(path, scanner):
  """The same specification in pypulseq."""
  pp = pytest.importorskip('pypulseq')
  system = pp.Opts(max_grad=40, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s',
                   rf_ringdown_time=0, rf_dead_time=0, adc_dead_time=0)
  # Amplitudes go in as Hz/m. Convert with the scanner's gammabar, which is
  # the constant the adapter reads them back with, so both sides mean the same
  # T/m instead of differing by the 9.4e-5 between the two gyromagnetic ratios.
  hz_per_mtm = scanner.gammabar.m_as('Hz/T') * 1e-3

  seq = pp.Sequence(system=system)
  for rep in range(N_REP):
    rf = pp.make_block_pulse(flip_angle=np.deg2rad(FLIP_DEG),
                             duration=RF_DUR_MS * 1e-3, system=system,
                             use='excitation')
    ramp = dict(rise_time=RISE_MS * 1e-3, flat_time=FLAT_MS * 1e-3,
                fall_time=RISE_MS * 1e-3, system=system)
    gx = pp.make_trapezoid('x', amplitude=GX_MTM * hz_per_mtm, **ramp)
    gy = pp.make_trapezoid('y', amplitude=GY_MTM * (rep + 1) * hz_per_mtm, **ramp)
    gz = pp.make_trapezoid('z', amplitude=GZ_MTM * hz_per_mtm, **ramp)
    adc = pp.make_adc(N_ADC, duration=FLAT_MS * 1e-3, delay=RISE_MS * 1e-3,
                      system=system)
    seq.add_block(rf)
    seq.add_block(gx, gy, gz, adc)
    seq.add_block(pp.make_delay(DELAY_MS * 1e-3))

  ok, errors = seq.check_timing()
  assert ok, f'the reference sequence is itself malformed: {errors}'
  seq.write(str(path))
  return seq


@pytest.fixture(scope='module')
def pair(tmp_path_factory):
  from feelmri.PulseqAdapter import import_pulseq
  scanner = _scanner()
  path = tmp_path_factory.mktemp('equiv') / 'equivalence.seq'
  _write_pulseq(path, scanner)
  return scanner, _native_sequence(scanner), import_pulseq(path, scanner=scanner)


def test_block_structure_matches(pair):
  _scan, native, imp = pair
  got = imp.feelmri_seq
  assert len(got.blocks) == len(native.blocks)
  assert abs(got.dur.m_as('ms') - native.dur.m_as('ms')) < 1e-9
  for i, (a, b) in enumerate(zip(native.blocks, got.blocks)):
    assert abs(a.dur.m_as('ms') - b.dur.m_as('ms')) < 1e-9, f'block {i} duration'
    assert abs(a.time_extent[0].m_as('ms')
               - b.time_extent[0].m_as('ms')) < 1e-9, f'block {i} start'
    assert len(a.gradients) == len(b.gradients), f'block {i} gradient count'
    assert len(a.rf_pulses) == len(b.rf_pulses), f'block {i} RF count'


def test_fields_match(pair):
  """The B1 and gradient waveforms the two solvers would integrate, sampled on
  the union of both blocks' rasters."""
  _scan, native, imp = pair
  worst_rf = worst_g = 0.0
  peak_g = 1.0
  for a, b in zip(native.blocks, imp.feelmri_seq.blocks):
    t = np.union1d(a.discrete_times.m_as('ms'), b.discrete_times.m_as('ms'))
    lo, hi = b.time_extent[0].m_as('ms'), b.time_extent[1].m_as('ms')
    t = t[(t >= lo) & (t <= hi)]
    rf_a, G_a, _ = a(t)
    rf_b, G_b, _ = b(t)
    worst_rf = max(worst_rf, float(np.abs(np.asarray(rf_a) - np.asarray(rf_b)).max()))
    for axis in range(3):
      ga = np.broadcast_to(np.asarray(G_a[axis], dtype=float), t.shape)
      gb = np.broadcast_to(np.asarray(G_b[axis], dtype=float), t.shape)
      worst_g = max(worst_g, float(np.abs(ga - gb).max()))
      peak_g = max(peak_g, float(np.abs(gb).max()))

  b1_peak = np.deg2rad(FLIP_DEG) / (_scan.gamma.m_as('rad/ms/mT') * RF_DUR_MS)
  assert worst_rf < RF_WRITE_REL * b1_peak, f'B1 differs by {worst_rf:.4g} mT'
  # Gradients are integers of Hz/m in the file, so these agree to round-off.
  assert worst_g < 1e-9 * peak_g, f'gradient differs by {worst_g:.4g} mT/m'


def _phantom(tmp_path):
  from feelmri.Phantom import FEMPhantom
  import sys
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  from _phantom_fixtures import make_minimal_tet_mesh
  mesh = tmp_path / 'equiv_tet.vtu'
  make_minimal_tet_mesh(mesh)
  phantom = FEMPhantom(path=str(mesh))
  phantom.set_assembler(voxel_size=5e-3, lorder=1, horder=2,
                        nodal_approximation=True, lumped=True)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 100.0, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))
  return phantom


def _solve(sequence, phantom, scanner):
  from feelmri.Bloch import BlochSolver
  solver = BlochSolver(sequence=sequence, phantom=phantom, scanner=scanner,
                       T1=Q_(1000.0, 'ms'), T2=Q_(100.0, 'ms'),
                       perfect_spoiling=False, dtype='float64')
  return solver.solve()


def test_magnetization_matches(pair, tmp_path):
  """The claim the adapter exists to support: a sequence taken through a .seq
  file evolves the magnetization the same way as the one built directly."""
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  try:
    from feelmri.Phantom import FEMPhantom  # noqa: F401
  except ImportError as exc:
    pytest.skip(f'feelmri C++ extensions not available: {exc}')

  scanner, native, imp = pair
  phantom = _phantom(tmp_path)

  # Flag the same blocks on both so the returned columns line up.
  native = native.copy()
  imported = imp.feelmri_seq.copy()
  for a, b in zip(native.blocks, imported.blocks):
    a.store_magnetization = True
    b.store_magnetization = True

  Mxy_n, Mz_n = _solve(native, phantom, scanner)
  Mxy_i, Mz_i = _solve(imported, phantom, scanner)

  assert Mxy_n.shape == Mxy_i.shape
  assert np.all(np.isfinite(Mxy_i)) and np.all(np.isfinite(Mz_i))
  scale = max(float(np.abs(Mxy_n).max()), 1e-12)
  rel_xy = float(np.abs(Mxy_n - Mxy_i).max()) / scale
  rel_z = float(np.abs(Mz_n - Mz_i).max()) / max(float(np.abs(Mz_n).max()), 1e-12)
  # Measured residual is 1.3e-6, set by the six significant digits the RF
  # amplitude is written with. Measured sensitivity, so do not loosen this
  # without re-deriving it: a 0.01% gradient amplitude error reads 9e-3, a
  # 0.1% one 8.9e-2, and removing the trapezoid ramps (an 11% area error, the
  # shape of the v1.5 first/last bug) reads 1.3. A rigid half-raster time
  # shift of a whole gradient is the weak case at 7e-5, because it barely
  # moves the end-of-block state; test_pulseq_timing.py catches that one
  # directly on the waveform instead.
  assert rel_xy < 1e-5, f'Mxy differs by {rel_xy:.3e} relative'
  assert rel_z < 1e-5, f'Mz differs by {rel_z:.3e} relative'


def test_assembled_signal_matches(pair, tmp_path):
  """Closing the loop: the same trajectory driven by either magnetization has
  to give the same k-space."""
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  try:
    from feelmri.Phantom import FEMPhantom  # noqa: F401
  except ImportError as exc:
    pytest.skip(f'feelmri C++ extensions not available: {exc}')
  from feelmri.PulseqAdapter import _reshape_signal_inputs

  scanner, native, imp = pair
  phantom = _phantom(tmp_path)

  native = native.copy()
  imported = imp.feelmri_seq.copy()
  for rw in imp.readouts:
    assert rw.m_storage_idx >= 0
  # The import already flags its anchors; mirror them onto the native copy.
  for a, b in zip(native.blocks, imported.blocks):
    a.store_magnetization = b.store_magnetization

  Mxy_n, _ = _solve(native, phantom, scanner)
  Mxy_i, _ = _solve(imported, phantom, scanner)

  assert imp.readouts
  for rw in imp.readouts:
    points, times = _reshape_signal_inputs(
        rw.kspace[:, 0], rw.kspace[:, 1], rw.kspace[:, 2], rw.times, None)
    phantom.update_magnetization(Mxy_n[:, rw.m_storage_idx])
    k_native = phantom.mri_signal(list(points), times, None)
    phantom.update_magnetization(Mxy_i[:, rw.m_storage_idx])
    k_imported = phantom.mri_signal(list(points), times, None)

    assert k_native.shape == k_imported.shape == (rw.times.size, 1, 1, 1)
    assert np.all(np.isfinite(k_imported))
    scale = max(float(np.abs(k_native).max()), 1e-30)
    rel = float(np.abs(k_native - k_imported).max()) / scale
    assert rel < 1e-5, f'k-space differs by {rel:.3e} relative'
