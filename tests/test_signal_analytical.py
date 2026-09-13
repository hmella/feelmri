"""Signal-assembly regression checks on minimal phantoms.

The C++ ``SignalAssembler`` (``cpp/feelmri/MRIAssemble.cpp``) computes

  S(k, t) = int_Omega rho(r) * M_xy(r) * exp(-t / T2(r))
                                       * exp(-i * dphi_B0(r) * t)
                                       * exp(-i * 2*pi * k . r) dV.

These tests assert *qualitative* properties of S(k) on small meshes,
not exact quadrature convergence:

* At k = 0, t = 0, with constant ``M_xy = 1`` and no relaxation, the
  signal is a single complex sample whose imaginary part is small
  and whose real part is positive.
* On a 1-D rod, ``|S(k_x)|`` falls off rapidly as ``k_x`` crosses
  ``1/L`` (the first sinc zero of an idealised rod's FT).
* On a 2-D disk, ``|S(k)|`` is monotone-decreasing across the first
  lobe of the disk-FT (Bessel-J1 proxy).

The tests catch wiring regressions in the FE-quadrature pipeline
without depending on the exact mesh volume (the hex-cell tet
decomposition in our test fixtures undercounts volume; what matters
is that the signal *shape* is correct)."""
from __future__ import annotations

import numpy as np
import pytest

from feelmri import FEMPhantom

from _phantom_fixtures import make_1d_rod_mesh, make_2d_disk_mesh


def _to_3d_inputs(kx, ky, kz, t):
  n = kx.size
  shape = (n, 1, 1)
  pts = [
    np.ascontiguousarray(kx.reshape(shape), dtype=np.float32),
    np.ascontiguousarray(ky.reshape(shape), dtype=np.float32),
    np.ascontiguousarray(kz.reshape(shape), dtype=np.float32),
  ]
  return pts, np.ascontiguousarray(t.reshape(shape), dtype=np.float32)


def _build_phantom_at_zero_field(phantom, voxel_size=5e-4):
  phantom.set_assembler(
    voxel_size=voxel_size,
    lorder=1, horder=4,
    nodal_approximation=True, lumped=True,
  )
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(
    T2=np.full(n, 1e9, dtype=np.float32),
    phi_dB0=np.zeros(n, dtype=np.float32),
  )


# ---------------------------------------------------------------------------
# k = 0: real, positive, finite signal
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 1-D rod: |S(k_x)| falls off across k_x = 1/L
# ---------------------------------------------------------------------------

def test_signal_for_1d_rod_falls_off_at_inverse_length(tmp_path):
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')

  L = 4e-3
  mesh_path = tmp_path / 'rod.vtu'
  make_1d_rod_mesh(mesh_path, length=L, n_segments=32,
                   transverse_width=2e-4)
  phantom = FEMPhantom(path=str(mesh_path))
  _build_phantom_at_zero_field(phantom, voxel_size=5e-4)

  n = phantom.local_nodes.shape[0]
  phantom.update_magnetization(np.ones((n, 1), dtype=np.complex64))

  ks = np.array([0.0, 0.5 / L, 1.0 / L], dtype=np.float32)
  pts, t3 = _to_3d_inputs(ks, np.zeros_like(ks), np.zeros_like(ks),
                           np.zeros_like(ks))
  S = np.abs(np.asarray(phantom.mri_signal(pts, t3, None)).reshape(-1))

  assert S[0] > 0.0
  # |S| must drop substantially as kx crosses 1/L (the sinc zero
  # of a uniform-rod FT). Allow generous tolerance because the rod
  # cross-section is finite.
  assert S[2] < 0.30 * S[0], (
    f'|S(1/L)|={S[2]:.3g} did not drop below 30% of |S(0)|={S[0]:.3g}'
  )
  assert S[1] < S[0], 'expected |S(0.5/L)| <= |S(0)|'


# ---------------------------------------------------------------------------
# 2-D disk: |S(k)| monotone in |k|
# ---------------------------------------------------------------------------

def test_signal_for_2d_disk_decays_monotonically_in_kr(tmp_path):
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')

  R = 3e-3
  mesh_path = tmp_path / 'disk.vtu'
  make_2d_disk_mesh(mesh_path, radius=R,
                     n_radial=4, n_angular=12, thickness=2e-4)
  phantom = FEMPhantom(path=str(mesh_path))
  _build_phantom_at_zero_field(phantom, voxel_size=3e-4)

  n = phantom.local_nodes.shape[0]
  phantom.update_magnetization(np.ones((n, 1), dtype=np.complex64))

  kr_band = 1.22 / (2.0 * R)
  ks = np.linspace(0.0, 0.5 * kr_band, 4, dtype=np.float32)
  pts, t3 = _to_3d_inputs(ks, np.zeros_like(ks), np.zeros_like(ks),
                           np.zeros_like(ks))
  S = np.abs(np.asarray(phantom.mri_signal(pts, t3, None)).reshape(-1))

  diffs = np.diff(S)
  assert np.all(diffs <= 1e-3 * S[0]), (
    f'|S(k)| not monotonically decreasing in the first lobe: {S}'
  )
  assert S[0] > 0.0


# ---------------------------------------------------------------------------
# Multi-species: uniform per-species T2 / off-resonance factor out of the integral
# ---------------------------------------------------------------------------

def test_uniform_relaxation_and_offresonance_factor_out(tmp_path):
  """A species whose T2 and off-resonance are spatially uniform can be evaluated
  with them removed from the integral and re-applied afterwards:

      S(k,t) = exp(-t/T2) exp(-i dw t) * INT Mxy exp(-i phi_shared t) exp(-i 2pi k.x) dV

  This is what lets several chemical species share one mesh, one partition and one
  assembler, carried as separate `nv` columns -- see
  `feelmri_paper_experiments/water_and_fat.py`. It holds ONLY while those two
  quantities are uniform per species; a spatially varying T2 needs its own
  assembler.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  from _phantom_fixtures import make_cube_mesh

  path, _ = make_cube_mesh(tmp_path / 'cube.vtu', 'tetra', n=4, scale=2e-3)
  T2, dw, phi_shared = 40.0, -0.44, 0.013      # ms, rad/ms, rad/ms

  rng = np.random.default_rng(0)
  S = 16
  pts = [np.ascontiguousarray(rng.uniform(-60, 60, (S, 1, 1)).astype(np.float32))
         for _ in range(3)]
  ts = np.ascontiguousarray(np.linspace(0.0, 3.0, S, dtype=np.float32).reshape(S, 1, 1))

  def run(T2_val, phi_val):
    phantom = FEMPhantom(path=str(path))
    phantom.set_assembler(voxel_size=0.0, lorder=2, horder=2,
                          nodal_approximation=False, lumped=False)
    n = phantom.local_nodes.shape[0]
    phantom.set_static_fields(T2=np.full(n, T2_val, dtype=np.float32),
                              phi_dB0=np.full(n, phi_val, dtype=np.float32))
    idx = np.arange(n)
    phantom.update_magnetization(
      (np.cos(idx * 0.1) + 1j * np.sin(idx * 0.07)).astype(np.complex64).reshape(-1, 1))
    return np.asarray(phantom.signal(pts, ts, None)).reshape(-1)

  inside = run(T2, phi_shared + dw)                       # both inside the integral
  factored = run(np.inf, phi_shared) * (
      np.exp(-ts.reshape(-1) / T2) * np.exp(-1j * dw * ts.reshape(-1)))

  assert np.max(np.abs(inside - factored)) / np.max(np.abs(inside)) < 1e-5


def test_offresonance_continues_across_the_solver_to_assembler_handoff(tmp_path):
  """The assembler must CONTINUE the solver's off-resonance precession, not
  reverse it.

  The solver turns `Mxy` as `exp(-i*gamma*delta_B*t)` and the assembler applies
  `exp(-i*phi*t)` to the snapshot it is handed, so one physical field described
  to both halves -- `delta_B` in mT, `phi_dB0 = 2*pi*gammabar*delta_B` in rad/ms
  -- must produce a single unbroken precession.

  Asserted as a splitting invariant, which needs no sign convention of its own:
  evolving for `TA` in the solver and a further `TB` in the assembler must equal
  evolving for `TA + TB` in the solver and reading out at once. A sign flip on
  either side turns the `TB` leg around and the two disagree by `2*omega*TB`.

  Until 2026-09-10 the assembler ran `exp(+i*phi*t)` while every caller passed
  `phi_dB0 = +2*pi*gammabar*delta_B0`, so this was the disagreeing pairing and
  no tracked test exercised the two halves together with a non-zero field.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  from pint import Quantity as Q_
  from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
  from feelmri.MRObjects import RF, Scanner
  from _phantom_fixtures import make_cube_mesh

  path, _volume = make_cube_mesh(tmp_path / 'cube.vtu', 'tetra', n=1, scale=2e-3)

  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')          # rad/ms/mT
  dB_mT = 1e-3
  omega = 2.0 * np.pi * scanner.gammabar.m_as('1/ms/mT') * dB_mT   # rad/ms
  TA, TB = 5.0, 3.0

  def hard90(dur_ms=0.1, n=64):
    t = np.linspace(0.0, dur_ms, n)
    amp = (np.pi / 2) / (gamma * dur_ms)
    return RF(waveform=Q_(np.full(n, amp, dtype=complex), 'mT'),
              timings=Q_(t, 'ms'))

  def snapshot_after(delay_ms):
    """Mxy at the end of a hard 90 followed by `delay_ms` of free precession."""
    phantom = FEMPhantom(path=str(path))
    phantom.set_assembler(voxel_size=0.0, lorder=2, horder=4,
                          nodal_approximation=False, lumped=False)
    seq = Sequence()
    seq.add_block(SequenceBlock(rf_pulses=[hard90()]))
    seq.add_block(SequenceBlock(dur=Q_(delay_ms, 'ms'), dt=Q_(0.01, 'ms'),
                                store_magnetization=True))
    Mxy, _Mz = BlochSolver(sequence=seq, phantom=phantom, M0=1.0,
                           T1=Q_(1e9, 'ms'), T2=Q_(1e9, 'ms'),
                           delta_B=dB_mT, dtype='float64',
                           perfect_spoiling=False).solve()
    return phantom, Mxy[:, -1]

  def readout(phantom, mxy, elapsed_ms):
    """One k = 0 sample `elapsed_ms` after the snapshot."""
    n = phantom.local_nodes.shape[0]
    phantom.set_static_fields(T2=np.full(n, 1e9, dtype=np.float32),
                              phi_dB0=np.full(n, omega, dtype=np.float32))
    phantom.update_magnetization(np.ascontiguousarray(mxy))
    zero = np.zeros((1, 1, 1), dtype=np.float32)
    t = np.full((1, 1, 1), elapsed_ms, dtype=np.float32)
    return complex(np.asarray(
        phantom.mri_signal((zero.copy(), zero.copy(), zero.copy()), t, None)).ravel()[0])

  ph_a, mxy_a = snapshot_after(TA)
  split = readout(ph_a, mxy_a, TB)                  # TA in the solver, TB in the assembler
  ph_b, mxy_b = snapshot_after(TA + TB)
  whole = readout(ph_b, mxy_b, 0.0)                 # TA + TB in the solver

  gap = float(abs(np.angle(split / whole)))
  assert gap < 1e-3, (
      f'the handoff is discontinuous by {gap:.4f} rad: {TA} ms in the solver plus '
      f'{TB} ms in the assembler disagrees with {TA + TB} ms in the solver. '
      f'A reversed leg would show 2*omega*TB = {2 * omega * TB:.4f} rad.')

  # The direction itself, so a matched pair of flips cannot pass silently.
  advance = float(np.angle(readout(ph_a, mxy_a, TB) / readout(ph_a, mxy_a, 0.0)))
  assert abs(advance - np.angle(np.exp(-1j * omega * TB))) < 1e-3, (
      f'the assembler advanced the phase by {advance:+.4f} rad over {TB} ms, '
      f'expected {np.angle(np.exp(-1j * omega * TB)):+.4f}')


def test_kspace_is_hermitian_for_a_real_object(tmp_path):
  """A real, non-negative object must give `S(k) = conj(S(-k))`.

  The box-transform test in `test_pulseq_analytical.py` checks only `|S(k)|`
  and so discards exactly the phase this identity lives in -- a sign error in
  the encoding exponent would leave the magnitude untouched and break this.
  Also re-pins `S(0) == volume` on the same samples, which is the one absolute
  scale the assembler has.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  from _phantom_fixtures import make_cube_mesh

  path, volume = make_cube_mesh(tmp_path / 'cube.vtu', 'tetra', n=2, scale=2e-3)
  phantom = FEMPhantom(path=str(path))
  phantom.set_assembler(voxel_size=0.0, lorder=2, horder=4,
                        nodal_approximation=False, lumped=False)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 1e9, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))
  phantom.update_magnetization(np.ones(n, dtype=np.complex64))   # real, positive

  # Symmetric about zero, odd count, so k and -k are both sampled exactly.
  k = np.linspace(-300.0, 300.0, 41).astype(np.float32)
  kx = np.ascontiguousarray(k.reshape(-1, 1, 1))
  zero = np.zeros_like(kx)
  t = np.zeros_like(kx)
  S = np.asarray(phantom.mri_signal((kx, zero.copy(), zero.copy()), t, None)).reshape(-1)

  scale = np.abs(S).max()
  worst = float(np.abs(S - np.conj(S[::-1])).max() / scale)
  assert worst < 1e-5, (
      f'|S(k) - conj(S(-k))| is {worst:.3e} of peak; a real object must have '
      f'Hermitian-symmetric k-space')

  centre = S[S.size // 2]
  assert abs(k[S.size // 2]) < 1e-9, 'the centre sample is not at k = 0'
  assert abs(centre.real - volume) < 1e-3 * volume, (
      f'S(k=0) is {centre.real:.6e}, cube volume is {volume:.6e}')
  assert abs(centre.imag) < 1e-5 * volume


def test_adc_frequency_offset_tunes_the_receiver(tmp_path):
  """A receiver tuned `+df` must bring the spins precessing `+df` faster to DC.

  This pins the SIGN of the receive chain, which nothing did before: the two
  demodulation tests elsewhere apply `exp(+i*phase)` and assert they recover
  the input, so they pass under either sign. The transmit sign is pinned the
  same way, by a measured POSITION, in
  `test_pulseq_analytical.py::test_rf_frequency_offset_shifts_the_slice`.

  The Pulseq specification calls `adc.freq` the "frequency offset of ADC
  receiver relative to the system frequency". So placing a narrow rod at `x0`
  and tuning the receiver to `gammabar*Gx*x0` must put it at the centre of the
  image -- that population IS what the receiver is listening to. Getting the
  sign wrong does not merely reverse the shift, it DOUBLES it: measured
  `+7.927 mm` for an `x0` of `+4 mm`, against `-0.008 mm` when correct.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  import meshio
  from feelmri.Bloch import apply_demodulation, demodulation_phase
  from feelmri.MRObjects import Scanner

  gammabar = Scanner().gammabar.m_as('Hz/T')
  x0_mm = 4.0
  path, _length = make_1d_rod_mesh(tmp_path / 'rod.vtu', length=2e-3,
                                   n_segments=8)
  mesh = meshio.read(str(path))
  mesh.points[:, 0] += x0_mm * 1e-3
  meshio.write(str(tmp_path / 'shifted.vtu'), mesh)

  phantom = FEMPhantom(path=str(tmp_path / 'shifted.vtu'))
  phantom.set_assembler(voxel_size=0.0, lorder=1, horder=3,
                        nodal_approximation=False, lumped=False)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 1e9, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))
  phantom.update_magnetization(np.ones(n, dtype=np.complex64))

  g_x, n_samples, dwell_ms = 10e-3, 256, 0.004
  t_ms = np.arange(n_samples) * dwell_ms
  k = (gammabar * g_x * t_ms * 1e-3).astype(np.float32)
  k = (k - k.mean()).astype(np.float32)
  zeros = np.zeros_like(k)
  signal = np.asarray(phantom.mri_signal(
      [k.reshape(-1, 1, 1), zeros.reshape(-1, 1, 1), zeros.reshape(-1, 1, 1)],
      np.zeros((n_samples, 1, 1), dtype=np.float32), None)).reshape(-1)

  def centre_mm(sig):
    img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(sig)))
    x = np.fft.fftshift(np.fft.fftfreq(n_samples, d=float(k[1] - k[0])))
    w = np.abs(img) ** 2
    return float((x * w).sum() / w.sum()) * 1e3

  assert centre_mm(signal) == pytest.approx(x0_mm, abs=0.3), (
      'the undemodulated rod is not where it was built')

  freq = gammabar * g_x * x0_mm * 1e-3
  tuned = centre_mm(apply_demodulation(
      signal, demodulation_phase(t_ms, freq_offset_hz=freq)))
  assert tuned == pytest.approx(0.0, abs=0.1), (
      f'a receiver tuned to gammabar*Gx*{x0_mm} mm = {freq:.1f} Hz left the '
      f'rod at {tuned:+.3f} mm instead of DC. About {2 * x0_mm:+.1f} mm means '
      f'the frequency term lost its negation and doubled the offset')


# ---------------------------------------------------------------------------
# The realism features, carried into k-space
# ---------------------------------------------------------------------------
#
# Every test of concomitant_fields and b1_map stops at the solver's returned
# magnetization; no example or test had ever called update_magnetization with
# one, so nothing the two features produce had reached the assembler at all.
#
# Both tests below pin the whole chain against a closed form by first measuring
# the assembler's own nodal weights. At k = 0 and t = 0 the signal is
# ``S = sum_n m_n Mxy_n`` with ``m_n = integral of basis function n``, which is
# linear in the nodal vector -- so handing it a unit vector per node recovers
# the weights, and the prediction that follows is exact rather than a
# tolerance. The weights differ from node to node on an irregular mesh, so a
# handoff that paired the wrong rows cannot pass.


def _irregular_phantom(path):
  """Five nodes at mutually incommensurate coordinates spread over ~20 cm, so a
  term quadratic in position is large and every nodal weight is distinct."""
  import meshio
  points = np.array([[0.11, -0.03, 0.07],
                     [-0.05, 0.12, 0.02],
                     [0.04, 0.06, -0.10],
                     [-0.09, -0.08, 0.05],
                     [0.02, -0.11, -0.06]])
  meshio.write(str(path), meshio.Mesh(points, [('tetra',
                                                np.array([[0, 1, 2, 3],
                                                          [1, 2, 3, 4]]))]))
  phantom = FEMPhantom(path=str(path))
  phantom.set_assembler(voxel_size=0.0, lorder=2, horder=4,
                        nodal_approximation=False, lumped=False)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 1e9, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))
  return phantom


def _signal_at_dc(phantom, mxy):
  phantom.update_magnetization(np.ascontiguousarray(mxy, dtype=np.complex64))
  zero = np.zeros((1, 1, 1), dtype=np.float32)
  return complex(np.asarray(phantom.mri_signal(
      (zero.copy(), zero.copy(), zero.copy()), zero.copy(), None)).ravel()[0])


def _nodal_weights(phantom):
  """``m_n = integral of basis function n``, read off the assembler itself."""
  n = phantom.local_nodes.shape[0]
  weights = np.zeros(n, dtype=np.complex128)
  for i in range(n):
    unit = np.zeros(n, dtype=np.complex64)
    unit[i] = 1.0
    weights[i] = _signal_at_dc(phantom, unit)
  return weights


def test_concomitant_phase_reaches_kspace(tmp_path):
  """Phase the solver accumulates from the Maxwell term must appear in the
  k-space signal, with the magnitude the closed form gives.

  The gradient is a BIPOLAR pair, so the linear term `G.x` refocuses exactly
  and everything left at the snapshot is concomitant. `mri_signal` has no B0
  argument and no quadratic spatial channel of its own, so this is the only
  way the term can reach a readout: carried on the magnetization.

  The pair is OBLIQUE on purpose. Driven on `Gz` alone, `Bc` collapses to
  `(Gz^2/4)(x^2 + y^2)`, which has no cross terms and is symmetric in x and y:
  on these nodes, negating a cross term or swapping the x and y position
  indices then changes `Bc` by exactly 0.000, so the test could not see either.
  With `G = (14, -9, 20)` the same two mistakes change it by 41x and 37x.

  The prediction is written in the FACTORED form `(Bx^2 + By^2) / (2 B0)`,
  which the library does not use -- it expands the square -- so the check is
  independent of the expression it is checking.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  from pint import Quantity as Q_
  from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
  from feelmri.MRObjects import Gradient, Scanner

  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')
  B0_mT = scanner.field_strength.m_as('mT')
  G, dur_ms = (14.0, -9.0, 20.0), 4.0

  def lobe(sign):
    gradients = [
      Gradient(timings=Q_(np.array([0.0, dur_ms]), 'ms'),
               amplitudes=Q_(np.array([sign * G[axis]] * 2), 'mT/m'),
               scanner=scanner, ref=Q_(0.0, 'ms'), time=Q_(0.0, 'ms'),
               axis=axis)
      for axis in range(3)]
    return SequenceBlock(gradients=gradients, dur=Q_(dur_ms, 'ms'),
                         dt=Q_(0.01, 'ms'), empty=False)

  def run(concomitant):
    phantom = _irregular_phantom(tmp_path / f'conc_{concomitant}.vtu')
    seq = Sequence()
    seq.add_block(lobe(+1.0))
    closing = lobe(-1.0)
    closing.store_magnetization = True
    seq.add_block(closing)
    Mxy, _Mz = BlochSolver(
      sequence=seq, phantom=phantom, M0=1.0, T1=Q_(1e9, 'ms'),
      T2=Q_(1e9, 'ms'), initial_Mxy=1.0 + 0.0j, initial_Mz=0.0,
      dtype='float64', perfect_spoiling=False,
      concomitant_fields=concomitant).solve()
    return phantom, Mxy[:, -1]

  # Control: with the term off the two lobes cancel and every node is still at
  # its initial value, so the signal is the mesh volume with no phase.
  phantom_off, mxy_off = run(False)
  weights = _nodal_weights(phantom_off)
  off = _signal_at_dc(phantom_off, mxy_off)
  assert abs(off - weights.sum()) < 1e-6 * abs(weights.sum()), (
    'the bipolar pair did not refocus with the concomitant term off')

  phantom_on, mxy_on = run(True)
  nodes = phantom_on.local_nodes.astype(np.float64)
  # Both lobes contribute the same amount because Bc goes as G^2, and the
  # amplitude is constant over each, so the time integral is just 2 * dur.
  x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
  Bx = G[0] * z - 0.5 * G[2] * x
  By = G[1] * z - 0.5 * G[2] * y
  bc_phase = -gamma * (2.0 * dur_ms) * (Bx**2 + By**2) / (2.0 * B0_mT)
  predicted = complex((weights * np.exp(1j * bc_phase)).sum())

  got = _signal_at_dc(phantom_on, mxy_on)
  scale = float(np.abs(weights).sum())
  assert abs(got - predicted) < 2e-5 * scale, (
    f'k-space carries {got:.6e} where the closed-form concomitant phase '
    f'predicts {predicted:.6e}')
  # ... and the term did something, so the agreement is not with zero phase.
  assert abs(got - off) > 0.1 * scale, (
    f'the concomitant term changed the DC signal by only '
    f'{abs(got - off) / scale:.2e} of the volume; this case cannot discriminate')


def test_a_b1_map_reaches_kspace_node_by_node(tmp_path):
  """A per-node transmit sensitivity must weight each node's contribution to
  the signal by `sin(b1_n * nominal flip)`, at that node and not another.

  The nodal weights differ by a factor of three on this mesh, so a map applied
  to the wrong rows changes the answer.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  from pint import Quantity as Q_
  from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
  from feelmri.MRObjects import RF, Scanner

  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')
  dur_ms, n_samples = 0.2, 64

  def run(phantom, b1_map):
    amplitude = (np.pi / 2) / (gamma * dur_ms)
    pulse = RF(waveform=Q_(np.full(n_samples, amplitude, dtype=complex), 'mT'),
               timings=Q_(np.linspace(0.0, dur_ms, n_samples), 'ms'))
    block = SequenceBlock(rf_pulses=[pulse], dur=Q_(dur_ms, 'ms'))
    block.store_magnetization = True
    seq = Sequence()
    seq.add_block(block)
    Mxy, _Mz = BlochSolver(
      sequence=seq, phantom=phantom, M0=1.0, T1=Q_(1e9, 'ms'),
      T2=Q_(1e9, 'ms'), initial_Mxy=0.0, initial_Mz=1.0, dtype='float64',
      perfect_spoiling=False, b1_map=b1_map).solve()
    return _signal_at_dc(phantom, Mxy[:, -1])

  phantom = _irregular_phantom(tmp_path / 'b1.vtu')
  weights = _nodal_weights(phantom)
  assert float(np.abs(weights).max() / np.abs(weights).min()) > 2.0, (
    'the nodal weights are too uniform for this test to localise the map')

  nominal = run(phantom, None)
  b1 = np.array([1.0, 0.8, 0.5, 0.3, 0.0])
  mapped = run(phantom, b1)

  # The flip is linear in b1 for a hard pulse, so node n ends at sin(b1_n*90).
  predicted = nominal * complex(
      (weights * np.sin(b1 * np.pi / 2)).sum() / weights.sum())
  scale = abs(nominal)
  assert abs(mapped - predicted) < 1e-5 * scale, (
    f'k-space reads {mapped:.6e} where the per-node flips predict '
    f'{predicted:.6e}')
  # A permuted map is the failure this is built to catch; show the case can
  # see one.
  permuted = complex(
      nominal * (weights * np.sin(b1[::-1] * np.pi / 2)).sum() / weights.sum())
  assert abs(permuted - predicted) > 1e-2 * scale, (
    'reversing the map changes nothing here, so the test cannot localise it')


def test_a_zero_or_non_finite_static_field_is_refused(tmp_path):
  """`T2 = 0` inverts to Inf, and `exp(-t*Inf)` is NaN even at `t = 0`, so ONE
  bad node used to turn EVERY k-space sample into NaN -- not just its own
  contribution. Nothing rejected it on either side of the boundary.

  A negative T2 is the worse case: it is finite, so there is no NaN to notice
  and the signal simply GROWS. Measured before the guard, one negative node of
  125 moved S(5 ms) from 5.888e-08 to 5.920e-08.

  The check cannot be `std::isnan` on the C++ side: the build is -Ofast, which
  implies -ffinite-math-only and folds that to false. It goes through the same
  IEEE bit-pattern helper the b1_map guard uses, now shared in `Numeric.h`.

  An INFINITE T2 is not an error -- it inverts to exactly zero and is the
  idiomatic way to switch relaxation off, which several tests here rely on.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  from _phantom_fixtures import make_cube_mesh

  path, _volume = make_cube_mesh(tmp_path / 'cube.vtu', 'tetra', n=2,
                                 scale=2e-3)
  phantom = FEMPhantom(path=str(path))
  phantom.set_assembler(voxel_size=0.0, lorder=2, horder=4,
                        nodal_approximation=False, lumped=False)
  n = phantom.local_nodes.shape[0]
  good_phi = np.zeros(n, dtype=np.float32)

  def t2_with(first):
    out = np.full(n, 60.0, dtype=np.float32)
    out[0] = first
    return out

  for bad in (0.0, -60.0, np.nan):
    with pytest.raises(ValueError, match='T2'):
      phantom.set_static_fields(T2=t2_with(bad), phi_dB0=good_phi)
  for bad in (np.nan, np.inf):
    poisoned = np.zeros(n, dtype=np.float32)
    poisoned[0] = bad
    with pytest.raises(ValueError, match='phi_dB0'):
      phantom.set_static_fields(T2=np.full(n, 60.0, dtype=np.float32),
                                phi_dB0=poisoned)

  # The C++ side refuses it too, reached directly so the Python check cannot
  # be what fires -- the same arrangement the b1_map guard's test uses.
  for assembler in phantom.assembler:
    with pytest.raises(Exception, match='T2'):
      assembler.set_static_fields(t2_with(0.0), good_phi)

  # A SHORT map is the silent case: the assembler indexes these by element
  # connectivity under -DNDEBUG, so it reads adjacent heap. Measured before
  # the length check, 22 entries for 27 nodes: 6.348e-08 against a true
  # 6.336e-08 -- and a DIFFERENT wrong value on a re-run, which is what
  # identifies it as a heap read rather than an arithmetic error.
  with pytest.raises(ValueError, match='local nodes'):
    phantom.set_static_fields(T2=np.full(n - 5, 60.0, dtype=np.float32),
                              phi_dB0=np.zeros(n - 5, dtype=np.float32))

  # (n,) and (n, 1) describe the same nodes and callers mix them freely --
  # examples/water_and_fat.py passes one of each in the same call, which a
  # shape-equality check broke. Only a differing ENTRY COUNT is an error.
  phantom.set_static_fields(T2=np.full((n, 1), 60.0, dtype=np.float32),
                            phi_dB0=good_phi)
  with pytest.raises(ValueError, match='entries'):
    phantom.set_static_fields(T2=np.full(n - 1, 60.0, dtype=np.float32),
                              phi_dB0=good_phi)

  # An infinite T2 is legitimate: no relaxation, so the signal does not decay.
  phantom.set_static_fields(T2=np.full(n, np.inf, dtype=np.float32),
                            phi_dB0=good_phi)
  phantom.update_magnetization(np.ones(n, dtype=np.complex64))
  zero = np.zeros((1, 1, 1), dtype=np.float32)
  def at(t_ms):
    t = np.full((1, 1, 1), t_ms, dtype=np.float32)
    return abs(complex(np.asarray(phantom.mri_signal(
        (zero.copy(), zero.copy(), zero.copy()), t, None)).ravel()[0]))
  assert abs(at(5.0) - at(0.0)) < 1e-6 * at(0.0), (
    'an infinite T2 should switch relaxation off, not decay')


# ---------------------------------------------------------------------------
# Concomitant fields DURING the readout
# ---------------------------------------------------------------------------
#
# The assembler's phase was linear in position: `-k.x` plus a static
# `-phi.t`. The Maxwell term is quadratic, so it cannot be folded into either,
# and concomitant phase accrued inside an ADC window was simply absent.
# Measured on epi_v142 at 1.5 T that is -5.50 rad at z = 5 cm and -22.0 rad at
# x = z = 10 cm over its 103 ms train.


def _maxwell_fixture(tmp_path, name, orientation=None):
  """Five nodes at incommensurate coordinates spread over ~20 cm, so all four
  Maxwell terms are live and none of them is degenerate.

  `orientation` is applied BEFORE `set_assembler`, which is the only valid
  order: the assembler captures the node coordinates in its constructor.
  """
  import meshio
  from pint import Quantity as _Q
  points = np.array([[0.11, -0.03, 0.07],
                     [-0.05, 0.12, 0.02],
                     [0.04, 0.06, -0.10],
                     [-0.09, -0.08, 0.05],
                     [0.02, -0.11, -0.06]])
  path = tmp_path / name
  meshio.write(str(path), meshio.Mesh(points, [('tetra', np.array([[0, 1, 2, 3],
                                                                  [1, 2, 4, 3]]))]))
  phantom = FEMPhantom(path=str(path))
  if orientation is not None:
    phantom.orient(orientation, _Q(np.zeros(3), 'm'))
  phantom.set_assembler(voxel_size=0.0, lorder=2, horder=4,
                        nodal_approximation=False, lumped=False)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 1e9, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))
  phantom.update_magnetization(np.ones(n, dtype=np.complex64))
  return phantom, points


def _constant_gradient_coefficients(amps, dur_ms, scanner, scale=1.0):
  from pint import Quantity as Q_

  from feelmri import maxwell_moments, maxwell_phase_coefficients
  from feelmri.Bloch import Sequence, SequenceBlock
  from feelmri.MRObjects import Gradient

  timings = np.array([0.0, dur_ms])
  gradients = [Gradient(timings=Q_(timings, 'ms'),
                        amplitudes=Q_(np.array([a, a]), 'mT/m'), scanner=scanner,
                        ref=Q_(0.0, 'ms'), time=Q_(0.0, 'ms'), axis=axis)
               for axis, a in enumerate(amps)]
  seq = Sequence()
  seq.add_block(SequenceBlock(gradients=gradients, dur=Q_(dur_ms, 'ms'),
                              dt=Q_(0.01, 'ms'), empty=False))
  times = np.array([0.0, dur_ms])
  coef = maxwell_phase_coefficients(maxwell_moments(seq, 0.0, times), scanner)
  return times, coef * scale


def _nodal_phase(coef_row, points):
  """The general quadratic form the assembler evaluates: six coefficients over
  x^2, y^2, z^2, xy, xz and yz."""
  x, y, z = points[:, 0], points[:, 1], points[:, 2]
  return (coef_row[0] * x ** 2 + coef_row[1] * y ** 2 + coef_row[2] * z ** 2
          + coef_row[3] * x * y + coef_row[4] * x * z + coef_row[5] * y * z)


@pytest.mark.parametrize('scale, tag', [(1.0, '1 rad'), (20.0, '20 rad')])
def test_the_concomitant_readout_term_matches_its_closed_form(tmp_path, scale, tag):
  """`signal_sum` is an unweighted sum over nodes, so `S = sum_n exp(i phi_n)`
  is exact -- no quadrature, no interpolation, nothing to approximate.

  **Do not use the quadrature path as a nodal reference.** It evaluates the
  phase AT the quadrature points, whereas a nodal prediction interpolates
  `exp(i phi)` from the nodes; at a 1 rad phase over a 20 cm element the two
  differ by 46%. That is physics, not a defect, and it cost a wrong alarm here.

  The 20 rad case is the one that matters: the assembler works in float32 and
  its phase already carries hundreds of radians of `k.x`, so what has to be
  shown is that a term of the size this feature exists to model survives.
  """
  from feelmri.MRObjects import Scanner

  phantom, points = _maxwell_fixture(tmp_path, f'maxwell_{tag.split()[0]}.vtu')
  scanner = Scanner()
  times, coef = _constant_gradient_coefficients((14.0, -9.0, 20.0), 3.0,
                                                scanner, scale=scale)
  zero = np.zeros((2, 1, 1), dtype=np.float32)
  t3 = times.reshape(2, 1, 1).astype(np.float32)
  pts = (zero.copy(), zero.copy(), zero.copy())

  got = np.asarray(phantom.signal_sum(pts, t3, None, maxwell=coef)).ravel()[1]
  phase = _nodal_phase(coef[1], points)
  expected = complex(np.exp(1j * phase).sum())

  assert np.abs(phase).max() > 0.5, 'this case produces no concomitant phase'
  assert abs(got - expected) < 1e-5 * points.shape[0], (
    f'{tag}: signal_sum reads {got:.6f} where the closed form over nodes gives '
    f'{expected:.6f}, at a phase span of {np.abs(phase).max():.3f} rad')


def test_zero_coefficients_are_bit_identical_to_the_term_being_off(tmp_path):
  """The new branch must not perturb anything by itself. With the feature off
  entirely the whole suite and the 24-case kernel A/B are unchanged; this pins
  the remaining case, where the branch runs but the coefficients are zero."""
  from feelmri.MRObjects import Scanner

  phantom, _points = _maxwell_fixture(tmp_path, 'maxwell_zero.vtu')
  times, coef = _constant_gradient_coefficients((14.0, -9.0, 20.0), 3.0, Scanner())
  zero = np.zeros((2, 1, 1), dtype=np.float32)
  t3 = times.reshape(2, 1, 1).astype(np.float32)
  pts = (zero.copy(), zero.copy(), zero.copy())

  off = np.asarray(phantom.signal_sum(pts, t3, None))
  zeros = np.asarray(phantom.signal_sum(pts, t3, None,
                                        maxwell=np.zeros_like(coef)))
  assert np.array_equal(off, zeros), 'zero coefficients changed the signal'
  # ... and the case is not vacuous: real coefficients do change it.
  live = np.asarray(phantom.signal_sum(pts, t3, None, maxwell=coef))
  assert not np.allclose(off, live)


def test_the_readout_term_follows_a_moving_phantom(tmp_path):
  """The term is quadratic in the CURRENT position, so under a POD trajectory
  it must be evaluated at the displaced coordinates -- exactly as `-k.x` is.
  A constant displacement must therefore be indistinguishable from building
  the phantom at the displaced position."""
  pytest.importorskip('meshio')
  import meshio
  from feelmri.Motion import POD
  from feelmri.MRObjects import Scanner

  phantom, points = _maxwell_fixture(tmp_path, 'maxwell_moving.vtu')
  shift = np.array([0.03, -0.02, 0.025])
  n = points.shape[0]
  data = np.zeros((n, 3, 4), dtype=np.float32)
  for axis in range(3):
    data[:, axis, :] = shift[axis]
  pod = POD(data=data, times=np.linspace(0.0, 5.0, 4), n_modes=1)

  shifted_path = tmp_path / 'maxwell_shifted.vtu'
  meshio.write(str(shifted_path), meshio.Mesh(
    points + shift, [('tetra', np.array([[0, 1, 2, 3], [1, 2, 4, 3]]))]))
  shifted = FEMPhantom(path=str(shifted_path))
  shifted.set_assembler(voxel_size=0.0, lorder=2, horder=4,
                        nodal_approximation=False, lumped=False)
  shifted.set_static_fields(T2=np.full(n, 1e9, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))
  shifted.update_magnetization(np.ones(n, dtype=np.complex64))

  # Scaled up: at a ~1 rad phase a 3 cm shift moves it by only 0.05 rad, which
  # is too weak to tell a moving phantom from a still one. At the ~20 rad scale
  # the feature exists to model, the same shift moves it by a full radian.
  times, coef = _constant_gradient_coefficients((14.0, -9.0, 20.0), 3.0,
                                                Scanner(), scale=20.0)
  zero = np.zeros((2, 1, 1), dtype=np.float32)
  t3 = times.reshape(2, 1, 1).astype(np.float32)
  pts = (zero.copy(), zero.copy(), zero.copy())

  moving = np.asarray(phantom.signal_sum(pts, t3, pod, maxwell=coef)).ravel()[1]
  static = np.asarray(shifted.signal_sum(pts, t3, None, maxwell=coef)).ravel()[1]
  assert abs(moving - static) < 1e-4 * n, (
    f'a moving phantom reads {moving:.6f} where the statically shifted one '
    f'gives {static:.6f}; the term is not following the displacement')
  # The displacement must actually matter. Checked on the PHASE rather than on
  # the summed signal: the sum partly cancels across nodes, so it understates
  # how different the two configurations are.
  moved_phase = _nodal_phase(coef[1], points + shift)
  rest_phase = _nodal_phase(coef[1], points)
  assert np.abs(moved_phase - rest_phase).max() > 0.1, (
    f'the shift changes the concomitant phase by only '
    f'{np.abs(moved_phase - rest_phase).max():.3e} rad, so this case cannot '
    f'tell a moving phantom from a still one')


@pytest.mark.parametrize('oblique', [False, True], ids=['axial', 'oblique'])
def test_the_concomitant_term_continues_across_the_solver_to_assembler_handoff(
        tmp_path, oblique):
  """The acceptance test for carrying concomitant fields into the readout.

  Evolving `TA` in the solver and handing the remaining `TB` to the assembler
  must equal evolving `TA + TB` in the solver and reading out at once. That is
  a splitting invariant: it needs no sign convention of its own, so a flipped
  sign, a missing factor of 4 on the `(x^2+y^2)` term or a wrong origin each
  break it. It is the same shape as
  `test_offresonance_continues_across_the_solver_to_assembler_handoff`.

  **Both moments have to cross the handoff.** The `TA + TB` run also winds the
  LINEAR gradient phase through `TB`, so the assembler must be given the k of
  that leg as well; comparing against `k = 0` measures a leg that had a
  gradient against one that did not, and reads 1.0 no matter what the
  concomitant term does. That cost a wrong alarm here.

  Measured: 8.5e-05 relative with the term, 4.9e-01 without it.

  **The oblique case is the one that pins the FRAME.** Run axially, the
  rotation is the identity and the whole six-coefficient apparatus collapses
  to the four the field naturally has -- so the axial arm passes whether or not
  either half knows about `FEMPhantom.orient`. With the phantom tilted, the
  solver must evaluate `Bc` on physical-frame coordinates and gradients while
  the assembler works in the imaging frame, and the two must still meet.
  Measured with the solver made frame-naive again: the axial arm is unmoved
  and the oblique one reads **3.11e-01**, against the 1e-3 gate.
  """
  pytest.importorskip('meshio')
  from pint import Quantity as Q_

  from feelmri import BlochSolver, maxwell_moments, maxwell_phase_coefficients
  from feelmri.Bloch import Sequence, SequenceBlock
  from feelmri.MRObjects import Gradient, Scanner
  from feelmri.PulseqAdapter import _gradient_moment_between

  scanner = Scanner()
  TA, TB, amps = 2.0, 3.0, (14.0, -9.0, 20.0)
  th = np.deg2rad(23.0)
  R = (np.array([[np.cos(th), -np.sin(th), 0.0],
                 [np.sin(th), np.cos(th), 0.0],
                 [0.0, 0.0, 1.0]])
       @ np.array([[np.cos(th), 0.0, np.sin(th)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(th), 0.0, np.cos(th)]])) if oblique else None

  def gradients(duration):
    return [Gradient(timings=Q_(np.array([0.0, duration]), 'ms'),
                     amplitudes=Q_(np.array([a, a]), 'mT/m'), scanner=scanner,
                     ref=Q_(0.0, 'ms'), time=Q_(0.0, 'ms'), axis=axis)
            for axis, a in enumerate(amps)]

  def solve_to(duration, tag):
    # The solver reads the orientation off the phantom, which is the whole
    # point: forgetting to pass it is how the frame defect arose.
    phantom, _points = _maxwell_fixture(
        tmp_path, f'handoff_{tag}_{"obl" if oblique else "ax"}.vtu',
        orientation=R)
    block = SequenceBlock(gradients=gradients(duration),
                          dur=Q_(duration, 'ms'), dt=Q_(0.002, 'ms'),
                          empty=False)
    block.store_magnetization = True
    seq = Sequence()
    seq.add_block(block)
    Mxy, _Mz = BlochSolver(seq, phantom, T1=Q_(1e9, 'ms'), T2=Q_(1e9, 'ms'),
                           initial_Mxy=1.0 + 0.0j, initial_Mz=0.0,
                           perfect_spoiling=False, dtype='float64',
                           concomitant_fields=True).solve()
    return phantom, Mxy[:, -1]

  def readout(phantom, mxy, moments, kvec):
    phantom.update_magnetization(np.ascontiguousarray(mxy))
    points = tuple(np.full((1, 1, 1), v, dtype=np.float32) for v in kvec)
    coef = (None if moments is None
            else maxwell_phase_coefficients(moments, scanner, rotation=R))
    return complex(np.asarray(phantom.signal_sum(
        points, np.zeros((1, 1, 1), dtype=np.float32), None,
        maxwell=coef)).ravel()[0])

  leg = Sequence()
  leg.add_block(SequenceBlock(gradients=gradients(TB), dur=Q_(TB, 'ms'),
                              dt=Q_(0.002, 'ms'), empty=False))
  moments = maxwell_moments(leg, 0.0, np.array([TB]), rotation=R)
  kvec = _gradient_moment_between(leg, 0.0, TB, scanner.gammabar.m_as('Hz/T'))

  phantom_a, mxy_a = solve_to(TA, 'split')
  split = readout(phantom_a, mxy_a, moments, kvec)
  phantom_b, mxy_b = solve_to(TA + TB, 'whole')
  whole = readout(phantom_b, mxy_b, None, (0.0, 0.0, 0.0))

  gap = abs(split - whole) / abs(whole)
  assert gap < 1e-3, (
    f'the handoff is discontinuous by {gap:.2e}: {TA} ms in the solver plus '
    f'{TB} ms in the assembler disagrees with {TA + TB} ms in the solver')
  # Dropping the readout term must break it, or the test is about the linear
  # term and says nothing about this feature.
  naked = abs(readout(phantom_a, mxy_a, None, kvec) - whole) / abs(whole)
  assert naked > 100 * gap, (
    f'omitting the concomitant readout term changes the answer by only '
    f'{naked:.2e} against {gap:.2e} with it; this case cannot see the term')


def test_an_oblique_orientation_gives_the_same_physics(tmp_path):
  """`Bc` is B0-aligned in the scanner's PHYSICAL frame, but `FEMPhantom.orient`
  leaves node coordinates in the imaging frame, where the same field is a
  general quadratic form. That is why the assembler takes six coefficients and
  not the four the field naturally has.

  The invariant: rotating the phantom and folding the same rotation into the
  coefficients must leave every node's phase unchanged, because both describe
  one physical spin in one physical field. phase_contrast.py's planning file is
  about 20 degrees off axial, so this is not a corner case.

  Without the rotation folded in, the Maxwell expression is evaluated in the
  imaging frame and treats the slice normal as B0 -- which the second half
  measures, so the test cannot pass by the rotation being irrelevant.
  """
  from feelmri.MRObjects import Scanner

  scanner = Scanner()

  # Rotation about two axes, comparable to a real oblique prescription.
  ca, sa = np.cos(0.35), np.sin(0.35)
  cb, sb = np.cos(-0.22), np.sin(-0.22)
  Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
  Ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
  R = Rz @ Ry

  from feelmri import maxwell_moments, maxwell_phase_coefficients
  from pint import Quantity as Q_
  from feelmri.Bloch import Sequence, SequenceBlock
  from feelmri.MRObjects import Gradient

  amps, dur = (14.0, -9.0, 20.0), 3.0
  gradients = [Gradient(timings=Q_(np.array([0.0, dur]), 'ms'),
                        amplitudes=Q_(np.array([a, a]), 'mT/m'),
                        scanner=scanner, ref=Q_(0.0, 'ms'), time=Q_(0.0, 'ms'),
                        axis=axis)
               for axis, a in enumerate(amps)]
  seq = Sequence()
  seq.add_block(SequenceBlock(gradients=gradients, dur=Q_(dur, 'ms'),
                              dt=Q_(0.01, 'ms'), empty=False))
  moments = maxwell_moments(seq, 0.0, np.array([0.0, dur]))

  physical = np.array([[0.11, -0.03, 0.07], [-0.05, 0.12, 0.02],
                       [0.04, 0.06, -0.10], [-0.09, -0.08, 0.05]])
  # x_physical = R @ x_imaging, the sense FEMPhantom.orient leaves behind.
  imaging = physical @ R

  direct = _nodal_phase(maxwell_phase_coefficients(moments, scanner)[1],
                        physical)
  rotated = _nodal_phase(
      maxwell_phase_coefficients(moments, scanner, rotation=R)[1], imaging)

  worst = float(np.abs(rotated - direct).max())
  assert worst < 1e-12, (
    f'the same spin in the same field reads {rotated} in the imaging frame '
    f'and {direct} in the physical one, differing by {worst:.2e}')
  assert np.abs(direct).max() > 0.1, 'this case produces almost no phase'

  # Ignoring the rotation is the mistake this exists to prevent, and it is a
  # large one: the expression then treats the slice normal as B0.
  naive = _nodal_phase(maxwell_phase_coefficients(moments, scanner)[1], imaging)
  assert np.abs(naive - direct).max() > 0.1 * np.abs(direct).max(), (
    'this orientation is too close to axial to show the frame matters')


# ---------------------------------------------------------------------------
# Receive coil sensitivity
# ---------------------------------------------------------------------------
#
# `b1_map` is TRANSMIT and reaches k-space through the magnetization, which the
# test above pins. Receive sensitivity is the other half and had no
# counterpart: nothing scaled what a coil hears from a magnetization already
# fixed. It rides the assembler's free `nv` axis, so these pin both the
# arithmetic and the column ORDER, which is the part a caller has to unpack.


def _dc_signal(phantom, mxy):
  """The full `nv` row at k = 0, t = 0, rather than its first entry."""
  phantom.update_magnetization(np.ascontiguousarray(mxy, dtype=np.complex64))
  zero = np.zeros((1, 1, 1), dtype=np.float32)
  return np.asarray(phantom.mri_signal(
      (zero.copy(), zero.copy(), zero.copy()), zero.copy(), None)).reshape(-1)


def test_a_receive_map_folds_onto_the_coil_axis(tmp_path):
  """`S[e*n_coils + c] = sum_n m_n C[n,c] Mxy[n,e]`, exactly.

  The nodal weights are measured off the assembler first, so this is a closed
  form and not a tolerance -- and they differ by 9x across this mesh, so a fold
  that paired the wrong rows cannot pass.

  It also pins the column ORDER, which is the part a caller has to unpack:
  coils vary FASTEST, so one C-order reshape to `(n_enc, n_coils)` recovers
  both. The transposed prediction is asserted to disagree, or this test would
  pass under either convention.
  """
  phantom = _irregular_phantom(tmp_path / 'coils.vtu')
  weights = _nodal_weights(phantom)
  n = phantom.local_nodes.shape[0]

  rng = np.random.default_rng(17)
  C = (rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))).astype(np.complex64)
  Mxy = (rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))).astype(np.complex64)

  phantom.set_receive_sensitivity(C)
  got = _dc_signal(phantom, Mxy)
  assert got.size == 6, f'expected nv = 2 encodings x 3 coils, got {got.size}'

  predicted = np.array([(weights * Mxy[:, e] * C[:, c]).sum()
                        for e in range(2) for c in range(3)])
  worst = float(np.abs(got - predicted).max() / np.abs(predicted).max())
  assert worst < 1e-6, f'the folded signal is off by {worst:.2e}'

  transposed = np.array([(weights * Mxy[:, e] * C[:, c]).sum()
                         for c in range(3) for e in range(2)])
  assert np.abs(got - transposed).max() > 0.1 * np.abs(predicted).max(), (
    'encodings-fastest and coils-fastest agree here, so this case cannot '
    'distinguish the two orderings')


def test_a_receive_map_weights_the_node_it_belongs_to(tmp_path):
  """Permuting the map must permute which node each coil hears, not merely
  change the answer. A fold that applied the map in the wrong row order, or
  broadcast one node's value everywhere, satisfies "the signal moved"."""
  phantom = _irregular_phantom(tmp_path / 'local.vtu')
  weights = _nodal_weights(phantom)
  n = phantom.local_nodes.shape[0]

  rng = np.random.default_rng(23)
  C = (rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))).astype(np.complex64)
  Mxy = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
  order = np.array([2, 4, 0, 3, 1])

  phantom.set_receive_sensitivity(C)
  plain = _dc_signal(phantom, Mxy)
  phantom.set_receive_sensitivity(C[order])
  permuted = _dc_signal(phantom, Mxy)

  expect = np.array([(weights * Mxy * C[order][:, c]).sum() for c in range(2)])
  assert np.abs(permuted - expect).max() < 1e-6 * np.abs(expect).max()
  assert np.abs(permuted - plain).max() > 0.1 * np.abs(plain).max(), (
    'this permutation left the signal unchanged, so it pins nothing')


def test_clearing_the_receive_map_restores_the_plain_signal(tmp_path):
  """`None` must leave no trace -- bit-identical, not merely close, since the
  fold is skipped rather than multiplied by ones."""
  phantom = _irregular_phantom(tmp_path / 'clear.vtu')
  n = phantom.local_nodes.shape[0]
  rng = np.random.default_rng(29)
  Mxy = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)

  before = _dc_signal(phantom, Mxy)
  phantom.set_receive_sensitivity(
    (rng.normal(size=(n, 4)) + 1j * rng.normal(size=(n, 4))).astype(np.complex64))
  assert _dc_signal(phantom, Mxy).size == 4
  phantom.set_receive_sensitivity(None)
  after = _dc_signal(phantom, Mxy)
  assert after.size == 1
  assert after[0] == before[0], (
    f'clearing the map changed the signal: {before[0]} -> {after[0]}')


@pytest.mark.parametrize('bad,match', [
  ('short', 'rows for'),
  ('nan', 'not finite'),
  ('rank3', 'expected'),
], ids=['short', 'nonfinite', 'three_dimensional'])
def test_a_bad_receive_map_is_refused(tmp_path, bad, match):
  """A short map is read out of bounds by the assembler under `-DNDEBUG` and a
  non-finite one turns every sample that coil touches into NaN, so neither may
  be accepted. Refused at the setter, before any redistribution."""
  phantom = _irregular_phantom(tmp_path / f'bad_{bad}.vtu')
  n = phantom.local_nodes.shape[0]
  C = np.ones((n, 2), dtype=np.complex64)
  if bad == 'short':
    C = C[:-1]
  elif bad == 'nan':
    C[2, 1] = np.nan
  else:
    C = C.reshape(n, 2, 1)
  with pytest.raises(ValueError, match=match):
    phantom.set_receive_sensitivity(C)


# ---------------------------------------------------------------------------
# The six-monomial sweep, the null line, and the paths the readout term had
# never been driven through.
#
# Written during audit 5. The `xy` coefficient was multiplied by zero in every
# existing test -- a gradient held on one axis gives |p3|/max|p| == 0.000
# exactly -- so `m3 * mxy` in all three kernels was executed by nothing.
# `signal_nodal` with `maxwell` had no test at all, and that gap was hiding a
# read of uninitialised memory.


def _maxwell_phantom(tmp_path, name, *, nodal, lumped=False):
  """The `_maxwell_fixture` node cloud, assembled either way.

  `nodal=True` gives `nodal_approximation`, which is what builds the
  mass-matrix projection `signal_nodal` reads; `nodal=False` is the quadrature
  configuration every other test here uses. `lumped` makes that projection
  DIAGONAL, which is what lets the nodal identity below be exact -- a
  consistent mass matrix couples neighbouring nodes, so applying the phase to
  the magnetization and letting the kernel apply it at the node are then
  genuinely different quantities.
  """
  import meshio
  points = np.array([[0.11, -0.03, 0.07],
                     [-0.05, 0.12, 0.02],
                     [0.04, 0.06, -0.10],
                     [-0.09, -0.08, 0.05],
                     [0.02, -0.11, -0.06]])
  path = tmp_path / name
  meshio.write(str(path), meshio.Mesh(points, [('tetra',
                                                np.array([[0, 1, 2, 3],
                                                          [1, 2, 4, 3]]))]))
  phantom = FEMPhantom(path=str(path))
  phantom.set_assembler(voxel_size=1e3 if nodal else 0.0, lorder=2, horder=4,
                        nodal_approximation=nodal, lumped=lumped)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 1e9, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))
  return phantom, points


def _six_coefficients():
  """A full quadratic form: all six monomials live, none degenerate.

  Hand-written rather than taken from a gradient, because a gradient whose
  direction is fixed in time produces `p3 == 0` identically -- which is exactly
  why the `xy` branch had no coverage.
  """
  # Scaled so the phase spans a few radians over this 20 cm cloud: the
  # coefficients are rad/m^2 and the coordinates are ~0.1 m.
  return 100.0 * np.array([[7.0, -4.0, 11.0, 5.0, -8.0, 3.0]],
                          dtype=np.float64)


def _monomial_phase(coef_row, points):
  x, y, z = points[:, 0], points[:, 1], points[:, 2]
  return (coef_row[0] * x ** 2 + coef_row[1] * y ** 2 + coef_row[2] * z ** 2
          + coef_row[3] * x * y + coef_row[4] * x * z + coef_row[5] * y * z)


def _dc_inputs():
  zero = np.zeros((1, 1, 1), dtype=np.float32)
  return (zero.copy(), zero.copy(), zero.copy()), zero.copy()


@pytest.mark.parametrize('path_name', ['signal_sum', 'signal_nodal'])
@pytest.mark.parametrize('n_coils', [0, 3], ids=['no_coils', 'three_coils'])
def test_every_maxwell_monomial_reaches_both_nodal_paths(tmp_path, path_name,
                                                         n_coils):
  """On a NODAL path the phase is evaluated at the node, so applying it to the
  magnetization instead must give the identical signal.

  `signal_nodal` is driven with a LUMPED mass matrix here, because that is
  what makes the projection diagonal in the node index. With the consistent
  matrix the two sides are different quantities -- `sum_n (M Mxy)_n e^(i phi_n)`
  against `sum_n (M (Mxy e^(i phi)))_n` -- and the linearity test below is what
  covers that configuration instead.

  The identity needs no knowledge of the mass matrix and no quadrature
  reference, and it is sharp: it fixes the sign, the factor and the position
  index of every one of the six monomials independently. The mutation below
  drives the point -- negating the `xy` coefficient alone has to be visible,
  and before this nothing in the suite multiplied `m3` by anything but zero.

  The coil arm runs the same identity per channel, which is the first test of
  `maxwell` together with `nv > 1`.
  """
  nodal = path_name == 'signal_nodal'
  phantom, points = _maxwell_phantom(tmp_path, f'{path_name}_{n_coils}.vtu',
                                     nodal=nodal, lumped=nodal)
  n = points.shape[0]
  coef = _six_coefficients()
  phase = _monomial_phase(coef[0], points)
  assert np.abs(phase).max() > 0.5, 'this form produces no phase to speak of'

  rng = np.random.default_rng(41)
  Mxy = (rng.normal(size=(n, 2))
         + 1j * rng.normal(size=(n, 2))).astype(np.complex64)
  if n_coils:
    C = (rng.normal(size=(n, n_coils))
         + 1j * rng.normal(size=(n, n_coils))).astype(np.complex64)
    phantom.set_receive_sensitivity(C)

  call = getattr(phantom, path_name)
  pts, t = _dc_inputs()

  phantom.update_magnetization(Mxy)
  got = np.asarray(call(pts, t, None, maxwell=coef)).reshape(-1)

  # The same phase, applied to the magnetization before the handoff.
  phantom.update_magnetization(
      np.ascontiguousarray(Mxy * np.exp(1j * phase)[:, None],
                           dtype=np.complex64))
  expected = np.asarray(call(pts, t, None)).reshape(-1)

  scale = float(np.abs(expected).max())
  assert got.size == 2 * max(n_coils, 1)
  assert float(np.abs(got - expected).max()) < 3e-6 * scale, (
    f'{path_name}: the kernel phase disagrees with the same phase applied to '
    f'the magnetization')

  # Not vacuous: each coefficient on its own has to change the answer.
  phantom.update_magnetization(Mxy)
  for j in range(6):
    flipped = coef.copy()
    flipped[0, j] = -flipped[0, j]
    moved = np.asarray(call(pts, t, None, maxwell=flipped)).reshape(-1)
    assert float(np.abs(moved - got).max()) > 1e-3 * scale, (
      f'{path_name}: coefficient {j} of the quadratic form changes nothing, '
      f'so this test cannot see it')


def test_every_maxwell_monomial_reaches_the_quadrature_path(tmp_path):
  """The quadrature path evaluates the phase AT the quadrature points, so the
  nodal identity above does not hold for it -- `exp(i phi)` interpolated from
  the nodes is not `exp(i phi)` integrated over the element, and at these
  element sizes the two differ by tens of percent.

  What is checkable exactly is that the term is still LINEAR in the nodal
  magnetization and factorises over the coil axis, which is what the fold and
  the `nv` bookkeeping have to get right, plus that all six coefficients are
  read.
  """
  phantom, points = _maxwell_phantom(tmp_path, 'quad_maxwell.vtu', nodal=False)
  n = points.shape[0]
  coef = _six_coefficients()
  pts, t = _dc_inputs()

  rng = np.random.default_rng(43)
  C = (rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))).astype(np.complex64)
  phantom.set_receive_sensitivity(C)

  # Weights of the quadratic-phase functional, measured off the assembler one
  # node at a time: w[j, c] = integral of N_j exp(i phi) C[j, c].
  w = np.zeros((n, 3), dtype=np.complex128)
  for j in range(n):
    unit = np.zeros((n, 1), dtype=np.complex64)
    unit[j, 0] = 1.0
    phantom.update_magnetization(unit)
    w[j] = np.asarray(phantom.signal(pts, t, None, maxwell=coef)).reshape(-1)

  Mxy = (rng.normal(size=(n, 2))
         + 1j * rng.normal(size=(n, 2))).astype(np.complex64)
  phantom.update_magnetization(Mxy)
  got = np.asarray(phantom.signal(pts, t, None, maxwell=coef)).reshape(-1)
  predicted = np.array([(w[:, c] * Mxy[:, e]).sum()
                        for e in range(2) for c in range(3)])
  scale = float(np.abs(predicted).max())
  assert float(np.abs(got - predicted).max()) < 1e-5 * scale

  # The measured weights must differ across nodes, or the mesh is too regular
  # for a mispaired row to show.
  assert float(np.abs(w[:, 0]).max() / np.abs(w[:, 0]).min()) > 2.0

  for j in range(6):
    flipped = coef.copy()
    flipped[0, j] = -flipped[0, j]
    moved = np.asarray(phantom.signal(pts, t, None,
                                      maxwell=flipped)).reshape(-1)
    assert float(np.abs(moved - got).max()) > 1e-3 * scale, (
      f'coefficient {j} changes nothing on the quadrature path')


def test_signal_nodal_refuses_a_phantom_that_never_built_its_projection(tmp_path):
  """`signal_nodal` reads `f_M_Mxy_nodes_`, which only
  `update_nodal_magnetization` writes -- and Python calls that only when the
  assembler was built with `nodal_approximation=True`.

  `Phantom.signal_nodal` is public and unconditional, so on a phantom built
  the other way every read was a `middleRows` on a 0 x 0 matrix multiplied
  against a `q_count`-long row -- a dimension mismatch that only `eigen_assert`
  would catch, and `-DNDEBUG` compiles that out. Measured on this build it
  returned `0+0j` on every run: a silently EMPTY k-space rather than a crash,
  which is the worst of the available outcomes.
  """
  phantom, _points = _maxwell_phantom(tmp_path, 'no_projection.vtu',
                                      nodal=False)
  n = phantom.local_nodes.shape[0]
  phantom.update_magnetization(np.ones((n, 1), dtype=np.complex64))
  pts, t = _dc_inputs()
  with pytest.raises(RuntimeError, match='update_nodal_magnetization'):
    phantom.signal_nodal(pts, t, None)


def test_the_signal_paths_refuse_an_assembler_with_no_magnetization(tmp_path):
  """`nv_` is set by the magnetization updaters and was uninitialised in the
  constructor, so a signal call before any update sized its output matrix from
  whatever was on the stack."""
  phantom, _points = _maxwell_phantom(tmp_path, 'no_magnetization.vtu',
                                      nodal=False)
  pts, t = _dc_inputs()
  for name in ('signal_sum', 'signal', 'signal_nodal'):
    with pytest.raises(RuntimeError, match='no magnetization'):
      getattr(phantom, name)(pts, t, None)


def _rod_along(path, direction, length=0.24, n_segments=6, width=1e-4):
  """The pseudo-1D rod of `_phantom_fixtures`, pointed along `direction`."""
  import meshio
  d = np.asarray(direction, dtype=np.float64)
  d = d / np.linalg.norm(d)
  # Any two unit vectors orthogonal to d; the rod's cross-section is 1e-4 m, so
  # nothing about the choice is observable.
  helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
  e1 = np.cross(d, helper); e1 /= np.linalg.norm(e1)
  e2 = np.cross(d, e1)
  s = np.linspace(-0.5 * length, 0.5 * length, n_segments + 1)
  h = 0.5 * width
  pts = np.zeros((4 * (n_segments + 1), 3))
  for i, si in enumerate(s):
    c = si * d
    pts[4 * i + 0] = c - h * e1 - h * e2
    pts[4 * i + 1] = c + h * e1 - h * e2
    pts[4 * i + 2] = c + h * e1 + h * e2
    pts[4 * i + 3] = c - h * e1 + h * e2
  cells = []
  for i in range(n_segments):
    a, b = 4 * i, 4 * (i + 1)
    cells += [[a + 0, a + 1, a + 2, b + 0], [a + 2, a + 3, a + 0, b + 0],
              [a + 2, b + 0, b + 1, b + 2], [a + 2, a + 3, b + 0, b + 3],
              [a + 2, b + 0, b + 2, b + 3]]
  meshio.write(str(path), meshio.Mesh(pts, [('tetra', np.array(cells))]))
  phantom = FEMPhantom(path=str(path))
  phantom.set_assembler(voxel_size=0.0, lorder=2, horder=4,
                        nodal_approximation=False, lumped=False)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 1e9, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))
  phantom.update_magnetization(np.ones(n, dtype=np.complex64))
  return phantom


def test_the_concomitant_phase_vanishes_on_its_null_line(tmp_path):
  """`Bc = [(u.r)^2 + (v.r)^2] / (2 B0)` with `u = (-Gz/2, 0, Gx)` and
  `v = (0, -Gz/2, Gy)`, so its matrix is `(u u^T + v v^T)/(2 B0)`: rank at most
  two, positive semidefinite, and identically zero along `(2Gx, 2Gy, Gz)`.

  This is the ONLY property that can catch a cross-term sign error. Negating
  `Gx*Gz*x*z` turns `(Gx z - Gz x/2)^2` into `(Gx z + Gz x/2)^2` -- still a sum
  of squares with the same eigenvalues, because the flip is conjugation by
  `diag(1, 1, -1)`, an orthogonal similarity. No rank, determinant or
  "can only retard" test can see it; what moves is the null LINE. Measured on
  this rod, where the correct form gives 5.6e-17 rad: negating the `xz`
  coefficient puts 0.801 rad on it and negating `yz` 0.331 rad.
  """
  from feelmri.MRObjects import Scanner

  scanner = Scanner()
  G = (14.0, -9.0, 20.0)
  times, coef = _constant_gradient_coefficients(G, 3.0, scanner)
  null = np.array([2.0 * G[0], 2.0 * G[1], G[2]])

  phantom = _rod_along(tmp_path / 'nullrod.vtu', null)
  nodes = phantom.local_nodes.astype(np.float64)
  on_line = _monomial_phase(coef[1], nodes)
  # The residual is the rod's own CROSS-SECTION, not the algebra: the form is
  # PSD and stationary on the line, so an offset `h` off it costs order
  # `lambda h^2` -- 3.2e-07 rad at the 1e-4 m width used here, and it does not
  # grow along the rod. On the line itself the form is zero to 5.6e-17.
  assert float(np.abs(on_line).max()) < 1e-6, (
    f'the analytic form is not zero on its own null line: '
    f'{np.abs(on_line).max():.3e} rad')
  exact = _monomial_phase(coef[1], np.outer(np.linspace(-0.12, 0.12, 7),
                                            null / np.linalg.norm(null)))
  assert float(np.abs(exact).max()) < 1e-15

  zero3 = np.zeros((2, 1, 1), dtype=np.float32)
  t3 = times.reshape(2, 1, 1).astype(np.float32)
  pts = (zero3.copy(), zero3.copy(), zero3.copy())
  plain = np.asarray(phantom.signal(pts, t3, None)).reshape(-1)
  with_term = np.asarray(phantom.signal(pts, t3, None,
                                        maxwell=coef)).reshape(-1)
  scale = float(np.abs(plain).max())
  assert float(np.abs(with_term - plain).max()) < 1e-5 * scale, (
    'the readout term is not zero along the null line')

  # A cross-term sign error moves the null line, and this is where it shows.
  for j in (4, 5):
    flipped = coef.copy()
    flipped[:, j] = -flipped[:, j]
    moved = np.asarray(phantom.signal(pts, t3, None,
                                      maxwell=flipped)).reshape(-1)
    assert float(np.abs(moved - plain).max()) > 0.1 * scale, (
      f'negating coefficient {j} leaves the null line where it was')

  # A rod along a DIFFERENT direction must see a large phase, or the fixture
  # would pass with the whole term switched off.
  off = _rod_along(tmp_path / 'offrod.vtu', np.array([1.0, 0.0, 0.0]))
  off_phase = _monomial_phase(coef[1], off.local_nodes.astype(np.float64))
  assert float(np.abs(off_phase).max()) > 0.1, (
    'the same coefficients produce no phase anywhere, so a rod that reads zero '
    'proves nothing')


def test_a_receive_map_does_not_survive_a_repartition(tmp_path):
  """The map is one value per LOCAL node, so a repartition invalidates it.

  `set_receive_sensitivity` did not mark the partition as bound, so
  `enable_dual_partition`'s own refusal never fired, and `distribute_mesh` did
  not clear the map either. At one rank the row counts coincide and nothing
  shows; at several ranks the map is silently paired with the wrong nodes, or
  raises a bare numpy broadcast error from inside
  `_update_magnetization_local` -- which is AFTER an `Alltoallv`, the hang
  shape this module works hard to avoid.
  """
  phantom, _points = _maxwell_phantom(tmp_path, 'repart.vtu', nodal=False)
  n = phantom.local_nodes.shape[0]
  phantom.set_receive_sensitivity(np.ones((n, 2), dtype=np.complex64))
  assert phantom._partition_bound is True

  # Refused up front, rather than left to fail inside a collective later.
  with pytest.raises(RuntimeError, match='already in use'):
    phantom.enable_dual_partition(voxel_size=0.0, lorder=2, horder=4,
                                  nodal_approximation=False)

  # And a repartition that does go ahead drops it rather than mispairing it.
  phantom.distribute_mesh(graph_type='nodal')
  assert phantom._receive_sensitivity is None
  phantom.update_magnetization(np.ones(phantom.local_nodes.shape[0],
                                       dtype=np.complex64))
  pts, t = _dc_inputs()
  assert np.asarray(phantom.signal_sum(pts, t, None)).size == 1


def test_orienting_after_set_assembler_is_refused(tmp_path):
  """The assembler captures the node coordinates in its constructor -- node
  positions, element sizes, the quadrature cache, the mass matrix and the
  ownership mask all come from them -- so moving the mesh afterwards leaves
  every one of them describing a phantom that no longer exists.

  Nothing downstream notices; the signal is simply computed at the old
  positions. Measured on a 23 deg tilt, the solver-to-assembler handoff came
  apart by 1.46 relative, which is how this was found.
  """
  from pint import Quantity as Q_
  th = np.deg2rad(23.0)
  R = np.array([[np.cos(th), 0.0, np.sin(th)],
                [0.0, 1.0, 0.0],
                [-np.sin(th), 0.0, np.cos(th)]])
  phantom, _points = _maxwell_fixture(tmp_path, 'orient_late.vtu')
  with pytest.raises(RuntimeError, match='set_assembler has already'):
    phantom.orient(R, Q_(np.zeros(3), 'm'))
  with pytest.raises(RuntimeError, match='set_assembler has already'):
    phantom.reorient(R, Q_(np.zeros(3), 'm'))

  # The right order is accepted and leaves the orientation where the solver
  # and the readout helpers both look for it.
  ok, _points = _maxwell_fixture(tmp_path, 'orient_early.vtu', orientation=R)
  assert np.allclose(ok._orientation, R)


def test_coils_motion_and_the_maxwell_term_compose_in_one_readout(tmp_path):
  """The three signal-side features at once, against the closed form.

  Each had been tested alone. Together they share one loop and one `nv` axis:
  the quadratic phase is evaluated at the DEFORMED position (it has to follow
  the material point, exactly as `-k.x` does), the coil map multiplies into the
  same axis the encodings use, and the whole thing is one `signal_sum`.

  `signal_sum` is an unweighted nodal sum, so
  `S[e*n_coils+c] = sum_n Mxy[n,e] C[n,c] exp(i phi(x_n + u_n))` is exact.
  """
  from feelmri.Motion import POD

  phantom, points = _maxwell_phantom(tmp_path, 'coil_pod_maxwell.vtu',
                                     nodal=False)
  n = points.shape[0]
  rng = np.random.default_rng(53)

  # A two-mode POD whose displacement is large enough to matter against a
  # phase that is quadratic in position.
  snap_times = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
  data = (0.02 * rng.normal(size=(n, 3, snap_times.size))).astype(np.float32)
  pod = POD(times=snap_times, data=data, n_modes=2, is_periodic=False)

  coef = np.vstack([_six_coefficients()[0],
                    -0.7 * _six_coefficients()[0]])
  sample_times = np.array([0.4, 2.6], dtype=np.float32)

  C = (rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))).astype(np.complex64)
  Mxy = (rng.normal(size=(n, 2))
         + 1j * rng.normal(size=(n, 2))).astype(np.complex64)
  phantom.set_receive_sensitivity(C)
  phantom.update_magnetization(Mxy)

  zero = np.zeros((2, 1, 1), dtype=np.float32)
  pts = (zero.copy(), zero.copy(), zero.copy())
  got = np.asarray(phantom.signal_sum(pts, sample_times.reshape(2, 1, 1),
                                      pod, maxwell=coef)).reshape(-1)
  assert got.size == 2 * 2 * 3

  # The deformed node positions, from the same modes and weights the kernel is
  # handed: x(t) = x0 + Phi w(t).
  modes = pod.get_modes(n).astype(np.float64)
  weights = pod.get_weights(sample_times).astype(np.float64)
  expected = []
  for s in range(sample_times.size):
    deformed = points + np.einsum('ncm,m->nc', modes, weights[s])
    phase = np.exp(1j * _monomial_phase(coef[s], deformed))
    for e in range(2):
      for c in range(3):
        expected.append((Mxy[:, e] * C[:, c] * phase).sum())
  expected = np.array(expected)

  scale = float(np.abs(expected).max())
  assert float(np.abs(got - expected).max()) < 3e-6 * scale

  # Not vacuous on any of the three: dropping each in turn must change it.
  still = np.asarray(phantom.signal_sum(pts, sample_times.reshape(2, 1, 1),
                                        None, maxwell=coef)).reshape(-1)
  assert float(np.abs(still - got).max()) > 1e-3 * scale, 'the motion does nothing'
  flat = np.asarray(phantom.signal_sum(pts, sample_times.reshape(2, 1, 1),
                                       pod)).reshape(-1)
  assert float(np.abs(flat - got).max()) > 1e-3 * scale, 'the maxwell term does nothing'
  assert float(np.abs(np.abs(C) - 1.0).max()) > 0.1, 'the coil map is trivial'
