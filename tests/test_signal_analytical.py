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
