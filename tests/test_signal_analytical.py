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

from _phantom_fixtures import (
  make_1d_rod_mesh,
  make_2d_disk_mesh,
  make_minimal_tet_mesh,
)


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

def test_signal_at_k0_is_real_positive(tmp_path):
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')

  mesh_path = tmp_path / 'tet.vtu'
  make_minimal_tet_mesh(mesh_path)
  phantom = FEMPhantom(path=str(mesh_path))
  _build_phantom_at_zero_field(phantom, voxel_size=5e-3)

  n = phantom.local_nodes.shape[0]
  phantom.update_magnetization(np.ones((n, 1), dtype=np.complex64))

  kx = np.array([0.0], dtype=np.float32)
  pts, t3 = _to_3d_inputs(kx, kx, kx, kx)
  S = phantom.mri_signal(pts, t3, None)

  s0 = complex(np.asarray(S).reshape(-1)[0])
  assert np.isfinite(s0.real) and np.isfinite(s0.imag)
  assert s0.real > 0.0, f'S(0) should be positive real; got {s0}'
  # At k=0 with constant Mxy=1 and no phase the imaginary part comes
  # only from FE quadrature noise; it must be much smaller than the
  # real part.
  assert abs(s0.imag) < 0.10 * abs(s0.real), (
    f'|imag(S(0))|={abs(s0.imag):.3g} too large vs real={s0.real:.3g}'
  )


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
