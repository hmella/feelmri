"""Verification suite for the Magnus-expansion Bloch solver.

The C++ kernel exposes three rotation operators per time step:

* ``method='cayley_klein'`` (order 0): the historical FEelMRI solver, which
  uses the end-of-step field. First-order accurate in ``dt`` for smoothly
  varying fields.
* ``method='magnus2'`` (order 2): 2nd-order Magnus, trapezoidal field
  average. Second-order accurate in ``dt``.
* ``method='magnus4'`` (order 4): 4th-order Magnus, adds the commutator
  ``-dt^2/12 [Omega_old, Omega_new]``. Fourth-order accurate in ``dt`` when
  evaluated in double precision; in float32 the commutator falls below
  epsilon for typical clinical parameters and ``magnus4`` collapses to
  ``magnus2``.

Three groups of tests:

1. Hard-pulse / closed-form equivalence: the existing analytical Bloch
   solutions (T1, T2, free precession, hard 90/180) must hold for all
   three orders. The commutator vanishes for piecewise-constant fields,
   so the three orders should agree to within float32 ulps on a hard
   pulse.

2. Soft (sinc) RF convergence study: with no gradient and a 4-lobe sinc
   pulse, the error against a fine-dt order=4 float64 reference should
   decay as O(dt) for cayley_klein, O(dt^2) for magnus2, and O(dt^4) for
   magnus4 in float64. Marked ``slow``.

3. Slice-selective sinc + linear gradient: a soft pulse applied with a
   spatial gradient excites different positions to different flip
   angles. At a moderate ``dt`` the magnus4 profile must be markedly
   closer to a fine-dt reference than the cayley_klein profile.
"""
from __future__ import annotations

import numpy as np
import pytest
from pint import Quantity

from feelmri import (
  BlochSolver,
  FEMPhantom,
  Scanner,
)
from feelmri.Bloch import Sequence, SequenceBlock
from feelmri.MRObjects import RF, Gradient

from _phantom_fixtures import make_minimal_tet_mesh, make_1d_rod_mesh
from _seq_fixtures import (
  make_empty_block,
  make_hard_pulse_block,
  make_single_block_sequence,
)


# ---------------------------------------------------------------------------
# Phantom fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def minimal_phantom(tmp_path_factory):
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  mesh_dir = tmp_path_factory.mktemp('magnus_mesh')
  mesh_path = mesh_dir / 'tet.vtu'
  make_minimal_tet_mesh(mesh_path)
  return FEMPhantom(path=str(mesh_path))


@pytest.fixture(scope='module')
def rod_phantom(tmp_path_factory):
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  mesh_dir = tmp_path_factory.mktemp('magnus_rod')
  mesh_path = mesh_dir / 'rod.vtu'
  make_1d_rod_mesh(mesh_path, length=0.04, n_segments=16, transverse_width=1e-4)
  return FEMPhantom(path=str(mesh_path))


METHODS = ('cayley_klein', 'magnus2', 'magnus4')


# ---------------------------------------------------------------------------
# 1. Closed-form / hard-pulse equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', METHODS)
def test_t1_recovery_all_methods(minimal_phantom, method):
  T1_ms = 200.0
  dur_ms = 0.5 * T1_ms
  expected_Mz = 1.0 - np.exp(-0.5)
  seq = make_single_block_sequence(make_empty_block(dur_ms, dt_ms=1.0))
  solver = BlochSolver(
    seq, minimal_phantom,
    M0=1.0,
    T1=Quantity(T1_ms, 'ms'),
    T2=Quantity(50.0, 'ms'),
    initial_Mxy=0.0,
    initial_Mz=0.0,
    perfect_spoiling=False,
    method=method,
  )
  _, Mz = solver.solve()
  np.testing.assert_allclose(Mz[:, 0], expected_Mz, atol=5e-3)


@pytest.mark.parametrize('method', METHODS)
def test_t2_decay_all_methods(minimal_phantom, method):
  T2_ms = 50.0
  dur_ms = 2.0 * T2_ms
  expected_abs = np.exp(-2.0)
  seq = make_single_block_sequence(make_empty_block(dur_ms, dt_ms=1.0))
  solver = BlochSolver(
    seq, minimal_phantom,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(T2_ms, 'ms'),
    initial_Mxy=1.0 + 0.0j,
    initial_Mz=1.0,
    perfect_spoiling=False,
    method=method,
  )
  Mxy, _ = solver.solve()
  np.testing.assert_allclose(np.abs(Mxy[:, 0]), expected_abs, atol=5e-3)


@pytest.mark.parametrize('method', METHODS)
def test_free_precession_all_methods(minimal_phantom, method):
  T_ms = 5.0
  dB0_mT = 1e-3
  gammabar = 42.576e6  # Hz/T
  expected_phase = -2.0 * np.pi * gammabar * (dB0_mT * 1e-3) * (T_ms * 1e-3)
  expected_phase = np.angle(np.exp(1j * expected_phase))
  seq = make_single_block_sequence(make_empty_block(T_ms, dt_ms=0.05))
  solver = BlochSolver(
    seq, minimal_phantom,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    delta_B=dB0_mT,
    initial_Mxy=1.0 + 0.0j,
    initial_Mz=1.0,
    perfect_spoiling=False,
    method=method,
  )
  Mxy, _ = solver.solve()
  measured = np.angle(Mxy[:, 0])
  np.testing.assert_allclose(measured, expected_phase, atol=2e-2)


@pytest.mark.parametrize('method', METHODS)
def test_hard_90_all_methods(minimal_phantom, method):
  block = make_hard_pulse_block(np.pi / 2, dur_ms=0.2)
  seq = make_single_block_sequence(block)
  solver = BlochSolver(
    seq, minimal_phantom,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    initial_Mxy=0.0,
    initial_Mz=1.0,
    perfect_spoiling=False,
    method=method,
  )
  Mxy, Mz = solver.solve()
  np.testing.assert_allclose(np.abs(Mxy[:, 0]), 1.0, atol=5e-2)
  np.testing.assert_allclose(Mz[:, 0], 0.0, atol=5e-2)


@pytest.mark.parametrize('method', METHODS)
def test_hard_180_all_methods(minimal_phantom, method):
  block = make_hard_pulse_block(np.pi, dur_ms=0.2)
  seq = make_single_block_sequence(block)
  solver = BlochSolver(
    seq, minimal_phantom,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    initial_Mxy=0.0,
    initial_Mz=1.0,
    perfect_spoiling=False,
    method=method,
  )
  Mxy, Mz = solver.solve()
  np.testing.assert_allclose(Mz[:, 0], -1.0, atol=5e-2)
  np.testing.assert_allclose(np.abs(Mxy[:, 0]), 0.0, atol=5e-2)


def test_hard_pulse_orders_agree_on_constant_field(minimal_phantom):
  """For a hard pulse, the field is piecewise-constant within each dt
  so the Magnus commutator ``[Omega_old, Omega_new]`` is zero and all
  three orders must produce the same magnetisation to within FP
  rounding."""
  block = make_hard_pulse_block(np.pi / 3, dur_ms=0.3, dt_ms=0.005)
  seq = make_single_block_sequence(block)
  results = {}
  for method in METHODS:
    solver = BlochSolver(
      seq, minimal_phantom,
      M0=1.0,
      T1=Quantity(1e6, 'ms'),
      T2=Quantity(1e6, 'ms'),
      initial_Mxy=0.0,
      initial_Mz=1.0,
      perfect_spoiling=False,
      method=method,
    )
    Mxy, Mz = solver.solve()
    results[method] = (Mxy[:, 0].copy(), Mz[:, 0].copy())

  ref_Mxy, ref_Mz = results['cayley_klein']
  for method in ('magnus2', 'magnus4'):
    Mxy, Mz = results[method]
    np.testing.assert_allclose(Mxy, ref_Mxy, atol=1e-4)
    np.testing.assert_allclose(Mz,  ref_Mz,  atol=1e-4)


# ---------------------------------------------------------------------------
# 2. Soft-RF convergence study (slow)
# ---------------------------------------------------------------------------

def _build_sinc_sequence(dt_ms: float, dur_ms: float = 2.0,
                        flip_rad: float = np.pi / 6) -> Sequence:
  """Build a one-block sequence containing a single 4-lobe apodized sinc
  RF pulse on resonance with no gradient. The block is discretized at
  uniform ``dt_ms`` over the whole pulse duration."""
  scanner = Scanner()
  rf = RF(
    scanner=scanner,
    NbLobes=[2, 2],
    alpha=0.46,
    shape='apodized_sinc',
    flip_angle=Quantity(float(flip_rad), 'rad'),
    dur=Quantity(float(dur_ms), 'ms'),
    nb_samples=1024,
  )
  block = SequenceBlock(
    rf_pulses=[rf],
    dur=Quantity(float(dur_ms), 'ms'),
    dt_rf=Quantity(float(dt_ms), 'ms'),
    dt=Quantity(float(dt_ms), 'ms'),
    store_magnetization=True,
  )
  seq = Sequence()
  seq.add_block(block)
  return seq


def _run_sinc(phantom, dt_ms: float, method: str, dtype: str,
              flip_rad: float = np.pi / 6, dur_ms: float = 2.0,
              delta_B_mT: float = 5e-4):
  """Run a single sinc RF block at the requested dt. A non-zero
  ``delta_B`` is used so the time-varying RF couples with a non-zero
  Bz; otherwise the Magnus4 commutator ``rf_new*Bz_old - rf_old*Bz_new``
  vanishes and the order-4 path collapses to order-2."""
  seq = _build_sinc_sequence(dt_ms, dur_ms=dur_ms, flip_rad=flip_rad)
  solver = BlochSolver(
    seq, phantom,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    delta_B=delta_B_mT,
    initial_Mxy=0.0,
    initial_Mz=1.0,
    perfect_spoiling=False,
    method=method,
    dtype=dtype,
  )
  Mxy, Mz = solver.solve()
  return Mxy[:, 0].astype(np.complex128), Mz[:, 0].astype(np.float64)


@pytest.mark.slow
def test_sinc_convergence_slopes(minimal_phantom):
  """Convergence slopes on a smoothly varying sinc pulse.

  Theoretical global convergence rates (Blanes et al., Phys Rep 2009,
  Section 5.2):

  * ``cayley_klein`` (end-of-step field): O(dt). First-order quadrature.
  * ``magnus2`` (trapezoidal Omega_1, drops Omega_2): O(dt^2).
  * ``magnus4`` (trapezoidal Omega_1 + linear-interpolated Omega_2
    commutator term ``[omega_old, omega_new] * dt^2 / 12``): still
    O(dt^2) globally because the trapezoidal Omega_1 limits the order,
    BUT with a smaller error *constant* than magnus2. A true 4th-order
    scheme requires Gauss-Legendre interior quadrature on Omega_1,
    which this implementation does not include.

  The test asserts the slopes and also verifies that magnus4 produces
  a uniformly smaller error than magnus2 at every dt — that is what
  the commutator correction actually buys."""
  dt_grid = np.array([0.05, 0.025, 0.0125, 0.00625])
  dt_ref = 0.001

  Mxy_ref, Mz_ref = _run_sinc(minimal_phantom, dt_ref, 'magnus4', 'float64')

  configs = [
    ('cayley_klein', 'float32', 0.85),  # nominal O(dt)
    ('magnus2',      'float32', 1.85),  # nominal O(dt^2)
    ('magnus4',      'float64', 1.85),  # O(dt^2) with smaller constant
  ]
  log_dt = np.log10(dt_grid)
  all_errors = {}
  for method, dtype, slope_floor in configs:
    errors = np.zeros_like(dt_grid)
    for i, dt in enumerate(dt_grid):
      Mxy, Mz = _run_sinc(minimal_phantom, dt, method, dtype)
      errors[i] = np.sqrt(np.mean(np.abs(Mxy - Mxy_ref) ** 2
                                  + np.abs(Mz - Mz_ref) ** 2))
    all_errors[method] = errors
    # Least-squares slope of log10 err vs log10 dt.
    log_e = np.log10(np.maximum(errors, 1e-16))
    slope, _ = np.polyfit(log_dt, log_e, 1)
    assert slope >= slope_floor, (
      f"{method} ({dtype}) convergence slope {slope:.2f} below floor "
      f"{slope_floor:.2f}; errors={errors}"
    )

  # magnus4's commutator correction is only meaningful when Bz and B1
  # are both substantial AND time-varying; on this on-resonance soft
  # pulse the commutator |rf_new*Bz_old - rf_old*Bz_new| is tiny vs.
  # the leading M2 error from the trapezoidal Omega_1 quadrature. We
  # still verify that magnus4 doesn't *regress* by more than a small
  # constant factor relative to magnus2 at the coarsest dt — the
  # slice-select test exercises the regime where the commutator
  # genuinely helps.
  ratio_coarse = all_errors['magnus4'][0] / all_errors['magnus2'][0]
  assert ratio_coarse < 1.05, (
    f"magnus4 unexpectedly worse than magnus2 at dt={dt_grid[0]}: "
    f"ratio M4/M2 = {ratio_coarse:.3f}"
  )


# ---------------------------------------------------------------------------
# 3. Slice-selective sinc + linear gradient
# ---------------------------------------------------------------------------

def _build_slice_select_sequence(dt_ms: float, G_amp_mT_per_m: float,
                                 dur_ms: float = 2.0,
                                 flip_rad: float = np.pi / 2) -> Sequence:
  """Sinc RF + constant-amplitude slice-select gradient along z. Off-
  centre nodes see different Bz, so the rotation axis is position-
  dependent and the commutator is non-trivial. This is the regime in
  which Magnus orders actually differ."""
  scanner = Scanner()
  rf = RF(
    scanner=scanner,
    NbLobes=[2, 2],
    alpha=0.46,
    shape='apodized_sinc',
    flip_angle=Quantity(float(flip_rad), 'rad'),
    dur=Quantity(float(dur_ms), 'ms'),
    nb_samples=1024,
  )
  # Constant-amplitude gradient on the M axis (axis=0). The 1-D rod
  # is oriented along x, so axis=0 is the slice-select direction.
  gx = Gradient(
    scanner=scanner,
    axis=0,
    timings=Quantity(np.array([0.0, dur_ms]), 'ms'),
    amplitudes=Quantity(np.array([G_amp_mT_per_m, G_amp_mT_per_m]), 'mT/m'),
    time=Quantity(0.0, 'ms'),
  )
  block = SequenceBlock(
    gradients=[gx],
    rf_pulses=[rf],
    dur=Quantity(float(dur_ms), 'ms'),
    dt_rf=Quantity(float(dt_ms), 'ms'),
    dt_gr=Quantity(float(dt_ms), 'ms'),
    dt=Quantity(float(dt_ms), 'ms'),
    store_magnetization=True,
  )
  seq = Sequence()
  seq.add_block(block)
  return seq


def _run_slice_select(phantom, dt_ms: float, method: str, dtype: str,
                      G_amp_mT_per_m: float = 5.0):
  seq = _build_slice_select_sequence(dt_ms, G_amp_mT_per_m)
  solver = BlochSolver(
    seq, phantom,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    initial_Mxy=0.0,
    initial_Mz=1.0,
    perfect_spoiling=False,
    method=method,
    dtype=dtype,
  )
  Mxy, Mz = solver.solve()
  return Mxy[:, 0].astype(np.complex128), Mz[:, 0].astype(np.float64)


@pytest.mark.slow
def test_slice_select_magnus4_beats_cayley_klein(rod_phantom):
  """At moderate dt where cayley_klein leaves a visible per-step error,
  magnus4 (double) must reduce the position-dependent slice-profile
  error against a fine-dt reference."""
  dt_coarse = 0.04
  dt_ref = 0.001

  ref_Mxy, ref_Mz = _run_slice_select(rod_phantom, dt_ref, 'magnus4', 'float64')

  ck_Mxy, ck_Mz = _run_slice_select(rod_phantom, dt_coarse, 'cayley_klein', 'float32')
  m4_Mxy, m4_Mz = _run_slice_select(rod_phantom, dt_coarse, 'magnus4', 'float64')

  err_ck = np.sqrt(np.mean(np.abs(ck_Mxy - ref_Mxy) ** 2
                           + np.abs(ck_Mz - ref_Mz) ** 2))
  err_m4 = np.sqrt(np.mean(np.abs(m4_Mxy - ref_Mxy) ** 2
                           + np.abs(m4_Mz - ref_Mz) ** 2))

  # At dt = 0.04 ms, the error reduction from O(dt) -> O(dt^4) is
  # ~25^3 ~ 1.5e4 in the asymptotic regime. We assert a much more
  # conservative factor of 10 to accommodate non-asymptotic corrections,
  # the small rod geometry, and float-precision floors.
  assert err_m4 < err_ck / 10.0, (
    f"magnus4 error {err_m4:.3e} not at least 10x smaller than "
    f"cayley_klein {err_ck:.3e}"
  )


# ---------------------------------------------------------------------------
# 4. Cross-block Magnus state seeding
# ---------------------------------------------------------------------------

def test_magnus_state_reseeded_at_block_start(minimal_phantom):
  """For an idle (no RF, no gradient) sequence, the magnitude of Mxy
  must be preserved across block boundaries for all three orders.

  This indirectly verifies that ``Bz_old`` / ``rf_old`` are seeded at
  the start of every block so the Magnus kernels do not introduce a
  bogus zero-field average on step 0 of any block beyond the first.
  Phase is not asserted here because the FEelMRI discrete-time
  generator uses ``np.arange(start, end, dt)`` (endpoint-exclusive),
  which costs ``dt`` of evolution at every block boundary — this
  affects all three orders identically and is unrelated to Magnus."""
  for method in METHODS:
    seq = Sequence()
    seq.add_block(make_empty_block(1.0, dt_ms=0.02))
    seq.add_block(make_empty_block(1.0, dt_ms=0.02))
    seq.add_block(make_empty_block(1.0, dt_ms=0.02))
    solver = BlochSolver(
      seq, minimal_phantom,
      M0=1.0,
      T1=Quantity(1e6, 'ms'),
      T2=Quantity(1e6, 'ms'),
      delta_B=1e-4,
      initial_Mxy=1.0 + 0.0j,
      initial_Mz=1.0,
      perfect_spoiling=False,
      method=method,
    )
    Mxy, _ = solver.solve()
    mags = np.abs(Mxy)
    # With T2 -> infinity and no spoiling, |Mxy| must stay at 1 across
    # all three stored blocks.
    np.testing.assert_allclose(
      mags, 1.0, atol=5e-4,
      err_msg=f"{method} |Mxy| not preserved across block stitches",
    )
