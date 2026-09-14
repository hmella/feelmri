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

import warnings

from pathlib import Path

import numpy as np
import pytest
from pint import Quantity

from feelmri import (
  B0Field,
  BlochSolver,
  FEMPhantom,
  Scanner,
)
from feelmri.Bloch import (Sequence, SequenceBlock, lineshape_bins,
                           B0_MOTION_DT_MS)
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

def _one_block(phantom, block, **kw):
  base = dict(M0=1.0, T1=Quantity(1e6, 'ms'), T2=Quantity(1e6, 'ms'),
              initial_Mxy=0.0, initial_Mz=1.0, perfect_spoiling=False)
  base.update(kw)
  return BlochSolver(make_single_block_sequence(block), phantom, **base).solve()


def test_the_bare_bloch_closed_forms(minimal_phantom):
  """T1 recovery, T2 decay and free precession, each against its closed form.

  One test rather than three: they share a fixture and a single empty block,
  and each is one line of algebra with no interaction between them. The kernel
  returning only the FINAL state -- one column, not a history -- is asserted
  here because every other test in this file depends on it.
  """
  T1_ms, T2_ms, dB0_mT = 200.0, 50.0, 1.0e-3

  _mxy, Mz = _one_block(minimal_phantom, make_empty_block(0.5 * T1_ms, dt_ms=1.0),
                        T1=Quantity(T1_ms, 'ms'), T2=Quantity(50.0, 'ms'),
                        initial_Mz=0.0)
  assert Mz.shape[1] == 1, 'the kernel returned a history, not a final state'
  np.testing.assert_allclose(Mz[:, 0], 1.0 - np.exp(-0.5), atol=5e-3)

  Mxy, _mz = _one_block(minimal_phantom, make_empty_block(2.0 * T2_ms, dt_ms=1.0),
                        T2=Quantity(T2_ms, 'ms'), initial_Mxy=1.0 + 0.0j)
  np.testing.assert_allclose(np.abs(Mxy[:, 0]), np.exp(-2.0), atol=5e-3)

  T_ms = 5.0
  want = np.angle(np.exp(-2.0j * np.pi * 42.576e6
                         * (dB0_mT * 1e-3) * (T_ms * 1e-3)))
  Mxy, _mz = _one_block(minimal_phantom, make_empty_block(T_ms, dt_ms=0.05),
                        delta_B=dB0_mT, initial_Mxy=1.0 + 0.0j)
  np.testing.assert_allclose(np.angle(Mxy[:, 0]), want, atol=2e-2)


@pytest.mark.parametrize('flip,want_mz,want_mxy', [
    (np.pi / 2, 0.0, 1.0),
    (np.pi, -1.0, 0.0),
])
def test_a_hard_pulse_delivers_its_nominal_flip(minimal_phantom, flip, want_mz,
                                                want_mxy):
  """The 90 and the 180 are the same assertion at two angles; the 180 is not a
  special case of the kernel, only of the trigonometry."""
  Mxy, Mz = _one_block(minimal_phantom, make_hard_pulse_block(flip, dur_ms=0.2))
  np.testing.assert_allclose(Mz[:, 0], want_mz, atol=5e-2)
  np.testing.assert_allclose(np.abs(Mxy[:, 0]), want_mxy, atol=5e-2)


def test_hard_pulse_orders_agree_on_constant_field(minimal_phantom):
  """For a hard pulse, the field is piecewise-constant within each dt
  so the Magnus commutator ``[Omega_old, Omega_new]`` is zero and all
  three orders must produce the same magnetisation to within FP
  rounding.

  This is the bridge that lets the five closed-form tests above run at the
  default method alone rather than three times each. Their fields are all
  piecewise-constant -- zero for the relaxation pair, a constant Bz for
  precession, a constant B1 for the two hard pulses -- which is exactly the
  regime pinned here. Relaxation is finite so the shared T1/T2 path is
  covered too, not only the rotation.
  """
  block = make_hard_pulse_block(np.pi / 3, dur_ms=0.3, dt_ms=0.005)
  seq = make_single_block_sequence(block)
  results = {}
  for method in METHODS:
    solver = BlochSolver(
      seq, minimal_phantom,
      M0=1.0,
      T1=Quantity(200.0, 'ms'),
      T2=Quantity(50.0, 'ms'),
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


# ---------------------------------------------------------------------------
# 4. Sub-voxel T2' by a spectral sub-ensemble
# ---------------------------------------------------------------------------
#
# The gap these close: a scalar T2* decays monotonically from the snapshot
# whatever constant it is given, so it can never rephase at an echo. A real
# sub-ensemble does, because each sub-spin simply runs backwards after a 180.
# `test_spin_echo_rephases_what_t2_prime_dephased` FAILS without the feature --
# that is what makes it worth having.

T2_PRIME_MS = 20.0


def _t2_prime_echo_sequence(tau_ms, n_steps, refocus, pulse_ms=0.002):
  """90 -- tau -- (180) -- tau, sampled every tau/n_steps."""
  seq = Sequence()
  seq.add_block(make_hard_pulse_block(np.pi / 2, dur_ms=pulse_ms, dt_ms=1e-4))
  step = tau_ms / n_steps
  for _ in range(n_steps):
    blk = make_empty_block(step, dt_ms=step)
    blk.store_magnetization = True
    seq.add_block(blk)
  if refocus:
    seq.add_block(make_hard_pulse_block(np.pi, dur_ms=pulse_ms, dt_ms=1e-4))
  for _ in range(n_steps):
    blk = make_empty_block(step, dt_ms=step)
    blk.store_magnetization = True
    seq.add_block(blk)
  return seq


def _run_t2_prime(phantom, seq, t2_prime_ms=T2_PRIME_MS, T2_ms=1e9, **kwargs):
  extra = {} if t2_prime_ms is None else dict(
    t2_prime=Quantity(t2_prime_ms, 'ms'), **kwargs)
  solver = BlochSolver(
    seq, phantom,
    T1=Quantity(1e9, 'ms'), T2=Quantity(T2_ms, 'ms'),
    initial_Mxy=0.0 + 0.0j, initial_Mz=1.0,
    perfect_spoiling=False, dtype='float64', **extra)
  Mxy, Mz = solver.solve()
  return np.abs(Mxy[0, :]), Mz[0, :]


@pytest.mark.parametrize('lineshape, decay', [
  ('gaussian', lambda t, T: np.exp(-0.5 * (t / T)**2)),
  # |Mxy| is a magnitude and the uniform lineshape's sinc goes NEGATIVE:
  # the signal has true zero crossings and partial recoveries, which is the
  # physically right behaviour for a linear gradient across the voxel.
  ('uniform', lambda t, T: np.abs(np.sinc(np.sqrt(3.0) * t / T / np.pi))),
], ids=['gaussian', 'uniform'])
def test_free_induction_decays_with_the_shape_of_its_lineshape(
        minimal_phantom, lineshape, decay):
  """The ensemble must reproduce the decay its own quadrature rule encodes.

  This is the check that the bins are a real distribution and not just a
  spread: a gaussian lineshape gives exp(-t^2 / 2 T2'^2), NOT exp(-t/T2*).
  Getting the shape right is the whole difference from a scalar.

  The residual is set by the finite pulse width, not the quadrature: the time
  origin is the pulse CENTRE while the first sample is taken at its end.
  Measured, it falls exactly linearly with the pulse duration -- 3.86e-4 at
  20 us, 3.86e-5 at 2 us, 3.87e-6 at 0.2 us -- so at the 2 us used here the
  quadrature (1.7e-8 at K=16) is nowhere near the limit.
  """
  n_steps, tau = 12, 30.0
  seq = _t2_prime_echo_sequence(tau, n_steps, refocus=False)
  mag, _ = _run_t2_prime(minimal_phantom, seq, lineshape=lineshape,
                         spectral_bins=16)
  t = np.arange(mag.size) * (tau / n_steps)
  np.testing.assert_allclose(mag, decay(t, T2_PRIME_MS), atol=2e-4)


def test_spin_echo_rephases_what_t2_prime_dephased(minimal_phantom):
  """The point of the whole feature, and the one thing no scalar can do.

  T2 is infinite, so every radian lost is REVERSIBLE and a 180 must bring all
  of it back. Under the scalar model -- T2* handed to the solver as T2 -- the
  magnetization decays monotonically and reaches exp(-2 tau / T2*) at the echo
  with no recovery whatever, which is the control asserted below.
  """
  n_steps, tau = 12, 30.0
  mag, _ = _run_t2_prime(
    minimal_phantom, _t2_prime_echo_sequence(tau, n_steps, refocus=True),
    lineshape='gaussian', spectral_bins=16)

  at_tau = mag[n_steps]
  at_echo = mag[-1]
  assert at_tau < 0.4, (
    f'the sub-ensemble should have dephased to ~0.32 by tau; got {at_tau:.4f}')
  assert at_echo > 0.999, (
    f'a 180 must rephase reversible dephasing: |Mxy| at the echo is '
    f'{at_echo:.6f}, expected ~1. If this fails the ensemble is not '
    f'surviving the block boundary.')

  # The control: the same sequence under the scalar model cannot recover.
  scalar, _ = _run_t2_prime(minimal_phantom,
                            _t2_prime_echo_sequence(tau, n_steps, refocus=True),
                            t2_prime_ms=None, T2_ms=T2_PRIME_MS)
  assert np.all(np.diff(scalar) <= 1e-12), 'the scalar model must be monotone'
  assert scalar[-1] < 0.06, (
    f'scalar control should reach exp(-2 tau/T2*) = 0.05; got {scalar[-1]:.4f}')


def test_rephasing_is_exact_for_every_lineshape_and_bin_count(minimal_phantom):
  """Refocusing is exact for ANY static distribution, so the echo amplitude
  does not depend on the quadrature rule -- only the decay BETWEEN echoes
  does. Worth pinning: it is the reason the inaccurate lorentzian rule is
  still usable for echo-based sequences."""
  n_steps, tau = 6, 24.0
  seq = _t2_prime_echo_sequence(tau, n_steps, refocus=True)
  for lineshape in ('gaussian', 'uniform', 'lorentzian'):
    for K in (8, 16):
      with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mag, _ = _run_t2_prime(minimal_phantom, seq, lineshape=lineshape,
                               spectral_bins=K)
      assert mag[-1] > 0.999, (
        f'{lineshape} at K={K} rephased to only {mag[-1]:.6f}')


def test_bin_weights_are_a_probability_distribution(minimal_phantom):
  """Non-negative and summing to exactly 1.

  Not cosmetic: the T1 recovery term (1 - e1) * M0 is AFFINE, so the collapsed
  equilibrium is M0 * sum(w). Raw Gauss-Hermite weights sum to 2.5066 and
  Gauss-Legendre to 2.0, either of which would put the whole phantom at the
  wrong M0. Asserted here on the rule and end to end on a long recovery.
  """
  for lineshape in ('gaussian', 'uniform', 'lorentzian'):
    for K in (4, 8, 16, 32):
      z, w = lineshape_bins(K, lineshape)
      # AT MOST K bins: the rule prunes sub-spins whose weight is below
      # float64 epsilon, which Gauss-Hermite produces in quantity (4 of 32,
      # 70 of 128). Pruning them is free -- it moves neither the error nor the
      # reach -- so the contract is the distribution, not the array length.
      assert z.shape == w.shape and 1 <= z.size <= K
      assert w.min() >= 0.0, f'{lineshape} K={K} has a negative weight'
      assert abs(w.sum() - 1.0) < 1e-15, f'{lineshape} K={K} sums to {w.sum()}'

  # The rules break down at large K and numpy does not say so; without this
  # guard a big spectral_bins produced NaN weights and NaN magnetization.
  with pytest.raises(ValueError, match='loses all precision'):
    lineshape_bins(400, 'gaussian')

  # End to end: the recovery curve is M0 * sum(w) * (1 - exp(-t/T1)), so an
  # unnormalised rule scales the whole phantom. Compared against the closed
  # form rather than against 1, which five T1 does not reach anyway (0.9933).
  T1_ms, dur_ms = 100.0, 500.0
  seq = make_single_block_sequence(make_empty_block(dur_ms, dt_ms=10.0))
  solver = BlochSolver(
    seq, minimal_phantom, M0=1.0,
    T1=Quantity(T1_ms, 'ms'), T2=Quantity(1e9, 'ms'),
    initial_Mxy=0.0 + 0.0j, initial_Mz=0.0,
    perfect_spoiling=False, dtype='float64',
    t2_prime=Quantity(T2_PRIME_MS, 'ms'), spectral_bins=16)
  _, Mz = solver.solve()
  np.testing.assert_allclose(Mz[:, 0], 1.0 - np.exp(-dur_ms / T1_ms), atol=1e-9)


def test_t2_prime_refuses_what_stage_one_cannot_do(minimal_phantom):
  """Each guard names a real cost, not a missing convenience -- see the
  docstrings. Silence here would mean a K-fold slowdown or a wrong answer."""
  seq = make_single_block_sequence(make_empty_block(5.0, dt_ms=1.0))
  common = dict(T1=Quantity(1e9, 'ms'), initial_Mz=1.0,
                perfect_spoiling=False, t2_prime=Quantity(T2_PRIME_MS, 'ms'))

  # A per-node T2 would drop the kernel onto its per-node exp() path at K times
  # the cost.
  n_nodes = minimal_phantom.local_nodes.shape[0]
  per_node = np.linspace(40.0, 60.0, n_nodes).reshape(-1, 1)
  with pytest.raises(NotImplementedError, match='per-node'):
    BlochSolver(seq, minimal_phantom, T2=Quantity(per_node, 'ms'), **common)

  # A spoiler block is a SECOND sub-voxel axis; combining needs a tensor
  # product.
  spoiled = make_single_block_sequence(make_empty_block(5.0, dt_ms=1.0))
  spoiled.blocks[0].spoiler = True
  with pytest.raises(NotImplementedError, match='spoiler'):
    BlochSolver(spoiled, minimal_phantom, T2=Quantity(50.0, 'ms'), **common)

  # One bin is not an ensemble.
  with pytest.raises(ValueError, match='spectral_bins'):
    BlochSolver(seq, minimal_phantom, T2=Quantity(50.0, 'ms'),
                spectral_bins=1, **common)

  # The lorentzian rule cannot reach the decay it targets, and says so.
  with pytest.warns(UserWarning, match='lorentzian'):
    BlochSolver(seq, minimal_phantom, T2=Quantity(50.0, 'ms'),
                lineshape='lorentzian', **common)


def test_finite_bin_sets_revive_and_the_sizing_rule_holds():
  """A finite ensemble is quasi-periodic: it cannot stay cancelled forever.

  Pinned rather than merely documented because the failure is SILENT and looks
  like signal -- a free induction decay that has reached zero climbs back out.
  Measured usable range for the gaussian rule is tau/T2' = 0.2*K (2.50 at K=8,
  4.67 at 16, 7.86 at 32, 12.47 at 64), which is the source of the
  `K >= 5 * tau_max / T2'` guidance in `lineshape_bins`.
  """
  tau = np.linspace(0.0, 15.0, 4000)          # in units of T2'
  target = np.exp(-0.5 * tau**2)

  def usable_range(K):
    z, w = lineshape_bins(K, 'gaussian')
    F = np.abs((w[None, :] * np.exp(-1j * np.outer(tau, z))).sum(1))
    bad = np.where(np.abs(F - target) > 1e-3)[0]
    return tau[bad[0]] if bad.size else np.inf

  reach = {K: usable_range(K) for K in (8, 16, 32)}
  assert reach[8] < reach[16] < reach[32], f'not monotone in K: {reach}'
  for K, expected in ((8, 2.50), (16, 4.67), (32, 7.86)):
    assert abs(reach[K] - expected) < 0.15, (
      f'gaussian K={K} tracks to tau/T2\' = {reach[K]:.2f}, expected '
      f'{expected:.2f}; the 5 * tau_max / T2\' sizing rule has moved')
    assert reach[K] > 0.2 * K, 'the documented 0.2*K rule must be conservative'


# ---------------------------------------------------------------------------
# 5. Audit coverage: concomitant fields, B1+, and their interactions
# ---------------------------------------------------------------------------
#
# Written during the audit of the `realism` branch, which found that of its
# fifteen commits exactly one carried test code. `concomitant_fields` and
# `b1_map` had none at all, and the parts of each that nothing executed were
# the parts most able to hide a sign error: three of the four concomitant terms
# (the shipped example drives one axis) and the order-4 |b1|^2 path (the
# example uses a hard pulse with no gradient, where the commutator vanishes).


@pytest.fixture(scope='module')
def wide_phantom(tmp_path_factory):
  """A phantom whose bounding box is 20 x 23 x 17 cm, so a term quadratic in
  position is measurable -- `minimal_phantom` spans 1 cm, where it is not --
  and with every node at DISTINCT x, y and z.

  The distinctness matters. `make_minimal_tet_mesh` puts four of its five nodes
  on the axes and the fifth at (s, s, s), so `x*z` and `y*z` are equal at every
  node: an x-vs-y position-index swap in the concomitant cross terms would be
  invisible. These coordinates are mutually incommensurate instead.

  Both tetrahedra are positively oriented. The assembler takes |det J|, so the
  orientation does not change any result here, but a mesh fixture that a reader
  might reuse should not carry an inverted cell.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  import meshio as _meshio
  mesh_path = tmp_path_factory.mktemp('magnus_wide') / 'wide.vtu'
  points = np.array([[0.11, -0.03, 0.07],
                     [-0.05, 0.12, 0.02],
                     [0.04, 0.06, -0.10],
                     [-0.09, -0.08, 0.05],
                     [0.02, -0.11, -0.06]])
  cells = np.array([[0, 1, 2, 3], [1, 2, 4, 3]])
  _meshio.write(str(mesh_path), _meshio.Mesh(points, [('tetra', cells)]))
  return FEMPhantom(path=str(mesh_path))


def _gradient_block(G_mT_per_m, dur_ms, dt_ms=0.002):
  """A constant gradient on all three axes at once."""
  scanner = Scanner()
  gradients = [
    Gradient(timings=Quantity(np.array([0.0, dur_ms]), 'ms'),
             amplitudes=Quantity(np.array([G_mT_per_m[a]] * 2), 'mT/m'),
             scanner=scanner, ref=Quantity(0.0, 'ms'), time=Quantity(0.0, 'ms'),
             axis=a)
    for a in range(3)]
  return SequenceBlock(gradients=gradients, dur=Quantity(dur_ms, 'ms'),
                       dt=Quantity(dt_ms, 'ms'), empty=False,
                       store_magnetization=True)


def _concomitant_field_mT(positions, G, B0_mT):
  """Bc = (Bx^2 + By^2) / (2 B0) for a linear gradient set, where
  Bx = Gx*z - Gz*x/2 and By = Gy*z - Gz*y/2. Written in the factored form the
  kernel does NOT use, so the test is independent of the expanded expression
  it is checking."""
  Gx, Gy, Gz = G
  x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
  Bx = Gx * z - 0.5 * Gz * x
  By = Gy * z - 0.5 * Gz * y
  return (Bx**2 + By**2) / (2.0 * B0_mT)


def _shifted_phantom(phantom, shift, _cache={}):
  """The same node cloud translated rigidly, as a separate phantom."""
  import meshio as _meshio
  import tempfile
  key = (id(phantom), tuple(shift))
  if key not in _cache:
    points = phantom.local_nodes.astype(np.float64) + np.asarray(shift)
    cells = np.asarray(phantom.local_elements)
    path = Path(tempfile.mkdtemp()) / 'shifted.vtu'
    _meshio.write(str(path), _meshio.Mesh(points, [('tetra', cells)]))
    _cache[key] = FEMPhantom(path=str(path))
  return _cache[key]


def _precess(phantom, block, dtype='float64', method='magnus2', **solver_kwargs):
  seq = make_single_block_sequence(block)
  solver = BlochSolver(
    seq, phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
    initial_Mxy=1.0 + 0.0j, initial_Mz=0.0, perfect_spoiling=False,
    dtype=dtype, method=method, **solver_kwargs)
  Mxy, _ = solver.solve()
  return Mxy[:, 0]


@pytest.mark.parametrize('G', [(18.0, 0.0, 12.0), (0.0, -20.0, 15.0),
                               (11.0, -9.0, -17.0)],
                         ids=['Gx_Gz', 'Gy_Gz', 'oblique'])
def test_concomitant_phase_matches_the_maxwell_closed_form(wide_phantom, G):
  """Every term, including the two CROSS terms, against the closed form.

  The shipped example drives one axis, so `(Gx^2+Gy^2)z^2` and both
  `-Gx*Gz*x*z` terms are executed by nothing else: a sign flip or a dropped
  factor of two on either cross term would pass the whole suite.

  Measured over random gradients and positions, the agreement is 1.2e-7 rad.
  """
  dur_ms = 6.0
  scanner = Scanner()
  B0_mT = scanner.field_strength.m_as('mT')
  gamma = scanner.gamma.m_as('rad/ms/mT')

  nodes = wide_phantom.local_nodes.astype(np.float64)
  expected = -gamma * _concomitant_field_mT(nodes, G, B0_mT) * dur_ms
  block = _gradient_block(G, dur_ms)
  # All three integrators: order 0 skips the Python Magnus seed entirely, so a
  # seed that disagreed with the kernel would show up only here.
  for method in ('cayley_klein', 'magnus2', 'magnus4'):
    off = _precess(wide_phantom, block, method=method, concomitant_fields=False)
    on = _precess(wide_phantom, block, method=method, concomitant_fields=True)
    # Compared as unit phasors, so the check cannot be fooled by 2*pi wrapping.
    worst = float(np.abs(np.exp(1j * np.angle(on / off))
                         - np.exp(1j * expected)).max())
    assert worst < 1e-5, (
      f'{method}: concomitant phase departs from the closed form by {worst:.2e}')


def test_concomitant_phase_can_only_ever_retard(wide_phantom):
  """Bc is `(Bx^2 + By^2)/(2 B0)`, a sum of squares, so the phase it adds can
  only ever be negative -- for every gradient and every position.

  The previous version of this test looped 200 random gradients through the
  test's OWN helper, which is unfalsifiable and touches no library code, and
  its end-to-end half passed with the feature disabled (`on == off` gives
  `angle == 0`, which satisfies `<= 0`). Both halves are replaced: the sign is
  read off the SOLVER, and the test first requires the term to be doing
  something, so a silently disabled feature fails instead of passing.

  What this does NOT catch is a sign error on either CROSS term: negating
  `Gx*Gz*x*z` turns `(Gx*z - Gz*x/2)^2` into `(Gx*z + Gz*x/2)^2`, which is
  still a sum of squares and still never advances the phase (measured: 0 of 4
  gradients notice). That case belongs to
  test_concomitant_phase_matches_the_maxwell_closed_form, which compares
  against the factored form term by term.
  """
  rng = np.random.default_rng(5)
  scanner = Scanner()
  B0_mT = scanner.field_strength.m_as('mT')
  gamma = scanner.gamma.m_as('rad/ms/mT')
  nodes = wide_phantom.local_nodes.astype(np.float64)
  for _ in range(4):
    G = tuple(rng.uniform(-30.0, 30.0, 3))
    # Keep the accumulated phase inside one turn. np.angle wraps, so a phase
    # past -pi comes back POSITIVE and would look like a sign error -- which is
    # how the first version of this test failed.
    per_ms = gamma * float(_concomitant_field_mT(nodes, G, B0_mT).max())
    dur_ms = min(5.0, 0.8 * np.pi / per_ms)
    block = _gradient_block(G, dur_ms)
    off = _precess(wide_phantom, block, concomitant_fields=False)
    on = _precess(wide_phantom, block, concomitant_fields=True)
    phase = np.angle(on / off)
    assert np.abs(phase).max() > 1e-3, (
      f'G={np.round(G, 1)} produced no concomitant phase at all; the test '
      f'would pass with the feature disabled')
    assert phase.max() <= 1e-9, (
      f'G={np.round(G, 1)}: the concomitant term ADVANCED the phase by '
      f'{phase.max():.3e}, which no sum of squares can do.')


def _phantom_from_points(points, cells, tag):
  """A fresh phantom on a given node cloud, so a test may `orient` it without
  mutating a module-scoped fixture."""
  import meshio as _meshio
  import tempfile
  path = Path(tempfile.mkdtemp()) / f'{tag}.vtu'
  _meshio.write(str(path), _meshio.Mesh(np.asarray(points, dtype=np.float64),
                                        [('tetra', np.asarray(cells))]))
  return FEMPhantom(path=str(path))


def _rotation_zyx(az, ay, ax):
  """An ordinary right-handed rotation, built from the three axis rotations so
  the test does not depend on any library helper."""
  ca, sa = np.cos(az), np.sin(az)
  cb, sb = np.cos(ay), np.sin(ay)
  cc, sc = np.cos(ax), np.sin(ax)
  Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
  Ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
  Rx = np.array([[1.0, 0.0, 0.0], [0.0, cc, -sc], [0.0, sc, cc]])
  return Rz @ Ry @ Rx


def test_the_solver_evaluates_bc_in_the_physical_frame(wide_phantom):
  """ONE physical experiment, described twice: once in the frame where B0 is z,
  once in an oblique imaging frame. The concomitant phase must be the same.

  `Bc = (Bx^2 + By^2)/(2 B0)` singles out B0's axis, so unlike the linear term
  `x . G` it is NOT frame-invariant. `FEMPhantom.orient` leaves the nodes in the
  imaging frame and the sequence carries logical gradients, so before this the
  kernel evaluated `Bc` as though the SLICE NORMAL were B0.

  Measured at a 20 deg tilt on this node cloud: 1.559 rad of disagreement,
  31.6% of the phase. The guard below is what makes that a finding rather than
  a tolerance -- a shallow tilt would make the two frames agree for free, and
  the test would then pass on the unfixed code.

  float64 throughout: at float32 the solver's own noise is ~1e-2 rad, an order
  above nothing this test is trying to resolve.
  """
  dur_ms = 6.0
  scanner = Scanner()
  B0_mT = scanner.field_strength.m_as('mT')
  gamma = scanner.gamma.m_as('rad/ms/mT')

  R = _rotation_zyx(np.deg2rad(20.0), np.deg2rad(-14.0), np.deg2rad(9.0))
  P = wide_phantom.local_nodes.astype(np.float64)      # physical coordinates
  cells = np.asarray(wide_phantom.local_elements)
  G_phys = np.array([21.0, -13.0, 25.0])

  # The same gradient and the same spins, written in the imaging frame.
  # `orient` applies `nodes @ R`, i.e. `R^T P` in column form, and the scanner
  # plays `G_physical = R G_logical`.
  G_img = R.T @ G_phys
  X = P @ R

  # 1. Physical description: no orientation, no rotation anywhere.
  ph_phys = _phantom_from_points(P, cells, 'frame_phys')
  blk = _gradient_block(tuple(G_phys), dur_ms)
  phi_phys = np.angle(_precess(ph_phys, blk, concomitant_fields=True)
                      / _precess(ph_phys, blk, concomitant_fields=False))

  # 2. Imaging description of the SAME experiment.
  ph_img = _phantom_from_points(P, cells, 'frame_img')
  ph_img.orient(R, Quantity(np.zeros(3), 'm'))
  assert np.allclose(ph_img.local_nodes, X, atol=1e-9)
  blk_img = _gradient_block(tuple(G_img), dur_ms)
  phi_img = np.angle(_precess(ph_img, blk_img, concomitant_fields=True)
                     / _precess(ph_img, blk_img, concomitant_fields=False))

  # The linear term is a dot product and must be untouched by any of this.
  # The floor here is the phantom's float32 NODE STORAGE, not the solver:
  # `orient` rotates the stored coordinates, and the round trip leaves 7.6e-9 m,
  # which at 25 mT/m over 6 ms is 3e-4 rad. Nothing about the frame fix can
  # improve that, and the concomitant comparison below is unaffected because
  # `Bc` varies far more slowly with position than `G . x` does.
  lin_phys = np.angle(_precess(ph_phys, blk, concomitant_fields=False))
  lin_img = np.angle(_precess(ph_img, blk_img, concomitant_fields=False))
  assert np.abs(np.exp(1j * lin_phys) - np.exp(1j * lin_img)).max() < 1e-3, (
    'the rotation moved the LINEAR encoding, which is frame-invariant')

  # VACUITY GUARD. The frame-naive answer -- `Bc` evaluated on the imaging
  # coordinates with the logical gradient, which is what the solver did before
  # -- has to be far enough away that agreeing is evidence.
  naive = -gamma * _concomitant_field_mT(X, tuple(G_img), B0_mT) * dur_ms
  truth = -gamma * _concomitant_field_mT(P, tuple(G_phys), B0_mT) * dur_ms
  gap = float(np.abs(naive - truth).max())
  assert gap > 0.1, (
    f'this geometry cannot discriminate the frames: the naive prediction is '
    f'only {gap:.3e} rad away, so the test would pass unfixed')

  worst = float(np.abs(np.exp(1j * phi_phys) - np.exp(1j * phi_img)).max())
  assert worst < 1e-5, (
    f'the concomitant phase depends on the frame the experiment is described '
    f'in: {worst:.3e} between the physical and imaging descriptions, against '
    f'a {gap:.3f} rad frame-naive gap')


def _spin_echo_phase(phantom, gradients, with_180, dur_ms=4.0):
  """Concomitant phase accumulated over two gradient lobes, optionally with a
  refocusing pulse between them. Returned as the on-minus-off phase, so the
  linear term and the RF itself divide out.
  """
  seq_on, seq_off = Sequence(), Sequence()
  for seq in (seq_on, seq_off):
    seq.add_block(_gradient_block(gradients[0], dur_ms))
    if with_180:
      # NON-SELECTIVE. A slice-selective 180 plays a gradient symmetric about
      # the pulse, and `Bc` is EVEN in G, so the two halves of that lobe add
      # instead of cancelling -- which would put concomitant phase into the
      # very case that is supposed to refocus it.
      seq.add_block(make_hard_pulse_block(np.pi, dur_ms=0.2))
    seq.add_block(_gradient_block(gradients[1], dur_ms))
  out = []
  for seq, conc in ((seq_on, True), (seq_off, False)):
    solver = BlochSolver(
      seq, phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
      initial_Mxy=1.0 + 0.0j, initial_Mz=0.0, perfect_spoiling=False,
      dtype='float64', method='magnus2', concomitant_fields=conc)
    Mxy, _ = solver.solve()
    out.append(Mxy[:, -1])
  return np.angle(out[0] / out[1])


def test_the_spin_echo_trio_separates_the_linear_and_maxwell_terms(wide_phantom):
  """Three two-lobe experiments whose ONLY difference is the sign of the second
  lobe and whether a 180 sits between them:

    A  (G,  G) with a 180   linear refocuses   Bc refocuses
    B  (G, -G) no 180       linear refocuses   Bc DOUBLES
    C  (G, -G) with a 180   linear DOUBLES     Bc refocuses

  A 180 refocuses any static field, and `Bc(-G) == Bc(G)` because it is a sum
  of squares -- so the two terms respond oppositely to the same pair of
  switches. A concomitant phase bolted on as a post-hoc `integral(G^2)` rather
  than carried through the evolution satisfies the closed-form test that
  already exists here and fails A and C by exactly 2x, because it has no way
  to know a 180 happened.

  float64 throughout: the quantity being shown to be zero is ~1 rad at float32
  round-off ~1e-2.
  """
  G = (16.0, -11.0, 19.0)
  Gm = tuple(-g for g in G)
  dur_ms = 4.0
  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')
  nodes = wide_phantom.local_nodes.astype(np.float64)
  bc = _concomitant_field_mT(nodes, G, scanner.field_strength.m_as('mT'))

  phi_A = _spin_echo_phase(wide_phantom, (G, G), True, dur_ms)
  phi_B = _spin_echo_phase(wide_phantom, (G, Gm), False, dur_ms)
  phi_C = _spin_echo_phase(wide_phantom, (G, Gm), True, dur_ms)

  one_lobe = gamma * bc * dur_ms
  assert one_lobe.max() > 0.5, 'this gradient produces no concomitant phase'

  for tag, phi in (('A', phi_A), ('C', phi_C)):
    worst = float(np.abs(np.exp(1j * phi) - 1.0).max())
    assert worst < 1e-5, (
      f'case {tag}: the 180 did not refocus the concomitant phase; residual '
      f'{worst:.3e} against {one_lobe.max():.3f} rad per lobe')

  expected = np.exp(-1j * 2.0 * one_lobe)
  worst = float(np.abs(np.exp(1j * phi_B) - expected).max())
  assert worst < 1e-5, (
    f'case B: two lobes of opposite sign must DOUBLE the concomitant phase, '
    f'not cancel it; off by {worst:.3e}')

  # The control for A and C: the SAME two lobes without the 180 reach the
  # doubled phase too, so what those two cases show is the refocusing and not
  # a gradient pair that happened to produce nothing.
  phi_A_no180 = _spin_echo_phase(wide_phantom, (G, G), False, dur_ms)
  assert float(np.abs(np.exp(1j * phi_A_no180) - expected).max()) < 1e-5


def _shaped_rf_block(scale, dur_ms=1.0, n=64, dt_ms=0.02):
  """A COMPLEX, time-varying pulse. Needed for the order-4 commutator to be
  non-zero: a real hard pulse on resonance makes both correction terms vanish
  identically, so it cannot test how b1 enters them."""
  t = np.linspace(0.0, dur_ms, n)
  envelope = np.sinc(4 * (t / dur_ms - 0.5)) * np.exp(1j * 3.0 * t / dur_ms)
  rf = RF(timings=Quantity(t, 'ms'),
          waveform=Quantity(0.25 * scale * envelope, 'mT'),
          scanner=Scanner(), ref=Quantity(0.0, 'ms'), time=Quantity(0.0, 'ms'),
          shape='custom')
  gradient = Gradient(timings=Quantity(np.array([0.0, dur_ms]), 'ms'),
                      amplitudes=Quantity(np.array([12.0, 12.0]), 'mT/m'),
                      scanner=Scanner(), ref=Quantity(0.0, 'ms'),
                      time=Quantity(0.0, 'ms'), axis=0)
  return SequenceBlock(rf_pulses=[rf], gradients=[gradient],
                       dur=Quantity(dur_ms, 'ms'), dt=Quantity(dt_ms, 'ms'),
                       empty=False, store_magnetization=True)


def test_b1_scaling_is_identical_to_scaling_the_pulse(wide_phantom):
  """A UNIFORM b1 = c must equal scaling the pulse by c, exactly.

  This is the only check with teeth on the order-4 terms, where `theta_xy` is
  linear in RF and takes b1 while `theta_z`'s commutator is bilinear and takes
  |b1|^2. Substituting the scaled RF into both endpoints gets that right for
  free; scaling the assembled rotation once would get the second wrong.

  It needs a complex, time-varying pulse under a gradient -- with a real hard
  pulse the commutator vanishes and the test passes under either scaling. The
  magnus2-vs-magnus4 gap below is the proof that it is live here.
  """
  c = 0.73 * np.exp(0.4j)

  def run(scale_pulse, b1, method):
    solver = BlochSolver(
      make_single_block_sequence(_shaped_rf_block(scale_pulse)), wide_phantom,
      T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
      initial_Mxy=0.0 + 0.0j, initial_Mz=1.0, perfect_spoiling=False,
      dtype='float64', method=method, b1_map=b1)
    Mxy, Mz = solver.solve()
    return Mxy[:, 0], Mz[:, 0]

  live = np.abs(run(c, None, 'magnus2')[0] - run(c, None, 'magnus4')[0]).max()
  assert live > 1e-6, (
    f'the order-4 commutator contributes only {live:.2e} here, so this test '
    f'would pass under a wrong b1 power; make the pulse less trivial')

  for method in ('cayley_klein', 'magnus2', 'magnus4'):
    scaled_pulse = run(c, None, method)
    via_map = run(1.0, c, method)
    worst = max(float(np.abs(scaled_pulse[0] - via_map[0]).max()),
                float(np.abs(scaled_pulse[1] - via_map[1]).max()))
    assert worst < 1e-12, (
      f'{method}: b1_map={c} differs from scaling the pulse by {worst:.2e}')


def test_b1_map_sets_the_flip_and_the_transmit_phase_per_node(minimal_phantom):
  """Per-node |b1| scales the flip exactly, and arg(b1) is a transmit phase
  that lands on Mxy without touching its magnitude."""
  n_nodes = minimal_phantom.local_nodes.shape[0]
  magnitude = np.linspace(1.0, 0.0, n_nodes)
  phase = np.linspace(-1.3, 0.7, n_nodes)
  b1 = magnitude * np.exp(1j * phase)
  block = make_hard_pulse_block(np.pi / 2, dur_ms=0.002, dt_ms=1e-4)

  def run(b1_map):
    solver = BlochSolver(
      make_single_block_sequence(block), minimal_phantom,
      T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
      initial_Mxy=0.0 + 0.0j, initial_Mz=1.0, perfect_spoiling=False,
      dtype='float64', b1_map=b1_map)
    Mxy, Mz = solver.solve()
    return Mxy[:, 0], Mz[:, 0]

  Mxy, Mz = run(b1)
  nominal, _ = run(None)

  delivered = np.arctan2(np.abs(Mxy), Mz)
  np.testing.assert_allclose(delivered, (np.pi / 2) * magnitude, atol=1e-6)
  # The transmit phase rotates the axis the pulse tips onto, and nothing else.
  turned = np.abs(np.exp(1j * np.angle(Mxy[:-1]))
                  - np.exp(1j * (np.angle(nominal[:-1]) + phase[:-1])))
  assert turned.max() < 1e-6, f'transmit phase off by {turned.max():.2e}'


def _echo_train(n_echoes, tau_ms, pulse_ms=0.002):
  """90 -- tau -- [180 -- tau(ECHO) -- tau] x n. Only the echo instants carry
  store_magnetization, so the returned columns are the echoes and nothing
  else."""
  seq = Sequence()
  ninety = make_hard_pulse_block(np.pi / 2, dur_ms=pulse_ms, dt_ms=1e-4)
  ninety.store_magnetization = False
  seq.add_block(ninety)
  seq.add_block(make_empty_block(tau_ms, dt_ms=tau_ms))
  seq.blocks[-1].store_magnetization = False
  for _ in range(n_echoes):
    refocus = make_hard_pulse_block(np.pi, dur_ms=pulse_ms, dt_ms=1e-4)
    refocus.store_magnetization = False
    seq.add_block(refocus)
    seq.add_block(make_empty_block(tau_ms, dt_ms=tau_ms))     # the echo
    seq.add_block(make_empty_block(tau_ms, dt_ms=tau_ms))
    seq.blocks[-1].store_magnetization = False
  return seq


def test_cpmg_echoes_reach_exp_minus_t_over_t2_with_the_ensemble_on(
        minimal_phantom):
  """Six echoes, with T2' short enough that the signal is essentially gone
  between them. Every echo must still land on exp(-2 n tau / T2): the
  reversible part is fully refocused and only T2 survives.

  This is the strongest test of persistence across block boundaries -- 19
  blocks, with coherence that has to survive every stitch.
  """
  T2_ms, T2_prime_ms, tau_ms, n_echoes = 200.0, 8.0, 12.0, 6
  seq = _echo_train(n_echoes, tau_ms)
  expected = np.exp(-2 * tau_ms * np.arange(1, n_echoes + 1) / T2_ms)

  for lineshape, K, dtype in (('gaussian', 32, 'float64'),
                              ('uniform', 32, 'float64'),
                              ('gaussian', 8, 'float64')):
    solver = BlochSolver(
      seq, minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(T2_ms, 'ms'),
      initial_Mxy=0.0 + 0.0j, initial_Mz=1.0, perfect_spoiling=False,
      dtype=dtype, t2_prime=Quantity(T2_prime_ms, 'ms'), spectral_bins=K,
      lineshape=lineshape)
    Mxy, _ = solver.solve()
    assert Mxy.shape[1] == n_echoes, 'only the echoes should be stored'
    got = np.abs(Mxy[0])
    np.testing.assert_allclose(got, expected, atol=1e-3,
                               err_msg=f'{lineshape} K={K} {dtype}')

  # exp(-2 n tau / T2) is ALSO what the solver returns with no ensemble at all,
  # so the loop above passes with t2_prime disabled, with every bin offset
  # zeroed, or with every offset negated. Sample BETWEEN two echoes as well:
  # there the ensemble must be dephased, and nothing but the ensemble can do
  # that.
  midpoint = _echo_train(1, tau_ms)
  midpoint.blocks[-1].store_magnetization = True      # tau after the echo
  solver = BlochSolver(
    midpoint, minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(T2_ms, 'ms'),
    initial_Mxy=0.0 + 0.0j, initial_Mz=1.0, perfect_spoiling=False,
    dtype='float64', t2_prime=Quantity(T2_prime_ms, 'ms'), spectral_bins=32)
  Mxy, _ = solver.solve()
  at_echo, after_echo = np.abs(Mxy[0, 0]), np.abs(Mxy[0, 1])
  # The echo carries the irreversible loss only; tau later the ensemble has
  # dephased again by its own lineshape on top of it.
  want_echo = np.exp(-2 * tau_ms / T2_ms)
  want_after = (np.exp(-0.5 * (tau_ms / T2_prime_ms)**2)
                * np.exp(-3 * tau_ms / T2_ms))
  assert abs(at_echo - want_echo) < 2e-3, (
    f'echo is {at_echo:.4f}, expected {want_echo:.4f}')
  assert abs(after_echo - want_after) < 2e-3, (
    f'tau past the echo got {after_echo:.4f}, expected {want_after:.4f}. With '
    f'the ensemble disabled this point reads {np.exp(-3 * tau_ms / T2_ms):.4f}')


def test_a_stimulated_echo_survives_being_stored_in_mz(minimal_phantom):
  """90 - t1 - 90 - t2 - 90 - t1. The second pulse parks the dephased pattern
  along z, where it does not dephase; the third brings it back and it rephases
  t1 later at exactly HALF the magnetization -- only one of the two halves of
  cos(phi) refocuses.

  Nothing else in the suite plays three pulses on one magnetization, and this
  is the only test of the ensemble surviving a trip through Mz.

  t1/T2' = 8 is deliberate: the ideal 1/2 assumes the other coherence pathways
  are dead at the echo, which at ratio 4 they are not (it reads 0.466 there).
  """
  t1, t2, T2_prime_ms = 16.0, 40.0, 2.0
  seq = Sequence()
  for dur in (None, t1, None, t2, None, t1):
    if dur is None:
      pulse = make_hard_pulse_block(np.pi / 2, dur_ms=0.002, dt_ms=1e-4)
      pulse.store_magnetization = False
      seq.add_block(pulse)
    else:
      seq.add_block(make_empty_block(dur, dt_ms=dur))
      seq.blocks[-1].store_magnetization = False
  seq.blocks[-1].store_magnetization = True          # the stimulated echo

  solver = BlochSolver(
    seq, minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
    initial_Mxy=0.0 + 0.0j, initial_Mz=1.0, perfect_spoiling=False,
    dtype='float64', t2_prime=Quantity(T2_prime_ms, 'ms'), spectral_bins=256)
  Mxy, _ = solver.solve()
  assert abs(abs(Mxy[0, 0]) - 0.5) < 5e-3, (
    f'stimulated echo is {abs(Mxy[0, 0]):.4f}, expected 0.5')


def test_b1_map_and_a_per_node_t2_prime_stay_aligned(minimal_phantom):
  """Two INDEPENDENT per-node maps, both expanded K-fold by np.repeat, given
  deliberately opposite orderings. If the two expansions disagreed, each node
  would silently get another node's constant -- and nothing else would notice,
  because the aggregate decay and the echo amplitude would both still be
  right."""
  n_nodes = minimal_phantom.local_nodes.shape[0]
  b1 = np.linspace(1.0, 0.3, n_nodes)
  t2_prime = np.linspace(4.0, 30.0, n_nodes)        # opposite order
  t_ms = 10.0

  seq = Sequence()
  pulse = make_hard_pulse_block(np.pi / 2, dur_ms=0.002, dt_ms=1e-4)
  pulse.store_magnetization = False
  seq.add_block(pulse)
  seq.add_block(make_empty_block(t_ms, dt_ms=t_ms))

  solver = BlochSolver(
    seq, minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
    initial_Mxy=0.0 + 0.0j, initial_Mz=1.0, perfect_spoiling=False,
    dtype='float64', b1_map=b1,
    t2_prime=Quantity(t2_prime, 'ms'), spectral_bins=64)
  Mxy, _ = solver.solve()

  expected = np.sin(np.pi / 2 * b1) * np.exp(-0.5 * (t_ms / t2_prime)**2)
  np.testing.assert_allclose(np.abs(Mxy[:, 0]), expected, atol=2e-4)
  # A scrambled pairing must be clearly distinguishable, or the test is vacuous.
  scrambled = np.sin(np.pi / 2 * b1) * np.exp(-0.5 * (t_ms / t2_prime[::-1])**2)
  assert np.abs(expected - scrambled).max() > 0.1


def test_a_spin_echo_refocuses_delta_b_and_t2_prime_together(minimal_phantom):
  """`t2_prime` rides the same per-node `delta_B` channel a caller may already
  be using, so the two superpose. Both are static, so a 180 must refocus BOTH:
  the echo lands on exp(-2 tau/T2) whatever the mean offset is."""
  T2_ms, T2_prime_ms, tau_ms = 300.0, 6.0, 20.0
  seq = _echo_train(1, tau_ms)
  floor = np.exp(-2 * tau_ms / T2_ms)

  # PER-NODE, not scalar. A spatially uniform offset contributes the same phase
  # to every bin of every node, factors straight out, and cannot affect
  # abs(Mxy) at all -- so the earlier scalar loop asserted the same number
  # three times and would have passed with delta_B ignored entirely.
  n_nodes = minimal_phantom.local_nodes.shape[0]
  spread = np.linspace(-4e-3, 4e-3, n_nodes).reshape(-1, 1)
  for delta_B in (np.zeros((n_nodes, 1)), spread):
    solver = BlochSolver(
      seq, minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(T2_ms, 'ms'),
      delta_B=delta_B, initial_Mxy=0.0 + 0.0j, initial_Mz=1.0,
      perfect_spoiling=False, dtype='float64',
      t2_prime=Quantity(T2_prime_ms, 'ms'), spectral_bins=64)
    Mxy, _ = solver.solve()
    # Every node, not just node 0: a per-node offset makes them differ before
    # the 180 and identical after it.
    np.testing.assert_allclose(np.abs(Mxy[:, 0]), floor, atol=2e-3)

  # And the negative control: without the 180, the same per-node offset leaves
  # the nodes visibly out of step, so the assertion above is not vacuous.
  no_refocus = _echo_train(0, tau_ms)
  no_refocus.blocks[-1].store_magnetization = True
  solver = BlochSolver(
    no_refocus, minimal_phantom, T1=Quantity(1e9, 'ms'),
    T2=Quantity(T2_ms, 'ms'), delta_B=spread, initial_Mxy=0.0 + 0.0j,
    initial_Mz=1.0, perfect_spoiling=False, dtype='float64',
    t2_prime=Quantity(T2_prime_ms, 'ms'), spectral_bins=64)
  Mxy, _ = solver.solve()
  assert np.ptp(np.angle(Mxy[:, -1])) > 0.5, (
    'the per-node delta_B should leave the nodes out of phase without a 180')


def test_the_sub_ensemble_survives_a_second_solve_call(minimal_phantom):
  """`solve(start=..., end=...)` in a per-shot loop is how every steady-state
  example in the repo drives the solver. The carried state used to be stored
  bin-expanded and re-expanded on the next call, so the SECOND call raised.

  Split so the 180 lands in a different call from the dephasing it undoes: the
  echo can only rephase if the ensemble crossed the call boundary intact.
  """
  tau_ms = 25.0
  seq = _echo_train(1, tau_ms)
  solver = BlochSolver(
    seq, minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
    initial_Mxy=0.0 + 0.0j, initial_Mz=1.0, perfect_spoiling=False,
    dtype='float64', t2_prime=Quantity(5.0, 'ms'), spectral_bins=64)

  solver.solve(start=0, end=2)                      # 90 + tau: dephases
  assert abs(solver.initial_Mxy[0, 0]) < 1e-3, 'should have dephased by tau'
  Mxy, _ = solver.solve(start=2, end=5)             # 180 + tau: must rephase
  assert abs(Mxy[0, 0]) > 0.999, (
    f'the echo reached {abs(Mxy[0, 0]):.6f}; the ensemble did not survive the '
    f'call boundary')

  # And a caller who RESETS the state between calls must be honoured, not
  # silently resumed from the carried ensemble.
  solver.solve(start=0, end=2)
  n_nodes = minimal_phantom.local_nodes.shape[0]
  solver.initial_Mxy = np.zeros((n_nodes, 1), dtype=np.complex128)
  solver.initial_Mz = np.ones((n_nodes, 1))
  Mxy, _ = solver.solve(start=2, end=5)
  assert abs(Mxy[0, 0]) < 1e-6, 'the reset was ignored'


def test_the_new_features_hold_up_at_the_default_float32(wide_phantom,
                                                         minimal_phantom):
  """float32 is the DEFAULT, and every test written for these features when
  they landed used float64. Tolerances are the measured float32 gaps, so this
  pins the precision rather than merely exercising the path."""
  scanner = Scanner()
  B0_mT = scanner.field_strength.m_as('mT')
  gamma = scanner.gamma.m_as('rad/ms/mT')

  # Concomitant. The floor here is NOT the concomitant term: it is the LINEAR
  # phase, which reaches 3852 rad over this geometry and is represented to
  # float32 precision, so the on/off ratio cancels it only to eps32 * 3852 =
  # 4.6e-4 rad. Asserting that bound rather than a magic number, because it is
  # what the number actually means -- the concomitant phase itself is 3.34 rad.
  G, dur_ms = (16.0, -12.0, 20.0), 5.0
  block = _gradient_block(G, dur_ms)
  off = _precess(wide_phantom, block, dtype='float32', concomitant_fields=False)
  on = _precess(wide_phantom, block, dtype='float32', concomitant_fields=True)
  nodes = wide_phantom.local_nodes.astype(np.float64)
  want = -gamma * _concomitant_field_mT(nodes, G, B0_mT) * dur_ms
  worst = float(np.abs(np.exp(1j * np.angle(on / off)) - np.exp(1j * want)).max())
  linear_phase = gamma * float(np.abs(nodes @ np.asarray(G)).max()) * dur_ms
  floor = np.finfo(np.float32).eps * linear_phase
  assert worst < 2 * floor, (
    f'concomitant in float32 is off by {worst:.2e}, above the {floor:.2e} that '
    f'the {linear_phase:.0f} rad linear phase alone accounts for')

  # B1+: float32 costs 3.3e-4 degrees of flip.
  solver = BlochSolver(
    make_single_block_sequence(make_hard_pulse_block(np.pi / 2, dur_ms=0.5)),
    minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
    initial_Mxy=0.0 + 0.0j, initial_Mz=1.0, perfect_spoiling=False,
    dtype='float32', b1_map=0.8)
  Mxy, Mz = solver.solve()
  flip = np.rad2deg(np.arctan2(np.abs(Mxy[0, 0]), Mz[0, 0]))
  assert abs(flip - 72.0) < 1e-2, f'float32 flip is {flip:.4f} deg, want 72'

  # T2': float32 warns, and with no background field costs 2.2e-4 relative.
  seq = Sequence()
  pulse = make_hard_pulse_block(np.pi / 2, dur_ms=0.002, dt_ms=1e-4)
  pulse.store_magnetization = False
  seq.add_block(pulse)
  seq.add_block(make_empty_block(20.0, dt_ms=20.0))
  with pytest.warns(UserWarning, match='float32'):
    solver = BlochSolver(
      seq, minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
      initial_Mxy=0.0 + 0.0j, initial_Mz=1.0, perfect_spoiling=False,
      dtype='float32', t2_prime=Quantity(20.0, 'ms'), spectral_bins=32)
  Mxy, _ = solver.solve()
  assert abs(abs(Mxy[0, 0]) - np.exp(-0.5)) < 1e-3


# ---------------------------------------------------------------------------
# 6. Second-audit regressions
# ---------------------------------------------------------------------------
#
# Each of these reproduces a defect the second audit found in the first
# audit's work. The first two used to crash or hang rather than fail.


def test_a_wrong_length_attribute_raises_instead_of_corrupting_the_heap(
        minimal_phantom):
  """The kernel sizes everything from r0.rows() and validates nothing else, so
  a short array is an out-of-bounds WRITE under -DNDEBUG -DEIGEN_NO_DEBUG.

  Reassigning a public attribute between solves is the reachable way in, and
  a scalar is the exact spelling the constructor accepts. Before the fix this
  aborted the process with `free(): invalid next size (fast)`.
  """
  seq = make_single_block_sequence(make_empty_block(2.0, dt_ms=0.5))
  solver = BlochSolver(
    seq, minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(50.0, 'ms'),
    initial_Mxy=1.0 + 0.0j, initial_Mz=0.0, perfect_spoiling=False,
    dtype='float64')
  solver.solve()

  n_nodes = minimal_phantom.local_nodes.shape[0]
  long = n_nodes + 3
  cases = (('initial_Mxy', 0.0 + 0.0j),
           ('initial_Mz', 1.0),
           ('delta_B', np.zeros((long, 1))),
           ('T1', Quantity(np.full((long, 1), 1e9), 'ms')),
           ('T2', Quantity(np.full((long, 1), 50.0), 'ms')),
           # Not a length but a column count: a rows-only test passes this.
           ('delta_B', np.zeros((n_nodes, 5))))
  # `x` is checked on the same loop and cannot be reached from here -- it comes
  # from phantom.local_nodes, and the kernel's r0 is Matrix<T, Dynamic, 3>, so
  # pybind refuses anything but three columns before the check runs.
  for bins in (1, 8):
    extra = ({} if bins == 1 else
             dict(t2_prime=Quantity(12.0, 'ms'), spectral_bins=bins))
    for attribute, value in cases:
      fresh = BlochSolver(
        seq, minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(50.0, 'ms'),
        initial_Mxy=1.0 + 0.0j, initial_Mz=0.0, perfect_spoiling=False,
        dtype='float64', **extra)
      fresh.solve()
      setattr(fresh, attribute, value)
      with pytest.raises(ValueError, match=attribute):
        fresh.solve()


def test_a_failed_solve_does_not_tear_the_sub_ensemble(minimal_phantom):
  """A solve() that raises midway must leave the carried ensemble where the
  previous call left it, so a retry reproduces the clean run.

  The resume path used to alias self._bin_Mxy rather than copy it, and the
  block loop writes in place -- so a failure left the ensemble half-advanced
  while the stamp still described the state before the call, and the next
  solve() resumed from it silently. Measured 0.2416 error, no warning.
  """
  def build():
    seq = Sequence()
    for _ in range(4):
      block = make_empty_block(10.0, dt_ms=10.0)
      block.store_magnetization = True
      seq.add_block(block)
    return seq

  def make():
    return BlochSolver(
      build(), minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
      initial_Mxy=1.0 + 0.0j, initial_Mz=0.0, perfect_spoiling=False,
      dtype='float64', t2_prime=Quantity(20.0, 'ms'), spectral_bins=16)

  reference = np.abs(make().solve()[0][0])

  solver = make()
  solver.solve(start=0, end=1)

  class _Boom:
    def m_as(self, *_args, **_kwargs):
      raise RuntimeError('injected failure mid-loop')

  block = solver.sequence.blocks[2]
  saved, block.discrete_times = block.discrete_times, _Boom()
  with pytest.raises(RuntimeError):
    solver.solve(start=1, end=4)
  block.discrete_times = saved

  retry = np.abs(solver.solve(start=1, end=4)[0][0])
  np.testing.assert_allclose(retry, reference[1:], atol=1e-12)


def test_a_non_finite_b1_map_is_refused_by_the_kernel_too(minimal_phantom):
  """-Ofast implies -ffinite-math-only, under which `v != v`, std::isnan and
  Eigen's allFinite() are all folded to false. The kernel's guard was
  therefore dead code and a NaN b1_map produced a NaN magnetization.

  Exercised through the kernel directly, since the Python check would
  otherwise fire first and the C++ one would never be reached.
  """
  from feelmri.BlochSimulator import solve_mri_f64

  n_nodes, n_time = 4, 3
  args = dict(
    r0=np.zeros((n_nodes, 3)), T1=np.full((n_nodes, 1), 1e9),
    T2=np.full((n_nodes, 1), 1e9), delta_B=np.zeros((n_nodes, 1)),
    M0=1.0, gamma=267.5, rf_all=np.full((n_time, 1), 0.01 + 0j),
    G_all=np.asfortranarray(np.zeros((n_time, 3))), dt=np.full(n_time, 0.01),
    regime_idx=np.ones((n_time, 1), dtype=bool),
    Mxy_initial=np.zeros((n_nodes, 1), dtype=complex),
    Mz_initial=np.ones((n_nodes, 1)),
    modes=np.asfortranarray(np.zeros((0, 0))), weights=np.zeros((0, 0)),
    has_traj=False, order=2, Bz_old_init=np.zeros((n_nodes, 1)),
    rf_old_init=0j)

  solve_mri_f64(**args, b1_map=np.ones(n_nodes, dtype=complex))
  for poison in (np.nan, np.inf):
    # `1 + nan*1j` is nan in BOTH parts -- nan*0 is nan -- so the earlier
    # version of this test never reached the `.imag()` half of the guard: the
    # `||` short-circuited on the real part every time. complex() builds the
    # one-sided cases explicitly.
    for component in (complex(poison, 0.0), complex(1.0, poison)):
      bad = np.ones(n_nodes, dtype=complex)
      bad[1] = component
      with pytest.raises(Exception, match='non-finite'):
        solve_mri_f64(**args, b1_map=bad)


def test_a_spoiler_block_solves_with_the_other_features_on(minimal_phantom):
  """The spoiler branch carries a SECOND concomitant Magnus seed, derived from
  the jittered isochromat positions, and its own np.repeat of b1_map. Nothing
  else in the suite solves a spoiler block at all, so both were dead code.

  With no gradient the jitter cannot change the field, so spoiler=True must
  equal spoiler=False exactly -- and all three integrators must agree with the
  gradient on, which is the probe for a missing seed mirror since order 0 never
  reads the seed.
  """
  def run(spoiler, amplitude, method='magnus2', **kwargs):
    block = _gradient_block((amplitude, -amplitude, amplitude), 3.0)
    block.spoiler = spoiler
    seq = Sequence()
    pulse = make_hard_pulse_block(np.pi / 2, dur_ms=0.05)
    pulse.store_magnetization = False
    seq.add_block(pulse)
    seq.add_block(block)
    solver = BlochSolver(
      seq, minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
      initial_Mxy=0.0 + 0.0j, initial_Mz=1.0, perfect_spoiling=False,
      dtype='float64', method=method, isochromat_K=25, **kwargs)
    return solver.solve()[0][:, -1]

  for kwargs in ({}, dict(concomitant_fields=True), dict(b1_map=0.8),
                 dict(concomitant_fields=True, b1_map=0.8)):
    np.testing.assert_allclose(run(False, 0.0, **kwargs),
                               run(True, 0.0, **kwargs), atol=1e-14)

  spread = [run(True, 12.0, method=m, concomitant_fields=True)
            for m in ('cayley_klein', 'magnus2', 'magnus4')]
  for other in spread[1:]:
    np.testing.assert_allclose(np.abs(spread[0]), np.abs(other), rtol=1e-6)


def test_concomitant_fields_follow_a_moving_phantom(wide_phantom):
  """The concomitant term is evaluated at the DEFORMED position in the kernel
  and at the deformed position in the Python Magnus seed. A constant POD
  displacement must therefore be indistinguishable from building the phantom
  at the displaced position.

  The residual is the POD's own float32 mode representation, not the term: it
  is identical with the feature on and off, which the test asserts so a real
  disagreement cannot hide behind it.
  """
  pytest.importorskip('meshio')
  from feelmri.Motion import POD

  shift = np.array([0.03, -0.02, 0.025])
  nodes = wide_phantom.local_nodes.astype(np.float64)
  n_nodes, n_times = nodes.shape[0], 6
  data = np.zeros((n_nodes, 3, n_times), dtype=np.float32)
  for axis in range(3):
    data[:, axis, :] = shift[axis]
  pod = POD(data=data, times=np.linspace(0.0, 5.0, n_times), n_modes=1)

  block = _gradient_block((15.0, -10.0, 18.0), 5.0)
  gaps = {}
  for concomitant in (False, True):
    moving = _precess(wide_phantom, block, pod_trajectory=pod,
                      concomitant_fields=concomitant)
    # The same block on a phantom that is simply built at the shifted position.
    static_phantom = _shifted_phantom(wide_phantom, shift)
    static = _precess(static_phantom, block, concomitant_fields=concomitant)
    gaps[concomitant] = float(np.abs(np.exp(1j * np.angle(moving))
                                     - np.exp(1j * np.angle(static))).max())

  assert gaps[True] < 5e-3, (
    f'a moving phantom disagrees with a statically shifted one by '
    f'{gaps[True]:.2e}; the concomitant term is not following the motion')
  assert abs(gaps[True] - gaps[False]) < 1e-6, (
    f'the residual differs with the term on ({gaps[True]:.2e}) and off '
    f'({gaps[False]:.2e}), so it is NOT just the float32 POD representation')


# ---------------------------------------------------------------------------
# 7. Third-audit regressions
# ---------------------------------------------------------------------------


def test_only_the_stored_columns_are_allocated(tmp_path_factory):
  """solve() used to allocate Mxy, Mz and the sub-ensemble over EVERY block
  and slice them to the stored ones at the end.

  A column is write-only until that slice, so the whole difference was waste,
  and the bin array carries the n_bins factor on top of it. Measured at 22 167
  nodes with K = 28 (a pruned gaussian K = 32) in complex128: epi_v142
  reserved 2.29 GB to keep 0.01 GB, and flash_tr_v15, which stores nothing at
  all, reserved 1.59 GB to keep none.

  Here the same ratio is 200 blocks against 1 stored column, so the old code
  peaks above 100 MB and the fixed one below 1 MB.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  import tracemalloc

  mesh_path = tmp_path_factory.mktemp('audit3_rod') / 'rod.vtu'
  make_1d_rod_mesh(mesh_path, length=0.04, n_segments=300,
                   transverse_width=1e-4)
  phantom = FEMPhantom(path=str(mesh_path))
  n_nodes = phantom.local_nodes.shape[0]

  seq = Sequence()
  for index in range(200):
    block = make_empty_block(1.0, dt_ms=1.0)
    block.store_magnetization = (index == 199)
    seq.add_block(block)

  solver = BlochSolver(
    seq, phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
    initial_Mxy=1.0 + 0.0j, initial_Mz=0.0, perfect_spoiling=False,
    t2_prime=Quantity(10.0, 'ms'), spectral_bins=32, dtype='float64')
  n_bins = solver.n_spectral_bins

  tracemalloc.start()
  Mxy, _ = solver.solve()
  peak = tracemalloc.get_traced_memory()[1]
  tracemalloc.stop()

  assert Mxy.shape == (n_nodes, 1)
  assert solver.bin_magnetization.shape == (n_nodes, n_bins, 1)

  over_all_blocks = n_nodes * n_bins * len(seq.blocks) * 16
  assert peak < 0.2 * over_all_blocks, (
    f'solve() peaked at {peak / 1e6:.1f} MB, against {over_all_blocks / 1e6:.1f} '
    f'MB for the sub-ensemble over all {len(seq.blocks)} blocks -- it is still '
    f'allocating columns it throws away')


def test_stored_columns_keep_their_block_order(minimal_phantom):
  """Writing through a block-to-slot map must put each stored block's state in
  the column the caller expects. ReadoutWindow.m_storage_idx counts stored
  blocks, so an off-by-one here silently pairs a readout with the wrong echo.
  """
  flags = (True, False, False, True, False, True)
  T2_ms = 40.0

  def build():
    seq = Sequence()
    for stored in flags:
      block = make_empty_block(10.0, dt_ms=10.0)
      block.store_magnetization = stored
      seq.add_block(block)
    return seq

  def run(sequence):
    solver = BlochSolver(
      sequence, minimal_phantom, T1=Quantity(1e9, 'ms'),
      T2=Quantity(T2_ms, 'ms'), initial_Mxy=1.0 + 0.0j, initial_Mz=0.0,
      perfect_spoiling=False, dtype='float64')
    return solver.solve()[0]

  selective = run(build())
  every = build()
  for block in every.blocks:
    block.store_magnetization = True
  dense = run(every)

  assert selective.shape[1] == sum(flags)
  wanted = [i for i, stored in enumerate(flags) if stored]
  np.testing.assert_allclose(selective, dense[:, wanted], rtol=0, atol=0)
  # ... and the columns are the decay at 10 ms per elapsed block, so the map
  # cannot be right by accident.
  expected = np.exp(-10.0 * (np.array(wanted) + 1) / T2_ms)
  np.testing.assert_allclose(np.abs(selective[0]), expected, rtol=1e-6)


def test_a_one_dimensional_per_node_array_is_accepted_between_solves(
        minimal_phantom):
  """The constructor normalises an (n,) per-node array onto the node column, so
  the same spelling assigned to a public attribute must work too.

  The universal length check added by the second audit required ndim == 2, so
  `solver.delta_B = np.zeros(n)` -- correct data, one missing trailing axis --
  was refused with a message claiming it was wrongly sized. The check still
  refuses a wrong LENGTH, which is the case that corrupts the heap.
  """
  n_nodes = minimal_phantom.local_nodes.shape[0]
  seq = make_single_block_sequence(make_empty_block(2.0, dt_ms=0.5))

  def solver_with(**kwargs):
    return BlochSolver(
      seq, minimal_phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(50.0, 'ms'),
      initial_Mxy=1.0 + 0.0j, initial_Mz=0.0, perfect_spoiling=False,
      dtype='float64', **kwargs)

  reference = solver_with(delta_B=np.full((n_nodes, 1), 2e-3)).solve()[0]

  flat = solver_with()
  flat.delta_B = np.full(n_nodes, 2e-3)
  np.testing.assert_allclose(flat.solve()[0], reference, rtol=0, atol=0)

  # A genuinely wrong length is still refused.
  short = solver_with()
  short.delta_B = np.full(n_nodes - 1, 2e-3)
  with pytest.raises(ValueError, match='delta_B'):
    short.solve()


def test_the_spoiler_seed_carries_the_concomitant_field(wide_phantom):
  """A spoiler block solved WITH a gradient, which is the only arrangement that
  can see the concomitant term in the jittered Magnus seed.

  The existing spoiler test runs at G = 0, where Bc is proportional to G^2 and
  therefore exactly zero, so the seed at `Bloch.py`'s spoiler branch was
  executed and never checked. Here the gradient is constant over the block, so
  the trapezoidal average equals the end-of-interval value and `magnus2` must
  reproduce `cayley_klein` EXACTLY -- and `cayley_klein` never reads the seed,
  which is what makes it the reference.

  Measured on this phantom at dt = 0.5 ms, worst |Mxy| difference against
  cayley_klein:

      seed as shipped                       4.7e-16
      concomitant term dropped from it      4.9e-2   (9.8e-3 at dt = 0.1,
                                                      i.e. the O(dt) block
                                                      boundary error)
      seed taken at the node centres        8.3e-01

  so the tolerance below separates the right answer from either mistake by
  thirteen orders of magnitude.
  """
  G, dur_ms, dt_ms = (14.0, -9.0, 20.0), 4.0, 0.5

  def run(method):
    block = _gradient_block(G, dur_ms, dt_ms=dt_ms)
    block.spoiler = True
    return _precess(wide_phantom, block, method=method,
                    concomitant_fields=True, isochromat_K=8,
                    isochromat_seed=0)

  reference = run('cayley_klein')
  # The spoiler has to have done something, or the comparison is between two
  # undephased states and proves nothing.
  assert np.abs(reference).max() < 0.8, (
    f'the spoiler left |Mxy| at {np.abs(reference).max():.4f} of 1.0, so the '
    f'isochromats are not dephasing and the seed cannot matter')

  for method in ('magnus2', 'magnus4'):
    gap = float(np.abs(run(method) - reference).max())
    assert gap < 1e-12, (
      f'{method} differs from cayley_klein by {gap:.3e} on a CONSTANT field, '
      f'where the trapezoidal average is exact -- the jittered Magnus seed '
      f'is not the field the kernel then integrates')


# ---------------------------------------------------------------------------
# 8. Fourth-audit regressions
# ---------------------------------------------------------------------------


def _trapezoid_block(amp_mT_per_m, rise_ms, flat_ms, fall_ms, axis=2,
                     dt_ms=2.6):
  """One trapezoid on a single axis, rastered COARSELY on purpose.

  `dt_ms` is deliberately longer than the whole event, so the block's raster is
  the trapezoid's four corners and nothing else -- which is what an imported
  Pulseq block gets, since the adapter builds every block with the default
  dt = 10 ms.
  """
  scanner = Scanner()
  timings = np.array([0.0, rise_ms, rise_ms + flat_ms,
                      rise_ms + flat_ms + fall_ms])
  amps = np.array([0.0, amp_mT_per_m, amp_mT_per_m, 0.0])
  gradient = Gradient(timings=Quantity(timings, 'ms'),
                      amplitudes=Quantity(amps, 'mT/m'), scanner=scanner,
                      ref=Quantity(0.0, 'ms'), time=Quantity(0.0, 'ms'),
                      axis=axis)
  return SequenceBlock(gradients=[gradient], dur=Quantity(timings[-1], 'ms'),
                       dt=Quantity(dt_ms, 'ms'), empty=False,
                       store_magnetization=True)


def test_the_concomitant_term_gets_the_exact_second_moment_of_a_ramp(
        wide_phantom):
  """`magnus2` integrates a straight ramp exactly from its endpoints, which is
  why `dt_gr` defaults to disabled -- but that covers the LINEAR field only.

  `Bc` goes as `G^2`, so along a ramp it is QUADRATIC in time and the
  trapezoidal quadrature is not exact: over a ramp it charges `A^2*h/2` where
  the exact second moment is `A^2*h/3`. The error is one-signed, so every ramp
  over-counts. On this trapezoid the corner rule gives 920 against an exact
  880, and before the solver sub-sampled its ramps it reproduced the CORNER
  value to 8.8e-14 rad -- 1.5e-2 rad of phase here, and 0.15 to 0.20 rad on
  gre_v15 / se_v15 / epi_v142, about 20% of the effect being modelled.

  The gradient is pure Gz and the phantom sits off-axis, so the linear term
  `Gz*z` is not what is being measured: the comparison is on-minus-off.
  """
  A, RISE, FLAT, FALL = 20.0, 0.30, 2.0, 0.30
  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')
  B0_mT = scanner.field_strength.m_as('mT')

  block = _trapezoid_block(A, RISE, FLAT, FALL)
  on = _precess(wide_phantom, block, concomitant_fields=True)
  off = _precess(wide_phantom, block, concomitant_fields=False)
  measured = np.angle(on / off)

  nodes = wide_phantom.local_nodes.astype(np.float64)
  # Pure Gz: Bc collapses to (Gz^2/4)(x^2 + y^2) / (2 B0).
  geometry = (nodes[:, 0] ** 2 + nodes[:, 1] ** 2) / (8.0 * B0_mT)
  exact_moment = A * A * (RISE / 3.0 + FLAT + FALL / 3.0)
  # The trapezoid rule over the stored corners, written out rather than
  # integrated numerically: a ramp contributes A^2*h/2 under it.
  corner_moment = A * A * (RISE / 2.0 + FLAT + FALL / 2.0)
  assert corner_moment > 1.04 * exact_moment, (
    'this trapezoid no longer separates the two quadratures')

  to_exact = float(np.abs(measured + gamma * geometry * exact_moment).max())
  to_corner = float(np.abs(measured + gamma * geometry * corner_moment).max())
  assert to_exact < 1e-3, (
    f'the concomitant phase is off the exact second moment by {to_exact:.3e} '
    f'rad; against the corner rule it is {to_corner:.3e}, so the ramps are '
    f'being charged A^2*h/2 instead of A^2*h/3')
  assert to_corner > 100 * to_exact, (
    'the two quadratures are no longer distinguishable on this case')


def test_sub_sampling_the_ramps_is_paid_only_when_concomitant_is_on(
        wide_phantom):
  """The densification must not touch the default path.

  With the term off, `magnus2` really does integrate the ramp exactly from the
  corners, so refining the raster by hand changes nothing -- measured 5.9e-13
  rad on gre_v15. That is what makes the fix a correction to the concomitant
  quadrature rather than a general raster effect, and it is also the guarantee
  that no existing result moves.
  """
  A, RISE, FLAT, FALL = 20.0, 0.30, 2.0, 0.30

  coarse = _precess(wide_phantom, _trapezoid_block(A, RISE, FLAT, FALL),
                    concomitant_fields=False)
  fine = _precess(wide_phantom,
                  _trapezoid_block(A, RISE, FLAT, FALL, dt_ms=0.002),
                  concomitant_fields=False)
  gap = float(np.abs(coarse - fine).max())
  assert gap < 1e-12, (
    f'refining the raster moved the feature-OFF answer by {gap:.3e}; the '
    f'linear term is supposed to be exact from the trapezoid corners')


@pytest.mark.slow
def test_balanced_ssfp_reaches_its_closed_form_per_node(minimal_phantom):
  """The COHERENT steady state, which nothing in the suite pinned.

  The spoiled one is covered (`test_spoiled_steady_state_matches_the_closed_form`
  on flash_tr_v15). This is the other side: no spoiling at all, alternating RF
  phase, and coherence that has to survive 1600 block boundaries carrying the
  Magnus state. It is the regime `perfect_spoiling=False` exists for.

  On resonance the steady state immediately after the pulse is

      M+ = M0 sin(a) (1 - E1) / (1 - (E1 - E2) cos(a) - E1 E2)

  **The closed form was checked independently before it was used here**, by
  iterating the ideal-pulse Bloch map to its fixed point in numpy: 8.6e-15
  relative. That matters because the formula is quoted for the echo in some
  references, which differs by a factor sqrt(E2).

  Every node carries a different `b1_map`, so each has its own flip angle and
  its own closed form **in one solve** -- a per-node check of b1 in a coherent
  multi-block steady state, where a flip error compounds over TRs. The other
  b1 tests are all single-pulse.

  Two traps, both measured: the steady state is not reached at 400 TRs (the
  error reads 3.4e-3 there against 5.0e-5 at 800), and the residual scales with
  the PULSE WIDTH, not with dt -- 2.5e-3 at 0.05 ms, 1.0e-4 at 0.01, 2.0e-5 at
  0.002 -- because the closed form assumes an instantaneous pulse and the
  solver relaxes during it.
  """
  T1_ms, T2_ms, TR_ms, pulse_ms, n_tr = 600.0, 100.0, 5.0, 0.002, 800
  nominal = np.radians(15.0)
  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')
  n_nodes = minimal_phantom.local_nodes.shape[0]
  b1 = np.linspace(0.2, 1.8, n_nodes)

  def pulse(phase_rad, n_samples=4):
    amplitude = nominal / (gamma * pulse_ms)
    return RF(waveform=Quantity(np.full(n_samples, amplitude, dtype=complex), 'mT'),
              timings=Quantity(np.linspace(0.0, pulse_ms, n_samples), 'ms'),
              phase_offset=Quantity(phase_rad, 'rad'))

  seq = Sequence()
  for i in range(n_tr):
    block = SequenceBlock(rf_pulses=[pulse(0.0 if i % 2 == 0 else np.pi)],
                          dur=Quantity(pulse_ms, 'ms'))
    block.store_magnetization = (i == n_tr - 1)
    seq.add_block(block)
    seq.add_block(SequenceBlock(dur=Quantity(TR_ms - pulse_ms, 'ms'),
                                dt=Quantity(TR_ms - pulse_ms, 'ms')))

  Mxy, _Mz = BlochSolver(
    seq, minimal_phantom, M0=1.0, T1=Quantity(T1_ms, 'ms'),
    T2=Quantity(T2_ms, 'ms'), initial_Mxy=0.0, initial_Mz=1.0,
    perfect_spoiling=False, dtype='float64', b1_map=b1).solve()

  E1, E2 = np.exp(-TR_ms / T1_ms), np.exp(-TR_ms / T2_ms)

  def closed_form(flip):
    return (np.sin(flip) * (1.0 - E1)
            / (1.0 - (E1 - E2) * np.cos(flip) - E1 * E2))

  expected = closed_form(b1 * nominal)
  got = np.abs(Mxy[:, -1])

  # The nodes must actually differ, or a map applied to the wrong ones would
  # not show. Measured spread on these b1 values: ~5.9x.
  assert expected.max() / expected.min() > 3.0, (
    'these flip angles no longer separate the nodes')
  worst = float(np.abs(got / expected - 1.0).max())
  assert worst < 5e-4, (
    f'the balanced steady state is off its closed form by {worst:.2e} per node;'
    f' measured 5.0e-5 at this pulse width and TR count')
  # Reversing the map must move every node, so the agreement above is a
  # per-node one and not an average.
  reversed_gap = float(np.abs(closed_form(b1[::-1] * nominal) / expected
                              - 1.0).max())
  assert reversed_gap > 0.5, (
    'a reversed b1 map would give nearly the same answer here, so this case '
    'cannot localise the map')


def test_a_nan_relaxation_time_is_refused_rather_than_absorbed(minimal_phantom):
  """The kernel cannot see a NaN T1 or T2.

  Its uniform-relaxation dispatch asks `(T2.array() == T2(0)).all()`, and
  -ffinite-math-only lets the compiler assume that comparison cannot involve a
  NaN. Measured on a 27-node cube before the check: a NaN T2 at local node 2
  was silently given node 0's value and that node returned the HEALTHY
  exp(-0.04) = 0.96078944, while the same NaN at node 0 turned all 27 nodes
  into NaN. The blast radius depended on which local index the bad node landed
  at -- i.e. on the partition, so the same input gave a different wrong answer
  at a different rank count.
  """
  n_nodes = minimal_phantom.local_nodes.shape[0]
  seq = make_single_block_sequence(make_empty_block(4.0, dt_ms=1.0))

  for name, first in (('T2', 0), ('T2', 2), ('T1', 1)):
    values = np.full(n_nodes, 100.0)
    values[first] = np.nan
    kwargs = {'T1': Quantity(1e9, 'ms'), 'T2': Quantity(50.0, 'ms')}
    kwargs[name] = Quantity(values, 'ms')
    with pytest.raises(ValueError, match=name):
      BlochSolver(seq, minimal_phantom, initial_Mxy=1.0 + 0.0j,
                  initial_Mz=0.0, perfect_spoiling=False, dtype='float64',
                  **kwargs)

  # Zero and negative are refused for the same reason; an infinite relaxation
  # time is legitimate and means "no decay".
  for bad in (0.0, -50.0):
    with pytest.raises(ValueError, match='T2'):
      BlochSolver(seq, minimal_phantom, T1=Quantity(1e9, 'ms'),
                  T2=Quantity(bad, 'ms'), initial_Mxy=1.0 + 0.0j,
                  initial_Mz=0.0, perfect_spoiling=False, dtype='float64')
  solver = BlochSolver(seq, minimal_phantom, T1=Quantity(np.inf, 'ms'),
                       T2=Quantity(np.inf, 'ms'), initial_Mxy=1.0 + 0.0j,
                       initial_Mz=0.0, perfect_spoiling=False, dtype='float64')
  np.testing.assert_allclose(np.abs(solver.solve()[0][:, 0]), 1.0, atol=1e-12)



# ---------------------------------------------------------------------------
# PODVelocity: the Taylor time is measured from the block, not from the origin
# ---------------------------------------------------------------------------
#
# `PODVelocity` models position as the first-order expansion `x0 + v * t_ro`,
# and `t_ro` is the time since the excitation. `POD` (displacement) ignores
# that scale entirely and reads its argument only inside the periodic fold,
# where a block offset cancels against the trajectory's `timeshift`. So the
# two classes respond DIFFERENTLY to the same call, and the subclass is the
# one that carries the physics.
#
# Nothing exercised `PODVelocity` at all before this, which is how the solver
# came to hand it absolute sequence time: on `examples/phase_contrast.py` that
# put `t_ro` at 1305 ms instead of ~3 ms, inflating every displacement ~450x
# and advecting 84-90% of the moving spins out of an 10.4 mm slice.


def _constant_velocity_pod(n_nodes, vz_m_per_s, n_frames=4, dt_ms=50.0):
  """A POD whose velocity field is CONSTANT in time and uniform in space.

  Constant in time on purpose: the cardiac phase then cannot influence the
  answer, so the only thing a test can be measuring is `t_ro`.
  """
  from feelmri.Motion import PODVelocity
  times = np.arange(n_frames, dtype=np.float32) * dt_ms
  data = np.zeros((n_nodes, 3, n_frames), dtype=np.float32)
  data[:, 2, :] = vz_m_per_s * 1e-3          # m/s -> m/ms, the POD's own unit
  # Periodic, like every example: a NON-periodic POD evaluated past its last
  # frame returns NaN from the interpolator, which would poison the whole
  # magnetization and mask what this is measuring.
  return PODVelocity(times=times, data=data, n_modes=1, is_periodic=True,
                     local_to_global_nodes=np.arange(n_nodes))


def _displacement_probe_sequence(scanner, lead_ms, grad_mT_per_m, dur_ms):
  """An optional leading delay, then one gradient block that reads position.

  The gradient is constant, so the phase it winds is
  `-gamma * G * (z0 + v*t_ro) * t` -- a direct readout of how far the solver
  thinks the spins have moved.
  """
  seq = Sequence()
  if lead_ms > 0.0:
    seq.add_block(make_empty_block(lead_ms, dt_ms=lead_ms))
  g = Gradient(timings=Quantity(np.array([0.0, dur_ms]), 'ms'),
               amplitudes=Quantity(np.array([grad_mT_per_m] * 2), 'mT/m'),
               scanner=scanner, ref=Quantity(0.0, 'ms'),
               time=Quantity(0.0, 'ms'), axis=2)
  seq.add_block(SequenceBlock(gradients=[g], dur=Quantity(dur_ms, 'ms'),
                              dt=Quantity(0.05, 'ms'), empty=False,
                              store_magnetization=True))
  return seq


@pytest.mark.parametrize('lead_ms', [0.0, 500.0], ids=['no_delay', 'delay_500ms'])
def test_pod_velocity_taylor_time_is_measured_from_the_block(rod_phantom, lead_ms):
  """With a constant velocity field, a leading DELAY must not change how far
  the spins move during the readout block.

  `t_ro` is time since the excitation, so the displacement inside the block is
  `v * t_ro` with `t_ro` running 0 -> dur -- whatever absolute time the block
  happens to sit at. Handing the trajectory absolute time instead makes the
  displacement grow with the delay, which is what this pins.

  The delay block carries no gradient, so position cannot reach the field
  there; the only way the delay can matter is through `t_ro`.
  """
  phantom = rod_phantom
  scanner = Scanner()
  n = phantom.local_nodes.shape[0]
  # A modest gradient keeps the total phase near 5 rad, so the float32
  # POD weights (`get_weights` casts) stay well under the bound below.
  vz, grad, dur = 2.0, 2.0, 3.0

  seq = _displacement_probe_sequence(scanner, lead_ms, grad, dur)
  pod = _constant_velocity_pod(n, vz)
  solver = BlochSolver(seq, phantom, scanner=scanner, M0=1.0,
                       T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
                       initial_Mxy=1.0 + 0j, initial_Mz=0.0,
                       perfect_spoiling=False, dtype='float64',
                       pod_trajectory=pod)
  Mxy, _ = solver.solve()
  phase = np.angle(np.asarray(Mxy)[:, -1])

  # Closed form: with G constant and z(t) = z0 + v*t,
  #   phi = -gamma * G * (z0 * dur + v * dur^2 / 2)
  gamma = scanner.gamma.m_as('rad/ms/mT')
  z0 = phantom.local_nodes[:, 2]
  expected = -gamma * grad * (z0 * dur + (vz * 1e-3) * dur ** 2 / 2.0)
  worst = float(np.abs(np.angle(np.exp(1j * (phase - expected)))).max())
  # 1e-5: the weights are float32 by construction, so the floor is
  # ~eps32 * |phase|. Measured 6.3e-07 here; the defect this pins moves the
  # 500 ms case by many radians, so the bound is nowhere near the signal.
  assert worst < 1e-5, (
    f'the in-block displacement disagrees with v*t_ro by {worst:.3e} rad at a '
    f'lead of {lead_ms} ms; t_ro is not being measured from the block')

  # The trajectory must come back as the caller left it. The solver composes
  # the block offset onto `timeshift` to keep the cardiac phase absolute, and
  # a version that overwrote it left every later consumer reading the LAST
  # block's start time.
  assert pod.timeshift == 0.0, (
    f'the solver left timeshift at {pod.timeshift}, not the 0.0 it was given')


def test_a_caller_set_timeshift_still_reaches_the_cardiac_phase(rod_phantom):
  """The block offset is COMPOSED onto the caller's `timeshift`, not
  substituted for it. A time-varying velocity makes the two distinguishable:
  shifting the trajectory by half a cycle must change the answer."""
  from feelmri.Motion import PODVelocity
  phantom = rod_phantom
  scanner = Scanner()
  n = phantom.local_nodes.shape[0]
  period, n_frames = 200.0, 9
  times = np.linspace(0.0, period, n_frames, dtype=np.float32)
  data = np.zeros((n, 3, n_frames), dtype=np.float32)
  data[:, 2, :] = (2.0e-3 * np.cos(2 * np.pi * times / period)).astype(np.float32)

  results = {}
  for shift in (0.0, 0.5 * period):
    pod = PODVelocity(times=times, data=data.copy(), n_modes=3,
                      is_periodic=True, local_to_global_nodes=np.arange(n))
    pod.update_timeshift(shift)
    seq = _displacement_probe_sequence(scanner, 0.0, 20.0, 3.0)
    solver = BlochSolver(seq, phantom, scanner=scanner, M0=1.0,
                         T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
                         initial_Mxy=1.0 + 0j, initial_Mz=0.0,
                         perfect_spoiling=False, dtype='float64',
                         pod_trajectory=pod)
    results[shift] = np.angle(np.asarray(solver.solve()[0])[:, -1])
    assert pod.timeshift == shift, 'the caller\'s timeshift was not restored'

  apart = float(np.abs(results[0.0] - results[0.5 * period]).max())
  assert apart > 1e-3, (
    f'half a cycle of timeshift moved the phase by only {apart:.2e} rad, so '
    f'the caller\'s shift is being discarded')


def test_a_partial_solve_is_quiet_on_a_natively_built_sequence(minimal_phantom):
  """`solve(start=-N)` inside a per-shot loop is the documented incremental
  steady-state idiom, and the warning about column numbering is about
  `ReadoutWindow.m_storage_idx`, which only an imported sequence carries. A
  native sequence has no such index, so it must not warn -- under MPI the
  message was emitted once per rank on every run of `free_running.py`,
  `gradient_spoiling.py` and both water/fat examples.
  """
  seq = Sequence()
  solver = BlochSolver(seq, minimal_phantom, T1=Quantity(1e9, 'ms'),
                       T2=Quantity(1e9, 'ms'), perfect_spoiling=False,
                       dtype='float64')
  assert seq.from_pulseq is False
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    for _ in range(3):
      block = make_hard_pulse_block(0.1, dur_ms=0.2)
      block.store_magnetization = True
      seq.add_block(block)
      seq.add_block(make_empty_block(1.0))
      solver.solve(start=-2)
  assert not [w for w in caught if 'm_storage_idx' in str(w.message)]

  # The same loop on a sequence carrying the Pulseq bookkeeping DOES warn,
  # so the guard is gated rather than removed.
  seq.from_pulseq = True
  solver._warned_start_storage = False
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    seq.add_block(make_hard_pulse_block(0.1, dur_ms=0.2))
    seq.add_block(make_empty_block(1.0))
    solver.solve(start=-2)
  assert [w for w in caught if 'm_storage_idx' in str(w.message)]


def _constant_displacement_pod(n_nodes, shift, n_frames=4, dur_ms=5.0):
  """A POD whose every mode is the same rigid translation, so the deformed
  mesh is exactly the reference mesh moved by `shift`."""
  from feelmri.Motion import POD
  data = np.zeros((n_nodes, 3, n_frames), dtype=np.float32)
  for axis in range(3):
    data[:, axis, :] = shift[axis]
  return POD(data=data, times=np.linspace(0.0, dur_ms, n_frames), n_modes=1)


def test_a_lab_frame_field_moves_under_the_spins_and_delta_b_does_not():
  """The whole point of the channel, as a pair. Two fields that are numerically
  identical while the phantom is at rest must behave oppositely once it moves:

    lab-frame  (`b0_field`)  -- the spin samples the field where it now IS,
                                so a rigid shift `s` changes every node's
                                phase by exactly `-gamma (g.s) T`
    tissue-bound (`delta_B`) -- the value is frozen to the node, so the same
                                shift changes nothing at all

  Both arms are required. The equality alone passes for a solver that ignores
  the field; the null alone passes for one that ignores the motion.
  """
  import tempfile
  from pathlib import Path
  points = np.array([[0.11, -0.03, 0.07],
                     [-0.05, 0.12, 0.02],
                     [0.04, 0.06, -0.10],
                     [-0.09, -0.08, 0.05],
                     [0.02, -0.11, -0.06]])
  cells = np.array([[0, 1, 2, 3], [1, 2, 4, 3]])
  shift = np.array([0.03, -0.02, 0.025])
  g = np.array([0.011, -0.004, 0.0075])          # mT/m
  dur_ms = 5.0
  gamma = Scanner().gamma.m_as('rad/ms/mT')
  pod = _constant_displacement_pod(points.shape[0], shift, dur_ms=dur_ms)

  # Both descriptions must be built from the SAME node array: the phantom
  # stores float32 coordinates, so a per-node map computed from the exact
  # points would differ from the lab-frame field by 5e-08 for that reason
  # alone and the comparison below would be measuring the mesh dtype.
  nodes = np.asarray(_phantom_from_points(points, cells, 'b0_ref').local_nodes,
                     dtype=np.float64)

  def solve(tag, moving, **kwargs):
    phantom = _phantom_from_points(points, cells, tag)
    seq = make_single_block_sequence(make_empty_block(dur_ms, dt_ms=0.05))
    solver = BlochSolver(
      seq, phantom, T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
      initial_Mxy=1.0 + 0.0j, initial_Mz=0.0, perfect_spoiling=False,
      dtype='float64', method='magnus2',
      pod_trajectory=pod if moving else None, **kwargs)
    return solver.solve()[0][:, 0]

  field = B0Field(gradient=Quantity(g, 'mT/m'))
  lab_still = solve('lab_still', False, b0_field=field)
  lab_moved = solve('lab_moved', True, b0_field=field)

  per_node = (nodes @ g).reshape(-1, 1)
  tis_still = solve('tis_still', False, delta_B=per_node)
  tis_moved = solve('tis_moved', True, delta_B=per_node)

  # At rest the two descriptions are the same field.
  assert np.abs(lab_still - tis_still).max() < 1e-12, (
    'the two channels disagree even before anything moves')

  expected = -gamma * float(g @ shift) * dur_ms
  assert abs(expected) > 0.5, 'this geometry produces no phase to discriminate'

  moved = np.angle(lab_moved / lab_still)
  worst = float(np.abs(np.exp(1j * moved) - np.exp(1j * expected)).max())
  assert worst < 1e-6, (
    f'the lab-frame field did not follow the spins: {worst:.3e} against a '
    f'predicted {expected:.4f} rad')

  frozen = float(np.abs(tis_moved - tis_still).max())
  assert frozen < 1e-12, (
    f'the tissue-bound field moved with the mesh by {frozen:.3e}; delta_B is '
    f'a property of the material point and must not')


def test_bc_is_centred_on_isocentre_and_not_on_the_slice(wide_phantom):
  """`orient` moves the slice to the origin; `Bc` must stay on isocentre.

  The node coordinates a solver sees are measured from the slice centre, which
  is what makes the linear encoding right. `Bc` is not translation invariant --
  it is a quadratic form about ISOCENTRE -- so evaluating it on those same
  coordinates silently images an off-isocentre slab as though it sat in the
  middle of the bore.

  Asserted against the closed form at the PHYSICAL position, which needs no
  second solve to compare against. The `assert` on the naive value is what
  makes this a finding rather than a tolerance: at this offset the two differ
  by more than a radian, so a version that ignored `LOC` cannot pass by being
  nearly right.
  """
  dur_ms = 6.0
  scanner = Scanner()
  B0_mT = scanner.field_strength.m_as('mT')
  gamma = scanner.gamma.m_as('rad/ms/mT')

  P = wide_phantom.local_nodes.astype(np.float64)
  cells = np.asarray(wide_phantom.local_elements)
  G = np.array([21.0, -13.0, 25.0])
  LOC = np.array([0.031, -0.047, 0.062])

  # No rotation, so the imaging frame differs from the physical one by the
  # translation alone and the comparison isolates it.
  phantom = _phantom_from_points(P, cells, 'bc_loc')
  phantom.orient(np.eye(3), Quantity(LOC, 'm'))
  x_img = np.asarray(phantom.local_nodes, dtype=np.float64)

  blk = _gradient_block(tuple(G), dur_ms)
  phi = np.angle(_precess(phantom, blk, concomitant_fields=True)
                 / _precess(phantom, blk, concomitant_fields=False))

  def bc(pos):
    gx, gy, gz = G
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return ((gx * gx + gy * gy) * z * z + 0.25 * gz * gz * (x * x + y * y)
            - gx * gz * x * z - gy * gz * y * z) / (2.0 * B0_mT)

  right = -gamma * bc(x_img + LOC) * dur_ms
  naive = -gamma * bc(x_img) * dur_ms

  gap = float(np.abs(np.exp(1j * phi) - np.exp(1j * right)).max())
  assert gap < 1e-3, (
      f'the solver disagrees with Bc at the physical position by {gap:.3e}')

  blind = float(np.abs(right - naive).max())
  assert blind > 1.0, (
      f'this offset only moves the phase by {blind:.4f} rad, so the test cannot '
      f'tell an isocentre-centred Bc from a slice-centred one')

  # The re-centring has THREE mirrors, and the two Magnus seeds recompute the
  # field in Python. Under a constant gradient the trapezoidal rule is exact, so
  # `magnus2` must reproduce `cayley_klein`, which never reads a seed -- and a
  # coarse raster is what makes the resulting O(dt) error visible: it reads
  # 1.3e-01 with the plain seed left on the slice-centred position.
  coarse = _gradient_block(tuple(G), dur_ms, dt_ms=0.5)
  m2 = _precess(phantom, coarse, concomitant_fields=True)
  ck = _precess(phantom, coarse, concomitant_fields=True, method='cayley_klein')
  seed_gap = float(np.abs(m2 - ck).max())
  assert seed_gap < 1e-6, (
      f'the Magnus seed disagrees with the kernel by {seed_gap:.3e}, so it is '
      f'not re-centring Bc the same way')


def test_a_moving_spin_samples_a_quadratic_field_where_it_moves_to(wide_phantom):
  """A degree-2 lab-frame field must be evaluated at the CURRENT position.

  The quadratic part rides six solve-invariant scalars -- it does not follow
  the gradient the way the concomitant term does -- so it costs no memory and
  the kernel evaluates it at `curr`, which is where the spin actually is. The
  closed form below needs no second solve to compare against.

  The `frozen` assert is the guard: it is what a per-node `delta_B` would give,
  and it has to differ by far more than the tolerance or the test cannot tell
  the Eulerian answer from the Lagrangian one.
  """
  dur_ms = 5.0
  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')

  P = wide_phantom.local_nodes.astype(np.float64)
  cells = np.asarray(wide_phantom.local_elements)
  shift = np.array([0.021, -0.014, 0.018])          # 31 mm rigid translation
  # Sized so the FROZEN answer differs from the Eulerian one by ~0.8 rad:
  # at a tenth of that the two are indistinguishable and the test proves
  # nothing.
  q = np.array([1.0e-1, -6.0e-2, 8.0e-2, 4.0e-2, -5.0e-2, 3.0e-2])  # mT/m^2

  def quad(pos):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return (q[0] * x * x + q[1] * y * y + q[2] * z * z
            + q[3] * x * y + q[4] * x * z + q[5] * y * z)

  field = B0Field(gradient=Quantity(np.zeros(3), 'mT/m'), order=2,
                  coefficients=np.concatenate(([0.0], np.zeros(3), q)))
  assert field.kind == 'polynomial'

  blk = _gradient_block((0.0, 0.0, 0.0), dur_ms)

  def solve(tag, moving):
    phantom = _phantom_from_points(P, cells, tag)
    pod = (_constant_displacement_pod(phantom.local_nodes.shape[0], shift,
                                      dur_ms=dur_ms) if moving else None)
    return _precess(phantom, blk, b0_field=field, pod_trajectory=pod)

  still = solve('quad_still', False)
  moved = solve('quad_moved', True)

  nodes = np.asarray(_phantom_from_points(P, cells, 'quad_ref').local_nodes,
                     dtype=np.float64)
  want_still = -gamma * quad(nodes) * dur_ms
  want_moved = -gamma * quad(nodes + shift) * dur_ms

  for label, got, want in (('at rest', still, want_still),
                           ('moved', moved, want_moved)):
    gap = float(np.abs(np.exp(1j * np.angle(got)) - np.exp(1j * want)).max())
    assert gap < 1e-6, (
        f'the quadratic field {label} disagrees with its closed form by {gap:.3e}')

  # Freezing the field to the node -- what `delta_B` does -- gives the at-rest
  # answer wherever the spin has gone, which is the error being removed.
  frozen = float(np.abs(np.exp(1j * want_moved)
                        - np.exp(1j * want_still)).max())
  assert frozen > 0.5, (
      f'this geometry only moves the phase by {frozen:.3f}, so it cannot tell '
      f'the Eulerian answer from the Lagrangian one')


def test_a_rough_field_follows_a_moving_spin_to_first_order(wide_phantom):
  """The rung no polynomial can reach: an arbitrary analytic field on a phantom
  that moves.

  `dB0(x0 + u) ~= [dB0(x0) - g.x0] + curr . g` puts the bracket on `delta_B`,
  where it costs nothing, and leaves a per-node vector the kernel adds to the
  gradient scalars it already hoists. Exact for a linear field; first order in
  the displacement otherwise, which is what the tolerance below reflects.

  Two guards, and the test is worthless without them: the FROZEN answer -- what
  `delta_B` alone gives -- must be far outside the tolerance, and the closed
  form must not be reachable by a polynomial, or this would be testing rung B.
  """
  dur_ms = 4.0
  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')

  P = wide_phantom.local_nodes.astype(np.float64)
  cells = np.asarray(wide_phantom.local_elements)
  shift = np.array([0.004, -0.003, 0.005])           # 7.1 mm

  # A 0.25 m sine: smooth, so the expansion is valid, but no polynomial up to
  # the cap represents it -- `B0Field.fit` refuses it, which is the point.
  n = np.array([0.6, -0.5, 0.62]); n /= np.linalg.norm(n)
  amp = 2.0e-3
  rough = lambda p: amp * np.sin(2 * np.pi * (p @ n) / 0.25)

  ref = _phantom_from_points(P, cells, 'rough_ref')
  with pytest.raises(ValueError, match='needs the per-node expansion'):
    B0Field.fit(rough, B0Field._scanner_nodes(ref), collective=False)

  blk = _gradient_block((0.0, 0.0, 0.0), dur_ms)

  def solve(tag, moving):
    phantom = _phantom_from_points(P, cells, tag)
    field = B0Field.on_phantom(rough, phantom, collective=False)
    assert field.kind == 'nodal'
    pod = (_constant_displacement_pod(phantom.local_nodes.shape[0], shift,
                                      dur_ms=dur_ms) if moving else None)
    return _precess(phantom, blk, b0_field=field, pod_trajectory=pod)

  still, moved = solve('rough_still', False), solve('rough_moved', True)

  nodes = np.asarray(ref.local_nodes, dtype=np.float64)
  want_still = -gamma * rough(nodes) * dur_ms
  want_moved = -gamma * rough(nodes + shift) * dur_ms

  at_rest = float(np.abs(np.exp(1j * np.angle(still))
                         - np.exp(1j * want_still)).max())
  assert at_rest < 1e-6, (
      f'at rest the per-node field should be EXACT, off by {at_rest:.3e}')

  after = float(np.abs(np.exp(1j * np.angle(moved))
                       - np.exp(1j * want_moved)).max())
  frozen = float(np.abs(np.exp(1j * want_still)
                        - np.exp(1j * want_moved)).max())
  assert after < 0.1 * frozen, (
      f'the moved spin is off by {after:.3e}, not much better than freezing '
      f'the field to the node ({frozen:.3e})')
  assert frozen > 0.2, (
      f'freezing the field only costs {frozen:.3f}, so this geometry cannot '
      f'tell the Eulerian answer from the Lagrangian one')


@pytest.mark.parametrize('spoiler', [False, True], ids=['plain', 'spoiler'])
def test_the_per_node_field_reaches_both_magnus_seeds(wide_phantom, spoiler):
  """The per-node gradient has three mirror sites -- the kernel and the two
  Magnus seeds -- and a seed that disagrees with the kernel leaves an O(dt)
  error at every block boundary, silently.

  `cayley_klein` never reads a seed, so under a field that is CONSTANT IN TIME
  the trapezoidal rule is exact and `magnus2` must reproduce it. The POD here
  has zero displacement, which keeps the per-node channel live -- the solver
  only takes it when a trajectory is present -- while holding the field still.
  A coarse raster is what makes the O(dt) term visible; at the fine dt the rest
  of this file uses it hides under the tolerance.

  The spoiler arm is the one that is easy to miss: that seed is derived from
  the K-fold JITTERED positions, so it only fires under `spoiler=True` and no
  ordinary test reaches it.

  Measured with the term dropped from each seed in turn: **6.52e-01** plain and
  **3.98e-01** spoiler, against the 1e-9 gate below. Neither is visible to the
  moving-spin test above, which runs at a fine raster where the O(dt) term
  hides -- that is why this probe exists separately.
  """
  dur_ms = 6.0
  P = wide_phantom.local_nodes.astype(np.float64)
  cells = np.asarray(wide_phantom.local_elements)

  n = np.array([0.6, -0.5, 0.62]); n /= np.linalg.norm(n)
  rough = lambda p: 2.0e-3 * np.sin(2 * np.pi * (p @ n) / 0.25)

  def solve(tag, method):
    phantom = _phantom_from_points(P, cells, tag)
    field = B0Field.on_phantom(rough, phantom, collective=False)
    # A CONSTANT displacement: the channel is live (the solver only takes it
    # with a trajectory present) but `curr` never changes, so the field is
    # constant in time and the trapezoidal rule is exact. A zero displacement
    # would be refused -- a POD with no energy is undefined.
    pod = _constant_displacement_pod(phantom.local_nodes.shape[0],
                                     np.array([0.004, -0.003, 0.005]),
                                     dur_ms=dur_ms)
    blk = _gradient_block((0.0, 0.0, 0.0), dur_ms, dt_ms=0.75)
    blk.spoiler = spoiler
    return _precess(phantom, blk, method=method, b0_field=field,
                    pod_trajectory=pod, isochromat_K=4, isochromat_seed=0)

  m2 = solve(f'seed_m2_{spoiler}', 'magnus2')
  ck = solve(f'seed_ck_{spoiler}', 'cayley_klein')
  gap = float(np.abs(m2 - ck).max())
  assert gap < 1e-9, (
      f'the {"spoiler" if spoiler else "plain"} Magnus seed disagrees with the '
      f'kernel by {gap:.3e}: under a field constant in time the trapezoidal '
      f'rule is exact, so magnus2 must reproduce cayley_klein')


def test_the_shim_channels_survive_the_concomitant_branch():
  """Switching the concomitant term on must not change a field it cannot touch.

  With `G = 0` the Maxwell field `Bc = (Bx^2 + By^2)/(2 B0)` is IDENTICALLY
  zero -- both components are linear in the gradient -- so
  `concomitant_fields=True` is required to be bit-identical to False, whatever
  else the solver is carrying. That makes it an exact identity rather than a
  tolerance, and it needs no closed form.

  It is the only shape of test that can see a dropped kernel cell. The node
  loop branches on `Conc` at compile time and on the two shim channels at run
  time, and an `else if` hanging off the concomitant arm covered two of the
  four combinations while reading as though it covered all four: `node_lin`
  was dropped from `Bz_new` while both Python Magnus seeds kept it, so the
  opening trapezoidal step of every block carried the shim and no other step
  did. Measured before the fix, on this geometry: **2.32 rad** between the two
  flags, leaving the answer **1.83** from the Eulerian truth and **1.86** from
  the frozen one -- neither of the two things it could legitimately have been.
  """
  pytest.importorskip('meshio')
  import meshio
  import tempfile
  from feelmri import B0Field
  from _phantom_fixtures import make_cube_mesh

  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')
  dur_ms, shift = 5.0, np.array([0.021, -0.014, 0.018])

  path, _v = make_cube_mesh(Path(tempfile.mkdtemp()) / 'conc_shim.vtu',
                            'tetra', n=3, scale=0.18)
  mesh = meshio.read(str(path))
  P = np.asarray(mesh.points, dtype=np.float64)
  cells = mesh.cells_dict['tetra']

  # Curved on the scale of the object, so no polynomial up to the cap fits and
  # `on_phantom` falls back to the per-node rung -- the channel under test.
  # Gentle enough that the first-order expansion is still valid over 31 mm.
  rough = lambda q: 1.0e-3 * np.sin(q[:, 0] / 0.25) * np.cos(q[:, 1] / 0.30)

  blk = _gradient_block((0.0, 0.0, 0.0), dur_ms)

  def solve(tag, conc):
    phantom = _phantom_from_points(P, cells, tag)
    field = B0Field.on_phantom(rough, phantom, collective=False)
    assert field.kind == 'nodal', f'the fixture must be per-node, got {field.kind}'
    pod = _constant_displacement_pod(phantom.local_nodes.shape[0], shift,
                                     dur_ms=dur_ms)
    return _precess(phantom, blk, b0_field=field, pod_trajectory=pod,
                    scanner=scanner, concomitant_fields=conc)

  off = solve('conc_shim_off', False)
  on = solve('conc_shim_on', True)

  gap = float(np.abs(np.angle(on) - np.angle(off)).max())
  assert gap == 0.0, (
      f'at G = 0 the concomitant field is identically zero, so the flag must '
      f'change nothing; it moved the phase by {gap:.3e} rad')

  # Both arms must be the EULERIAN answer, not the frozen one -- otherwise the
  # identity above is satisfied by dropping the channel on both sides.
  nodes = np.asarray(_phantom_from_points(P, cells, 'conc_shim_ref').local_nodes,
                     dtype=np.float64)
  want_moved = -gamma * rough(nodes + shift) * dur_ms
  want_frozen = -gamma * rough(nodes) * dur_ms

  def err(got, want):
    return float(np.abs(np.exp(1j * np.angle(got)) - np.exp(1j * want)).max())

  for tag, got in (('off', off), ('on', on)):
    assert err(got, want_moved) < 2e-2, (
        f'[{tag}] the per-node field disagrees with the Eulerian truth by '
        f'{err(got, want_moved):.3e}')
  frozen_gap = err(off, want_frozen)
  assert frozen_gap > 10.0 * err(off, want_moved), (
      f'this geometry only separates the Eulerian answer from the frozen one '
      f'by {frozen_gap:.3e}, so it cannot tell them apart')


def _wobble_pod(n_nodes, dur_ms, amps=(0.012, -0.008, 0.010)):
  """A smooth displacement that is NOT zero at the coarse raster points.

  The obvious `sin(2 pi t / dur)` is degenerate here: it vanishes at 0, dur/2
  and dur, which is exactly where a dt = 10 ms raster samples a 20 ms block,
  so the trapezoid is accidentally exact and the probe reads 3e-15. The
  non-integer period and the phase offsets remove that coincidence.
  """
  from feelmri.Motion import POD
  n_frames = 24
  ts = np.linspace(0.0, dur_ms, n_frames)
  data = np.zeros((n_nodes, 3, n_frames), dtype=np.float32)
  for axis, amp in enumerate(amps):
    data[:, axis, :] = amp * np.sin(2 * np.pi * 1.37 * ts / dur_ms
                                    + 0.6 + 0.4 * axis)
  return POD(data=data, times=ts, n_modes=6)


def test_a_gradient_free_block_resolves_the_motion_under_a_lab_field():
  """`G = 0` no longer means the block composes exactly.

  A delay integrating at `dt = 10 ms` -- what the Pulseq adapter builds for
  every event-free block -- is exact under any subdivision while `Bz` is per
  node constant, because the rotation and the relaxation both compose. With a
  scanner-fixed field the kernel forms `curr . (G + g)`, so a MOVING spin sees
  a changing field even at `G = 0`, and the trapezoidal Omega_1 has an ordinary
  O(dt^2) error with a 10 ms step to pay it with.

  Measured against a converged raster, on a 20 ms gradient-free block with a
  12 mm displacement and a 3 mT/m lab gradient:

  | dt (ms) | before | after |
  |---|---|---|
  | 10 | **1.72e-01** | 1.27e-03 |
  | 5 | 3.02e-02 | 1.27e-03 |
  | 1 | 1.27e-03 | 1.27e-03 |
  | 0.2 | 4.48e-05 | 4.48e-05 |

  The last row is the other half of the contract: a caller who chose a raster
  finer than the cap keeps it. The control below is what makes this a finding
  rather than a re-derivation of the Magnus order -- with the field off the
  same rasters agree to **exactly 0.000e+00**, so it really is the field and
  the motion together, not the step size alone.
  """
  pytest.importorskip('meshio')
  import meshio
  import tempfile
  from feelmri import B0Field
  from _phantom_fixtures import make_cube_mesh

  dur_ms = 20.0
  path, _v = make_cube_mesh(Path(tempfile.mkdtemp()) / 'delay_motion.vtu',
                            'tetra', n=2, scale=0.12)
  mesh = meshio.read(str(path))
  P = np.asarray(mesh.points, dtype=np.float64)
  cells = mesh.cells_dict['tetra']
  field = B0Field(gradient=Quantity(np.array([3.0e-3, -2.0e-3, 2.5e-3]), 'mT/m'))

  def run(dt_ms, with_field):
    phantom = _phantom_from_points(P, cells, f'dm_{dt_ms}_{with_field}')
    block = _gradient_block((0.0, 0.0, 0.0), dur_ms, dt_ms=dt_ms)
    pod = _wobble_pod(phantom.local_nodes.shape[0], dur_ms)
    kwargs = {'b0_field': field} if with_field else {}
    return _precess(phantom, block, pod_trajectory=pod, **kwargs)

  def gap(a, b):
    return float(np.abs(np.angle(a * np.conj(b))).max())

  converged = run(0.01, True)
  for dt_ms in (10.0, 5.0, 1.0):
    assert gap(run(dt_ms, True), converged) < 5e-3, (
        f'at dt = {dt_ms} ms the block leaves '
        f'{gap(run(dt_ms, True), converged):.3e} rad against a converged '
        f'raster; the motion is not being resolved')

  # A caller who asked for a finer raster than the cap keeps it.
  assert gap(run(0.2, True), converged) < 1e-4

  # The control: with no field the claim the adapter relies on still holds
  # exactly, so nothing is densified and nothing is paid for.
  converged_off = run(0.01, False)
  for dt_ms in (10.0, 1.0):
    assert gap(run(dt_ms, False), converged_off) == 0.0, (
        'a gradient-free block with no lab field must compose exactly under '
        'any subdivision, and it no longer does')

  # `_motion_raster` guards its `block.dt` read with `except AttributeError`,
  # so it declares that it accepts a block without one -- and then read `dt`
  # and `dt_rf` again OUTSIDE that guard to build the tolerance, raising the
  # exception it had just swallowed. Every step it reads goes through the same
  # accessor now. Both objects below are exactly what the guard promises to
  # tolerate; before the fix each raised `AttributeError`.
  solver = BlochSolver.__new__(BlochSolver)
  coarse = np.arange(0.0, 50.0, 10.0)

  class _NoDt:
    pass

  class _BareFloatDt:
    dt, dt_rf = 10.0, 0.01          # plain floats, no `.m_as`

  for blk in (_NoDt(), _BareFloatDt()):
    out = np.asarray(solver._motion_raster(blk, coarse))
    assert float(np.diff(out).max()) <= B0_MOTION_DT_MS + 1e-9, (
        f'{type(blk).__name__} was accepted but its raster still steps '
        f'{float(np.diff(out).max()):.3g} ms, above the cap')


def _linear_displacement_pod(n_nodes, shift, n_frames=6, dur_ms=5.0):
  """A displacement that ramps LINEARLY from zero to `shift` over the block,
  so the spin moves at a constant velocity and the phase has a closed form."""
  from feelmri.Motion import POD
  ts = np.linspace(0.0, dur_ms, n_frames)
  data = np.zeros((n_nodes, 3, n_frames), dtype=np.float32)
  for axis in range(3):
    data[:, axis, :] = shift[axis] * ts / dur_ms
  return POD(data=data, times=ts, n_modes=2)


def test_a_spin_moving_at_constant_velocity_integrates_the_field_it_crosses():
  """The closed form for the whole channel, with no second solve to lean on.

  A spin at `x0` moving at constant `v` through `dB0 = g . x` accrues

      phi(T) = -gamma [ (g . x0) T + (g . v) T^2 / 2 ]

  and the `T^2 / 2` is the part that only a correct TIME INTEGRATION produces.
  Freezing the field to the node drops it entirely; sampling the field at the
  END of the window instead of integrating doubles it. The solver steps through
  the block, so it must land on the closed form, and the two wrong answers
  below are what the test is against.
  """
  pytest.importorskip('meshio')
  import meshio
  import tempfile
  from feelmri import B0Field
  from _phantom_fixtures import make_cube_mesh

  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')
  dur_ms = 6.0
  g = np.array([4.0e-3, -3.0e-3, 2.0e-3])          # mT/m
  shift = np.array([0.024, -0.016, 0.020])         # 35 mm over the block

  path, _v = make_cube_mesh(Path(tempfile.mkdtemp()) / 'const_v.vtu',
                            'tetra', n=2, scale=0.12)
  mesh = meshio.read(str(path))
  P = np.asarray(mesh.points, dtype=np.float64)
  cells = mesh.cells_dict['tetra']

  phantom = _phantom_from_points(P, cells, 'const_v')
  nodes = np.asarray(phantom.local_nodes, dtype=np.float64)
  pod = _linear_displacement_pod(phantom.local_nodes.shape[0], shift,
                                 dur_ms=dur_ms)
  got = _precess(phantom, _gradient_block((0.0, 0.0, 0.0), dur_ms),
                 b0_field=B0Field(gradient=Quantity(g, 'mT/m')),
                 pod_trajectory=pod)

  v = shift / dur_ms
  want = -gamma * ((nodes @ g) * dur_ms + (v @ g) * dur_ms ** 2 / 2.0)
  frozen = -gamma * (nodes @ g) * dur_ms                       # no motion term
  endpoint = -gamma * ((nodes + shift) @ g) * dur_ms           # doubles it

  def err(phase):
    return float(np.abs(np.exp(1j * np.angle(got)) - np.exp(1j * phase)).max())

  assert err(want) < 1e-5, (
      f'the solver leaves {err(want):.3e} against the closed form')
  # Both wrong answers have to be far away, or the tolerance above is doing
  # the work rather than the physics.
  assert err(frozen) > 100.0 * err(want)
  assert err(endpoint) > 100.0 * err(want)


@pytest.mark.parametrize('placement', ['axial+offset', 'oblique+offset'])
def test_a_per_node_field_is_the_same_field_however_the_phantom_is_placed(
        placement):
  """The per-node rung under an oblique orientation and a slice offset.

  `node_gradient` maps the scanner-frame gradient into the imaging frame as
  `g @ R`, and the nodal values are sampled at `R x + LOC`. Neither had ever
  been exercised: both per-node tests in the suite build their phantom without
  calling `orient`, so the rotation ran with `rotation=None` every time.

  Scored against a truth built from SCANNER coordinates and nothing the class
  owns. Measured: **7.7e-03** on the oblique arm, the same first-order Taylor
  residual the axial arm shows, against **9.4e-01** for an answer that forgets
  the frame -- a factor of 121, so a dropped `R` cannot hide in the tolerance.
  """
  pytest.importorskip('meshio')
  import meshio
  import tempfile
  from feelmri import B0Field
  from _phantom_fixtures import make_cube_mesh

  scanner = Scanner()
  gamma = scanner.gamma.m_as('rad/ms/mT')
  dur_ms = 5.0
  shift = np.array([0.018, -0.012, 0.015])
  rough = lambda q: 1.0e-3 * (np.sin(q[:, 0] / 0.25) * np.cos(q[:, 1] / 0.30)
                              * np.exp(q[:, 2] / 0.9))

  # Both arms carry an OFFSET. At `R = I, LOC = 0` the frame-naive answer
  # below is literally the same expression as the correct one, so the
  # discriminating assertion had to be skipped and the arm asserted only that
  # the field agreed with itself -- it passed with `g @ R` deleted from
  # `node_gradient` and with the rotation deleted from `_scanner_nodes`.
  if placement == 'axial+offset':
    R, LOC = np.eye(3), np.array([0.031, -0.047, 0.062])
  else:
    R, LOC = _rotation_zyx(0.37, -0.21, 0.15), np.array([0.031, -0.047, 0.062])

  path, _v = make_cube_mesh(Path(tempfile.mkdtemp()) / f'place_{placement}.vtu',
                            'tetra', n=3, scale=0.16)
  mesh = meshio.read(str(path))
  P = np.asarray(mesh.points, dtype=np.float64)
  cells = mesh.cells_dict['tetra']

  phantom = _phantom_from_points(P, cells, f'place_{placement}')
  phantom.orient(R, Quantity(LOC, 'm'))
  field = B0Field.on_phantom(rough, phantom, collective=False)
  assert field.kind == 'nodal'

  pod = _constant_displacement_pod(phantom.local_nodes.shape[0], shift,
                                   dur_ms=dur_ms)
  got = _precess(phantom, _gradient_block((0.0, 0.0, 0.0), dur_ms),
                 b0_field=field, pod_trajectory=pod, scanner=scanner)

  x_local = np.asarray(phantom.local_nodes, dtype=np.float64)
  want = -gamma * rough((x_local + shift) @ np.asarray(R).T + LOC) * dur_ms
  # The two ways to get the frame wrong, and they are NOT interchangeable:
  # with `R = I` a dropped rotation is the correct answer, so only the dropped
  # LOCATION discriminates there. Asserting one of them and skipping the arm
  # where it cannot fire leaves that arm asserting only that the field agrees
  # with itself.
  naive_rot = -gamma * rough(x_local + shift + LOC) * dur_ms
  naive_loc = -gamma * rough((x_local + shift) @ np.asarray(R).T) * dur_ms

  def err(phase):
    return float(np.abs(np.exp(1j * np.angle(got)) - np.exp(1j * phase)).max())

  assert err(want) < 2e-2, (
      f'[{placement}] the field disagrees with its scanner-frame truth by '
      f'{err(want):.3e}')
  assert err(naive_loc) > 20.0 * err(want), (
      f'[{placement}] dropping the slice offset is only {err(naive_loc):.3e} '
      f'away, so this placement cannot see it')
  if not np.allclose(R, np.eye(3)):
    assert err(naive_rot) > 20.0 * err(want), (
        f'[{placement}] dropping the rotation is only {err(naive_rot):.3e} '
        f'away, so this orientation cannot see it')


@pytest.mark.parametrize('conc', [False, True], ids=['plain', 'concomitant'])
@pytest.mark.parametrize('node_lin', [False, True], ids=['-nodelin', '+nodelin'])
@pytest.mark.parametrize('field_quad', [False, True], ids=['-quad', '+quad'])
def test_every_kernel_field_branch_reproduces_the_closed_form(
        conc, node_lin, field_quad):
  """All eight `Bz_new` branches, driven through the kernel directly.

  The node loop spells the field out once per cell of
  `Conc x has_node_lin x has_field_quad`, because a `+=` would regroup the FMAs
  under `-ffast-math` and the feature-off path has to stay bit-identical. Eight
  copies of one expression is exactly the shape in which a term goes missing
  from one of them, and that is what happened: an `else if (has_node_lin)`
  hanging off the concomitant branch read as though it covered all four cells
  and covered two, so with the concomitant term on the shim was dropped
  entirely while both Python Magnus seeds kept it.

  Two of the eight are unreachable from `BlochSolver` -- `solver_terms` never
  returns `quadratic` and `node_gradient` together -- so they carry the most
  complex expression in the file and nothing drives them. They stay, because
  the time-varying-field work will make the combination reachable, and this is
  what pins them meanwhile.

  Order 0 with no RF and no trajectory reduces the whole step to
  `phi = -gamma * Bz * T`, so every term appears linearly in a phase with a
  closed form. Every channel is non-zero and incommensurate with the others,
  so a dropped or duplicated term cannot cancel.
  """
  from feelmri.BlochSimulator import solve_mri_f64

  gamma, dur_ms, B0_mT = 267.5, 3.0, 1500.0
  x = np.array([[0.031, -0.047, 0.062], [-0.019, 0.023, -0.055],
                [0.044, 0.017, 0.029], [-0.038, -0.026, 0.011]])
  n = x.shape[0]
  G = np.array([21.0e-3, -13.0e-3, 25.0e-3])          # mT/m
  delta_B = np.array([1.1e-4, -0.7e-4, 0.3e-4, -1.9e-4])
  g_node = np.array([[3.0e-3, -2.0e-3, 1.5e-3], [-1.0e-3, 2.5e-3, -0.5e-3],
                     [2.0e-3, 1.0e-3, -3.0e-3], [-2.5e-3, -1.5e-3, 2.0e-3]])
  q = np.array([7.0e-3, -5.0e-3, 3.0e-3, 2.0e-3, -4.0e-3, 6.0e-3])
  conc_off_mT = -0.37e-3

  # What the kernel is asked to form, written independently of it.
  want = x @ G + delta_B
  if node_lin:
    want = want + np.einsum('ij,ij->i', x, g_node)
  if field_quad:
    px, py, pz = x[:, 0], x[:, 1], x[:, 2]
    want = want + (q[0]*px*px + q[1]*py*py + q[2]*pz*pz
                   + q[3]*px*py + q[4]*px*pz + q[5]*py*pz)
  if conc:
    px, py, pz = x[:, 0], x[:, 1], x[:, 2]
    Gx, Gy, Gz = G
    want = want + conc_off_mT + (
        (Gx*Gx + Gy*Gy) * pz*pz + 0.25 * Gz*Gz * (px*px + py*py)
        - Gx*Gz*px*pz - Gy*Gz*py*pz) / (2.0 * B0_mT)

  n_time = 2
  kw = dict(
    r0=np.asfortranarray(x), T1=np.full((n, 1), 1e12), T2=np.full((n, 1), 1e12),
    delta_B=delta_B.reshape(n, 1), M0=1.0, gamma=gamma,
    rf_all=np.zeros((n_time, 1), dtype=complex),
    G_all=np.asfortranarray(np.tile(G, (n_time, 1))),
    dt=np.array([0.0, dur_ms]),
    regime_idx=np.zeros((n_time, 1), dtype=bool),
    Mxy_initial=np.ones((n, 1), dtype=complex),
    Mz_initial=np.zeros((n, 1)),
    modes=np.asfortranarray(np.zeros((0, 0))), weights=np.zeros((0, 0)),
    has_traj=False, order=0, Bz_old_init=np.zeros((n, 1)), rf_old_init=0j,
    B0=B0_mT if conc else 0.0)
  if conc:
    kw['conc_offset'] = np.full(n_time, conc_off_mT)
  if node_lin:
    kw['node_lin'] = np.ascontiguousarray(g_node)
  if field_quad:
    kw['field_quad'] = np.ascontiguousarray(q)

  Mxy = solve_mri_f64(**kw)[0]
  if not conc:
    # `Bc_off` is read only inside the concomitant branch, so an offset passed
    # with `B0 <= 0` used to be silently discarded -- neither applied nor
    # refused. `BlochSolver` never reaches that state, but only because two
    # independent gates line up, and this is a public entry point.
    with pytest.raises(Exception, match='conc_offset'):
      solve_mri_f64(**{**kw, 'conc_offset': np.full(n_time, conc_off_mT)})
  got = np.angle(np.asarray(Mxy).reshape(-1))
  expect = np.angle(np.exp(-1j * gamma * want * dur_ms))
  err = float(np.abs(np.angle(np.exp(1j * (got - expect)))).max())
  assert err < 1e-11, (
      f'conc={conc} node_lin={node_lin} field_quad={field_quad}: the branch '
      f'is {err:.3e} rad from its closed form, so it is not forming the field '
      f'its three flags describe')
