"""The time-integrated gradient products behind the concomitant readout term.

`Bc` is a quadratic form in position whose coefficients depend only on time, so
its time integral factorises exactly into four scalars times four fixed spatial
monomials. Those four scalars are to the concomitant term what the three
components of `k` are to the linear one: a second trajectory.

Everything here is checked against a reference that is refined until it
converges ONTO the helper, rather than against a fixed tolerance -- the point
is that the segment rule is exact, not merely close.
"""
from __future__ import annotations

import numpy as np
import pytest
from pint import Quantity as Q_

from feelmri.Bloch import Sequence, SequenceBlock, _concomitant_mT
from feelmri.MRObjects import Gradient, Scanner
from feelmri.PulseqAdapter import (
  maxwell_moments,
  maxwell_moments_from_kspace,
  maxwell_phase_coefficients,
)


SCANNER = Scanner()
GAMMA = SCANNER.gamma.m_as('rad/ms/mT')
B0_MT = SCANNER.field_strength.m_as('mT')
NODES = np.array([[0.11, -0.03, 0.07],
                  [-0.05, 0.12, 0.02],
                  [0.04, 0.06, -0.10]])


def _gradient(timings, amplitudes, axis):
  return Gradient(timings=Q_(np.asarray(timings, dtype=float), 'ms'),
                  amplitudes=Q_(np.asarray(amplitudes, dtype=float), 'mT/m'),
                  scanner=SCANNER, ref=Q_(0.0, 'ms'), time=Q_(0.0, 'ms'),
                  axis=axis)


def _sequence(gradients, dur_ms):
  seq = Sequence()
  seq.add_block(SequenceBlock(gradients=gradients, dur=Q_(dur_ms, 'ms'),
                              dt=Q_(0.01, 'ms'), empty=False))
  return seq


def _trapezoid(amps, rise, flat, fall):
  """One trapezoid per axis, sharing corners. Ramps are the point: that is
  where a corner trapezoid rule is 50% out."""
  ts = [0.0, rise, rise + flat, rise + flat + fall]
  return _sequence([_gradient(ts, [0.0, a, a, 0.0], ax)
                    for ax, a in enumerate(amps)], ts[-1])


def _phase_from_helpers(seq, t0, sample_times, nodes=NODES):
  coef = maxwell_phase_coefficients(maxwell_moments(seq, t0, sample_times),
                                    SCANNER)
  x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
  # Six coefficients over x^2, y^2, z^2, xy, xz, yz -- the general symmetric
  # form, which collapses to the B0-aligned four when no rotation is given.
  return (coef[:, 0:1] * x ** 2 + coef[:, 1:2] * y ** 2 + coef[:, 2:3] * z ** 2
          + coef[:, 3:4] * x * y + coef[:, 4:5] * x * z + coef[:, 5:6] * y * z)


def _phase_by_dense_quadrature(seq, t0, t1, n_points, nodes=NODES):
  """-gamma * integral(Bc) dt, integrating the library's OWN expression."""
  fine = np.linspace(t0, t1, n_points)
  G = np.zeros((fine.size, 3))
  for block in seq.blocks:
    _rf, g, _adc = block(fine)
    for axis in range(3):
      G[:, axis] += np.broadcast_to(np.asarray(g[axis]), fine.shape)
  bc = np.array([_concomitant_mT(nodes, G[i], B0_MT) for i in range(fine.size)])
  return -GAMMA * np.trapezoid(bc, fine, axis=0)


def test_the_segment_rules_are_exact():
  """The two closed forms the whole helper rests on.

      integral A^2 dt = h (A0^2 + A0 A1 + A1^2)/3
      integral A B dt = h (2 A0 B0 + A0 B1 + A1 B0 + 2 A1 B1)/6

  Both integrands are quadratic on a piecewise-linear segment, so a quadrature
  rule exact for quadratics reproduces them; a trapezoid does not.
  """
  rng = np.random.default_rng(11)
  s = np.linspace(0.0, 1.0, 200001)
  worst_square = worst_cross = 0.0
  for _ in range(200):
    h = float(rng.uniform(1e-4, 2.0))
    A0, A1, B0, B1 = rng.uniform(-30.0, 30.0, 4)
    A, B = A0 + (A1 - A0) * s, B0 + (B1 - B0) * s
    worst_square = max(worst_square, abs(
      h * (A0 * A0 + A0 * A1 + A1 * A1) / 3.0
      / (np.trapezoid(A * A, s) * h) - 1.0))
    worst_cross = max(worst_cross, abs(
      h * (2 * A0 * B0 + A0 * B1 + A1 * B0 + 2 * A1 * B1) / 6.0
      / (np.trapezoid(A * B, s) * h) - 1.0))
  assert worst_square < 1e-9, f'the square rule is off by {worst_square:.2e}'
  assert worst_cross < 1e-7, f'the cross rule is off by {worst_cross:.2e}'


def test_a_ramp_is_charged_one_third_and_not_one_half():
  """The trap this helper exists to avoid.

  Over a ramp `0 -> A` of length h the exact second moment is `A^2 h/3`; a
  trapezoid rule over the stored corners charges `A^2 h/2`, a one-signed 50%
  over-count of every ramp. That is the same error the solver had to
  sub-sample around (Bloch.CONCOMITANT_DT_GR_MS) -- here it is simply absent.
  """
  A, rise = 20.0, 0.30
  seq = _sequence([_gradient([0.0, rise], [0.0, A], 2)], rise)
  moments = maxwell_moments(seq, 0.0, np.array([0.0, rise]))

  exact = A * A * rise / 3.0
  corner_trapezoid = A * A * rise / 2.0
  assert abs(moments[1, 1] / exact - 1.0) < 1e-12, (
    f'the Gz^2 moment of a ramp is {moments[1, 1]:.6f}, exact {exact:.6f}; '
    f'the corner trapezoid would give {corner_trapezoid:.6f}')
  assert abs(corner_trapezoid / exact - 1.5) < 1e-12, (
    'this case no longer separates the two rules')


@pytest.mark.parametrize('amps', [(14.0, -9.0, 20.0), (0.0, 0.0, 25.0),
                                  (18.0, 11.0, 0.0)],
                         ids=['oblique', 'Gz_only', 'in_plane'])
def test_the_moments_reproduce_the_concomitant_field(amps):
  """The factorisation, end to end, against the library's own `_concomitant_mT`.

  The reference is refined rather than compared at a fixed tolerance: it must
  CONVERGE ONTO the helper, which is what shows the helper is the exact answer
  and the gap is the reference's own quadrature error. Measured on the oblique
  case: 2.4e-9 at 20k points, 4.9e-11 at 200k, 5.4e-14 at 2M.
  """
  seq = _trapezoid(amps, 0.30, 2.0, 0.30)
  end = 2.60
  got = _phase_from_helpers(seq, 0.0, np.array([0.0, end]))[1]

  errors = [float(np.abs(got / _phase_by_dense_quadrature(seq, 0.0, end, n) - 1.0).max())
            for n in (20001, 200001)]
  assert errors[1] < errors[0] / 10.0, (
    f'refining the reference did not converge onto the helper: {errors}')
  assert errors[1] < 1e-9, f'worst relative deviation {errors[1]:.2e}'


def test_two_gradients_on_one_axis_are_summed_before_squaring():
  """`integral (Gx + Gx')^2 != integral Gx^2 + integral Gx'^2`.

  The linear moment helper may accumulate per gradient OBJECT because the
  integral is linear. A squared moment may not, and the cross terms cannot be
  decomposed per axis at all. Two overlapping gradients on one axis is the
  smallest case that tells the two apart.
  """
  ts = [0.0, 1.0]
  split = _sequence([_gradient(ts, [10.0, 10.0], 0),
                     _gradient(ts, [6.0, 6.0], 0)], 1.0)
  merged = _sequence([_gradient(ts, [16.0, 16.0], 0)], 1.0)

  a_split = maxwell_moments(split, 0.0, np.array([0.0, 1.0]))[1, 0]
  a_merged = maxwell_moments(merged, 0.0, np.array([0.0, 1.0]))[1, 0]
  assert abs(a_split / a_merged - 1.0) < 1e-12, (
    f'two gradients on one axis gave {a_split:.4f} against {a_merged:.4f} for '
    f'the same total field -- they are being squared before they are summed')
  # ... and the wrong answer is far away, so this can see the mistake.
  per_object = 10.0 ** 2 + 6.0 ** 2
  assert abs(per_object / a_merged - 1.0) > 0.2, 'the case is degenerate'


def test_a_shaped_gradient_is_integrated_on_its_own_corners():
  """A non-trapezoid waveform, where a rule that assumed four corners fails."""
  ts = np.linspace(0.0, 2.0, 21)
  amps = 18.0 * np.sin(np.pi * ts / 2.0)
  seq = _sequence([_gradient(ts, amps, 0), _gradient(ts, 0.6 * amps, 2)], 2.0)
  got = _phase_from_helpers(seq, 0.0, np.array([0.0, 2.0]))[1]
  errors = [float(np.abs(got / _phase_by_dense_quadrature(seq, 0.0, 2.0, n) - 1.0).max())
            for n in (20001, 200001)]
  assert errors[1] < errors[0] / 10.0 and errors[1] < 1e-9, (
    f'shaped-gradient moments do not converge onto the reference: {errors}')


def test_the_moments_start_at_zero_and_refuse_a_sample_before_the_origin():
  """The origin is the magnetization snapshot, and there is no anchor
  correction to make: everything before it is already carried on the
  magnetization by the solver, so `moments[0]` is structurally zero."""
  seq = _trapezoid((14.0, -9.0, 20.0), 0.30, 2.0, 0.30)
  moments = maxwell_moments(seq, 0.5, np.array([0.5, 1.0, 2.0]))
  assert np.abs(moments[0]).max() == 0.0
  assert np.all(np.diff(moments[:, 0]) >= 0.0), (
    'the z^2 moment is an integral of a square and cannot decrease')
  with pytest.raises(ValueError, match='precedes the origin'):
    maxwell_moments(seq, 1.0, np.array([0.5, 1.5]))


def test_the_phase_can_only_ever_retard():
  """`Bc` is `(Bx^2 + By^2)/(2 B0)`, a sum of squares, so the phase it adds is
  never positive -- for any gradient and any position. This pins the sign the
  coefficients carry, which is the one thing in this chain that no amount of
  internal consistency would catch.
  """
  rng = np.random.default_rng(5)
  for _ in range(8):
    amps = tuple(rng.uniform(-25.0, 25.0, 3))
    seq = _trapezoid(amps, 0.2, 1.0, 0.2)
    phase = _phase_from_helpers(seq, 0.0, np.array([0.0, 1.4]))[1]
    assert phase.max() <= 1e-12, (
      f'gradients {np.round(amps, 1)} ADVANCED the phase by {phase.max():.3e}')
    assert np.abs(phase).max() > 1e-6, 'this case produced no phase at all'


def test_the_kspace_helper_agrees_where_both_are_exact():
  """A constant readout gradient makes k linear in t, so the finite-difference
  recovery `G = (dk/dt)/gammabar` is exact and the two helpers must agree."""
  amp, dur = 20.0, 2.0
  seq = _sequence([_gradient([0.0, dur], [amp, amp], 0),
                   _gradient([0.0, dur], [0.5 * amp, 0.5 * amp], 2)], dur)
  times = np.linspace(0.0, dur, 33)

  gammabar = SCANNER.gammabar.m_as('1/ms/mT')
  kx = amp * gammabar * times
  kz = 0.5 * amp * gammabar * times
  from_k = maxwell_moments_from_kspace(kx, np.zeros_like(times), kz, times,
                                       SCANNER)
  from_gradients = maxwell_moments(seq, 0.0, times)
  # Compared on a shared absolute scale, not per column: this case drives no
  # Gy, so the integral(Gy Gz) column is identically zero in both and a
  # relative test there is 0/0.
  scale = float(np.abs(from_gradients).max())
  worst = float(np.abs(from_k - from_gradients).max() / scale)
  assert worst < 1e-9, (
    f'the k-derived moments differ from the gradient-derived ones by '
    f'{worst:.2e} of the largest moment')
  # The three columns this case does drive must be non-trivial, or the
  # agreement above is agreement about zero.
  assert np.abs(from_gradients[-1, [0, 1, 2]]).min() > 0.1 * scale


# ---------------------------------------------------------------------------
# The native-trajectory path: `Trajectory.maxwell_coefficients`
# ---------------------------------------------------------------------------

def _oblique(deg=25.0):
  th = np.deg2rad(deg)
  return np.array([[np.cos(th), 0.0, np.sin(th)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(th), 0.0, np.cos(th)]])


def _cartesian(MPS_ori=None, t_start_ms=1.0, res=(16, 4, 1)):
  from feelmri import CartesianStack
  return CartesianStack(FOV=Q_(np.array([0.30, 0.30, 0.008]), 'm'),
                        res=np.asarray(res), oversampling=1, lines_per_shot=1,
                        scanner=SCANNER, t_start=Q_(t_start_ms, 'ms'),
                        MPS_ori=MPS_ori)


def _dense_phase_from_trajectory(traj, t1, nodes_img, n=40001):
  """-gamma integral(Bc) dt from the library's own `_concomitant_mT`, in the
  PHYSICAL frame: logical gradients rotated by MPS_ori, imaging-frame nodes
  rotated the same way. That is the statement the two rotations inside
  `maxwell_coefficients` have to add up to."""
  R = np.asarray(traj.MPS_ori, dtype=float)
  fine = np.linspace(0.0, t1, n)
  G = np.zeros((n, 3))
  for g in traj.gradients:
    G[:, g.axis] += np.interp(fine,
                              np.asarray(g.timings.m_as('ms'), dtype=float),
                              np.asarray(g.amplitudes.m_as('mT/m'), dtype=float),
                              left=0.0, right=0.0)
  G_phys = G @ R.T
  nodes_phys = np.asarray(nodes_img, dtype=float) @ R.T
  bc = np.array([_concomitant_mT(nodes_phys, G_phys[i], B0_MT) for i in range(n)])
  return -GAMMA * np.trapezoid(bc, fine, axis=0)


@pytest.mark.parametrize('oblique', [False, True], ids=['axial', 'oblique'])
def test_a_native_readout_reproduces_the_concomitant_field(oblique):
  """End to end on a real `CartesianStack`: the retained waveforms, the
  prephaser, and BOTH rotations, against the library's own field expression
  integrated densely in the physical frame.

  The oblique case is the one that matters. `Bc` is `(Bx^2 + By^2)/(2 B0)`, so
  it singles out z and there is no rotation under which an oblique acquisition
  reduces to an axial one -- the two parametrisations below genuinely differ
  (asserted), and both have to be right separately.
  """
  R = _oblique() if oblique else np.eye(3)
  traj = _cartesian(MPS_ori=R)
  coef = traj.maxwell_coefficients(SCANNER)
  times = np.asarray(traj.times.m_as('ms'), dtype=float).reshape(-1)
  nodes = np.array([[0.12, -0.06, 0.004],
                    [-0.09, 0.10, -0.003],
                    [0.05, 0.05, 0.002]])

  x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
  got = (coef[:, 0:1] * x ** 2 + coef[:, 1:2] * y ** 2 + coef[:, 2:3] * z ** 2
         + coef[:, 3:4] * x * y + coef[:, 4:5] * x * z + coef[:, 5:6] * y * z)

  worst = 0.0
  for idx in (0, times.size // 2, times.size - 1):
    ref = _dense_phase_from_trajectory(traj, times[idx], nodes)
    assert np.abs(ref).max() > 1e-6, 'this sample carries no phase at all'
    worst = max(worst, float(np.abs(got[idx] - ref).max()
                             / np.abs(ref).max()))
  # MPS_ori is stored float32, which sets the floor at ~7e-8.
  assert worst < 1e-6, f'the trajectory coefficients are off by {worst:.2e}'


def test_an_oblique_readout_is_not_an_axial_one_in_disguise():
  """Guards the test above against passing for the wrong reason: if the two
  parametrisations agreed, dropping either rotation would still pass."""
  nodes = np.array([[0.12, -0.06, 0.004]])
  x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]

  def phase(traj):
    c = traj.maxwell_coefficients(SCANNER)
    return (c[:, 0] * x ** 2 + c[:, 1] * y ** 2 + c[:, 2] * z ** 2
            + c[:, 3] * x * y + c[:, 4] * x * z + c[:, 5] * y * z)

  axial, oblique = phase(_cartesian()), phase(_cartesian(MPS_ori=_oblique()))
  rel = float(np.abs(axial - oblique).max() / np.abs(axial).max())
  assert rel > 0.05, (
    f'the oblique and axial readouts differ by only {rel:.2e}; this geometry '
    f'no longer separates the rotated case from the unrotated one')


def test_the_retained_waveforms_carry_the_prephaser():
  """The reason `CartesianStack` keeps its gradients at all.

  `maxwell_moments_from_kspace` recovers `G` from the SAMPLED k, so it starts
  at the first ADC sample and the prephaser is simply not in its window. The
  moment is already non-zero there, and on the geometry
  `examples/phase_contrast.py` uses the prephasers carry 52% of the whole
  window.
  """
  traj = _cartesian()
  # ONE readout line. Flattening the whole stack interleaves lines whose
  # clocks restart, and `np.gradient` over that produces division by zero
  # rather than a gradient -- against which any assertion passes.
  times = np.asarray(traj.times.m_as('ms'), dtype=float)[:, 0, 0]
  assert np.all(np.diff(times) > 0.0), 'the line is not monotonic in time'

  moments = maxwell_moments(traj.gradients, 0.0, times)
  a_first, a_last = moments[0, 0], moments[-1, 0]
  assert a_first > 0.0, 'nothing was integrated before the first ADC sample'
  assert a_first / a_last > 0.01, (
    f'the pre-sample share is {a_first / a_last:.4f}; this case no longer '
    f'demonstrates what the k-derived helper misses')

  from_k = maxwell_moments_from_kspace(
    np.asarray(traj.points[0], dtype=float)[:, 0, 0],
    np.asarray(traj.points[1], dtype=float)[:, 0, 0],
    np.asarray(traj.points[2], dtype=float)[:, 0, 0], times, SCANNER)
  assert np.all(np.isfinite(from_k)), 'the k-derived moments are not finite'
  # Both start their clock at their own origin, so both read zero at the first
  # sample. What the waveform integral has that this one cannot is everything
  # BEFORE that sample, and it is the whole of `a_first`.
  assert from_k[0, 0] == 0.0
  assert abs(from_k[-1, 0] - (a_last - a_first)) / a_last < 0.05, (
    f'the k-derived moment over the sampled window is {from_k[-1, 0]:.3f} '
    f'against {a_last - a_first:.3f} from the waveform; the two should differ '
    f'only by the readout ramp the sampling does not resolve')


def test_a_trajectory_without_retained_gradients_refuses():
  """RadialStack and SpiralStack build their k analytically and keep no
  waveform, so the exact moments are out of reach. Refused, not approximated
  behind the caller's back."""
  from feelmri import RadialStack
  traj = RadialStack(FOV=Q_(np.array([0.30, 0.30, 0.008]), 'm'),
                     res=np.array([16, 4, 1]), oversampling=1,
                     lines_per_shot=1, scanner=SCANNER)
  with pytest.raises(NotImplementedError, match='does not retain'):
    traj.maxwell_coefficients(SCANNER)


def test_gradient_activity_before_the_origin_is_warned_about():
  """The prephasers run over `[t_start - dur, t_start]`, so a trajectory built
  without `t_start` puts them at NEGATIVE times, outside any forward
  integration from 0 -- and the result is then quietly missing the larger part
  of the moment. The trajectory does not know where the excitation was, so it
  warns rather than guessing an origin."""
  traj = _cartesian(t_start_ms=0.0)
  earliest = min(float(np.asarray(g.timings.m_as('ms')).min())
                 for g in traj.gradients)
  assert earliest < 0.0, 'this case no longer places activity before 0 ms'
  with pytest.warns(UserWarning, match='before the integration origin'):
    traj.maxwell_coefficients(SCANNER)


def test_two_overlapping_gradient_SETS_cannot_be_added_after_squaring():
  """`Bc` is quadratic in G, so `Bc(G_a + G_b) != Bc(G_a) + Bc(G_b)` wherever
  the two overlap -- the cross term belongs to neither set.

  This is the control for the `carried` argument below: it shows the split is
  wrong only in the overlap, and exactly right outside it. Disjoint sets add to
  0.0e+00; overlapping ones do not.
  """
  ts = np.array([0.0, 0.2, 1.2, 1.4])
  amp = np.array([0.0, 18.0, 18.0, 0.0])
  # Same support -> the two sets overlap completely.
  first = _gradient(ts, amp, axis=0)
  second = _gradient(ts, -0.6 * amp, axis=2)
  t1 = float(ts[-1])
  joint = maxwell_moments([first, second], 0.0, np.array([t1]))
  split = (maxwell_moments([first], 0.0, np.array([t1]))
           + maxwell_moments([second], 0.0, np.array([t1])))
  assert np.abs(joint - split).max() > 0.1 * np.abs(joint).max()

  # Shifted clear of each other, the same two sets DO add.
  shifted = _gradient(ts + t1 + 1.0, -0.6 * amp, axis=2)
  t2 = float(ts[-1] + t1 + 1.0)
  joint2 = maxwell_moments([first, shifted], 0.0, np.array([t2]))
  split2 = (maxwell_moments([first], 0.0, np.array([t2]))
            + maxwell_moments([shifted], 0.0, np.array([t2])))
  assert np.abs(joint2 - split2).max() == 0.0


def test_carried_gradients_are_integrated_as_one_field_with_the_readout():
  """The solver integrates its block's gradients and the assembler the
  trajectory's, so where the two OVERLAP in time their cross term is computed
  by neither. `carried` closes that: the whole field is integrated once and
  what the solver already applied is subtracted back off.

  Checked against the definition rather than against the implementation --
  `moments(union) - moments(carried up to the snapshot)` -- and shown to differ
  from the naive sum, or the argument would be doing nothing.
  """
  traj = _cartesian(t_start_ms=2.0)
  scanner = SCANNER
  # A gradient that is still playing when the prephasers start.
  carried = [_gradient([0.0, 0.4, 1.6, 1.9], [0.0, 22.0, 22.0, 0.0], axis=2)]
  t_snap = float(traj.t_start.m_as('ms'))
  times = np.asarray(traj.times.m_as('ms'), dtype=float).reshape(-1)
  R = np.asarray(traj.MPS_ori, dtype=float)

  got = traj.maxwell_coefficients(scanner, carried=carried)
  expected = maxwell_phase_coefficients(
      maxwell_moments(list(traj.gradients) + carried, 0.0, times, rotation=R)
      - maxwell_moments(carried, 0.0, np.array([t_snap]), rotation=R)[0],
      scanner, rotation=R)
  assert np.abs(got - expected).max() < 1e-9 * np.abs(expected).max()

  plain = traj.maxwell_coefficients(scanner)
  assert np.abs(got - plain).max() > 1e-3 * np.abs(plain).max(), (
    'this carried set does not overlap the readout, so the argument is '
    'untested here')

  # The correction is CONSTANT across the window, because the overlap ends at
  # the snapshot and the carried gradients contribute nothing after it.
  delta = got - plain
  assert np.abs(delta - delta[0]).max() < 1e-9 * np.abs(delta).max()

  # A carried set that finishes before the trajectory starts changes nothing.
  early = [_gradient([-4.0, -3.6, -3.2], [0.0, 22.0, 0.0], axis=2)]
  assert np.abs(traj.maxwell_coefficients(scanner, carried=early)
                - plain).max() < 1e-12 * np.abs(plain).max()


def test_a_stack_retains_its_partition_encode_gradient():
  """`CartesianStack` modelled kz as a pure k-space offset with no waveform, so
  the concomitant moments saw a stack as though nothing were played along z --
  and `Bc` weights `Gz` most heavily of the three, through `(Gz^2/4)(x^2+y^2)`
  and both cross terms.

  Timing is unchanged by construction: the partition encode is START-aligned
  with the other two prephasers, so `enc_time`, the echo time and every sample
  time are the same as before.
  """
  flat = _cartesian(t_start_ms=4.0, res=(16, 4, 1))
  stack = _cartesian(t_start_ms=4.0, res=(16, 4, 5))
  assert not any(g.axis == 2 for g in flat.gradients), (
    'a single partition needs no z encode')
  assert sum(g.axis == 2 for g in stack.gradients) == 1

  # It carries real second moment: the z column of the moments is non-zero.
  times = np.asarray(stack.times.m_as('ms'), dtype=float).reshape(-1)
  with_z = maxwell_moments(stack.gradients, 0.0, times)
  without_z = maxwell_moments([g for g in stack.gradients if g.axis != 2],
                              0.0, times)
  assert without_z[:, 1].max() == 0.0, 'the fixture already had a z gradient'
  assert with_z[:, 1].max() > 0.0, 'the partition encode contributes nothing'

  assert float(flat.echo_time.m_as('ms')) == pytest.approx(
      float(stack.echo_time.m_as('ms'))), 'the timing moved'
