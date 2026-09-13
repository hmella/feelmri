"""Trajectory behaviour of feelmri.Motion: the fold, the wrap and the sum.

`PODSum` and `RespiratoryMotion` had no tests at all, and `POD`'s periodic
fold had none either -- `is_periodic=True` is also the setting that disables
the out-of-range guard, so a wrong period could not announce itself. All
fixtures are synthetic and tiny: no mesh, no MPI, no file I/O.
"""
import numpy as np
import pytest

from feelmri.Motion import POD, PODSum, PODVelocity, RespiratoryMotion

N_FRAMES = 30
DT_MS = 40.0
TIMES = (np.arange(N_FRAMES) * DT_MS).astype(np.float32)


def _cine_snapshots(n_nodes=12):
    """One full turn over N frames, so frame N IS frame 0.

    A pure circular motion is the sharpest probe available: it is exactly
    periodic with period `N*dt` and has no other period, so any error in the
    cycle length shows up directly as a discontinuity at the wrap.
    """
    ang = 2.0 * np.pi * np.arange(N_FRAMES) / N_FRAMES
    amp = np.linspace(1.0, 2.0, n_nodes)
    data = np.zeros((n_nodes, 3, N_FRAMES), dtype=np.float32)
    data[:, 0, :] = np.cos(ang)[None, :] * amp[:, None]
    data[:, 1, :] = np.sin(ang)[None, :] * amp[:, None]
    return data


def _cine_pod(**kwargs):
    kwargs.setdefault('n_modes', 2)
    kwargs.setdefault('is_periodic', True)
    return POD(TIMES, _cine_snapshots(), **kwargs)


def test_the_period_is_n_times_dt_not_the_last_timestamp():
    """A cine of N frames covers one cycle, so its period is `N*dt`.

    Taking `times[-1] = (N-1)*dt` replays the cycle 1/N short -- 3.3% here --
    which is a JUMP at every wrap and a cumulative drift: ten cycles of this
    30-frame record land ten full frames away from where they belong.
    """
    pod = _cine_pod()
    assert pod._period == pytest.approx(N_FRAMES * DT_MS)
    # The old spelling, as a control: it is a different number, and by exactly
    # one sampling interval.
    assert float(TIMES[-1]) == pytest.approx((N_FRAMES - 1) * DT_MS)
    assert pod.period - float(TIMES[-1]) == pytest.approx(DT_MS)


def test_the_period_is_exposed_for_cycle_synchronisation():
    """`period` is the cycle the examples pad their sequences by. `PODSum`
    has one only when both children agree, and a non-periodic trajectory has
    none."""
    pod = _cine_pod()
    signal = np.cos(2.0 * np.pi * np.arange(N_FRAMES) / N_FRAMES)
    resp = RespiratoryMotion(TIMES, signal.astype(np.float32), is_periodic=True)
    assert pod.period == pytest.approx(N_FRAMES * DT_MS)
    assert resp.period == pytest.approx(N_FRAMES * DT_MS)
    assert PODSum(pod, resp).period == pytest.approx(N_FRAMES * DT_MS)
    assert POD(TIMES, _cine_snapshots(), n_modes=2,
               is_periodic=False).period is None
    other = RespiratoryMotion((TIMES * 1.7).astype(np.float32),
                              signal.astype(np.float32), is_periodic=True)
    assert PODSum(pod, other).period is None


def test_the_weights_are_continuous_across_the_wrap():
    """Frame N and frame 0 are the same instant, so the fold must not jump.

    Measured with the period at `times[-1]`, this probe reads 0.208 -- 21% of
    the weight range -- because folding arrives at frame 0 while the motion is
    still one frame short of completing its turn.
    """
    pod = _cine_pod()
    eps = 1e-3
    T = N_FRAMES * DT_MS
    before = pod.get_weights(np.array([T - eps], dtype=np.float32))
    after = pod.get_weights(np.array([T + eps], dtype=np.float32))
    span = float(np.abs(pod.weights).max())
    jump = float(np.abs(after - before).max()) / span
    assert jump < 1e-3, f'the fold jumps by {jump:.3f} of the weight range'


@pytest.mark.parametrize('t_ms', [77.0, 613.0, 1213.0, -95.0])
def test_folding_reproduces_the_same_cycle_phase(t_ms):
    """`t` and `t + k*period` are the same instant, for any integer k."""
    pod = _cine_pod()
    T = N_FRAMES * DT_MS
    here = pod.get_weights(np.array([t_ms], dtype=np.float32))
    there = pod.get_weights(np.array([t_ms + 3 * T], dtype=np.float32))
    assert np.abs(here - there).max() < 1e-5


@pytest.mark.parametrize('method', ['Pchip', 'CubicSpline', 'AkimaSpline'])
def test_respiratory_motion_refuses_a_time_it_has_no_data_for(method):
    """It is not a `POD` subclass, so the guard written for `POD` never
    reached it -- and the three interpolators fail three different ways.

    Measured on the unguarded code at t = 2000 ms against a record ending at
    1160 ms, on a signal bounded by +-1: Pchip extrapolates to **-52.6** and
    CubicSpline to **+82.0** -- silently, and with opposite signs -- while
    Akima answers NaN, which multiplies into every node and every k-space
    sample rather than into that one time point.
    """
    signal = np.cos(2.0 * np.pi * np.arange(N_FRAMES) / N_FRAMES)
    resp = RespiratoryMotion(TIMES, signal.astype(np.float32),
                             is_periodic=False, interpolation_method=method)
    inside = resp.get_weights(np.array([500.0], dtype=np.float32))
    assert np.isfinite(inside).all()
    with pytest.raises(ValueError, match='no motion data'):
        resp.get_weights(np.array([2000.0], dtype=np.float32))


def test_respiratory_motion_is_periodic_over_n_times_dt():
    """The same period rule as `POD`, and the same continuity at the wrap."""
    signal = np.cos(2.0 * np.pi * np.arange(N_FRAMES) / N_FRAMES)
    resp = RespiratoryMotion(TIMES, signal.astype(np.float32),
                             is_periodic=True)
    assert resp._period == pytest.approx(N_FRAMES * DT_MS)
    eps, T = 1e-3, N_FRAMES * DT_MS
    before = resp.get_weights(np.array([T - eps], dtype=np.float32))
    after = resp.get_weights(np.array([T + eps], dtype=np.float32))
    assert float(np.abs(after - before).max()) < 1e-3


def test_podsum_shifts_its_children_without_destroying_their_own():
    """`PODSum.timeshift` is an OFFSET on top of each child's shift.

    Cardiac and respiratory phases are independent, so the two children
    legitimately carry different shifts -- and `PODSum.timeshift` starts at
    0.0 knowing nothing about them. Assigning the argument to both children
    therefore wiped that out on the FIRST block: `BlochSolver.solve` reads
    0.0, writes `0.0 + block_start` into both, then restores 0.0.
    """
    pod = _cine_pod()
    resp = RespiratoryMotion(TIMES, np.ones(N_FRAMES, dtype=np.float32),
                             is_periodic=True)
    pod.update_timeshift(5.0)
    resp.update_timeshift(-3.0)
    total = PODSum(pod, resp)
    assert total.timeshift == 0.0

    total.update_timeshift(100.0)
    assert (pod.timeshift, resp.timeshift) == (105.0, 97.0)
    # The compose/restore both callers use has to leave the children alone.
    total.update_timeshift(0.0)
    assert (pod.timeshift, resp.timeshift) == (5.0, -3.0)


def test_podsum_modes_times_weights_is_the_sum_of_its_parts():
    """The kernel only ever sees `modes @ weights`, so that product -- not the
    concatenation -- is what has to equal the sum of the two displacements.

    Checked on children whose mode counts DIFFER (2 and 1), so a mis-sliced
    concatenation cannot line up by accident.
    """
    pod = _cine_pod()
    signal = np.sin(2.0 * np.pi * np.arange(N_FRAMES) / N_FRAMES)
    resp = RespiratoryMotion(TIMES, signal.astype(np.float32),
                             is_periodic=True,
                             direction=np.array([0.0, 0.0, 1.0],
                                                dtype=np.float32))
    total = PODSum(pod, resp)
    n_nodes = pod._modes.shape[0]
    t = np.array([37.0, 413.0, 991.0], dtype=np.float32)

    both = np.einsum('ncm,tm->tnc', total.get_modes(n_nodes),
                     total.get_weights(t))
    apart = (np.einsum('ncm,tm->tnc', pod.get_modes(n_nodes),
                       pod.get_weights(t))
             + np.einsum('ncm,tm->tnc', resp.get_modes(n_nodes),
                         resp.get_weights(t)))
    assert np.abs(both - apart).max() < 1e-5
    # Not vacuous: both children have to contribute something.
    assert np.abs(pod.get_weights(t)).max() > 1e-3
    assert np.abs(resp.get_weights(t)).max() > 1e-3


def test_pod_velocity_separates_the_cardiac_phase_from_the_taylor_time():
    """`PODVelocity` scales by `t_ro`, the time since the EXCITATION, while the
    fold uses absolute time. `pod(t)` conflated them and kept its own TODO
    after `get_weights` was corrected.

    The check: two calls one full period apart are at the same cardiac phase,
    so with the same `t_ro` they must give the same displacement -- which the
    `t_ro = t` spelling cannot do, since its displacement grows with absolute
    time (here by a factor of 4.1).
    """
    pod = PODVelocity(TIMES, _cine_snapshots(), n_modes=2, is_periodic=True)
    T = N_FRAMES * DT_MS
    a = pod._evaluate_trajectory(387.0, t_ro=2.5).copy()
    b = pod._evaluate_trajectory(387.0 + T, t_ro=2.5).copy()
    assert np.abs(a).max() > 1e-4, 'no displacement at all; nothing is tested'
    assert np.abs(a - b).max() < 1e-5

    # The default is still `t_ro = t`, which is what the plotting callers rely
    # on; it is the CONFLATION that is now the caller's choice, not a silent
    # property of the class.
    assert np.abs(pod._evaluate_trajectory(387.0 + T)
                  - pod._evaluate_trajectory(387.0)).max() > 1e-3
