"""Tests for the POD energy-retention estimate in feelmri.Motion.

All fixtures are synthetic and tiny: no mesh, no MPI, no file I/O.
"""
import numpy as np
import pytest

from feelmri.Motion import (POD, modes_for_energy, plot_pod_energy,
                            pod_energy_spectrum, pod_frame_errors)


def _random_snapshots(n_nodes=20, n_comp=3, n_times=12, seed=0):
    """Full-rank random displacement snapshots of shape (P, C, T)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_nodes, n_comp, n_times))


def _rank_k_snapshots(n_nodes=50, n_comp=3, n_times=20, k=3, seed=1):
    """Snapshots built as exactly k separable space-time products."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_times)
    # k temporal profiles, linearly independent by construction
    temporal = np.stack([np.sin(2.0 * np.pi * (j + 1) * t) for j in range(k)])
    spatial = rng.standard_normal((n_nodes * n_comp, k))
    flat = spatial @ temporal
    return flat.reshape(n_nodes, n_comp, n_times)


def test_eigenvalues_match_squared_singular_values():
    """The spectrum is sigma^2 of the snapshot matrix, to round-off."""
    data = _random_snapshots()
    eigen_values, _ = pod_energy_spectrum(data)

    flat = data.reshape(-1, data.shape[-1])
    sigma = np.linalg.svd(flat, compute_uv=False)

    assert eigen_values.shape == (data.shape[-1],)
    np.testing.assert_allclose(eigen_values, sigma ** 2, rtol=1e-10, atol=1e-12)


def test_rank_deficient_data_concentrates_all_energy():
    """A rank-3 field puts everything in the first three modes."""
    data = _rank_k_snapshots(k=3)
    eigen_values, cumulative = pod_energy_spectrum(data)

    assert cumulative[2] >= 1.0 - 1e-12
    # The trailing spectrum is numerically zero relative to the leader.
    assert np.all(eigen_values[3:] / eigen_values[0] < 1e-12)


def test_cumulative_curve_shape():
    """Non-decreasing, bounded by (0, 1], and exactly 1 at the end."""
    data = _random_snapshots()
    eigen_values, cumulative = pod_energy_spectrum(data)

    assert cumulative.shape == eigen_values.shape
    assert np.all(np.diff(cumulative) >= -1e-15)
    assert cumulative[0] > 0.0
    assert np.all(cumulative <= 1.0 + 1e-12)
    assert cumulative[-1] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize('n_modes', [1, 4, 8])
def test_object_matches_standalone(n_modes):
    """POD.energy_ratio agrees with the standalone spectrum."""
    data = _random_snapshots(n_times=12)
    times = np.linspace(0.0, 1.0, data.shape[-1])
    _, cumulative = pod_energy_spectrum(data)

    pod = POD(times=times, data=data.copy(), n_modes=n_modes)

    assert pod.n_modes == n_modes
    assert pod.n_modes_max == data.shape[-1]
    assert pod.energy_ratio() == pytest.approx(cumulative[n_modes - 1], rel=1e-12)
    assert pod.energy_ratio(2) == pytest.approx(cumulative[1], rel=1e-12)
    np.testing.assert_allclose(pod.cumulative_energy(), cumulative, rtol=1e-12)


def test_cumulative_energy_returns_a_copy():
    """Callers cannot corrupt the stored spectrum through the getter."""
    data = _random_snapshots()
    pod = POD(times=np.linspace(0.0, 1.0, data.shape[-1]), data=data, n_modes=3)

    curve = pod.cumulative_energy()
    curve[:] = -1.0

    assert pod.energy_ratio() > 0.0


@pytest.mark.parametrize('target', [0.5, 0.9, 0.99])
def test_modes_for_energy_round_trip(target):
    """n is the smallest count reaching the target, and no smaller."""
    data = _random_snapshots(n_times=16)
    _, cumulative = pod_energy_spectrum(data)

    n = modes_for_energy(data, target)

    assert 1 <= n <= data.shape[-1]
    assert cumulative[n - 1] >= target
    if n > 1:
        assert cumulative[n - 2] < target


def test_modes_for_energy_full_target_is_bounded():
    """target=1.0 cannot run off the end of the spectrum."""
    data = _random_snapshots(n_times=9)
    assert modes_for_energy(data, 1.0) == 9


def test_modes_for_energy_rejects_out_of_range_target():
    data = _random_snapshots()
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match='target'):
            modes_for_energy(data, bad)


def test_energy_ratio_rejects_out_of_range_mode_count():
    data = _random_snapshots(n_times=10)
    pod = POD(times=np.linspace(0.0, 1.0, 10), data=data, n_modes=3)

    for bad in (0, -1, 11):
        with pytest.raises(ValueError, match='n_modes'):
            pod.energy_ratio(bad)


def test_rank_deficient_request_is_clamped_and_finite(capsys):
    """Regression: n_modes above the rank used to yield silent NaN modes.

    Motion.py scales the eigenvectors by 1/sqrt(eigenvalue); past the
    numerical rank that divides by ~0 and poisons every downstream
    simulation without a warning.
    """
    data = _rank_k_snapshots(k=3, n_times=20)
    times = np.linspace(0.0, 1.0, data.shape[-1])

    pod = POD(times=times, data=data, n_modes=10)

    assert pod.n_modes == 3
    assert pod.modes.shape[-1] == 3
    assert np.all(np.isfinite(pod.modes))
    assert np.all(np.isfinite(pod.weights))
    assert 'Clamping to 3' in capsys.readouterr().out

    # The truncated object stays self-consistent for the assembler.
    assert pod.get_modes(data.shape[0]).shape[-1] == 3
    assert pod.get_weights(times).shape == (times.size, 3)


def test_request_above_time_step_count_is_clamped(capsys):
    """n_modes > T is reported as a time-step limit, not a rank deficiency.

    This used to reach `phi.reshape((P, -1, n_modes))` with fewer columns
    than n_modes and fail there.
    """
    data = _random_snapshots(n_nodes=30, n_times=8)
    pod = POD(times=np.linspace(0.0, 1.0, 8), data=data, n_modes=30)

    assert pod.n_modes == 8
    assert pod.n_modes_max == 8
    assert np.all(np.isfinite(pod.modes))
    assert 'only has 8 time steps' in capsys.readouterr().out


def test_remove_mean_does_not_mutate_caller_data():
    data = _random_snapshots()
    before = data.copy()

    pod_energy_spectrum(data, remove_mean=True)

    np.testing.assert_array_equal(data, before)


def test_zero_field_is_rejected():
    with pytest.raises(ValueError, match='no energy'):
        pod_energy_spectrum(np.zeros((5, 3, 4)))


def _truncated_svd_residual(data, n):
    """Explicit reconstruction, used as ground truth for the error metrics."""
    flat = data.reshape(-1, data.shape[-1]).astype(np.float64)
    u, s, vt = np.linalg.svd(flat, full_matrices=False)
    return flat, flat - (u[:, :n] * s[:n]) @ vt[:n]


@pytest.mark.parametrize('n', [1, 3, 7])
def test_reconstruction_error_is_sqrt_of_energy_shortfall(n):
    data = _random_snapshots(n_times=12)
    pod = POD(times=np.linspace(0.0, 1.0, 12), data=data.copy(), n_modes=n)

    assert pod.reconstruction_error() == pytest.approx(
        np.sqrt(1.0 - pod.energy_ratio()), rel=1e-12)


@pytest.mark.parametrize('n', [1, 3, 7])
def test_reconstruction_error_matches_explicit_svd(n):
    """The headline number is the true relative L2 error of the truncation."""
    data = _random_snapshots(n_times=12)
    pod = POD(times=np.linspace(0.0, 1.0, 12), data=data.copy(), n_modes=n)

    flat, residual = _truncated_svd_residual(data, n)
    expected = np.linalg.norm(residual) / np.linalg.norm(flat)

    assert pod.reconstruction_error() == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize('n', [1, 3, 7])
def test_frame_errors_match_explicit_svd(n):
    """Per-snapshot curve, from the spectrum alone, equals the real residual."""
    data = _random_snapshots(n_times=12)
    pod = POD(times=np.linspace(0.0, 1.0, 12), data=data.copy(), n_modes=n)

    flat, residual = _truncated_svd_residual(data, n)
    expected = np.linalg.norm(residual, axis=0) / np.linalg.norm(flat, axis=0)

    np.testing.assert_allclose(pod.frame_errors(), expected, rtol=1e-10)


def test_frame_errors_shape_and_bounds():
    data = _random_snapshots(n_times=14)
    pod = POD(times=np.linspace(0.0, 1.0, 14), data=data, n_modes=4)

    errors = pod.frame_errors()

    assert errors.shape == (14,)
    assert np.all(errors >= 0.0)
    assert np.all(errors <= 1.0 + 1e-12)


def test_error_metrics_vanish_at_full_rank():
    """Keeping every usable mode reproduces the field.

    The floor is sqrt(machine eps), not eps: the method of snapshots forms
    X^T X, which squares the condition number, and the error metric then
    takes a square root of the residual energy.
    """
    data = _rank_k_snapshots(k=3, n_times=20)
    pod = POD(times=np.linspace(0.0, 1.0, 20), data=data, n_modes=3)

    assert pod.reconstruction_error() < 1e-7
    assert np.all(pod.frame_errors() < 1e-7)


def test_frame_errors_survive_an_empty_snapshot():
    """A zero-norm frame reports 0 rather than dividing by zero."""
    data = _random_snapshots(n_times=10)
    data[..., 4] = 0.0
    pod = POD(times=np.linspace(0.0, 1.0, 10), data=data, n_modes=3)

    errors = pod.frame_errors()

    assert np.all(np.isfinite(errors))
    assert errors[4] == 0.0


def test_frame_errors_reject_out_of_range_mode_count():
    data = _random_snapshots(n_times=10)
    pod = POD(times=np.linspace(0.0, 1.0, 10), data=data, n_modes=3)

    for bad in (0, -2, 11):
        with pytest.raises(ValueError, match='n_modes'):
            pod.frame_errors(bad)

    for bad in (0, 11):
        with pytest.raises(ValueError, match='n_modes'):
            pod_frame_errors(data, bad)


@pytest.mark.parametrize('n', [1, 4])
def test_standalone_frame_errors_match_the_method(n):
    data = _random_snapshots(n_times=12)
    pod = POD(times=np.linspace(0.0, 1.0, 12), data=data.copy(), n_modes=n)

    np.testing.assert_allclose(pod_frame_errors(data, n), pod.frame_errors(),
                               rtol=1e-12)


def test_low_energy_frames_are_reconstructed_worse():
    """The point of frame_errors: a good global number hides bad frames.

    Two-phase field, one loud phase and one quiet phase with an
    independent spatial pattern. A single mode captures the loud phase and
    abandons the quiet one, while global energy still reads high.
    """
    rng = np.random.default_rng(7)
    n_nodes, n_times = 200, 12
    loud_shape = rng.standard_normal((n_nodes * 3, 1))
    quiet_shape = rng.standard_normal((n_nodes * 3, 1))

    amplitude = np.zeros((1, n_times))
    amplitude[0, :6] = 1.0          # loud phase
    quiet = np.zeros((1, n_times))
    quiet[0, 6:] = 0.02             # quiet phase, 2% amplitude

    flat = loud_shape @ amplitude + quiet_shape @ quiet
    data = flat.reshape(n_nodes, 3, n_times)

    pod = POD(times=np.linspace(0.0, 1.0, n_times), data=data, n_modes=1)

    assert pod.energy_ratio() > 0.99          # global figure looks excellent
    assert np.all(pod.frame_errors()[:6] < 1e-4)   # loud phase is captured
    assert np.all(pod.frame_errors()[6:] > 0.99)   # quiet phase is destroyed


def test_plot_pod_energy_accepts_both_sources():
    plt = pytest.importorskip('matplotlib.pyplot')
    plt.switch_backend('Agg')

    data = _random_snapshots(n_times=12)
    pod = POD(times=np.linspace(0.0, 1.0, 12), data=data.copy(), n_modes=4)

    ax_from_pod = plot_pod_energy(pod, show=False)
    ax_from_data = plot_pod_energy(data, target=0.95, show=False)

    assert ax_from_pod is not None
    assert ax_from_data is not None
    plt.close('all')
