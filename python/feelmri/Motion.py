"""
Motion trajectory models for MRI simulation.

Provides :class:`RespiratoryMotion` for scalar 1-D respiratory signals
projected onto a direction vector, and :class:`POD` / :class:`PODVelocity`
for efficient Proper Orthogonal Decomposition representations of
spatially-varying motion fields. :class:`PODSum` combines two trajectory
objects additively.

Inspired by the CMRSim toolbox.
"""
# TODO: add additional information about the original authors and license
import time
from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.interpolate import (Akima1DInterpolator, CubicSpline,
                               PchipInterpolator, PPoly)

from feelmri.MPIUtilities import MPI_print, MPI_rank
from feelmri.PODHelper import tensordot_modes_weights


def _snapshot_eigenspectrum(flat_sv: np.ndarray):
    """Eigen-decompose the snapshot covariance, largest eigenvalue first.

    Parameters
    ----------
    flat_sv : np.ndarray
        Snapshot matrix of shape ``(P*C, T)``.

    Returns
    -------
    tuple
        ``(eigen_values, eigen_vectors)`` for the ``(T, T)`` covariance
        ``flat_sv.T @ flat_sv``, sorted by descending eigenvalue.
        Eigenvalues are clipped at zero: the covariance is positive
        semi-definite, so any negative value is round-off and would
        otherwise corrupt the energy sums.
    """
    covariance_matrix = np.dot(flat_sv.T, flat_sv)
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

    descending_sort_idx = np.argsort(eigen_values)[::-1]
    eigen_values = np.clip(eigen_values[descending_sort_idx], 0.0, None)
    eigen_vectors = eigen_vectors[:, descending_sort_idx]

    return eigen_values, eigen_vectors


def _cumulative_energy(eigen_values: np.ndarray) -> np.ndarray:
    """Cumulative fraction of the total energy, one entry per mode."""
    total = float(eigen_values.sum())
    if total <= 0.0:
        raise ValueError(
            "Snapshot data carries no energy (all eigenvalues are zero); "
            "the POD is undefined.")
    return np.cumsum(eigen_values) / total


def _modes_for_energy(cumulative_energy: np.ndarray, target: float) -> int:
    """Smallest mode count whose cumulative energy reaches ``target``."""
    target = float(target)
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target must lie in (0, 1], got {target}.")
    # searchsorted returns the first index with cum >= target; the clamp
    # covers target == 1.0 landing just past the end on round-off.
    idx = int(np.searchsorted(cumulative_energy, target))
    return min(idx + 1, int(cumulative_energy.size))


def _frame_errors(eigen_values: np.ndarray, eigen_vectors: np.ndarray,
                  n_modes: int) -> np.ndarray:
    """Relative truncation error of each snapshot, from the spectrum alone.

    Writing ``X = sum_i sigma_i u_i v_i^T`` with orthonormal ``u_i``, the
    residual of snapshot ``t`` after keeping ``n`` modes is

    .. math::

       \\|x_t - x_t^{(n)}\\|^2 = \\sum_{i > n} \\lambda_i\\, V_{ti}^2,
       \\qquad
       \\|x_t\\|^2 = \\sum_i \\lambda_i\\, V_{ti}^2

    so the whole per-snapshot curve follows from the ``(T,)`` eigenvalues
    and the ``(T, T)`` eigenvectors, with no reconstruction and no access
    to the snapshot data.

    Parameters
    ----------
    eigen_values : np.ndarray
        Descending spectrum of shape ``(T,)``.
    eigen_vectors : np.ndarray
        Matching eigenvectors of shape ``(T, T)``, columns ordered to
        match ``eigen_values``.
    n_modes : int
        Number of retained modes.

    Returns
    -------
    np.ndarray
        Relative error per snapshot, shape ``(T,)``, each entry in
        ``[0, 1]``. Snapshots with zero norm report 0.
    """
    # (T, T): energy that mode i contributes to snapshot t.
    per_mode = eigen_values[np.newaxis, :] * eigen_vectors ** 2

    total = per_mode.sum(axis=1)
    dropped = per_mode[:, n_modes:].sum(axis=1)

    # A snapshot with no signal must not divide. The test has to be
    # relative: an exactly-zero frame still leaves round-off in `total`
    # after the eigendecomposition, and dividing two round-off quantities
    # returns noise rather than the 0 the frame deserves.
    errors = np.zeros_like(total)
    floor = total.size * np.finfo(np.float64).eps * total.max(initial=0.0)
    valid = total > floor

    ratio = np.zeros_like(total)
    ratio[valid] = dropped[valid] / total[valid]
    errors[valid] = np.sqrt(np.clip(ratio[valid], 0.0, 1.0))

    return errors


class RespiratoryMotion:
    """Scalar respiratory motion projected onto a spatial direction.

    Interpolates a 1-D time-series of respiratory amplitudes and evaluates
    them at arbitrary time points, optionally with periodic folding and a
    time shift. The scalar output is broadcast onto ``direction`` to yield
    a 3-D displacement vector.

    Parameters
    ----------
    times : np.ndarray
        Time samples of the motion signal (s or ms, consistent units).
    data : np.ndarray
        Motion amplitude values at each time sample.
    timeshift : float, optional
        Time offset added before evaluation. Default is 0.0.
    is_periodic : bool, optional
        If True, the signal is treated as periodic with period
        ``times[-1]``. Default is False.
    remove_mean : bool, optional
        If True, remove the temporal mean before building the interpolator.
        Default is False.
    direction : np.ndarray, optional
        3-element direction vector. The scalar amplitude is multiplied by
        the unit vector in this direction. Default is ``[1, 0, 0]``.
    interpolation_method : str, optional
        Interpolation method: ``'AkimaSpline'``, ``'CubicSpline'``, or
        ``'Pchip'``. Default is ``'Pchip'``.
    """

    def __init__(self, times: np.ndarray,
                 data: np.ndarray,
                 timeshift: np.float32 = 0.0,
                 is_periodic: bool = False,
                 remove_mean: bool = False,
                 direction: np.ndarray = np.array([1, 0, 0], dtype=np.float32),
                 interpolation_method: Literal['AkimaSpline', 'CubicSpline', 'Pchip'] = 'Pchip'
                 ):
        self.times = times.astype(np.float32)
        self.data = data.astype(np.float32)
        self.timeshift = timeshift
        self.is_periodic = is_periodic
        self.remove_mean = remove_mean
        self.direction = (direction.reshape((1, 3)) / np.linalg.norm(direction)).astype(np.float32)
        self.interpolation_method = interpolation_method
        self.interpolator = self.calculate_interpolator()
        self._period = self.times[-1] if self.is_periodic else None

        if self._period is None:
            self._fold_time = lambda x: x
        else:
            T = self._period
            def _fold(x, T=T):
                # stable float "mod" without using % (avoids some edge cases)
                return (x - T * np.floor(x / T))
            self._fold_time = _fold

    def __add__(self, other):
        """Return a :class:`PODSum` combining this motion with another trajectory.

        Parameters
        ----------
        other : RespiratoryMotion, POD, or callable
            Second trajectory to add.

        Returns
        -------
        PODSum
            Combined trajectory evaluated as the sum of both.
        """
        return PODSum(self, other)

    def __call__(self, t: float):
        """Evaluate the displacement vector at time ``t``.

        Parameters
        ----------
        t : float
            Evaluation time (same unit as ``times``).

        Returns
        -------
        np.ndarray
            Displacement array of shape ``(1, 3)``.
        """
        trajectory = self._evaluate_motion(t)

        # Reshape trajectory to match the direction
        trajectory = trajectory * self.direction

        return trajectory

    def calculate_interpolator(self):
        """Build the 1-D interpolator from the time-series data.

        Returns
        -------
        scipy.interpolate interpolator
            Fitted interpolator instance.
        """
        # Data for interpolation
        times = self.times
        data = self.data

        # Remove mean if requested
        if self.remove_mean:
            data_mean = np.mean(self.data, axis=0, dtype=np.float32)
            data -= data_mean

        # Obtain the interpolator using the specified method
        if self.interpolation_method == 'AkimaSpline':
            interpolator = Akima1DInterpolator(times, data)
        elif self.interpolation_method == 'CubicSpline':
            interpolator = CubicSpline(times, data, bc_type='natural')
        elif self.interpolation_method == 'Pchip':
            interpolator = PchipInterpolator(times, data)
        else:
            raise ValueError(
                f"Interpolation method '{self.interpolation_method}' not recognized. "
                "Choose from 'AkimaSpline', 'CubicSpline', or 'Pchip'.")

        return interpolator

    def _evaluate_motion(self, t: float):
        """Return the interpolated scalar amplitude at time ``t``.

        Parameters
        ----------
        t : float
            Evaluation time (shifted and folded internally).

        Returns
        -------
        np.ndarray
            Scalar amplitude cast to ``float32``.
        """
        # Apply time shift if necessary
        t = self._fold_time(t + self.timeshift)

        return self.interpolator(t).astype(np.float32)

    def update_timeshift(self, timeshift: float):
        """Update the time shift applied before interpolation.

        Parameters
        ----------
        timeshift : float
            New time shift value.
        """
        self.timeshift = timeshift

    def get_modes(self, n_nodes: int) -> np.ndarray:
        """Return the global direction vector cast as a 1-mode displacement matrix."""
        # Output shape: (N_nodes, 3 components, 1 mode)
        modes = np.zeros((n_nodes, 3, 1), dtype=np.float32)
        modes[:, :, 0] = self.direction  # Broadcasts the (1, 3) vector to all nodes
        return modes

    def get_weights(self, t_array: np.ndarray) -> np.ndarray:
        """Evaluate the motion amplitude for all time points."""
        t_eff = self._fold_time(t_array + self.timeshift)
        weights = self.interpolator(t_eff).astype(np.float32)
        # Output shape: (N_times, 1 mode)
        return weights.reshape(-1, 1)


class POD:
    """Proper Orthogonal Decomposition trajectory for spatially-varying motion.

    Decomposes a sequence of full-field displacement snapshots into spatial
    modes and time-dependent weights. At evaluation time, weights are
    obtained by spline interpolation and the trajectory is reconstructed via
    a single tensor contraction with the modes.

    Parameters
    ----------
    times : np.ndarray
        1-D array of time samples, shape ``(T,)``.
    data : np.ndarray
        Displacement snapshots, shape ``(P, C, T)`` where ``P`` is the
        number of nodes, ``C`` the number of spatial components, and ``T``
        the number of time steps.
    global_to_local : np.ndarray, optional
        Index array mapping global node indices to the local MPI partition.
        If given, modes are extracted for the local partition only.
    n_modes : int, optional
        Number of POD modes to retain. Default is 5. Reduced automatically,
        with a warning, when it exceeds the numerical rank of ``data``.
        Use :func:`modes_for_energy` to pick this from the data instead of
        by hand.
    is_periodic : bool, optional
        If True, the trajectory is treated as periodic with period
        ``times[-1]``. Default is False.
    interpolation_method : str, optional
        Spline type for weight interpolation: ``'AkimaSpline'``,
        ``'CubicSpline'``, or ``'Pchip'``. Default is ``'Pchip'``.
    timeshift : float, optional
        Time offset added before evaluation. Default is 0.0.
    """

    def __init__(self, times: np.ndarray,
                 data: np.ndarray,
                 global_to_local: np.ndarray = None,
                 n_modes: int = 5,
                 is_periodic: bool = False,
                 interpolation_method: Literal['AkimaSpline', 'CubicSpline', 'Pchip'] = 'Pchip',
                 timeshift: np.float32 = 0.0):
        self.times = times          # (t,)
        self.data = data            # (P, C, t)
        self.local_to_global_map = global_to_local
        self.n_modes = n_modes
        self.timeshift = timeshift
        self.is_periodic = is_periodic
        self.interpolation_method = interpolation_method
        self.modes, self.weights = self.calculate_pod(remove_mean=False)
        self.spline_coeffs = self.spline_fit()
        pps = []
        for s in self.spline_coeffs:
            if not isinstance(s, PPoly):
                raise TypeError("Use CubicSpline/Pchip/Akima (PPoly subclasses)")
            pps.append(s)
        x0 = pps[0].x
        assert all(np.array_equal(pp.x, x0) for pp in pps), "knots differ"
        C = np.stack([pp.c for pp in pps], axis=-1)
        self._pp_batch = PPoly(C, x0, extrapolate=False)
        self._modes = np.asarray(self.modes, dtype=np.float32, order='C')
        self._period = self.times[-1] if self.is_periodic else None
        self._weights = np.zeros([self.n_modes, ], dtype=np.float32, order='C')

        if self._period is None:
            self._fold_time = lambda x: x
        else:
            T = self._period
            def _fold(x, T=T):
                # stable float "mod" without using % (avoids some edge cases)
                return x - T * np.floor(x / T)
            self._fold_time = _fold

    def __repr__(self):
        return f"POD(n_modes={self.n_modes}, interpolation_method='{self.interpolation_method}', is_periodic={self.is_periodic})"

    def __add__(self, other):
        """Return a :class:`PODSum` combining this POD with another trajectory.

        Parameters
        ----------
        other : POD, RespiratoryMotion, or callable
            Second trajectory to add.

        Returns
        -------
        PODSum
            Combined trajectory evaluated as the element-wise sum.
        """
        return PODSum(self, other)

    def __call__(self, t: float):
        """Evaluate the displacement field at time ``t``.

        Parameters
        ----------
        t : float
            Evaluation time.

        Returns
        -------
        np.ndarray
            Displacement array of shape ``(P_local, C)``.
        """
        trajectory = self._evaluate_trajectory(t)
        return trajectory

    def calculate_pod(self, remove_mean: bool = False):
        """Compute POD modes and time-weight matrix from the snapshot data.

        Parameters
        ----------
        remove_mean : bool, optional
            If True, subtract the temporal mean before decomposition.
            Default is False.

        Returns
        -------
        tuple
            ``(modes, weights)`` where ``modes`` has shape
            ``(P_local, C, n_modes)`` and ``weights`` has shape
            ``(T, n_modes)``.

        Notes
        -----
        Also sets ``self.eigenvalues`` (the full ``(T,)`` spectrum),
        ``self.energy`` (its cumulative fraction) and ``self.n_modes_max``.
        ``self.n_modes`` is reduced in place when it exceeds the numerical
        rank of the snapshots.
        """
        start = time.perf_counter()
        MPI_print(f"[POD] Calculating POD with {self.n_modes} modes and {self.interpolation_method} interpolation")

        n_tsteps = self.times.shape[0]
        flat_sv = self.data.reshape(-1, n_tsteps)

        # Remove mean if requested
        if remove_mean:
            sv_temporal_mean = np.mean(flat_sv, axis=1, keepdims=True)
            flat_sv -= sv_temporal_mean

        # Full spectrum of the (t, t) covariance matrix. Its eigenvalues are
        # the squared singular values of flat_sv, so partial sums give the
        # retained energy, and t is the largest mode count the data supports.
        eigen_values, eigen_vectors = _snapshot_eigenspectrum(flat_sv)
        self.eigenvalues = eigen_values
        self.energy = _cumulative_energy(eigen_values)
        self.n_modes_max = n_tsteps
        # (T, T) and therefore cheap to keep; it is what lets frame_errors
        # report the per-snapshot breakdown without a reconstruction.
        self._eigenvectors = eigen_vectors

        # Modes beyond the numerical rank would be scaled by 1/sqrt(~0) below
        # and come out as inf/nan without a word, so clamp instead.
        tol = eigen_values[0] * n_tsteps * np.finfo(np.float64).eps
        rank = int(np.count_nonzero(eigen_values > tol))
        if self.n_modes > rank:
            if rank < n_tsteps:
                reason = (f"the snapshot rank is {rank} (eigenvalue {rank + 1} is "
                          f"{eigen_values[rank] / eigen_values[0]:.1e} of the "
                          f"leading one)")
            else:
                reason = f"the data only has {n_tsteps} time steps"
            MPI_print(f"[POD] Requested {self.n_modes} modes but {reason}. "
                      f"Clamping to {rank}.")
            self.n_modes = rank

        # Keep the leading n_modes
        eigen_values = eigen_values[0:self.n_modes]
        eigen_vectors = eigen_vectors[:, 0:self.n_modes]

        # Scale eigen-vectors with inverse sqrt of eigen-value:
        modes_cut = eigen_vectors / np.sqrt(eigen_values).reshape(1, -1)

        # (P*ch, t) @ (t, N) -> (P*ch, N)
        phi = np.dot(flat_sv, modes_cut)

        weights = np.einsum('pn, pt -> nt', flat_sv, phi)

        # Reshape and distribute modes
        phi = phi.reshape((self.data.shape[0], -1, self.n_modes))
        if self.local_to_global_map is not None:
            phi = phi[self.local_to_global_map, :, :]

        MPI_print(f"[POD] Finished POD calculation in {time.perf_counter() - start:.2f} seconds")

        return phi, weights

    def spline_fit(self):
        """Fit a spline to the weight time series for each POD mode.

        Returns
        -------
        list
            List of ``n_modes`` :class:`scipy.interpolate.PPoly`-compatible
            interpolator objects, one per mode.
        """
        # Choose interpolation method
        if self.interpolation_method == 'AkimaSpline':
            interpolator = Akima1DInterpolator
        elif self.interpolation_method == 'CubicSpline':
            interpolator = CubicSpline
        elif self.interpolation_method == 'Pchip':
            interpolator = PchipInterpolator
        else:
            raise ValueError(
                f"Interpolation method '{self.interpolation_method}' not recognized. "
                "Choose from 'AkimaSpline', 'CubicSpline', or 'Pchip'.")

        # Fit spline to each mode's weights
        spline_coefficients = [interpolator(self.times, self.weights[:, i]) for i in range(self.n_modes)]

        return spline_coefficients

    def _evaluate_weights(self, t):
        self._weights[:] = self._pp_batch(t).astype(self._weights.dtype, copy=False)

    def _evaluate_trajectory(self, t: float) -> np.ndarray:
        """Evaluate the full spatial displacement at time ``t``.

        Steps: apply shift + periodic folding, evaluate mode weights via
        Horner's method, then contract weights with modes.

        Parameters
        ----------
        t : float
            Evaluation time.

        Returns
        -------
        np.ndarray
            Displacement field of shape ``(P_local, C)``.
        """
        # Apply shift and verify periodicity
        t_eff = self._fold_time(t + self.timeshift)

        # Evaluate weights at time t
        self._evaluate_weights(t_eff)

        return tensordot_modes_weights(self._modes, self._weights)

    def update_timeshift(self, timeshift: np.float32):
        """Update the time shift applied before evaluation.

        Parameters
        ----------
        timeshift : float
            New time shift value.
        """
        self.timeshift = timeshift

    def get_modes(self, n_nodes: int) -> np.ndarray:
        """Return the pre-calculated static POD modes."""
        if self._modes.shape[0] != n_nodes:
            raise ValueError(f"POD mesh size ({self._modes.shape[0]}) does not match assembler ({n_nodes}).")
        # Output shape: (N_nodes, 3 components, M_modes)
        return self._modes

    def get_weights(self, t_array: np.ndarray) -> np.ndarray:
        """Evaluate the spline interpolator for all time points simultaneously."""
        t_eff = self._fold_time(t_array + self.timeshift)
        # scipy's PPoly natively returns (N_times, M_modes) when evaluated with an array
        return self._pp_batch(t_eff).astype(np.float32)

    def energy_ratio(self, n_modes: int = None) -> float:
        """Fraction of the snapshot energy retained by the leading modes.

        Energy is a *squared* quantity, so this reads more reassuring than
        it is: ``energy_ratio() == 0.99`` is a 10% error on the field, not
        1%. Use :meth:`reconstruction_error` for the error itself and
        :meth:`frame_errors` for its distribution over time.

        Parameters
        ----------
        n_modes : int, optional
            Mode count to evaluate. Defaults to the number this object
            actually keeps.

        Returns
        -------
        float
            Value in ``(0, 1]``. Its complement is the relative squared
            reconstruction error of the truncated field.
        """
        n = self.n_modes if n_modes is None else int(n_modes)
        if n < 1 or n > self.n_modes_max:
            raise ValueError(
                f"n_modes must lie in [1, {self.n_modes_max}], got {n}.")
        return float(self.energy[n - 1])

    def cumulative_energy(self) -> np.ndarray:
        """Retained-energy curve over every available mode count.

        Returns
        -------
        np.ndarray
            Array of shape ``(T,)``; entry ``i`` is the energy kept by the
            leading ``i + 1`` modes, and the last entry is 1.
        """
        return self.energy.copy()

    def modes_for_energy(self, target: float) -> int:
        """Smallest mode count reaching ``target`` of the total energy.

        Parameters
        ----------
        target : float
            Desired energy fraction in ``(0, 1]``.

        Returns
        -------
        int
            Mode count, never larger than ``self.n_modes_max``.
        """
        return _modes_for_energy(self.energy, target)

    def reconstruction_error(self, n_modes: int = None) -> float:
        """Relative L2 error of the truncated field.

        This is ``sqrt(1 - energy_ratio(n))``, i.e.
        ``||X - X_n||_F / ||X||_F``. Because energy is squared, the error
        is much larger than the energy shortfall suggests: 99% energy is a
        10% error, 99.9% is 3%.

        Parameters
        ----------
        n_modes : int, optional
            Mode count to evaluate. Defaults to the number this object
            actually keeps.

        Returns
        -------
        float
            Value in ``[0, 1)``.
        """
        return float(np.sqrt(max(0.0, 1.0 - self.energy_ratio(n_modes))))

    def frame_errors(self, n_modes: int = None) -> np.ndarray:
        """Relative truncation error of each individual snapshot.

        :meth:`reconstruction_error` is an energy-weighted average over
        time, so it is dominated by the high-amplitude snapshots. Frames
        carrying little energy can be reconstructed far worse than the
        global figure implies — on pulsatile flow, a truncation that holds
        systole to a few percent routinely leaves diastole at tens of
        percent. Check this curve before trusting a mode count.

        Parameters
        ----------
        n_modes : int, optional
            Mode count to evaluate. Defaults to the number this object
            actually keeps.

        Returns
        -------
        np.ndarray
            Relative error per snapshot time, shape ``(T,)``. Entries lie
            in ``[0, 1]``; an all-zero snapshot reports 0.
        """
        n = self.n_modes if n_modes is None else int(n_modes)
        if n < 1 or n > self.n_modes_max:
            raise ValueError(
                f"n_modes must lie in [1, {self.n_modes_max}], got {n}.")
        return _frame_errors(self.eigenvalues, self._eigenvectors, n)


class PODVelocity(POD):
    """POD trajectory for velocity fields.

    Inherits from :class:`POD` and overrides the trajectory evaluation so
    that mode weights are scaled by the readout time ``t_ro``, converting
    integrated displacement modes into instantaneous velocity modes.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to :class:`POD`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate_trajectory(self, t: float):
        """Evaluate the velocity field at time ``t``.

        Parameters
        ----------
        t : float
            Evaluation time.

        Returns
        -------
        np.ndarray
            Velocity field of shape ``(P_local, C)``.
        """
        # Apply shift
        t_eff = self._fold_time(t + self.timeshift)

        # Check if t is within the bounds of the time array
        # TODO: verify if this is necessary (t_ro = t?)
        t_ro = t

        # Evaluate weights at time t
        self._evaluate_weights(t_eff)
        self._weights *= t_ro

        return tensordot_modes_weights(self._modes, self._weights)

    def get_weights(self, t_array: np.ndarray) -> np.ndarray:
        """Evaluate velocity weights (integrated displacement scaled by t_ro)."""
        t_eff = self._fold_time(t_array + self.timeshift)
        weights = self._pp_batch(t_eff).astype(np.float32)
        # Scale by readout time to convert displacement modes to instantaneous velocity
        weights *= t_array[:, np.newaxis]
        return weights

class PODSum:
    """Sum of two trajectory objects evaluated at the same time point.

    Allows combining, e.g., a :class:`POD` displacement field with a
    :class:`RespiratoryMotion` term via the ``+`` operator.

    Parameters
    ----------
    pod1 : POD or RespiratoryMotion
        First trajectory.
    pod2 : POD, RespiratoryMotion, or callable
        Second trajectory.
    """

    def __init__(self, pod1: POD, pod2: Callable[[np.float32], np.ndarray]):
        self.pod1 = pod1
        self.pod2 = pod2
        self.timeshift = 0.0

    def __call__(self, t: np.float32):
        """Evaluate the combined trajectory at time ``t``.

        Parameters
        ----------
        t : float
            Evaluation time.

        Returns
        -------
        np.ndarray
            Element-wise sum of both trajectory evaluations.
        """
        # Evaluate the trajectory at time t
        return self.pod1(t) + self.pod2(t)

    def update_timeshift(self, timeshift: np.float32):
        """Propagate a new time shift to both constituent trajectories.

        Parameters
        ----------
        timeshift : float
            New time shift value forwarded to ``pod1`` and ``pod2``.
        """
        self.pod1.update_timeshift(timeshift)
        self.pod2.update_timeshift(timeshift)
        self.timeshift = timeshift

    def get_modes(self, n_nodes: int) -> np.ndarray:
        """Concatenate the modes of both constituent trajectories."""
        m1 = self.pod1.get_modes(n_nodes)
        m2 = self.pod2.get_modes(n_nodes)
        # Combine along the 'modes' axis
        return np.concatenate([m1, m2], axis=2)

    def get_weights(self, t_array: np.ndarray) -> np.ndarray:
        """Concatenate the weights of both constituent trajectories."""
        w1 = self.pod1.get_weights(t_array)
        w2 = self.pod2.get_weights(t_array)
        # Combine along the 'modes' axis
        return np.concatenate([w1, w2], axis=1)


def pod_energy_spectrum(data: np.ndarray, *, remove_mean: bool = False):
    """Energy spectrum of a snapshot array, without building a POD.

    Decomposes the snapshots the same way :meth:`POD.calculate_pod` does
    and reports how much of the field each mode count captures, so
    ``n_modes`` can be sized before the POD is constructed. The
    eigenvalues are the squared singular values of the snapshot matrix,
    hence the retained energy for ``n`` modes is

    .. math::

       E(n) \\;=\\; \\frac{\\sum_{i \\le n} \\lambda_i}
                          {\\sum_{i \\le T} \\lambda_i}
             \\;=\\; 1 - \\frac{\\|X - X_n\\|_F^2}{\\|X\\|_F^2}

    so ``1 - E(n)`` is the relative squared reconstruction error of the
    truncated field. Mind the square: ``E = 0.99`` is a 10% error on the
    field, not 1%. :func:`pod_frame_errors` gives the error directly, and
    per snapshot.

    Parameters
    ----------
    data : np.ndarray
        Snapshots of shape ``(P, C, T)``. Any shape works as long as the
        last axis is time; it is flattened to ``(-1, T)``.
    remove_mean : bool, optional
        Subtract the temporal mean before decomposing. Default False,
        matching :class:`POD`, which decomposes the raw field so the mean
        counts toward the energy.

    Returns
    -------
    tuple of np.ndarray
        ``(eigenvalues, cumulative_energy)``, both of shape ``(T,)``.
        Eigenvalues descend; the cumulative curve ends at 1.

    See Also
    --------
    modes_for_energy : Invert the curve for a target energy.
    POD.energy_ratio : Same quantity from an existing POD.
    """
    data = np.asarray(data)
    n_tsteps = data.shape[-1]
    # Copy: remove_mean would otherwise write through the reshape view
    # into the caller's array.
    flat_sv = data.reshape(-1, n_tsteps).astype(np.float64, copy=True)

    if remove_mean:
        flat_sv -= np.mean(flat_sv, axis=1, keepdims=True)

    eigen_values, _ = _snapshot_eigenspectrum(flat_sv)

    return eigen_values, _cumulative_energy(eigen_values)


def modes_for_energy(data: np.ndarray, target: float, *,
                     remove_mean: bool = False) -> int:
    """Number of POD modes needed to retain ``target`` of the energy.

    Energy is squared, so a ``target`` of 0.99 leaves a 10% error on the
    field. Follow up with :func:`pod_frame_errors` to see how that error
    is spread over time before settling on the count.

    Parameters
    ----------
    data : np.ndarray
        Snapshots of shape ``(P, C, T)``.
    target : float
        Desired energy fraction in ``(0, 1]``, e.g. ``0.99``.
    remove_mean : bool, optional
        Forwarded to :func:`pod_energy_spectrum`.

    Returns
    -------
    int
        Mode count to pass as ``POD(n_modes=...)``, never larger than the
        number of time steps ``T``.
    """
    _, cumulative = pod_energy_spectrum(data, remove_mean=remove_mean)
    return _modes_for_energy(cumulative, target)


def pod_frame_errors(data: np.ndarray, n_modes: int, *,
                     remove_mean: bool = False) -> np.ndarray:
    """Per-snapshot truncation error, without building a POD.

    The companion to :func:`modes_for_energy`: once a mode count is on the
    table, this shows where the resulting error actually lands. A single
    global figure is an energy-weighted average and hides low-amplitude
    phases, which are often the ones a truncation sacrifices.

    Parameters
    ----------
    data : np.ndarray
        Snapshots of shape ``(P, C, T)``.
    n_modes : int
        Mode count to evaluate.
    remove_mean : bool, optional
        Subtract the temporal mean first. Default False, matching
        :class:`POD`.

    Returns
    -------
    np.ndarray
        Relative error per snapshot, shape ``(T,)``.

    See Also
    --------
    POD.frame_errors : Same curve from an existing POD.
    """
    data = np.asarray(data)
    n_tsteps = data.shape[-1]

    n = int(n_modes)
    if n < 1 or n > n_tsteps:
        raise ValueError(f"n_modes must lie in [1, {n_tsteps}], got {n}.")

    flat_sv = data.reshape(-1, n_tsteps).astype(np.float64, copy=True)
    if remove_mean:
        flat_sv -= np.mean(flat_sv, axis=1, keepdims=True)

    eigen_values, eigen_vectors = _snapshot_eigenspectrum(flat_sv)

    return _frame_errors(eigen_values, eigen_vectors, n)


def plot_pod_energy(source, *, target: float = 0.99, ax=None, show: bool = True,
                    export_to=None, title: str = None):
    """Plot the POD eigenvalue scree and the cumulative energy curve.

    Draws the normalised eigenvalues on a log axis and the cumulative
    retained energy on a twinned axis, with the ``target`` level and the
    mode count that first reaches it marked.

    Parameters
    ----------
    source : POD or np.ndarray
        An existing :class:`POD` (its stored spectrum is reused, nothing
        is recomputed) or a raw ``(P, C, T)`` snapshot array.
    target : float, optional
        Energy fraction to mark. Default 0.99.
    ax : matplotlib axis, optional
        Axes to draw into. A fresh figure is created when None.
    show : bool, optional
        Call ``plt.show()`` after rendering. Default True.
    export_to : str or path-like, optional
        Save the figure to this path.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib axis or None
        The axis drawn into, or None on non-root MPI ranks.
    """
    if MPI_rank != 0:
        return None

    # Local import: this is the only plotting entry point in the module,
    # and importing pyplot at module scope would cost every MPI rank.
    import matplotlib.pyplot as plt

    if isinstance(source, POD):
        eigen_values = source.eigenvalues
        cumulative = source.energy
    else:
        eigen_values, cumulative = pod_energy_spectrum(source)

    modes = np.arange(1, eigen_values.size + 1)
    n_target = _modes_for_energy(cumulative, target)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.5))

    ax.semilogy(modes, eigen_values / eigen_values[0], marker='o',
                markersize=4, color='steelblue', label='eigenvalue')
    ax.set_xlabel('POD mode')
    ax.set_ylabel('normalised eigenvalue', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')

    twin = ax.twinx()
    twin.plot(modes, 100.0 * cumulative, marker='s', markersize=4,
              color='indianred', label='cumulative energy')
    twin.axhline(100.0 * target, ls='--', lw=1.0, color='grey')
    twin.set_ylabel('retained energy [%]', color='indianred')
    twin.tick_params(axis='y', labelcolor='indianred')
    twin.set_ylim(0.0, 102.0)

    ax.axvline(n_target, ls=':', lw=1.0, color='grey')
    ax.annotate(f'{n_target} modes for {100.0 * target:.4g}%',
                xy=(n_target, 1.0), xycoords=('data', 'axes fraction'),
                xytext=(4, -12), textcoords='offset points', fontsize=9)

    ax.set_title(title if title is not None else 'POD energy retention')

    if export_to is not None:
        ax.figure.savefig(export_to, bbox_inches='tight')
    if show:
        plt.show()

    return ax