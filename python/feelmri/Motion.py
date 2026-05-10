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
        Number of POD modes to retain. Default is 5.
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
        """
        start = time.perf_counter()
        MPI_print(f"[POD] Calculating POD with {self.n_modes} modes and {self.interpolation_method} interpolation")

        n_tsteps = self.times.shape[0]
        flat_sv = self.data.reshape(-1, n_tsteps)

        # Remove mean if requested
        if remove_mean:
            sv_temporal_mean = np.mean(flat_sv, axis=1, keepdims=True)
            flat_sv -= sv_temporal_mean

        # Calculate covariance matrix: (P*ch, t) @ (t, P*ch) -> (P*ch, P*ch)
        covariance_matrix = np.dot(flat_sv.T, flat_sv)

        # Calculate eigenvalues and eigenvectors
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order:
        descending_sort_idx = np.argsort(eigen_values)[::-1][0:self.n_modes]
        eigen_values = eigen_values[descending_sort_idx]
        eigen_vectors = eigen_vectors[:, descending_sort_idx]

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