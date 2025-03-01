"""Contains the implementation of a proper orthogonal decomposition module for trajectory modules"""

__all__ = ["PODTrajectoryModule"]

from typing import Dict, Tuple

import tensorflow as tf
import numpy as np

from cmrsim.trajectory._taylor import TaylorTrajectoryN
from cmrsim.trajectory._base import BaseTrajectoryModule


# pylint: disable=abstract-method
class PODTrajectoryModule(BaseTrajectoryModule):
    # pylint: disable=anomalous-backslash-in-string
    """Captures the trajectories of an arbitrary set of particles  (e.g. Nodes of structured meshes)
    by computing the proper orthogonal decomposition (POD) from a set of snapshots.

    .. math::

      u(t) = \Sigma_j^{N_{modes}} \phi_j w_j(t),

    where :math:`\phi_j` are the computed basis functions (modes) $w_j(t)$ are the corresponding
    mode-weights as function of time. To allow the motion state to be reconstructed from the
    low-rank representation at arbitrary time-points within the interval of specified snapshots,
    the mode-weights are represented as Taylor-series.

    .. dropdown:: Example Usage:

        .. code::

            data, t = ...  # get snapshots of states and corresponding time-points
            # Shapes: data.shape == (#particles, #snapshots, #channels)
            #         t.shape == (#steps, )

            pod_module = cmrsim.trajectory.PODTrajectoryModule(data, t, n_modes=5, poly_oder=8)

            new_time_grid = np.linspace(t[0].m, t[-1].m, 100).astype(np.float32)

            reconstructed_states, _ = pod_module(new_time_grid)
            ## reconstructed_states.shape == (#particles, 100, #channels)


    :param time_grid: (#time_steps)
    :param trajectories: (#particles, #time_steps, 3)
    :param n_modes: Number of modes used for reduce-order representation
    :param poly_order: Order of the Taylor-series used to fit the mode-weights
    :param additional_data: Dictionary containing additional field data of shape (#particles, #time_steps, #channels)
    :param batch_size: If not None, the output of call and increment positions will be batched in given size. 
                        Which batch is returned depends on the argument `batch_index` as decribed below
    :param is_periodic: If true, the mode-weights are periodically interpolated for times exceeding the 
                        specified time-grid.
    """
    #: Number of modes used for reduce-order representation
    n_modes: tf.Variable
    #: Computed basis-functions (modes) :math:`\phi_j` used to represent the input data in
    #: a reduced order. Shape (#particles * #channels, #modes)
    basis_function: tf.Variable
    #: Keeps track of the current timing when increment_particles is called.
    current_time_ms: tf.Variable
    #: Allows to only evaluate the position for a batch of stored particle trajectories
    current_batch_idx: tf.Variable
    #: Together with self.current_batch_size determines the subset of particle trajectories that
    #: is evaluated on call and increment_particles
    batch_size: tf.Variable
    #: Keeps track of the current timing when increment_particles is called.
    current_time_ms: tf.Variable
    #: Start time of the specified time grid, used for internal calculations
    ref_time: tf.Variable
    #: End time of the specified time grid, used for internal calculations
    end_time: tf.Variable
    #:
    _nchannels: tf.Variable
    #:
    _additional_keys: Tuple[str]
    #:
    _additional_channels: Tuple[int]

    def __init__(self, time_grid: np.ndarray,
                 trajectories: np.ndarray,
                 n_modes: int, poly_order: int,
                 additional_data: Dict[str, np.ndarray] = None,
                 batch_size: int = None,
                 is_periodic: bool = False):

        self.ref_time = tf.Variable(time_grid[0].astype(np.float32),
                                    dtype=tf.float32, shape=(), trainable=False)
        self.end_time = tf.Variable(time_grid[-1].astype(np.float32),
                                    dtype=tf.float32, shape=(), trainable=False)
        self.n_modes = tf.Variable(n_modes, shape=(), dtype=tf.int32)

        if additional_data is None: additional_data = {}
        self._additional_keys = tuple(additional_data.keys())
        self._additional_channels = tuple([additional_data[k].shape[-1] for k in self._additional_keys])
        self._additional_channel_idx = tuple([3, ] + np.cumsum(np.array(self._additional_channels)+3).tolist())
        print(self._additional_channel_idx)
        data = np.concatenate([trajectories, ] + [additional_data[k] for k in self._additional_keys], axis=-1)

        self._nchannels = tf.Variable(data.shape[2], shape=(), dtype=tf.int32)
        phi, mode_weights = self.calculate_pod(time_grid, data, n_modes, remove_mean=False)
        self.basis_function = tf.Variable(phi, shape=(None, n_modes), dtype=tf.float32)

        if batch_size is not None:
            self.batch_size = tf.Variable(batch_size, dtype=tf.int32, shape=(), trainable=False)
        else:
            self.batch_size = tf.Variable(data.shape[0], dtype=tf.int32, shape=(), trainable=False)
        self.current_batch_idx = tf.Variable(0, dtype=tf.int32, shape=(), trainable=False)
        self._taylor_module = TaylorTrajectoryN(order=poly_order, time_grid=time_grid,
                                                particle_trajectories=mode_weights.T.reshape(n_modes, -1, 1),
                                                batch_size=None, fit_on_init=True,
                                                is_periodic=False)
        self.current_time_ms = self._taylor_module.current_time_ms
        self.is_periodic = tf.constant(is_periodic, dtype=tf.bool)

    @staticmethod
    def calculate_pod(time_grid: np.ndarray, data: np.ndarray, n_modes: int,
                      remove_mean: bool = False) -> (np.ndarray, np.ndarray):
        """Computes the proper orthogonal decomposition of data snapshots at points defined in
        `time_grid`. Returns only the `n_modes` number of most significant modes

        :param time_grid: (#time_steps) time-points corresponding to snapshots
        :param data: (#particles, #time_steps, #channels) snapshots of data
        :param n_modes: number of most significant modes to return
        :param remove_mean: if true removes the temporal mean of all snapshots before computing POD
        :return: - POD base modes, shape: (#particles * #channels, n\_modes),
                 - scaling of modes per time-step (#time_steps, n\_modes)
        """
        n_tsteps = time_grid.shape[0]
        flat_sv = data.transpose(0, 2, 1).reshape(-1, n_tsteps)

        if remove_mean:
            sv_temporal_mean = np.mean(flat_sv, axis=1, keepdims=True)
            flat_sv -= sv_temporal_mean

        # compute square matrix C of shape (t, P) @ (P, t) -> (t, t)
        covariance_matrix = np.dot(flat_sv.T, flat_sv)

        # shapes: (t, ), (t, t)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues in descending order and take N largest -> shapes: (N,), (t, N)
        descending_sort_idx = np.argsort(eigen_values)[::-1][0:n_modes]
        eigen_values = eigen_values[descending_sort_idx]
        eigen_vectors = eigen_vectors[:, descending_sort_idx]

        # Scale eigen-vectors with inverse sqrt of eigen-value:
        # modes_cut = eigen_vectors.dot(np.diag(np.power(eigen_values, -0.5)))
        modes_cut = eigen_vectors / np.sqrt(eigen_values).reshape(1, -1)

        # (P*ch, t) @ (t, N) -> (P*ch, N)
        phi = np.dot(flat_sv, modes_cut)

        weights = np.einsum('pn, pt -> nt', flat_sv, phi)
        return phi, weights

    def __call__(self, initial_positions: tf.Tensor, timing: tf.Tensor,
                 batch_index: int = 0, **kwargs) -> (tf.Tensor, dict):
        """Reconstructs the data state at given times t, by evaluating the taylor series
        of mode-weights and computing the weighted sum.

        :param initial_positions: Unused argument which can be set to None, 
                                    specified only to adhere to base implementaton
        :param timing: (#timesteps) in milliseconds
        :param batch_index: determines which batch to return if `batch_size` was 
                            specified on instantiation (sets `self.current_batch_idx`) 
        :return: (#particles, #timesteps, self._channels), {k: v} - additional data
        """
        idx_before = self.current_batch_idx.read_value()
        self.current_batch_idx.assign(batch_index)
        data = self._evaluate_trajectory(timing)
        if len(self._additional_keys) > 0:
            additional_data = {k: data[..., self._additional_channel_idx[i]:self._additional_channel_idx[i+1]]
                               for i, k in enumerate(self._additional_keys)}
        else:
            additional_data = {}

        self.current_batch_idx.assign(idx_before)
        return data[:, :, :3], additional_data

    @tf.function
    def increment_particles(self, particle_positions: tf.Tensor,
                            dt: tf.Tensor, **kwargs) -> (tf.Tensor, dict):
        """Evaluates the particle position for the time self.current_time_ms + dt and adds the
        delta t to the current_time_ms variable

        :param particle_positions: unused parameter (to adhere to calling signature of trajectory modules)
        :param dt: temporal step lengths
        :param kwargs: unused parameter (to adhere to calling signature of trajectory modules)
        :return: (#batch, self._channels),  {k: v} - additional data
        """
        self._taylor_module.current_time_ms.assign_add(dt)
        _t = tf.reshape(self._taylor_module.current_time_ms, [1, ])
        data = self._evaluate_trajectory(_t)
        if len(self._additional_keys) > 0:
            additional_data = {k: data[..., self._additional_channel_idx[i]:self._additional_channel_idx[i + 1]]
                               for i, k in enumerate(self._additional_keys)}
        else:
            additional_data = {}

        return tf.ensure_shape(data[:, 0, :3], (None, 3)), additional_data

    @tf.function(jit_compile=False, reduce_retracing=True)
    def _evaluate_trajectory(self, t: tf.Tensor) -> tf.Tensor:
        """Reconstructs the data state at given times t, by evaluating the taylor series
        of mode-weights and computing the weighted sum.

        :param t: (#steps) time-points to reconstruct at
        :return: data-states (#particles, #steps, #channels)
        """
        t = t - self.ref_time
        if self.is_periodic:
            t = tf.math.floormod(t, self.end_time - self.ref_time)

        batch_start = self.current_batch_idx * self.batch_size * self._nchannels
        batch_end = batch_start + self.batch_size * self._nchannels
        # Evaluate mode weights from taylor-expansion -> shape: (t, N)
        mode_weights = tf.transpose(self._taylor_module._evaluate_trajectory(t)[:, :, 0], [1, 0])
        # Compute weighted sum of basis functions / modes to reconstruct data
        values_at_t = tf.einsum('pn, tn -> pt', self.basis_function[batch_start:batch_end],
                                mode_weights)
        # Reshape according to input shapes (#particles, #steps, #channels)
        _shape = tf.stack([-1, self._nchannels, tf.shape(mode_weights)[0]], axis=0)
        return tf.transpose(tf.reshape(values_at_t, _shape), [0, 2, 1])
