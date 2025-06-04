# This piece of code was inspired by CMRSim toolbox
# TODO: add additional information about the original authors and license

import numpy as np
from FEelMRI.MPIUtilities import MPI_print


class PODTrajectory:
  def __init__(self, time_array: np.ndarray, data: np.ndarray, n_modes: int = 5, taylor_order: int = 10, is_periodic: bool = False):
      self.time_array = time_array
      self.data = data
      self.n_modes = n_modes
      self.taylor_order = taylor_order
      self.modes, self.weights = self.calculate_pod(remove_mean=False)
      self.taylor_coefficients = self.fit()
      self.timeshift = 0.0
      self.is_periodic = is_periodic

  def __add__(self, other):
      """ Combines two PODTrajectory objects by summing their modes and weights.
      
      :param other: another PODTrajectory object
      :return: new PODTrajectory object with combined modes and weights
      """
      if not isinstance(other, PODTrajectory):
          raise TypeError("Can only add another PODTrajectory object.")
      
      return PODSum(self, other)

  def __call__(self, t: float):
      """ Evaluates the trajectory at time t using the POD modes and the Taylor coefficients.

      :param t: time at which to evaluate the trajectory
      :return: evaluated trajectory at time t
      """
      return self._evaluate_trajectory(t)

  def calculate_pod(self, remove_mean: bool = False):
    """Computes the proper orthogonal decomposition of data snapshots at points defined in
    `time_grid`. Returns only the `n_modes` number of most significant modes

    :param time_grid: (#time_steps) time-points corresponding to snapshots
    :param data: (#particles, #time_steps, #channels) snapshots of data
    :param n_modes: number of most significant modes to return
    :param remove_mean: if true removes the temporal mean of all snapshots before computing POD
    :return: - POD base modes, shape: (#particles * #channels, n\_modes),
              - scaling of modes per time-step (#time_steps, n\_modes)
    """
    n_tsteps = self.time_array.shape[0]
    flat_sv = self.data.reshape(-1, n_tsteps)

    if remove_mean:
      sv_temporal_mean = np.mean(flat_sv, axis=1, keepdims=True)
      flat_sv -= sv_temporal_mean

    # compute square matrix C of shape (t, P) @ (P, t) -> (t, t)
    covariance_matrix = np.dot(flat_sv.T, flat_sv)

    # shapes: (t, ), (t, t)
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues in descending order and take N largest -> shapes: (N,), (t, N)
    descending_sort_idx = np.argsort(eigen_values)[::-1][0:self.n_modes]
    eigen_values = eigen_values[descending_sort_idx]
    eigen_vectors = eigen_vectors[:, descending_sort_idx]

    # Scale eigen-vectors with inverse sqrt of eigen-value:
    # modes_cut = eigen_vectors.dot(np.diag(np.power(eigen_values, -0.5)))
    modes_cut = eigen_vectors / np.sqrt(eigen_values).reshape(1, -1)

    # (P*ch, t) @ (t, N) -> (P*ch, N)
    phi = np.dot(flat_sv, modes_cut)

    weights = np.einsum('pn, pt -> nt', flat_sv, phi)

    return phi.reshape((self.data.shape[0], -1, self.n_modes)), weights

  def fit(self):
    """ Fits a Taylor polynomial of order self.order to the weights of the POD modes
    """
    flat_coefficients = np.polynomial.polynomial.polyfit(self.time_array, self.weights, deg=self.taylor_order)
    return flat_coefficients

  def _evaluate_weights(self, t: float):
      """ Evaluates the taylor expansion for the current batch of particles at the specified
      times t.

      :param t: (#timesteps)
      :return: (#particles, #timesteps, 3)
      """
      factors = self.taylor_coefficients
      exponents = np.arange(0, self.taylor_order + 1, dtype=np.float32)
      exponents = np.tile(exponents, (self.n_modes, 1)).T
      t_pow_n = t ** exponents  # (order, time)
      out = np.sum(factors * t_pow_n, axis=0)
      return out
  
  def _evaluate_trajectory(self, t: float):
    """ Evaluates the trajectory at time t using the POD modes and the Taylor coefficients.

    :param t: time at which to evaluate the trajectory
    :return: evaluated trajectory at time t
    """
    t = t + self.timeshift  # Apply time shift if necessary
    if self.is_periodic:
      t = t % self.time_array[-1]

    # Evaluate the weights at time t
    weights = self._evaluate_weights(t)[np.newaxis, np.newaxis, :]

    return np.sum(self.modes * weights, axis=-1)
  

class PODSum:
  def __init__(self, pod1: PODTrajectory, pod2: PODTrajectory):
    self.pod1 = pod1
    self.pod2 = pod2

  def __call__(self, t: float):
    """ Evaluates the sum of all PODTrajectory objects at time t.

    :param t: time at which to evaluate the sum of trajectories
    :return: sum of evaluated trajectories at time t
    """
    return self.pod1(t) + self.pod2(t)