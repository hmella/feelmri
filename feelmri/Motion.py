# This piece of code was inspired by CMRSim toolbox
# TODO: add additional information about the original authors and license

from collections.abc import Callable

import numpy as np
from scipy.interpolate import interp1d
from feelmri.MPIUtilities import MPI_rank, MPI_print


class RespiratoryMotion:
  """ Class to handle respiratory motion using interpolation of time series data.
  This class allows for the evaluation of motion at any given time point using
  interpolation methods such as linear, nearest, zero, quadratic, and cubic.
  It can also handle periodic motion and apply a time shift to the motion data.
  The motion can be defined in a specific direction, and the mean can be removed
  from the data if desired.
  """
  def __init__(self, time_array: np.ndarray,
               data: np.ndarray,
               timeshift: float = 0.0,
               is_periodic: bool = False,
               remove_mean: bool = False,
               direction: np.ndarray = np.array([1, 0, 0]),
               interpolation_method: str = 'cubic'
               ):
    self.time_array = time_array
    self.data = data
    self.timeshift = timeshift
    self.is_periodic = is_periodic
    self.remove_mean = remove_mean
    self.direction = direction.reshape((1, 3))/np.linalg.norm(direction)
    self.interpolation_method = interpolation_method
    self.interpolator = self.calculate_interpolator()

  def __add__(self, other):
      """ Overloads the addition operator to allow for the addition of another
      POD or a callable object that returns a trajectory.
      :param other: another POD or a callable object
      :return: a new PODSum object that represents the sum of the two trajectories
      """      
      return PODSum(self, other)

  def __call__(self, t: float):
      """ Evaluates the trajectory at time t using the interpolator.
      :param t: time at which to evaluate the trajectory
      :return: evaluated trajectory at time t
      """
      trajectory = self._evaluate_motion(t)

      # Reshape trajectory to match the direction
      trajectory = trajectory * self.direction

      return trajectory

  def calculate_interpolator(self):
    """ Calculates the interpolator for the motion data.
    :return: scipy.interpolate.interp1d object for the motion data
    """
    # Data for interpolation
    times = self.time_array
    data  = self.data

    # Remove mean if requested
    if self.remove_mean:
      data_mean = np.mean(self.data, axis=0)
      data -= data_mean

    # Obtain the interpolator using the specified method
    interpolator = interp1d(times, data, kind=self.interpolation_method, bounds_error=False)

    return interpolator

  def _evaluate_motion(self, t: float):
    """ Evaluates the motion at time t using the interpolator.
    :param t: time at which to evaluate the motion
    :return: evaluated motion at time t
    """
    # Apply time shift if necessary
    t = t + self.timeshift  # Apply time shift if necessary
    if self.is_periodic:
      t = t % self.time_array[-1]

    return self.interpolator(t)

  def update_timeshift(self, timeshift: float):
    """ Updates the timeshift of the POD.
    :param timeshift: new timeshift value
    """
    self.timeshift = timeshift


class POD:
  def __init__(self, time_array: np.ndarray, 
               data: np.ndarray,
               global_to_local: np.ndarray = None,
               n_modes: int = 5,
               fit_type: str = 'polynomial', 
               taylor_order: int = 10, 
               is_periodic: bool = False,
               timeshift: float = 0.0):
      self.time_array = time_array
      self.data = data
      self.global_to_local = global_to_local
      self.n_modes = n_modes
      self.fit_type = fit_type
      self.taylor_order = taylor_order
      self.modes, self.weights = self.calculate_pod(remove_mean=False)
      self.taylor_coefficients = self.fit()
      self.timeshift = timeshift
      self.is_periodic = is_periodic
      self.fit_type = fit_type

  def __repr__(self):
      """ Returns a string representation of the POD object.
      :return: string representation of the POD object
      """
      return f"POD(n_modes={self.n_modes}, taylor_order={self.taylor_order}, is_periodic={self.is_periodic})"

  def __add__(self, other):
      """ Overloads the addition operator to allow for the addition of another
      POD or a callable object that returns a trajectory.
      :param other: another POD or a callable object
      :return: a new PODSum object that represents the sum of the two trajectories
      """      
      return PODSum(self, other)

  def __call__(self, t: float):
      """ Evaluates the trajectory at time t using the POD modes and the Taylor coefficients.
      :param t: time at which to evaluate the trajectory
      :return: evaluated trajectory at time t
      """
      # Evaluate the trajectory at time t
      trajectory = self._evaluate_trajectory(t)

      return trajectory

  def calculate_pod(self, remove_mean: bool = False):
    """ Calculates the POD modes and weights from the provided data.
    :param remove_mean: if True, removes the temporal mean from the data before calculating POD
    :return: tuple of (modes, weights) where modes are the POD modes and weights are the corresponding weights
    """
    n_tsteps = self.time_array.shape[0]
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
    # modes_cut = eigen_vectors.dot(np.diag(np.power(eigen_values, -0.5)))
    modes_cut = eigen_vectors / np.sqrt(eigen_values).reshape(1, -1)

    #  (P*ch, t) @ (t, N) -> (P*ch, N)
    phi = np.dot(flat_sv, modes_cut)
    
    weights = np.einsum('pn, pt -> nt', flat_sv, phi)

    # Reshape and distribute modes
    phi = phi.reshape((self.data.shape[0], -1, self.n_modes))
    if self.global_to_local is not None:
      phi = phi[self.global_to_local, :, :]

    return phi, weights

  def fit(self):
    """ Fits a Taylor polynomial of order self.order to the weights of the POD modes
    """
    if self.fit_type == 'polynomial':
      # Fit a polynomial of order self.taylor_order to the weights of the i-th mode
      # flat_coefficients = np.polynomial.polynomial.polyfit(self.time_array, self.weights, deg=self.taylor_order)
      flat_coefficients = np.zeros((self.taylor_order + 1, self.n_modes), dtype=np.float32)
      for i in range(self.n_modes):
        # Fit a polynomial of order self.taylor_order to the weights of the i-th mode
        flat_coefficients[:, i] = np.polynomial.Polynomial.fit(self.time_array, self.weights[:, i], deg=self.taylor_order, domain=[self.time_array[0], self.time_array[-1]]).convert().coef

    elif self.fit_type == 'legendre':
      # Fit a polynomial of order self.taylor_order to the weights of the i-th mode
      flat_coefficients = np.polynomial.legendre.legfit(self.time_array, self.weights, deg=self.taylor_order)

    elif self.fit_type == 'hermite':
      # Evaluate Legendre polynomial at time t
      flat_coefficients = np.polynomial.hermite.hermfit(self.time_array, self.weights, deg=self.taylor_order)

    return flat_coefficients

  def _evaluate_weights(self, t: float) -> np.ndarray:
    """
    Evaluate all mode‐weights at time t using Horner's method.
    self.taylor_coefficients has shape (order+1, n_modes).
    Returns array of length n_modes.
    """
    coeffs = self.taylor_coefficients
    weights = np.zeros([self.n_modes,], dtype=np.float32)
    if self.fit_type == 'polynomial':
      # Evaluate polynomial at time t using Horner's method
      for m in range(self.n_modes):
        weights[m] = np.polyval(coeffs[::-1, m], t)
    elif self.fit_type == 'legendre':
      # Evaluate Legendre polynomial at time t
      for m in range(self.n_modes):
        weights[m] = np.polynomial.legendre.legval(t, coeffs[::-1, m])
    elif self.fit_type == 'hermite':
      # Evaluate Legendre polynomial at time t
      for m in range(self.n_modes):
        weights[m] = np.polynomial.hermite.hermval(t, coeffs[::-1, m])

    return weights

  def _evaluate_trajectory(self, t: float) -> np.ndarray:
    """
    Evaluate the full trajectory at time t:
      1) apply shift + periodic
      2) eval weights with Horner
      3) one tensordot over modes
    """
    # Apply shift and verify periodicity
    t_eff = t + self.timeshift
    if self.is_periodic:
        t_eff %= self.time_array[-1]

    # Evaluate weights at time t
    w = self._evaluate_weights(t_eff)  # shape (n_modes,)

    # Combine modes (shape (..., n_modes)) with weights result will have shape of self.modes[...,0] (space dims)
    return np.tensordot(self.modes, w, axes=([2], [0]))

  def update_timeshift(self, timeshift: float):
    """ Updates the timeshift of the POD.
    :param timeshift: new timeshift value
    """
    self.timeshift = timeshift


class PODVelocity(POD):
  """ Class to handle the POD of velocity data.
  This class inherits from the POD class and is specifically designed to handle
  velocity data, allowing for the evaluation of the trajectory at any given time point.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def _evaluate_trajectory(self, t: float):
    """ Evaluates the trajectory at time t using the POD modes and weights. It uses a first-order Taylor expansion to approximate the trajectory at time t.
    :param t: time at which to evaluate the trajectory
    :return: evaluated trajectory at time t
    """
    # Apply shift
    t_eff = t + self.timeshift  # Apply time shift if necessary

    # Check if t is within the bounds of the time array
    if (t_eff - self.timeshift) <= 0:
      t_ro = t_eff
    else:
      t_ro = (t_eff - self.timeshift)

    # Apply periodicity if necessary
    if self.is_periodic:
      t_eff = self.timeshift % self.time_array[-1]

    # Evaluate the weights at time t
    weights = t_ro * self._evaluate_weights(t_eff)

    return np.tensordot(self.modes, weights, axes=([2], [0]))

class PODSum:
  """ Class to handle the sum of a POD and a callable object.
  This class allows for the evaluation of the sum of a POD and a callable
  object that returns a trajectory at any given time point.
  """
  def __init__(self, pod1: POD, pod2: Callable[[float], np.ndarray]):
    self.pod1 = pod1
    self.pod2 = pod2
    self.timeshif = 0.0

  def __call__(self, t: float):
    """ Evaluates the sum of the POD and the callable object at time t.
    :param t: time at which to evaluate the sum
    :return: evaluated sum at time t
    """
    # Evaluate the trajectory at time t
    return self.pod1(t) + self.pod2(t)
  
  def update_timeshift(self, timeshift: float):
    """ Updates the timeshift of the POD.
    :param timeshift: new timeshift value
    """
    self.pod1.timeshift = timeshift
    self.pod2.timeshift = timeshift
    self.timeshift = timeshift