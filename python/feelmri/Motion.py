# This piece of code was inspired by CMRSim toolbox
# TODO: add additional information about the original authors and license
import time
from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.interpolate import (Akima1DInterpolator, CubicSpline,
                               PchipInterpolator, PPoly)

from feelmri.MPIUtilities import MPI_print, MPI_rank
from feelmri.POD import tensordot_modes_weights


class RespiratoryMotion:
  """ Class to handle respiratory motion using interpolation of time series data.
  This class allows for the evaluation of motion at any given time point using
  interpolation methods such as linear, nearest, zero, quadratic, and cubic.
  It can also handle periodic motion and apply a time shift to the motion data.
  The motion can be defined in a specific direction, and the mean can be removed
  from the data if desired.
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
    self.direction = (direction.reshape((1, 3))/np.linalg.norm(direction)).astype(np.float32)
    self.interpolation_method = interpolation_method
    self.interpolator = self.calculate_interpolator()
    self._period = self.times[-1] if self.is_periodic else None

    if self._period is None:
        self._fold_time = lambda x: x
    else:
        T = self._period
        def _fold(x, T=T):
            # stable float "mod" without using % (avoids some edge cases)
            return (x - T*np.floor(x / T))
        self._fold_time = _fold

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
    times = self.times
    data  = self.data

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
      raise ValueError(f"Interpolation method '{self.interpolation_method}' not recognized. Choose from 'AkimaSpline', 'CubicSpline', or 'Pchip'.")

    return interpolator

  def _evaluate_motion(self, t: float):
    """ Evaluates the motion at time t using the interpolator.
    :param t: time at which to evaluate the motion
    :return: evaluated motion at time t
    """
    # Apply time shift if necessary
    t = self._fold_time(t + self.timeshift)

    return self.interpolator(t).astype(np.float32)

  def update_timeshift(self, timeshift: float):
    """ Updates the timeshift of the POD.
    :param timeshift: new timeshift value
    """
    self.timeshift = timeshift


class POD:
  def __init__(self, times: np.ndarray, 
               data: np.ndarray,
               global_to_local: np.ndarray = None,
               n_modes: int = 5,
               is_periodic: bool = False,
               interpolation_method: Literal['AkimaSpline', 'CubicSpline', 'Pchip'] = 'Pchip',
               timeshift: np.float32 = 0.0):
      self.times = times  # (t,)
      self.data = data              # (P, C, t)
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
              return x - T*np.floor(x / T)
          self._fold_time = _fold

  def __repr__(self):
      """ Returns a string representation of the POD object.
      :return: string representation of the POD object
      """
      return f"POD(n_modes={self.n_modes}, interpolation_method='{self.interpolation_method}', is_periodic={self.is_periodic})"

  def __add__(self, other):
      """ Overloads the addition operator to allow for the addition of another
      POD or a callable object that returns a trajectory.
      :param other: another POD or a callable object
      :return: a new PODSum object that represents the sum of the two trajectories
      """      
      return PODSum(self, other)

  def __call__(self, t: float):
      """ Evaluates the trajectory at time t using the POD modes and weights.
      :param t: time at which to evaluate the trajectory
      :return: evaluated trajectory at time t
      """
      trajectory = self._evaluate_trajectory(t)

      return trajectory

  def calculate_pod(self, remove_mean: bool = False):
    """ Calculates the POD modes and weights from the provided data.
    :param remove_mean: if True, removes the temporal mean from the data before calculating POD
    :return: tuple of (modes, weights) where modes are the POD modes and weights are the corresponding weights
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
    # modes_cut = eigen_vectors.dot(np.diag(np.power(eigen_values, -0.5)))
    modes_cut = eigen_vectors / np.sqrt(eigen_values).reshape(1, -1)

    #  (P*ch, t) @ (t, N) -> (P*ch, N)
    phi = np.dot(flat_sv, modes_cut)
    
    weights = np.einsum('pn, pt -> nt', flat_sv, phi)

    # Reshape and distribute modes
    phi = phi.reshape((self.data.shape[0], -1, self.n_modes))
    if self.local_to_global_map is not None:
      phi = phi[self.local_to_global_map, :, :]

    MPI_print(f"[POD] Finished POD calculation in {time.perf_counter() - start:.2f} seconds")

    return phi, weights

  def spline_fit(self):
    """ Fits a spline to the weights of the POD modes
    """
    # Choose interpolation method
    if self.interpolation_method == 'AkimaSpline':
      interpolator = Akima1DInterpolator
    elif self.interpolation_method == 'CubicSpline':
      interpolator = CubicSpline
    elif self.interpolation_method == 'Pchip':
      interpolator = PchipInterpolator
    else:
      raise ValueError(f"Interpolation method '{self.interpolation_method}' not recognized. Choose from 'AkimaSpline', 'CubicSpline', or 'Pchip'.")

    # Fit spline to each mode's weights
    spline_coefficients = [interpolator(self.times, self.weights[:, i]) for i in range(self.n_modes)]

    return spline_coefficients

  def _evaluate_weights(self, t):
      self._weights[:] = self._pp_batch(t).astype(self._weights.dtype, copy=False)

  def _evaluate_trajectory(self, t: float) -> np.ndarray:
    """
    Evaluate the full trajectory at time t:
      1) apply shift + periodic
      2) eval weights with Horner
      3) one tensordot over modes
    """
    # Apply shift and verify periodicity
    t_eff = self._fold_time(t + self.timeshift)

    # Evaluate weights at time t
    self._evaluate_weights(t_eff)

    # return np.tensordot(self.modes, self._weights, axes=([2], [0]))
    return tensordot_modes_weights(self._modes, self._weights)

  def update_timeshift(self, timeshift: np.float32):
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
    """ Evaluate the full trajectory at time t:
        1) apply shift + periodic
        2) eval weights using interpolation method
        3) one tensordot over modes
    """
    # Apply shift
    t_eff = self._fold_time(t + self.timeshift)  # Apply time shift if necessary

    # Check if t is within the bounds of the time array
    # TODO: verify if this is necessary (t_ro = t?)
    t_ro = t

    # Evaluate weights at time t
    self._evaluate_weights(t_eff)
    self._weights *= t_ro

    # return np.tensordot(self.modes, self._weights, axes=([2], [0]))
    return tensordot_modes_weights(self._modes, self._weights)


class PODSum:
  """ Class to handle the sum of a POD and a callable object.
  This class allows for the evaluation of the sum of a POD and a callable
  object that returns a trajectory at any given time point.
  """
  def __init__(self, pod1: POD, pod2: Callable[[np.float32], np.ndarray]):
    self.pod1 = pod1
    self.pod2 = pod2
    self.timeshift = 0.0

  def __call__(self, t: np.float32):
    """ Evaluates the sum of the POD and the callable object at time t.
    :param t: time at which to evaluate the sum
    :return: evaluated sum at time t
    """
    # Evaluate the trajectory at time t
    return self.pod1(t) + self.pod2(t)
  
  def update_timeshift(self, timeshift: np.float32):
    """ Updates the timeshift of the POD.
    :param timeshift: new timeshift value
    """
    self.pod1.update_timeshift(timeshift)
    self.pod2.update_timeshift(timeshift)
    self.timeshift = timeshift