import time
from pathlib import Path

import numpy as np

from FEelMRI.Math import Rx, Ry, Rz
from FEelMRI.Parameters import ParameterHandler
from FEelMRI.Phantom import FEMPhantom
from FEelMRI.IO import XDMFFile

class PODTrajectory:
  def __init__(self, time_array: np.ndarray, data: np.ndarray, n_modes: int = 5, taylor_order: int = 10):
      self.time_array = time_array
      self.data = data
      self.n_modes = n_modes
      self.taylor_order = taylor_order
      self.modes, self.weights = self.calculate_pod(remove_mean=False)
      self.taylor_coefficients = self.fit()

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
    # coefficients = np.swapaxes(
    #     flat_coefficients.reshape(self._int_order + 1, n_particles, n_dims),
    #     0, 1)
    # self.optimal_parameters.assign(coefficients.astype(np.float32))
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
    weights = self._evaluate_weights(t)[np.newaxis, np.newaxis, :]
    # # Reshape weights to match the shape of modes
    # weights = weights.reshape(-1, self.n_modes)
    # # Evaluate trajectory
    return np.sum(self.modes * weights, axis=-1)


if __name__ == '__main__':

  # Import imaging parameters
  parameters = ParameterHandler('parameters/aorta_volume.yaml')

  # Imaging orientation paramters
  theta_x = np.deg2rad(parameters.theta_x)
  theta_y = np.deg2rad(parameters.theta_y)
  theta_z = np.deg2rad(parameters.theta_z)
  MPS_ori = Rz(theta_z)@Rx(theta_x)@Ry(theta_y)
  LOC = parameters.LOC


  # Create FEM phantom object
  phantom = FEMPhantom(path='phantoms/aorta_CFD.xdmf', scale_factor=1.0)

  # Translate phantom to obtain the desired slice location
  phantom.orient(MPS_ori, LOC)

  # # Distribute phantom across MPI ranks
  # phantom.distribute_mesh()

  # Iterate over cardiac phases
  t0 = time.time()
  trajectory = np.zeros([phantom.nodes.shape[0], phantom.nodes.shape[1],phantom.Nfr], dtype=np.float32)
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr
    phantom.read_data(fr)
    displacement = phantom.point_data['velocity'] @ MPS_ori
    trajectory[..., fr] = displacement

  # Define POD object
  times = np.linspace(0, (phantom.Nfr-1)*parameters.TimeSpacing, phantom.Nfr)
  pod_trajectory = PODTrajectory(time_array=times,
                                 data=trajectory,
                                 n_modes=10,
                                 taylor_order=10)

  # Test class
  times = np.linspace(0, (phantom.Nfr-1)*parameters.TimeSpacing, 100)
  file = XDMFFile('test_velocity.xdmf', nodes=phantom.nodes, elements=phantom.all_elements)
  for i, t in enumerate(times):
    ut = pod_trajectory(t)
    file.write(pointData={'velocity': ut}, time=t)
  file.close()