import warnings

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity as Q_
from scipy.integrate import RK45
from scipy.interpolate import interp1d

from FEelMRI.MPIUtilities import MPI_rank
from FEelMRI.MRObjects import RF, Gradient, Scanner
from FEelMRI.Units import *


class Bloch:
  def __init__(self, gamma=42.58, z0=0.0, z=0.0, eval_gradient=None, B1e=None, small_angle=False):
    self.gamma = gamma
    self.z0 = z0
    self.z = z
    self.eval_gradient = eval_gradient
    self.B1e = B1e
    self.small_angle = small_angle
    self.equation = self.bloch_small if self.small_angle else self.bloch

  def __call__(self, t, M):
    return self.equation(t, M)

  def bloch(self, t, M):
    # Frequency offset
    dw = self.gamma*self.eval_gradient(t)*(self.z - self.z0)

    # Bloch equations
    dMxdt = dw*M[1]
    dMydt = self.gamma*self.B1e(t)*M[2] - dw*M[0]
    dMzdt = -self.gamma*self.B1e(t)*M[1]

    return np.array([dMxdt, dMydt, dMzdt]).reshape((3,))

  def bloch_small(self, t, M):
    # Frequency offset
    dw = self.gamma*self.eval_gradient(t)*(self.z - self.z0)

    # Bloch equations
    dMxdt = dw*M[1]
    dMydt = self.gamma*self.B1e(t)*M[2] - dw*M[0]
    dMzdt = -self.gamma*self.B1e(t)*M[1]*0.0

    return np.array([dMxdt, dMydt, dMzdt]).reshape((3,))


class SliceProfile:
  """
  A class to simulate the slice profile of an MRI pulse sequence using the Bloch equations.

  Attributes:
    z0 (float): Initial position of the slice center (default is 0.0).
    delta_z (float): Slice thickness (default is 0.008).
    gammabar (float): Gyromagnetic ratio in MHz/T (default is 42.58).
    Gz (float): Gradient strength in T/m (default is 1.0).
    RFShape (str): Shape of the RF pulse ('sinc', 'apodized_sinc', 'hard') (default is 'sinc').
    NbLobes[0] (int): Number of left lobes for the sinc pulse (default is 2).
    NbLobes[1] (int): Number of right lobes for the sinc pulse (default is 2).
    alpha (float): Apodization factor for the apodized sinc pulse (default is 0.46 for a Hamming window. Use 0.5 for a Hanning window).
    flip_angle (float): Flip angle in radians (default is np.deg2rad(10.0)).
    dt (float): Time step for the simulation (default is 1e-4).
    profile_samples (int): Number of points in the slice profile (default is 150).
    plot (bool): Whether to plot the results (default is False).
    small_angle (bool): Whether to use the small angle approximation (default is False).
    refocusing_area_frac (float): Fraction of the refocusing gradient area (default is 0.5).

  Methods:
    bloch(t, M):
      Computes the derivatives of the magnetization vector M at time t using the Bloch equations.

    ss_gradient(t):
      Computes the slice selection gradient at time t.

    _rf_sinc(t):
      Computes the sinc RF pulse at time t.

    _rf_apodized_sinc(t):
      Computes the apodized sinc RF pulse at time t.

    _rf_hard(t):
      Computes the hard RF pulse at time t.

    B1e_norm(t):
      Computes the normalized B1 field at time t.

    calculate(y0=np.array([0,0,1]).reshape((3,))):
      Calculates the slice profile by solving the Bloch equations.
  """
  def __init__(self, z0=Q_(0.0,'m'), delta_z=Q_(0.008,'m'), bandwidth='maximum', rf=RF(), dt=Q_(1e-4,'ms'), profile_samples=150, plot=False, small_angle=False, refocusing_area_frac=0.5, scanner=Scanner(), dtype=np.float32):
    self.z0 = z0
    self.delta_z = delta_z
    self.rf = rf
    self.profile_samples = profile_samples
    self.plot = plot
    self.dt = dt.to('ms')
    self.small_angle = small_angle
    self.refocusing_area_frac = refocusing_area_frac
    self.scanner = scanner
    self.bandwidth = self._check_bandwidth(bandwidth)
    self.dtype = dtype
    self.interp_profile = self.calculate()

  def _check_bandwidth(self, bandwidth):
    """
    Check the bandwidth of the RF pulse.

    Raises:
    ValueError: If the bandwidth is less than the minimum required.
    """
    max_bw = self.scanner.gammabar.to('Hz/T')*self.scanner.gradient_strength.to('T/m')*self.delta_z.to('m')
    if bandwidth == 'maximum':
      return max_bw
    elif bandwidth > max_bw:
      raise ValueError('Bandwidth is greater than the maximum allowed by the slice selection gradient and desired slice thickness')
    else:
      return bandwidth

  def optimize(self, frac_start=1.0, frac_end=2.0, N=10, profile_samples=100):
    """
      Optimize the refocusing area fraction for a given number of points.
    """
    # Replace profile_samples and plot (temporary)
    _profile_samples = self.profile_samples
    _plot = self.plot
    self.profile_samples = profile_samples
    self.plot = False

    # Slice positions
    z_min = self.z0 - 2*self.delta_z
    z_max = self.z0 + 2*self.delta_z
    z_arr = np.linspace(z_min, z_max, self.profile_samples)

    # Optimize refocusing area fraction
    Mx = []
    My = []
    area_fracs = np.linspace(frac_start, frac_end, N)
    for i, frac in enumerate(area_fracs):
      self.refocusing_area_frac = frac
      self.interp_profile = self.calculate()
      Mx.append(np.abs(np.imag(self.interp_profile(z_arr))).max())
      My.append(np.abs(np.real(self.interp_profile(z_arr))).max())
      if MPI_rank == 0:
        print("[iter ", i, "]", "Refocusing area fraction: ", frac, "Max. Mx : ", Mx[-1], ", Max. My : ", My[-1])

    # Verify which fraction gives the best result
    idx = np.argmin(Mx)
    self.refocusing_area_frac = area_fracs[idx]
    self.interp_profile = self.calculate()

    if MPI_rank == 0:
      print("Optimal refocusing area fraction: ", self.refocusing_area_frac)

    # profile_samples back to original value
    self.profile_samples = _profile_samples
    self.plot = _plot

  def calculate(self, y0=np.array([0,0,1]).reshape((3,))):
    """
    Calculate the slice profile for MRI imaging.

    Parameters:
    -----------
    y0 : np.ndarray, optional
      Initial magnetization vector. Default is np.array([0,0,1]).reshape((3,)).

    Returns:
    --------
    interp_profile : scipy.interpolate.interp1d
      Interpolated profile of the slice selection.

    Notes:
    ------
    This function calculates the slice profile by solving the Bloch equations using the Runge-Kutta method.
    It also plots the B1 field, gradient waveforms, and magnetization components if plotting is enabled.
    """

    # Calculate gradient amplitude needed for the desired slice thickness and bandwidth
    Gz = Q_(np.min([self.bandwidth.m_as('Hz')/self.scanner.gammabar.m_as('Hz/T')/self.delta_z.m_as('m'), self.scanner.gradient_strength.m_as('T/m')]),'T/m')

    # RF pulse durations based on required bandwidth
    dur1 = ((self.rf.NbLobes[0]+1)/self.bandwidth).to('ms')
    dur2 = ((self.rf.NbLobes[1]+1)/self.bandwidth).to('ms')
    rf_ss = RF(scanner=self.scanner, NbLobes=self.rf.NbLobes, alpha=self.rf.alpha, shape=self.rf.shape, flip_angle=self.rf.flip_angle, dur=dur1+dur2, t_ref=self.rf.t_ref)

    # # Calculate area needed for dephasing condition
    # area  = 1.0/(self.delta_z * self.Gss.scanner.gammabar) # Ec. 16.5 in Brown's book
    # slope = self._Gz/self.Gss.Gr_sr_
    # dur   = (area - 0.5*slope*self._Gz)/self._Gz

    # Create slice selection gradient objects
    dephasing = Gradient(G=Gz.to('mT/m'), scanner=self.scanner, lenc=(dur1 + dur2).to('ms'), t_ref=(rf_ss.t_ref-dur1).to('ms'), axis=2)
    rephasing = Gradient(scanner=self.scanner, t_ref=dephasing.timings[-2].to('ms'), axis=2)

    # Match area lobe of the second gradient
    half_area = dephasing.area(t0=rf_ss.t_ref).to('mT*ms/m')
    rephasing.match_area(-half_area * self.refocusing_area_frac)

    # Fix timings
    dephasing.update_reference(dephasing.t_ref - dephasing.slope)
    rephasing.update_reference(rephasing.t_ref - dephasing.slope)
    self.dephasing = dephasing
    self.rephasing = rephasing
    self.rf = rf_ss

    def ss_gradient(t):
      return dephasing(t) + rephasing(t)

    # Integration bounds
    t0 = dephasing.timings[0].to('ms')
    t_bound = rephasing.timings[-1].to('ms')

    # Slice positions
    z_min = self.z0 - 2*self.delta_z
    z_max = self.z0 + 2*self.delta_z
    z_arr = np.linspace(z_min.m_as('m'), z_max.m_as('m'), self.profile_samples)

    # Bloch equation
    bloch = Bloch(gamma=self.scanner.gamma.m_as('rad/mT/ms'), z0=self.z0.m_as('m'), eval_gradient=ss_gradient, B1e=rf_ss, small_angle=self.small_angle)

    # Solve loop
    M = np.zeros([3, len(z_arr)])
    for (i, z) in enumerate(z_arr):

      # Solve
      bloch.z = z
      solver = RK45(bloch, t0.m, y0, t_bound.m, vectorized=False, first_step=self.dt.m_as('ms'), max_step=100*self.dt.m_as('ms'))

      # collect data
      t  = [t0.m,]
      B1 = [rf_ss(t0.m),]
      while solver.status not in ['finished','failed']:
        # get solution step state
        solver.step()
        t.append(solver.t)
        B1.append(rf_ss(t[-1]))

      M[:,i] = solver.y
    t  = np.array(t)
    B1 = np.array(B1)

    if self.plot and MPI_rank==0:
      # plt.rcParams['text.usetex'] = True
      plt.rcParams.update({'font.size': 16})
      
      # Plot RF and slice selection gradients
      fig, ax = plt.subplots(2, 1, figsize=(12, 4))
      ax[0].plot(t, B1)
      ax[0].set_xlim([t0.m, t_bound.m])
      ax[0].legend(['B1'])
      ax[0].set_ylabel('RF [mT/m]?')
      ax[0].set_xticklabels([])

      ax[1].plot(dephasing.timings.m_as('ms'), dephasing.amplitudes.m_as('mT/m'))
      ax[1].plot(rephasing.timings.m_as('ms'), rephasing.amplitudes.m_as('mT/m'))
      ax[1].set_xlim([t0.m, t_bound.m])
      ax[1].set_xlabel('Time [{:%s}]'.format(rephasing.timings.units))
      ax[1].set_ylabel('G [{:%s}]'.format(rephasing.amplitudes.units))
      ax[1].legend(['Dephasing','Rephasing'])
      fig.tight_layout()

      # Plot slice profiles
      fig, ax = plt.subplots(1, 2, figsize=(12, 4))
      ax[0].plot(z_arr, M[0,:])
      ax[0].plot(z_arr, M[1,:])
      ax[0].plot(z_arr, np.abs(M[0,:] + 1j*M[1,:]))
      ax[0].legend(['$M_x$','$M_y$','$M_{xy}$'])
      ax[0].vlines(x=[self.z0.m - 0.5*self.delta_z.m, self.z0.m + 0.5*self.delta_z.m], ymin=0, ymax=M[1,:].max(), colors='r', linestyles='dashed')
      ax[0].set_xlabel('z coordinate [m]')
      ax[0].set_ylabel('Magnetization')

      ax[1].plot(z_arr, M[2,:])
      ax[1].legend(['$M_z$'])
      ax[1].set_xlabel('z coordinate [m]')
      ax[1].set_ylabel('Magnetization')
      fig.tight_layout()
      plt.show()

    # Interpolator
    p = M[1,:].astype(self.dtype) + 1j*M[0,:].astype(self.dtype)
    interp_profile = interp1d(z_arr.astype(self.dtype), p, kind='linear', bounds_error=False, fill_value=0.0)

    return interp_profile


class VelocityEncoding:
  def __init__(self, VENC, directions, dtype=np.float32):
    """
    Initializes the MRImaging object with VENC and directions.

    Parameters:
      VENC (float or list of floats): The velocity encoding (VENC) value(s). If a single float is provided, it will be broadcasted to match the number of directions. If a list is provided, its length must match the number of directions.
      directions (numpy.ndarray): A 2D array representing the directions. The number of rows should match the number of VENC values.

    Raises:
      ValueError: If the length of VENC does not match the number of directions.
    """
    if isinstance(VENC, float):
      self.VENC = (np.ones(directions.shape[0])*VENC).astype(dtype)
    elif len(VENC) == directions.shape[0]:
      self.VENC = (np.array(VENC)).astype(dtype)
    else:
      raise ValueError('VENC and directions must have the same length')
    self.directions = directions.astype(dtype)
    self.nb_directions = directions.shape[0]
    self.normalize_directions()
    self.dtype = dtype

  def normalize_directions(self):
    """
    Normalize the directions vector.

    This method normalizes the `directions` attribute of the object by dividing it by its L2 norm (Euclidean norm). This ensures that the `directions` vector has a unit length.

    Returns:
      None
    """
    for i in range(self.nb_directions):
      norm = np.linalg.norm(self.directions[i,:], 2)
      if norm != 0:
        self.directions[i,:] /= norm

  def encode(self, velocity, delta_phi = 0.0):
    """
    Encodes the given velocity using the phase contrast MRI method.

    Parameters:
    velocity (numpy.ndarray): A numpy array representing the velocity vector.

    Returns:
    numpy.ndarray: The encoded velocity as a numpy array.
    """
    phi_v = np.zeros([velocity.shape[0], self.nb_directions], dtype=self.dtype)
    for i in range(self.nb_directions):
      phi_v[:,i] = np.pi * np.dot(velocity, self.directions[i,:].T) / self.VENC[i] + delta_phi
    return  phi_v


class PositionEncoding:
  def __init__(self, ke, directions, dtype=np.float32):
    """
    Initializes the MRImaging object with VENC and directions.

    Parameters:
      ke (float or list of floats): The position encoding (ke) value(s). If a single float is provided, it will be broadcasted to match the number of directions. If a list is provided, its length must match the number of directions.
      directions (numpy.ndarray): A 2D array representing the directions. The number of rows should match 
                    the number of VENC values.

    Raises:
      ValueError: If the length of VENC does not match the number of directions.
    """
    if isinstance(ke, float):
      self.ke = (np.ones(directions.shape[0])*ke).astype(dtype)
    elif len(ke) == directions.shape[0]:
      self.ke = np.array(ke).astype(dtype)
    else:
      raise ValueError('ke and directions must have the same length')
    self.directions = directions
    self.nb_directions = directions.shape[0]
    self.normalize_directions()
    self.dtype = dtype

  def normalize_directions(self):
    """
    Normalize the directions vector.

    This method normalizes the `directions` attribute of the object by dividing it by its L2 norm (Euclidean norm). This ensures that the `directions` vector has a unit length.

    Returns:
      None
    """
    for i in range(self.nb_directions):
      norm = np.linalg.norm(self.directions[i,:], 2)
      if norm != 0:
        self.directions[i,:] /= norm

  def encode(self, displacement):
    """
    Encodes the given velocity using the phase contrast MRI method.

    Parameters:
    velocity (numpy.ndarray): A numpy array representing the velocity vector.

    Returns:
    numpy.ndarray: The encoded velocity as a numpy array.
    """
    phi_x = np.zeros([displacement.shape[0], self.nb_directions])
    for i in range(self.nb_directions):
      phi_x[:,i] = np.dot(displacement, self.directions[i,:].T) * self.ke[i]
    return phi_x