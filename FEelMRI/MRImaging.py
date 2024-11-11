import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import RK45
from scipy.interpolate import interp1d

from FEelMRI.Helpers import order
from FEelMRI.KSpaceTraj import Gradient
from FEelMRI.MPIUtilities import MPI_rank
import warnings


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
    ZPoints (int): Number of points in the slice profile (default is 150).
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
  def __init__(self, z0=0.0, delta_z=0.008, gammabar=42.58, bandwidth='maximum', RFShape='sinc', NbLobes=[2, 2], alpha=0.46, flip_angle=np.deg2rad(10.0), dt=1e-4, ZPoints=150, plot=False, small_angle=False, refocusing_area_frac=0.5, Gss = Gradient(Gr_max=30.0, Gr_sr=195.0, gammabar=42.58)):
    self.z0 = z0
    self.delta_z = delta_z
    self.gammabar = gammabar                # [MHz/T]
    self._gammabar = 1e+6*gammabar          # [Hz/T]
    self.Gss = Gss
    self.bandwidth = self._check_bandwidth(bandwidth)
    self._Gz = 0.0
    self.RFShape = RFShape
    self.NbLobes = NbLobes
    self.alpha = alpha # 0.46 for Hamming and 0.5 for Hanning
    self.flip_angle = flip_angle
    self.ZPoints = ZPoints
    self.plot = plot
    self.z_rk = self.z0 # temporary assignment
    self.dt = dt
    self.small_angle = small_angle
    self.refocusing_area_frac = refocusing_area_frac
    if self.RFShape == 'sinc':
      self.RFPulse = self._rf_apodized_sinc
      if self.alpha != 0.0 and MPI_rank == 0:
        warnings.warn("For 'sinc' RFShape, the alpha parameter is automatically set to 0.0")
      self.alpha = 0.0
    elif self.RFShape == 'apodized_sinc':
      self.RFPulse = self._rf_apodized_sinc
    elif self.RFShape == 'hard':
      self.RFPulse = self._rf_hard
    self.ref_time = 0.0
    self.rf_dur = 0.0
    self.rf_dur2 = 0.0
    self.interp_profile = self.calculate()

  def _check_bandwidth(self, bandwidth):
    """
    Check the bandwidth of the RF pulse.

    Raises:
    ValueError: If the bandwidth is less than the minimum required.
    """
    if bandwidth == 'maximum':
      return self._gammabar*self.Gss._Gr_max*self.delta_z
    elif bandwidth > self._gammabar*self.Gss._Gr_max*self.delta_z:
      raise ValueError('Bandwidth is greater than the maximum allowed by the slice selection gradient and desired slice thickness')
    else:
      return bandwidth

  def _rf_apodized_sinc(self, t):
    """
    Generate an apodized sinc RF pulse.

    Parameters:
    t (numpy.ndarray): Time array over which the RF pulse is calculated.

    Returns:
    numpy.ndarray: The calculated RF pulse values.

    Description:
    This method calculates an apodized sinc RF pulse based on the given time array `t`. 
    The pulse is defined by several parameters:
    - `delta_f`: Pulse frequency needed for the desired slice thickness.
    - `tau_l`: Pulse duration and time window for the left lobes.
    - `tau_r`: Pulse duration and time window for the right lobes.
    - `N`: Maximum number of lobes between left and right.

    The RF pulse `B1e` is computed using a combination of a sinc function and a cosine apodization 
    window, constrained within the time window defined by `tau_l` and `tau_r`.
    """
    # Calculate gradient amplitude needed for the desired slice thickness
    self._Gz = np.min([self.bandwidth/(self._gammabar*self.delta_z), self.Gss._Gr_max])

    # Pulse duration and time window
    tau_l = (self.NbLobes[0]+1)/self.bandwidth
    tau_r = (self.NbLobes[1]+1)/self.bandwidth

    # Maximun number of lobes
    N = np.max([self.NbLobes[0], self.NbLobes[1]])

    # RF pulse definition
    B1e = (1/self.bandwidth)*((1-self.alpha) + self.alpha*np.cos(np.pi*self.bandwidth*t/N))*np.sinc(self.bandwidth*t)*(t >= -tau_l)*(t <= tau_r)

    return B1e

  def _rf_hard(self, t):
    """
    Generate a hard RF pulse for MRI imaging.

    Parameters:
    -----------
    t : float
      The time at which the RF pulse is evaluated.

    Returns:
    --------
    B1e : float
      The amplitude of the RF pulse at time t.

    Notes:
    ------
    - The pulse frequency is determined by the desired slice thickness.
    - The pulse duration is calculated based on the number of left and right lobes.
    - The RF pulse is defined as a rectangular pulse within the calculated time window.
    """
    # Calculate gradient amplitude needed for the desired slice thickness
    self._Gz = np.min([self.bandwidth/(self._gammabar*self.delta_z), self.Gss._Gr_max])

    # Pulse duration and time window
    tau_l = (self.NbLobes[0]+1)/self.bandwidth
    tau_r = (self.NbLobes[1]+1)/self.bandwidth

    # RF pulse definition
    B1e = 1.0*(t >= -tau_l)*(t <= tau_r)

    return B1e

  def bloch(self, t, M):
    """
    Compute the time derivative of the magnetization vector M according to the Bloch equations.

    Parameters:
    -----------
    t : float
      Time variable.
    M : array-like
      Magnetization vector [Mx, My, Mz].

    Returns:
    --------
    numpy.ndarray
      The time derivative of the magnetization vector [dMx/dt, dMy/dt, dMz/dt].

    Notes:
    ------
    - The gyromagnetic constant is calculated using the class attribute `self._gammabar`.
    - The frequency offset `dw` is computed using the gradient field and the positions `self.z_rk` and `self.z0`.
    - If `self.small_angle` is True, the small angle approximation is used for the Bloch equations.
    - The function `self.B1e_norm(t)` is used to compute the normalized B1 field at time `t`.
    """
    # Gyromagnetic constant
    gamma = 2.0*np.pi*self._gammabar

    # Frequency offset
    dw = gamma*1e-03*self.ss_gradient(1000.0*t)*(self.z_rk - self.z0)

    # Bloch equations
    if self.small_angle:
      dMxdt = dw*M[1]
      dMydt = gamma*self.B1e_norm(t)*M[2] - dw*M[0]
      dMzdt = -gamma*self.B1e_norm(t)*M[1]*0.0
    else:
      dMxdt = dw*M[1]
      dMydt = gamma*self.B1e_norm(t)*M[2] - dw*M[0]
      dMzdt = -gamma*self.B1e_norm(t)*M[1]

    return np.array([dMxdt, dMydt, dMzdt]).reshape((3,))

  def ss_gradient(self, t):
    return self.g1.evaluate(t) - self.g2.evaluate(t)

  def B1e_norm(self, t):
    """
    Calculate the normalized B1 field at a given time.

    This method computes the normalized B1 field by multiplying the RF pulse 
    at time `t` with the flip angle factor.

    Parameters:
    t (float): The time at which to evaluate the RF pulse.

    Returns:
    float: The normalized B1 field at the specified time.
    """
    return self.RFPulse(t)*self.flip_angle_factor

  def optimize(self, frac_start=1.0, frac_end=2.0, N=10, ZPoints=100):
    """
      Optimize the refocusing area fraction for a given number of points.
    """
    # Replace ZPoints and plot (temporary)
    _ZPoints = self.ZPoints
    _plot = self.plot
    self.ZPoints = ZPoints
    self.plot = False

    # Slice positions
    z_min = self.z0 - 2*self.delta_z
    z_max = self.z0 + 2*self.delta_z
    z_arr = np.linspace(z_min, z_max, self.ZPoints)

    # Optimize refocusing area fraction
    Mx = []
    area_fracs = np.linspace(frac_start, frac_end, N)
    for i, frac in enumerate(area_fracs):
      self.refocusing_area_frac = frac
      self.interp_profile = self.calculate()
      Mx.append(np.abs(np.imag(self.interp_profile(z_arr))).max())
      if MPI_rank == 0:
        print("[iter ", i, "]", "Refocusing area fraction: ", frac, "Max. Mx : ", Mx[-1])

    # Verify which fraction gives the best result
    idx = np.argmin(Mx)
    self.refocusing_area_frac = area_fracs[idx]
    self.interp_profile = self.calculate()

    if MPI_rank == 0:
      print("Optimal refocusing area fraction: ", self.refocusing_area_frac)

    # ZPoints back to original value
    self.ZPoints = _ZPoints
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

    # Calculate gradient amplitude needed for the desired slice thickness
    self._Gz = np.min([self.bandwidth/(self._gammabar*self.delta_z), self.Gss._Gr_max])
    self.Gz = self._Gz * 1e+3

    # RF pulse bounds
    tau_l = (self.NbLobes[0]+1)/self.bandwidth
    tau_r = (self.NbLobes[1]+1)/self.bandwidth

    # # Calculate area needed for dephasing condition
    # area  = 1.0/(self.delta_z * self._gammabar) # Ec. 16.5 in Brown's book
    # slope = self._Gz/self.Gss.Gr_sr_
    # dur   = (area - 0.5*slope*self._Gz)/self._Gz

    # Create slice selection gradient objects
    g1 = Gradient(G=self.Gz, Gr_max=self.Gss.Gr_max, Gr_sr=self.Gss.Gr_sr, lenc=(tau_l + tau_r)*1000.0, t_ref=-1000*tau_l)
    g2 = Gradient(Gr_max=self.Gss.Gr_max, Gr_sr=self.Gss.Gr_sr, t_ref=g1.timings[-2])
    g2.match_area(self.refocusing_area_frac*(g1.area() - (0.5*g1._slope*g1._G + tau_l*g1._G)))

    # Fix timings
    g1.t_ref -= g1.slope
    g1.timings, g1.amplitudes, _ = g1.group_timings()
    g2.t_ref -= g1.slope
    g2.timings, g2.amplitudes, _ = g2.group_timings()
    self.g1 = g1
    self.g2 = g2

    # Integration bounds
    t0 = g1.timings[0]/1000.0
    t_bound = g2.timings[-1]/1000.0

    # Store timings
    self.ref_time = 0.0
    self.rf_dur  = 1000.0*(tau_l + tau_r)
    self.rf_dur2 = 1000.0*tau_r

    # Calculate factor to accoundt for flip angle
    t = np.linspace(t0, t_bound, int((t_bound - t0)/self.dt))
    self.flip_angle_factor = self.flip_angle/(2.0*np.pi*self._gammabar*self.RFPulse(t).sum()*self.dt)

    # Slice positions
    z_min = self.z0 - 2*self.delta_z
    z_max = self.z0 + 2*self.delta_z

    z_arr = np.linspace(z_min, z_max, self.ZPoints)
    M = np.zeros([3, len(z_arr)])
    for (i, z) in enumerate(z_arr):

      # Solve
      self.z_rk = z
      solver = RK45(self.bloch, t0, y0, t_bound, vectorized=False, first_step=self.dt, max_step=100*self.dt)

      # collect data
      t  = [t0,]
      B1 = [self.B1e_norm(t0),]
      while solver.status not in ['finished','failed']:
        # get solution step state
        solver.step()
        t.append(solver.t)
        B1.append(self.B1e_norm(t[-1]))

      M[:,i] = solver.y
    t  = np.array(t)
    B1 = np.array(B1)

    if self.plot and MPI_rank==0:
      plt.rcParams['text.usetex'] = True
      plt.rcParams.update({'font.size': 16})
      
      # Plot RF and slice selection gradients
      fig, ax = plt.subplots(2, 1, figsize=(12, 4))
      ax[0].plot(1000*t, B1)
      ax[0].set_xlim([1000*t0, 1000*t_bound])
      ax[0].legend(['B1'])
      ax[0].set_ylabel('RF [mT/m]?')
      ax[0].set_xticklabels([])

      ax[1].plot(g1.timings, g1.amplitudes)
      ax[1].plot(g2.timings, -g2.amplitudes)
      ax[1].set_xlim([1000*t0, 1000*t_bound])
      ax[1].set_xlabel('Time [ms]')
      ax[1].set_ylabel('G [mT/m]')
      ax[1].legend(['Dephasing','Rephasing'])
      fig.tight_layout()

      # Plot slice profiles
      fig, ax = plt.subplots(1, 2, figsize=(12, 4))
      ax[0].plot(z_arr, M[0,:])
      ax[0].plot(z_arr, M[1,:])
      ax[0].plot(z_arr, np.abs(M[0,:] + 1j*M[1,:]))
      ax[0].legend(['$M_x$','$M_y$','$M_{xy}$'])
      ax[0].vlines(x=[self.z0 - 0.5*self.delta_z, self.z0 + 0.5*self.delta_z], ymin=0, ymax=M[1,:].max(), colors='r', linestyles='dashed')
      ax[0].set_xlabel('z coordinate [m]')
      ax[0].set_ylabel('Magnetization')

      ax[1].plot(z_arr, M[2,:])
      ax[1].legend(['$M_z$'])
      ax[1].set_xlabel('z coordinate [m]')
      ax[1].set_ylabel('Magnetization')
      fig.tight_layout()
      plt.show()

    # Interpolator
    p = M[1,:] + 1j*M[0,:]
    interp_profile = interp1d(z_arr, p, kind='linear', bounds_error=False, fill_value=0.0)

    return interp_profile


class VelocityEncoding:
  def __init__(self, VENC, directions):
    """
    Initializes the MRImaging object with VENC and directions.

    Parameters:
      VENC (float or list of floats): The velocity encoding (VENC) value(s). If a single float is provided, it will be broadcasted to match the number of directions. If a list is provided, its length must match the number of directions.
      directions (numpy.ndarray): A 2D array representing the directions. The number of rows should match the number of VENC values.

    Raises:
      ValueError: If the length of VENC does not match the number of directions.
    """
    if isinstance(VENC, float):
      self.VENC = np.ones(directions.shape[0])*VENC
    elif len(VENC) == directions.shape[0]:
      self.VENC = np.array(VENC)
    else:
      raise ValueError('VENC and directions must have the same length')
    self.directions = directions
    self.nb_directions = directions.shape[0]
    self.normalize_directions()

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
    phi_v = np.zeros([velocity.shape[0], self.nb_directions])
    for i in range(self.nb_directions):
      phi_v[:,i] = np.pi * np.dot(velocity, self.directions[i,:].T) / self.VENC[i] + delta_phi
    return  phi_v


class PositionEncoding:
  def __init__(self, ke, directions):
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
      self.ke = np.ones(directions.shape[0])*ke
    elif len(ke) == directions.shape[0]:
      self.ke = np.array(ke)
    else:
      raise ValueError('ke and directions must have the same length')
    self.directions = directions
    self.nb_directions = directions.shape[0]
    self.normalize_directions()

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