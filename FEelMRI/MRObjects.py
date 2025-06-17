import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity as Q_
from scipy.interpolate import interp1d

from FEelMRI.MPIUtilities import MPI_rank


# Scanner class
class Scanner:
  """
  A class to represent an MRI Scanner.

  Attributes
  ----------
  field_strength : float
    The magnetic field strength of the scanner in Tesla (default is 1.5 T).
  gradient_strength : float
    The gradient strength of the scanner in millitesla per meter (default is 33 mT/m).
  gradient_slew_rate : float
    The gradient slew rate of the scanner in millitesla per meter per millisecond (default is 200 mT/m/ms).
  gammabar : float
    The gyromagnetic ratio in Hz/T (default is 42.58e6 Hz/T).

  Methods
  -------
  __init__(self, field_strength=1.5, gradient_strength=33, gradient_slew_rate=200)
    Initializes the Scanner with the given field strength, gradient strength, and gradient slew rate.
  """
  def __init__(self, field_strength: Q_ = Q_(1.5, 'T'), 
              gradient_strength: Q_ = Q_(33,'mT/m'),
              gradient_slew_rate: Q_ = Q_(180,'mT/m/ms')):
    self.field_strength = field_strength
    self.gradient_strength = gradient_strength
    self.gradient_slew_rate = gradient_slew_rate
    self.gammabar = Q_(42.58e6, 'Hz/T')
    self.gamma = Q_(42.58e6*2*np.pi, 'rad*Hz/T')


# Gradient
class Gradient:
  def __init__(self, slope=None, lenc=Q_(1.0,'ms'), G=None, scanner=Scanner(), t_ref=Q_(0.0,'ms'), axis=0):
    self.scanner = scanner
    self.Gr_max = scanner.gradient_strength  # [mT/m]
    self.Gr_sr = scanner.gradient_slew_rate  # [mT/m/ms]
    self.G = self.Gr_max if G is None else G # [mT/m]
    self.lenc = lenc                         # [ms]
    self.slope = np.abs(self.G)/self.Gr_sr if slope is None else slope # [ms]
    self.t_ref = t_ref                       # [ms]
    self.dur, self.timings, self.amplitudes, self.interpolator = self.group_timings()
    self.axis = axis                         # M: 0, P: 1, S: 2

  def __copy__(self):
    return copy.deepcopy(self)

  def __repr__(self):
    return f"Gradient(slope={self.slope}, lenc={self.lenc}, G={self.G}, Gr_max={self.Gr_max}, Gr_sr={self.Gr_sr}, t_ref={self.t_ref})"

  def __call__(self, t):
    return self.interpolator(t)
  
  def __mul__(self, other):
    """
    Multiply the gradient by a scalar.
    This method allows for scaling the gradient amplitude by a scalar value.
    Parameters:
    other (float): The scalar value to multiply the gradient by.
    Returns:
    Gradient: A new Gradient object with the scaled amplitude.
    """
    if isinstance(other, (int, float)):
      return Gradient(slope=self.slope, lenc=self.lenc, G=self.G * other, scanner=self.scanner, t_ref=self.t_ref, axis=self.axis)
    else:
      raise TypeError("Gradient can only be multiplied by a scalar (int or float).")

  def evaluate(self, t):
    """
    Evaluate the gradient interpolator at a given time point.

    Parameters:
    t (float): The time point at which to evaluate the interpolator.

    Returns:
    float: The interpolated value at the given time point.
    """
    return self.interpolator(t)

  def group_timings(self):
    """
    Groups the timing and amplitude values for gradient evaluation based on the 
    length of the encoding (lenc) and slope parameters.

    If `self.lenc` is less than or equal to 0.0, the timings and amplitudes are 
    calculated for a simple slope. Otherwise, they are calculated for a slope 
    with an encoding length.

    Timings and amplitudes are adjusted by the reference time (`self.t_ref`).

    Returns:
      tuple: A tuple containing:
        - timings (np.ndarray): Array of timing values.
        - amplitudes (np.ndarray): Array of amplitude values.
        - interp (interp1d): Interpolator for gradient evaluation.
    """
    # TODO
    if self.lenc <= 0.0:
      timings = Q_(np.array([0.0, 
                          self.slope.m,
                          self.slope.m+self.slope.m]), self.slope.u)
      amplitudes = Q_(np.array([0.0, 
                            self.G.m,
                            0.0]), self.G.u)
    else:
      timings = Q_(np.array([0.0, 
                          self.slope.m, 
                          self.slope.m+self.lenc.m, 
                          self.slope.m+self.lenc.m+self.slope.m]), self.slope.u)
      amplitudes = Q_(np.array([0.0, 
                            self.G.m, 
                            self.G.m, 
                            0.0]), self.G.u)

    # Duration
    dur = timings[-1]

    # Add reference time
    timings += self.t_ref

    # Define interpolator for gradient evaluation
    interp = interp1d(timings.m, amplitudes.m, kind='linear', fill_value=0.0, bounds_error=False)

    return dur, timings, amplitudes, interp
  
  def update_reference(self, t_ref):
    """
    Update the reference time for the gradient.

    Parameters:
    t_ref (float): The new reference time.

    Returns:
    None
    """
    self.t_ref = t_ref
    self.dur, self.timings, self.amplitudes, self.interpolator = self.group_timings()

  def calculate(self, k_bw, receiver_bw=None, ro_samples=None, ofac=None):
    """
    Calculate gradient based on slope, slew rate, and amplitude parameters.

    Parameters:
    -----------
    k_bw : float
      Bandwidth in 1/m.
    receiver_bw : float, optional
      Receiver bandwidth in Hz. If provided, lenc is fixed and calculated accordingly.
    ro_samples : int, optional
      Number of readout samples.
    ofac : float, optional
      Oversampling factor.

    Returns:
    --------
    None

    Notes:
    ------
    This method calculates the gradient parameters including slope, gradient amplitude (G),
    and length (lenc) based on the provided bandwidth and optional receiver bandwidth.
    If the receiver bandwidth is provided, the length is fixed and calculated accordingly.
    Otherwise, the method calculates the parameters assuming only the ramps are needed.

    The calculated parameters are stored in the instance variables:
    - self.lenc : Length in ms
    - self.G : Gradient amplitude in mT/m
    - self.slope : Slope in ms
    - self.dur : Duration in ms

                                             __________
                _1_ /|\                    /|          |\
     slew rate |  /  |  \                /  |          |  \
               |/    G    \            /    G          G    \
              /      |      \        /      |          |      \
               slope   slope          slope     lenc     slope

    The method also updates the timings and amplitudes arrays using the group_timings method.
    """

    # Calculate gradient
    if receiver_bw is not None:
      ''' If the receiver bandwidth is given, the lenc should be fixed and
      calculated accordingly
      '''
      self.lenc  = (ro_samples/ofac)/receiver_bw.to('1/ms')
      self.G     = k_bw.to('1/m')/(self.scanner.gammabar.to('1/mT/ms')*self.lenc.to('ms'))
      if self.G > self.Gr_max:
        self.G = self.Gr_max
        self.lenc = k_bw.to('1/m')/(self.scanner.gammabar.to('1/mT/s')*self.G.to('mT/m'))
        receiver_bw = ((ro_samples/ofac)/self.lenc).to('Hz')
        if MPI_rank == 0:
          warnings.warn("The required gradient amplitude exceeds the maximum allowed. Adjusting to maximum gradient amplitude. Receiver bandwidth allowed with current configuration is {:.0f} Hz".format(receiver_bw.m_as('Hz')))
      self.slope = np.abs(self.G)/self.Gr_sr
    else:
      ''' Calcualte everything as if only the ramps are needed '''
      # Time needed to reach the maximun amplitude
      slope_req = np.sqrt(np.abs(k_bw.to('1/m'))/(self.scanner.gammabar.to('1/mT/ms')*self.Gr_sr.to('mT/m/ms')))

      # Time needed for the maximun amplitude
      slope = self.Gr_max/self.Gr_sr

      # Build gradient
      if slope_req < slope:
        self.slope = slope_req
        self.G     = self.Gr_sr*slope_req
        self.lenc  = self.slope - slope_req
      else:
        # Assign slope and gradient amplitude
        self.slope = slope
        self.G     = self.Gr_max

        # Calculate lenc
        k_bw_slopes = self.scanner.gammabar.to('1/mT/ms')*self.Gr_sr.to('mT/m/ms')*slope.to('ms')**2
        self.lenc = (np.abs(k_bw.to('1/m')) - k_bw_slopes.to('1/m'))/(self.G.to('mT/m')*self.scanner.gammabar.to('1/mT/ms'))

      # Account for area sign
      self.G *= np.sign(k_bw)

    # Update gradient duration
    if self.lenc < 0:
      self.dur = self.slope + self.slope
    else:
      self.dur = self.slope + self.lenc + self.slope

    # Update timings and amplitudes in array
    self.dur, self.timings, self.amplitudes, self.interpolator = self.group_timings()

  def make_bipolar(self, VENC: Q_):
    ''' Calculate the time needed to apply the velocity encoding gradients
    based on the values of Gr_max and Gr_sr'''

    # Store sign of VENC and make it positive to do all calculations
    VENC_sign = np.sign(VENC)
    VENC = np.abs(VENC)

    # The phase generated by the bipolar gradients is given by:
    #   phi = gamma*v*\int_0^{2*slope}{G(t)*t}*dt
    # If the both gradient lobes are traingular, the phase is given by:
    #   phi = 2*G*gamma*v*slope^2 = (2*slope)*gamma*v*(G*slope) = dt*gamma*v*Ag
    # where Ag is the area of the gradient and dt is the duration of one lobe 
    # of the bipolar gradients

    # Bipolar lobes areas without rectangle part
    # If lenc = 0, pi = 2*gamma*G(t)*VENC*slope^2
    # which is equivalent to: pi = 2*gamma*SR*VENC*slope^3
    slope = (self.Gr_max/self.Gr_sr).to('ms')
    slope_req = np.cbrt(Q_(np.pi,'rad')/(2*self.scanner.gamma.to('rad/ms/mT')*self.Gr_sr.to('mT/m/ms')*VENC.to('m/ms')))

    # Check if rectangle parts of the gradient are needed
    if slope_req <= slope:
      # Calculate duration of the first velocity encoding gradient lobe
      self.slope = slope_req.to('ms')
      self.G     = -self.Gr_sr.to('mT/m/ms')*slope_req.to('ms')
      self.lenc  = self.slope.to('ms') - slope_req.to('ms')
    else:
      # Calculate lenc based on the intergral of the velocity encoding gradients
      a = self.scanner.gamma.to('rad/ms/mT')*VENC.to('m/ms')*self.Gr_max.to('mT/m')
      b = 3*self.scanner.gamma.to('rad/ms/mT')*VENC.to('m/ms')*self.Gr_max.to('mT/m')*slope.to('ms')
      c = 2*self.scanner.gamma.to('rad/ms/mT')*VENC.to('m/ms')*self.Gr_max.to('mT/m')*slope.to('ms')**2 - np.pi
      lenc_req = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

      # Gradients parameters
      self.slope = slope.to('ms')
      self.G     = -self.Gr_max.to('mT/m') * VENC_sign
      self.lenc  = lenc_req.to('ms')

    # Upadte duration
    if self.lenc < 0:
      self.dur =(self.slope + self.slope).to('ms')
    else:
      self.dur =(self.slope + self.lenc + self.slope).to('ms')

    # Update timings and amplitudes in array
    self.dur, self.timings, self.amplitudes, self.interpolator = self.group_timings()

    # Create the second gradient lobe
    g = self.__copy__()
    g.G = -self.G
    g.update_reference(self.t_ref + self.dur)

    return g

  def area(self, t0: Q_ = Q_(0.0, 'ms'), nb_samples=1000):
    ''' Calculate the area of the gradient '''
    # Calculate area
    time = np.linspace(t0.m_as('ms'), (self.t_ref + self.dur).m_as('ms'), nb_samples)
    area = np.trapz(self.interpolator(time), time)

    return Q_(area, 'mT*ms/m')

  def match_area(self, area):
    # # Gradient area without rectangle part
    # current_area = self.G*self.slope

    # if current_area < area:
    #   # Add required rectangle area
    #   req_area = area - current_area
    #   self.lenc = req_area/self.G
    # # else:
    # #   # Remove remaining triangle area
    # #   self.G = area/self.slope
    # #   slope_req_ = area/self.G

    # Check sign and store it
    area_sign = np.sign(area)
    area = np.abs(area)

    # Minimun slope posible to achieve tha maximun gradient amplitude
    slope_min = self.Gr_max.to('mT/m')/self.Gr_sr.to('mT/m/ms')

    # Gradient area using maximun amplitude and slewrate (if lenc = 0, area = G*slope)
    area_max = slope_min*self.Gr_max.to('mT/m')

    # Check if rectangle parts of the gradient are needed
    if area.to('mT*ms/m') <= area_max.to('mT*ms/m'):
      # Calculate needed gradient amplitude
      ratio = area/area_max
      self.slope = slope_min.to('ms')*np.sqrt(ratio)
      self.G     = self.Gr_max.to('mT/m')*np.sqrt(ratio)
      self.lenc  = self.slope.to('ms') - slope_min.to('ms')*np.sqrt(ratio) 
    else:
      # Calculate missing area needed
      area_needed = area.to('mT*ms/m') - area_max.to('mT*ms/m')

      # Estimate remaining area 
      # If lenc != 0:
      #     area = area_ramps + Gmax*lenc
      # Gradients parameters
      self.slope = slope_min.to('ms')
      self.G     = self.Gr_max.to('mT/m')
      self.lenc  = area_needed.to('mT*ms/m')/self.Gr_max.to('mT/m')

    # Upadte duration
    if self.lenc < 0:
      self.dur = 2*self.slope.to('ms')
    else:
      self.dur = 2*self.slope.to('ms') + self.lenc.to('ms')

    # Restore sign
    self.G *= area_sign

    # Update timings and amplitudes in array
    _, self.timings, self.amplitudes, self.interpolator = self.group_timings()

  def change_dur(self, new_dur: Q_):
    """
    Change the duration of the gradient while keeping the area.

    Parameters:
    dur (float): The new duration in milliseconds.

    Returns:
    None
    """
    if new_dur.to('ms') > self.dur.to('ms'):
      # Calculate new gradient amplitude and lenc
      self.G = self.area()/new_dur.to('ms')
      self.lenc = new_dur.to('ms') - 2*self.slope.to('ms')
      self.dur = new_dur.to('ms')

      # Update timings and amplitudes in array
      _, self.timings, self.amplitudes, self.interpolator = self.group_timings()

    elif new_dur.to('ms') == self.dur.to('ms'):
      # If the duration is the same, do nothing
      if MPI_rank == 0:
        warnings.warn("The gradient duration is already set to the requested value. No changes made.")  

    else:
      # At the moment, we do not allow to change the duration to a smaller value as the gradients are calculated considering the maximum gradient amplitude and slew rate
      if MPI_rank == 0:
        warnings.warn("Changing the gradient duration to a smaller value is not allowed. The gradient parameters are calculated considering the maximum gradient amplitude and slew rate. Gradient duration remains unchanged.")

  def rotate(self, direction: np.ndarray):
    """
    Rotate the gradient direction.

    Parameters:
    direction (np.ndarray): The new direction vector for the gradient.

    Returns:
    None
    """
    if direction.shape != (3,):
      raise ValueError("Direction must be a 3-element vector with [M, P, S] directions.")

    # Normalize the direction vector
    norm = np.linalg.norm(direction)
    if norm != 0:
      direction = direction / norm

    # Gradient area and sign
    area = self.area()

    # Create gradients around each axis given by the direction vector
    gradients = []
    if norm == 0:
      g = self.__copy__() * 0.0
      gradients.append(g)
    else:
      for i, axis in enumerate(direction):
        if axis != 0:
          g = self.__copy__()
          g.match_area(axis*area)
          g.axis = i
          gradients.append(g)

    # Maximun gradient duration
    max_dur = max([g.dur for g in gradients])

    # Update all gradients to have the same duration
    [g.change_dur(max_dur) for g in gradients]

    return gradients

  def plot(self, linestyle='-', axes=[]):
    ''' Plot gradient '''
    fig = plt.figure()
    plt.plot(self.timings, self.amplitudes, linestyle)
    plt.show()

    return fig


class RF:
  def __init__(self, scanner=Scanner(), NbLobes=[2,2], alpha=0.46, shape='apodized_sinc', flip_angle=Q_(np.pi/2,'rad'), dur=Q_(2.0,'ms'), t_ref=Q_(0.0,'ms'), nb_samples=1000):
    self.scanner = scanner
    self.NbLobes = NbLobes
    self.alpha = alpha
    self.shape = shape
    if self.shape == 'sinc':
      self._pulse = self._rf_apodized_sinc
      if self.alpha != 0.0 and MPI_rank == 0:
        warnings.warn("For 'sinc' shape, the alpha parameter is automatically set to 0.0")
      self.alpha = 0.0
    elif self.shape == 'apodized_sinc':
      self._pulse = self._rf_apodized_sinc
    elif self.shape == 'hard':
      self._pulse = self._rf_hard
    self.flip_angle = flip_angle.to('rad')
    self.t_ref = t_ref.to('ms')
    self.dur1 = (self.NbLobes[0] + 1)/(np.sum(self.NbLobes) + 2)*dur.to('ms')
    self.dur2 = (self.NbLobes[1] + 1)/(np.sum(self.NbLobes) + 2)*dur.to('ms')
    self.dur = dur.to('ms')
    self.nb_samples = nb_samples
    self.interpolator = self._get_interpolator()

  def __str__(self):
    return f"RF(NbLobes={self.NbLobes}, alpha={self.alpha}, shape={self.shape}, flip_angle={self.flip_angle}, dur={self.dur}, t_ref={self.t_ref}, nb_samples={self.nb_samples})"

  def __copy__(self):
    return copy.deepcopy(self)

  def __repr__(self):
    return self.__str__()

  def __call__(self, t):
    return self.interpolator(t)

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
    # Maximun number of lobes
    N = np.max([self.NbLobes[0], self.NbLobes[1]])

    # RF pulse definition
    bw = (self.NbLobes[0] + self.NbLobes[1] + 2)/self.dur.to('ms')
    B1e = (1/bw.m)
    B1e *= ((1-self.alpha) + self.alpha*np.cos(np.pi*bw.m*t/N))
    B1e *= np.sinc(bw.m*t)
    B1e *= (t >= -self.dur1.m)*(t <= self.dur2.m)

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
    # RF pulse definition
    B1e = 1.0*(t >= -self.dur1.m)*(t <= self.dur2.m)

    return B1e

  def _flip_angle_factor(self):
    # Timings
    dur1 = self.dur1.m_as('ms')
    dur2 = self.dur2.m_as('ms')
    time = np.linspace(-dur1, dur2, self.nb_samples)
    dt = time[1] - time[0]

    # Calculate factor to accoundt for flip angle
    flip_angle_factor = self.flip_angle.m_as('rad')/(2.0*np.pi*self.scanner.gammabar.m_as('1/mT/ms') * self._pulse(time).sum() * dt)

    return flip_angle_factor

  def _get_interpolator(self):
    time = np.linspace(-self.dur1.m_as('ms'), self.dur2.m_as('ms'), self.nb_samples)
    interpolator = interp1d(time + self.t_ref.m_as('ms'), self._flip_angle_factor()*self._pulse(time), kind='linear', fill_value=0.0, bounds_error=False)
    return interpolator
  
  def update_reference(self, t_ref):
    """
    Update the reference time for the RF pulse.

    Parameters:
    t_ref (float): The new reference time.

    Returns:
    None
    """
    self.t_ref = t_ref
    self.interpolator = self._get_interpolator()