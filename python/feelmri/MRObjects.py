import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity
from scipy.interpolate import interp1d

from feelmri.MPIUtilities import MPI_print, MPI_rank


# Scanner class
class Scanner:
  """
  A class to represent an MRI Scanner.

  Attributes
  ----------
  field_strength: float
    The magnetic field strength of the scanner in Tesla (default is 1.5 T).
  gradient_strength: float
    The gradient strength of the scanner in millitesla per meter (default is 33 mT/m).
  gradient_slew_rate: float
    The gradient slew rate of the scanner in millitesla per meter per millisecond (default is 200 mT/m/ms).
  gammabar: float
    The gyromagnetic ratio in Hz/T (default is 42.58e6 Hz/T).

  Methods
  -------
  __init__(self, field_strength=1.5, gradient_strength=33, gradient_slew_rate=200)
    Initializes the Scanner with the given field strength, gradient strength, and gradient slew rate.
  """
  def __init__(self, field_strength: Quantity = Quantity(1.5, 'T'), 
              gradient_strength: Quantity = Quantity(33,'mT/m'),
              gradient_slew_rate: Quantity = Quantity(180,'mT/m/ms')):
    self.field_strength = field_strength
    self.gradient_strength = gradient_strength
    self.gradient_slew_rate = gradient_slew_rate
    self.gammabar = Quantity(42.58e6, 'Hz/T')
    self.gamma = Quantity(42.58e6*2*np.pi, 'rad*Hz/T')


# Gradient
class Gradient:
  def __init__(self, slope=None, 
              lenc=Quantity(1.0,'ms'), 
              strength=None, 
              scanner=Scanner(), 
              ref=Quantity(0.0,'ms'), 
              time=Quantity(0.0,'ms'), 
              axis=0):
    self.scanner = scanner
    self.Gr_max = scanner.gradient_strength  # [mT/m]
    self.Gr_sr = scanner.gradient_slew_rate  # [mT/m/ms]
    self.strength = self.Gr_max if strength is None else strength # [mT/m]
    self.lenc = lenc                         # [ms]
    self.slope = np.abs(self.strength)/self.Gr_sr if slope is None else slope # [ms]
    self.ref = ref.to('ms')
    self.time = time.to('ms')
    self.dur = (self.slope + self.lenc + self.slope if self.lenc > 0 else 2*self.slope).to('ms')
    self.dur2 = (self.dur - self.ref).to('ms')
    self.timings, self.amplitudes, self.interpolator = self.group_timings()
    self.axis = axis                         # M: 0, P: 1, S: 2

  def __copy__(self):
    return copy.deepcopy(self)

  def __repr__(self):
    return f"Gradient(slope={self.slope}, lenc={self.lenc}, strength={self.strength}, Gr_max={self.Gr_max}, Gr_sr={self.Gr_sr}, ref={self.ref}, time={self.time}, dur={self.dur}, axis={self.axis})"

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
    if isinstance(other, (int, float, np.number)):
      return Gradient(slope=self.slope, 
                      lenc=self.lenc, 
                      strength=self.strength * other, 
                      scanner=self.scanner, 
                      ref=self.ref,
                      time=self.time,
                      axis=self.axis)
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

    Timings and amplitudes are adjusted by the reference time (`self.ref`).

    Returns:
      tuple: A tuple containing:
        - timings (np.ndarray): Array of timing values.
        - amplitudes (np.ndarray): Array of amplitude values.
        - interp (interp1d): Interpolator for gradient evaluation.
    """
    # TODO
    if self.lenc <= 0.0:
      timings = Quantity(np.array([0.0, 
                          self.slope.m,
                          self.slope.m+self.slope.m], dtype=np.float32), self.slope.u)
      amplitudes = Quantity(np.array([0.0, 
                            self.strength.m,
                            0.0], dtype=np.float32), self.strength.u)
    else:
      timings = Quantity(np.array([0.0, 
                          self.slope.m, 
                          self.slope.m+self.lenc.m, 
                          self.slope.m+self.lenc.m+self.slope.m], dtype=np.float32), self.slope.u)
      amplitudes = Quantity(np.array([0.0, 
                            self.strength.m, 
                            self.strength.m, 
                            0.0], dtype=np.float32), self.strength.u)

    # Add time reference wrt the sequence object
    timings += self.time - self.ref

    # Define interpolator for gradient evaluation
    interp = interp1d(timings.m, amplitudes.m, kind='linear', fill_value=0.0, bounds_error=False)

    return timings, amplitudes, interp

  def change_ref(self, ref: Quantity):
    """
    Update the reference time for the gradient.

    Parameters:
    ref (float): The new reference time.

    Returns:
    None
    """
    self.ref = ref.to('ms')
    self.dur2 = (self.dur - self.ref).to('ms')

  def change_time(self, time):
    """
    Update the time for the gradient.

    Parameters:
    time (float): The new time.

    Returns:
    None
    """
    self.time = time.to('ms')
    self.timings, self.amplitudes, self.interpolator = self.group_timings()

  def calculate(self, k_bw, receiver_bw=None, ro_samples=None, ofac=None):
    """
    Calculate gradient based on slope, slew rate, and amplitude parameters.

    Parameters:
    -----------
    k_bw: float
      Bandwidth in 1/m.
    receiver_bw: float, optional
      Receiver bandwidth in Hz. If provided, lenc is fixed and calculated accordingly.
    ro_samples: int, optional
      Number of readout samples.
    ofac: float, optional
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
    - self.lenc: Length in ms
    - self.strength: Gradient amplitude in mT/m
    - self.slope: Slope in ms
    - self.dur: Duration in ms

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
      self.strength     = k_bw.to('1/m')/(self.scanner.gammabar.to('1/mT/ms')*self.lenc.to('ms'))
      if self.strength > self.Gr_max:
        self.strength = self.Gr_max
        self.lenc = k_bw.to('1/m')/(self.scanner.gammabar.to('1/mT/s')*self.strength.to('mT/m'))
        receiver_bw = ((ro_samples/ofac)/self.lenc).to('Hz')
        if MPI_rank == 0:
          warnings.warn("The required gradient amplitude exceeds the maximum allowed. Adjusting to maximum gradient amplitude. Receiver bandwidth allowed with current configuration is {:.0f} Hz".format(receiver_bw.m_as('Hz')))
      self.slope = np.abs(self.strength)/self.Gr_sr
    else:
      ''' Calcualte everything as if only the ramps are needed '''
      # Time needed to reach the maximun amplitude
      slope_req = np.sqrt(np.abs(k_bw.to('1/m'))/(self.scanner.gammabar.to('1/mT/ms')*self.Gr_sr.to('mT/m/ms')))

      # Time needed for the maximun amplitude
      slope = self.Gr_max/self.Gr_sr

      # Build gradient
      if slope_req < slope:
        self.slope = slope_req
        self.strength     = self.Gr_sr*slope_req
        self.lenc  = self.slope - slope_req
      else:
        # Assign slope and gradient amplitude
        self.slope = slope
        self.strength     = self.Gr_max

        # Calculate lenc
        k_bw_slopes = self.scanner.gammabar.to('1/mT/ms')*self.Gr_sr.to('mT/m/ms')*slope.to('ms')**2
        self.lenc = (np.abs(k_bw.to('1/m')) - k_bw_slopes.to('1/m'))/(self.strength.to('mT/m')*self.scanner.gammabar.to('1/mT/ms'))

      # Account for area sign
      self.strength *= np.sign(k_bw)

    # Update gradient duration
    if self.lenc < 0:
      self.dur = self.slope + self.slope
    else:
      self.dur = self.slope + self.lenc + self.slope
    self.dur2 = (self.dur - self.ref).to('ms')

    # Update timings and amplitudes in array
    self.timings, self.amplitudes, self.interpolator = self.group_timings()

  def make_bipolar(self, VENC: Quantity):
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
    slope_req = np.cbrt(Quantity(np.pi,'rad')/(2*self.scanner.gamma.to('rad/ms/mT')*self.Gr_sr.to('mT/m/ms')*VENC.to('m/ms')))

    # Check if rectangle parts of the gradient are needed
    if slope_req <= slope:
      # Calculate duration of the first velocity encoding gradient lobe
      self.slope = slope_req.to('ms')
      self.strength     = -self.Gr_sr.to('mT/m/ms')*slope_req.to('ms')
      self.lenc  = self.slope.to('ms') - slope_req.to('ms')
    else:
      # Calculate lenc based on the intergral of the velocity encoding gradients
      a = self.scanner.gamma.to('rad/ms/mT')*VENC.to('m/ms')*self.Gr_max.to('mT/m')
      b = 3*self.scanner.gamma.to('rad/ms/mT')*VENC.to('m/ms')*self.Gr_max.to('mT/m')*slope.to('ms')
      c = 2*self.scanner.gamma.to('rad/ms/mT')*VENC.to('m/ms')*self.Gr_max.to('mT/m')*slope.to('ms')**2 - np.pi
      lenc_req = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

      # Gradients parameters
      self.slope = slope.to('ms')
      self.strength     = -self.Gr_max.to('mT/m') * VENC_sign
      self.lenc  = lenc_req.to('ms')

    # Upadte duration
    if self.lenc < 0:
      self.dur =(self.slope + self.slope).to('ms')
    else:
      self.dur =(self.slope + self.lenc + self.slope).to('ms')
    self.dur2 = (self.dur - self.ref).to('ms')

    # Update timings and amplitudes in array
    self.timings, self.amplitudes, self.interpolator = self.group_timings()

    # Create the second gradient lobe
    g = self.__copy__()
    g *= -1.0  # Invert the gradient amplitude
    g.change_time(self.time + self.dur)

    return g

  def area(self, t0: Quantity = None, nb_samples=1000):
    ''' Calculate the area of the gradient '''
    # Calculate area
    if t0 is None:
      t0 = self.time - self.ref
    time = np.linspace(t0.m_as('ms'), (self.time - self.ref + self.dur).m_as('ms'), nb_samples)
    area = np.trapz(self.interpolator(time), time)

    return Quantity(area, 'mT*ms/m')


  def match_area(self, area: Quantity, dur: Quantity = None):
    """
    Adjust self.slope, self.lenc and self.strength so that
    1) the gradient net area equals `area`, and
    2) if `dur` is given, total duration equals `dur` (otherwise minimal duration).
    """
    # Use absolute value of area and store sign
    sign = np.sign(area)
    area = abs(area).to('mT*ms/m')

    # Maximum slope (ms) at max gradient and slew‐rate
    slope_max = (self.Gr_max/self.Gr_sr).to('ms')  

    # Calculate gradient
    if dur is not None:
      dur = dur.to('ms')
      # Case A: only triangular lobes (no plateau)
      if dur < 2 * slope_max:
        # Slope for desired duration
        self.slope = dur / 2
        self.lenc = Quantity(0.0, 'ms')
        # area = G * slope  =>  G = area / slope
        self.strength = (area / self.slope).to(self.Gr_max.u)
        if self.strength > self.Gr_max:
          raise ValueError(f"Cannot achieve area={area} in dur={dur}: "
                           f"required G={self.strength} > Gmax={self.Gr_max}")
        self.dur = dur

      # Case B: add plateau so slope = slope_max
      else:
        self.slope = slope_max
        # Plateau length to fill out duration
        self.lenc = dur - 2 * slope_max
        # area = G*(slope + lenc)  =>  G = area / (slope + lenc)
        self.strength = (area / (self.slope + self.lenc)).to(self.Gr_max.u)
        if self.strength > self.Gr_max:
          raise ValueError(f"Cannot achieve area={area} in dur={dur}: "
                           f"required G={self.strength} > Gmax={self.Gr_max}")
        self.dur = dur

    else:
      # Fallback to original behaviour when dur is not specified
      # triangle-only if area <= area_max, else add plateau
      slope_min = slope_max
      area_max = slope_min * self.Gr_max.to('mT/m')
      if area <= area_max:
        ratio = (area / area_max).m
        self.slope     = (slope_min * np.sqrt(ratio)).to('ms')
        self.strength  = (self.Gr_max * np.sqrt(ratio)).to(self.Gr_max.u)
        self.lenc      = (self.slope - slope_min * np.sqrt(ratio)).to('ms')
      else:
        area_needed = area - area_max
        self.slope     = slope_min
        self.strength  = self.Gr_max
        self.lenc      = (area_needed / self.Gr_max).to('ms')
      # Update duration
      if self.lenc.m <= 0:
        self.dur = 2 * self.slope
      else:
        self.dur = 2 * self.slope + self.lenc

    # Restore sign on amplitude
    self.strength *= sign

    # Recompute derived quantities
    self.dur2 = (self.dur - self.ref).to('ms')
    self.timings, self.amplitudes, self.interpolator = self.group_timings()

  def rotate(self, directions: np.ndarray, normalize_dirs: bool = False):
    """
    Rotate the gradient direction.

    Parameters:
    direction (np.ndarray): The new direction vector for the gradient.

    Returns:
    None
    """
    # Check if directions is a 3-element vector
    directions = directions.reshape((-1, 3))
    if directions.shape[1] != 3:
      raise ValueError("Direction must be a 3-element vector with [M, P, S] directions.")
    
    # Number of directions
    nb_dirs = directions.shape[0]

    # Calculate rotated gradients
    gradients = [[] for _ in range(nb_dirs)]
    for d in range(nb_dirs):

      # Direction vector
      direction = directions[d, :]

      # Normalize the direction vector
      norm = np.linalg.norm(direction)
      if norm != 0 and normalize_dirs:
        direction = direction / norm

      # Gradient area
      area = self.area()

      # Create gradients around each axis given by the direction vector
      for i, fraction in enumerate(direction):
        if fraction != 0.0:
          g = self.__copy__()
          g.axis = i
          g.match_area(fraction * area)
          gradients[d].append(g)
        elif fraction == 0.0 and norm == 0.0:
          g = self.__copy__()
          g *= fraction  # Scale the gradient by the direction fraction
          gradients[d].append(g)
          break

    # Update gradient durations based on the maximum duration of the gradients
    max_dur = max([
      g.dur if g.strength != 0.0 else Quantity(0.0, 'ms')
      for d in range(nb_dirs)
      for g in gradients[d]
    ])

    # Update all gradients to have the same duration
    [g.match_area(g.area(), max_dur) for d in range(nb_dirs) for g in gradients[d]]

    return gradients[0] if nb_dirs == 1 else gradients

  def plot(self, linestyle='-', axes=[]):
    ''' Plot gradient '''
    fig = plt.figure()
    plt.plot(self.timings, self.amplitudes, linestyle)
    plt.show()

    return fig


class RF:
  def __init__(self, scanner=Scanner(),
              NbLobes=[2,2], 
              alpha=0.46, 
              shape='apodized_sinc', 
              flip_angle=Quantity(np.pi/2,'rad'), 
              dur=Quantity(2.0,'ms'), 
              ref=Quantity(0.0,'ms'), 
              time=Quantity(0.0,'ms'), 
              nb_samples=200, 
              phase_offset=Quantity(0.0,'rad'), 
              frequency_offset=Quantity(0.0,'Hz')):
    self.scanner = scanner
    self.NbLobes = NbLobes
    self.alpha = alpha
    self.shape = shape
    if self.shape == 'sinc':
      self._pulse = self._unit_sinc
      if self.alpha != 0.0 and MPI_rank == 0:
        warnings.warn("For 'sinc' shape, the alpha parameter is automatically set to 0.0")
      self.alpha = 0.0
    elif self.shape == 'apodized_sinc':
      self._pulse = self._unit_sinc
    elif self.shape == 'hard':
      self._pulse = self._unit_hard
    self.flip_angle = flip_angle.to('rad')
    self.ref = ref.to('ms')
    self.time = time.to('ms')
    self.dur = dur.to('ms')
    self.dur2 = (self.dur - self.ref).to('ms')
    self.half1 = (self.NbLobes[0] + 1)/(np.sum(self.NbLobes) + 2)*dur.to('ms')
    self.half2 = (self.NbLobes[1] + 1)/(np.sum(self.NbLobes) + 2)*dur.to('ms')
    self.phase_offset = phase_offset.to('rad')
    self.frequency_offset = frequency_offset.to('Hz')
    self.nb_samples = nb_samples
    self.interp_real, self.interp_imag = self._interpolator()

  def __str__(self):
    return f"RF(NbLobes={self.NbLobes}, alpha={self.alpha}, shape={self.shape}, flip_angle={self.flip_angle}, dur={self.dur}, ref={self.ref}, nb_samples={self.nb_samples})"

  def __copy__(self):
    return copy.deepcopy(self)

  def __repr__(self):
    return self.__str__()

  def __call__(self, t):
    return self.interp_real(t) + 1j*self.interp_imag(t)
  
  def _window(self, t):
    start = (self.time - self.ref).m_as('ms')
    end   = (self.time - self.ref + self.dur).m_as('ms')
    return (t >= start)*(t <= end)

  def _unit_sinc(self, t):
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

    # Shifted time
    # t_shift = t - self.half1.m_as('ms')
    t_shift = t - (self.time - self.ref).m_as('ms') - self.half1.m_as('ms')

    # RF pulse definition
    bw = (self.NbLobes[0] + self.NbLobes[1] + 2)/self.dur.to('ms')
    B1e = (1/bw.m)
    # B1e *= ((1-self.alpha) + self.alpha*np.cos(np.pi*bw.m*(t-self.time.m_as('ms'))/N))
    # B1e *= np.sinc(bw.m*(t-self.time.m_as('ms')))
    B1e *= ((1-self.alpha) + self.alpha*np.cos(np.pi*bw.m*t_shift/N))
    B1e *= np.sinc(bw.m*t_shift)
    B1e *= self._window(t)
    B1 = B1e + 1j*np.zeros(B1e.shape)

    # Add phase and frequency offsets
    if self.phase_offset.m != 0.0 or self.frequency_offset.m != 0.0:
      B1 *= np.exp(1j*(self.phase_offset.m_as('rad') + 2*np.pi*self.frequency_offset.m_as('kHz')*t_shift))

    return B1

  def _unit_hard(self, t):
    """
    Generate a hard RF pulse for MRI imaging.

    Parameters:
    -----------
    t: float
      The time at which the RF pulse is evaluated.

    Returns:
    --------
    B1e: float
      The amplitude of the RF pulse at time t.

    Notes:
    ------
    - The pulse frequency is determined by the desired slice thickness.
    - The pulse duration is calculated based on the number of left and right lobes.
    - The RF pulse is defined as a rectangular pulse within the calculated time window.
    """
    # Shifted time
    t_shift = t - (self.time - self.ref).m_as('ms') - self.half1.m_as('ms')

    # RF pulse definition
    B1e = 1.0
    B1e *= self._window(t)
    B1 = B1e + 1j*np.zeros(B1e.shape)

    # Add phase and frequency offsets
    if self.phase_offset.m != 0.0 or self.frequency_offset.m != 0.0:
      B1 *= np.exp(1j*(self.phase_offset.m_as('rad') + 2*np.pi*self.frequency_offset.m_as('kHz')*t_shift))

    return B1

  def _flip_angle_factor(self, t):
    # Time step
    dt = t[1] - t[0]

    # Calculate factor to account for flip angle
    amplitude = self._pulse(t)
    unit_flip_angle = np.sum((amplitude[1:] + amplitude[:-1])/2) * dt \
      * self.scanner.gamma.m_as('rad/mT/ms')  # [mT*ms/m]

    return self.flip_angle.m_as('rad')/unit_flip_angle

  def _interpolator(self):
    # Start and end times of the pulse
    start = (self.time - self.ref).m_as('ms')
    end   = (self.time - self.ref + self.dur).m_as('ms')

    # Time array for interpolation
    t = np.linspace(start, end, self.nb_samples)

    # Interpolators for real and imaginary parts of the pulse
    scaling = self._flip_angle_factor(t)
    interp_real = interp1d(t, np.abs(scaling) * np.real(self._pulse(t)), kind='linear', fill_value=0.0, bounds_error=False)
    interp_imag = interp1d(t, np.abs(scaling) * np.imag(self._pulse(t)), kind='linear', fill_value=0.0, bounds_error=False)

    return interp_real, interp_imag

  def change_ref(self, ref: Quantity):
    """
    Update the reference time for the gradient.

    Parameters:
    ref (float): The new reference time.

    Returns:
    None
    """
    self.ref = ref.to('ms')
    self.dur2 = (self.dur - self.ref).to('ms')
    self.interp_real, self.interp_imag = self._interpolator()

  def change_time(self, time):
    """
    Update the time for the gradient.

    Parameters:
    time (float): The new time.

    Returns:
    None
    """
    self.time = time.to('ms')
    self.interp_real, self.interp_imag = self._interpolator()

  def plot(self, linestyle='-', axes=[]):
    ''' Plot RF pulse '''
    # Timings
    start = (self.time - self.ref).m_as('ms')
    end   = (self.time - self.ref + self.dur).m_as('ms')
    t = np.linspace(start, end, self.nb_samples)
    fig = plt.figure()
    plt.plot(t, np.real(self._pulse(t)), linestyle)
    plt.plot(t, np.imag(self._pulse(t)), linestyle)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude {:s}'.format('mT'))
    plt.legend(['Real', 'Imaginary'])
    plt.show()

    return fig