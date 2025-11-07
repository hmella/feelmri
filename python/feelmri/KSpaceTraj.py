from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from feelmri.MPIUtilities import MPI_comm, MPI_rank
from feelmri.MRObjects import Gradient, Scanner

# plt.rcParams['text.usetex'] = True

# Generic tracjectory
class Trajectory:
  def __init__(self, FOV: Quantity = Quantity(np.array([0.3, 0.3, 0.08]), 'm'), 
               res: np.ndarray = np.array([100, 100, 1]), 
               oversampling: int = 2, 
               lines_per_shot: int = 1, 
               scanner: Scanner = Scanner(), 
               t_start: Quantity = Quantity(0, 'ms'), 
               receiver_bw: Quantity = Quantity(128.0e+3, 'Hz'), 
               plot_seq: bool = False, 
               MPS_ori: np.ndarray = np.eye(3), 
               LOC: np.ndarray = np.zeros([3,]), 
               dtype: np.dtype = np.float32):
    self.scanner = scanner
    self.FOV = FOV
    self.res = res
    self.oversampling = oversampling
    self.oversampling_arr = np.array([oversampling, 1.0, 1.0])
    self.Gr_max = scanner.gradient_strength   # [mT/m]
    self.Gr_sr  = scanner.gradient_slew_rate  # [mT/(m*ms)]
    self.gammabar = scanner.gammabar  # [Hz/T]
    self.lines_per_shot = lines_per_shot
    self.ro_samples = self.oversampling * self.res[0] # number of readout samples
    self.ph_samples = self.check_ph_enc_lines(self.res[1])
    self.slices = self.res[2]                  # number of slices
    self.nb_shots = self.ph_samples // self.lines_per_shot
    self.shots = [[None, ] * self.lines_per_shot for _ in range(self.nb_shots)]
    self.pxsz = FOV / res
    self.k_bw = 1.0 / self.pxsz
    self.k_spa = 1.0 / (self.oversampling_arr * FOV)
    self.kx_extent = (np.array([0, self.ro_samples - 1]) - self.ro_samples // 2) * self.k_spa[0] + float((self.res[0] % 2 != 0)) * self.k_spa[0]
    self.ky_extent = (np.array([0, self.ph_samples - 1]) - self.ph_samples // 2) * self.k_spa[1]
    self.kz_extent = (np.array([0, self.slices - 1]) - self.slices // 2) * self.k_spa[2]
    self.t_start = t_start.astype(dtype)
    self.plot_seq = plot_seq
    self.receiver_bw = receiver_bw          # [Hz]
    self.MPS_ori = MPS_ori.astype(dtype)  # orientation
    self.LOC = LOC.astype(dtype)          # location
    self.dtype = dtype

  def check_ph_enc_lines(self, ph_samples):
    ''' Verify if the number of lines in the phase encoding direction
    satisfies the multishot factor '''
    return np.int32(self.lines_per_shot * (ph_samples // self.lines_per_shot))
  
  def plot_trajectory(self, figsize=(12, 5), tight_layout=True, export_to=None):
    ''' Show kspace points and time map'''
    if MPI_rank == 0:
      # plt.rcParams['text.usetex'] = True
      # plt.rcParams.update({'font.size': 16})

      # Plot kspace locations and times
      fig, ax = plt.subplots(1, 2, figsize=figsize)
      for shot in self.shots:
        kxx = np.concatenate((np.array([0]), self.points[0][:,shot,0].flatten('F')))
        kyy = np.concatenate((np.array([0]), self.points[1][:,shot,0].flatten('F')))
        ax[0].plot(kxx,kyy)
      ax[0].set_xlabel('$k_x ~(1/m)$')
      ax[0].set_ylabel('$k_y ~(1/m)$')

      im = ax[1].scatter(self.points[0][:,:,0], self.points[1][:,:,0], c=self.times[:,:,0].m_as('ms'), s=2.5, cmap='turbo')
      ax[1].set_xlabel('$k_x ~(1/m)$')
      ax[1].set_yticklabels([])
      if tight_layout:
        plt.tight_layout()
      cbar = fig.colorbar(im, orientation='vertical', ax=ax)
      cbar.ax.set_title('Time [ms]')
      if export_to is not None:
        plt.savefig(export_to, bbox_inches='tight')
      plt.show()

    # Synchronize all processes
    MPI_comm.Barrier()

# CartesianStack trajectory
class CartesianStack(Trajectory):
    def __init__(self, shot_coverage: Literal["full", "partial"] = "full", *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.shot_coverage = shot_coverage
      self.ph_samples = self.check_ph_enc_lines(self.res[1])
      self.nb_shots = self.ph_samples // self.lines_per_shot
      (self.points, self.times) = self.kspace_points()

    def kspace_points(self):
      ''' Get kspace points '''
      # k-space positioning gradients
      ph_grad = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
      ph_grad.calculate(0.5*self.k_bw[1].to('1/m'))

      ro_grad0 = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
      ro_grad0.calculate(-0.5*self.k_bw[0].to('1/m') - 0.5*ro_grad0.scanner.gammabar.to('1/mT/ms')*ro_grad0.strength.to('mT/m')*ro_grad0.slope.to('ms'))

      blip_grad = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
      blip_grad.calculate(-self.k_bw[1].to('1/m')/self.ph_samples)

      # Gradient duration is not used because can be shorter than second half of the slice selection gradient
      enc_time = Quantity(self.t_start.m_as('ms') - np.max([ph_grad.dur.m_as('ms'), ro_grad0.dur.m_as('ms')]), 'ms')
      
      # Update timings
      ph_grad.change_time(enc_time)
      ro_grad0.change_time(enc_time)

      enc_gradients = []
      ro_gradients = [ro_grad0, ]
      ph_gradients = [ph_grad, ]
      for i in range(self.lines_per_shot):
        # Calculate readout gradient
        ro_grad = Gradient(time=ro_gradients[i].timings[-1], scanner=self.scanner)
        ro_grad.calculate((-1)**i*self.k_bw[0].to('1/m'), receiver_bw=self.receiver_bw.to('Hz'), ro_samples=self.ro_samples, ofac=self.oversampling)
        ro_gradients.append(ro_grad)

        # Calculate blip gradient
        if self.lines_per_shot > 1 and i < self.lines_per_shot - 1:
          ref = ro_gradients[-1].time + ro_gradients[-1].dur - 0.5*blip_grad.dur
          blip_grad = Gradient(time=ref, scanner=self.scanner)
          blip_grad.calculate(-self.k_bw[1].to('1/m')/self.ph_samples)
          ph_gradients.append(blip_grad)

      if self.plot_seq:
        if MPI_rank == 0:
          # plt.rcParams['text.usetex'] = True
          plt.rcParams.update({'font.size': 16})

          fig, ax = plt.subplots(2, 1, figsize=(8,4))

          # Phase encoding gradients
          for gr in ph_gradients:
            ax1 = ax[1].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'r-', linewidth=2)

          # Readout gradients
          for gr in ro_gradients:
            ax2 = ax[0].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'b-', linewidth=2)

          # Encoding gradients
          for gr in enc_gradients:
            ax3 = ax[0].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'k--', linewidth=3, zorder=100)
            ax4 = ax[1].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'k--', linewidth=3, zorder=100)

          # Add ADC readout
          for i in range(1, self.lines_per_shot+1):
            a = [(ro_gradients[i].time + ro_gradients[i].slope).m_as('ms'),
                (ro_gradients[i].time + ro_gradients[i].slope + ro_gradients[i].lenc).m_as('ms')]
            ax5 = ax[0].plot(a, [0, 0], 'm-', linewidth=4, zorder=101)

          # Set legend labels
          ax1[0].set_label('PH')
          ax2[0].set_label('RO')
          ax5[0].set_label('ADC')

          # Format plots
          for i in range(len(ax)):
            ax[i].hlines(y=[0], xmin=0, xmax=[100], colors=['0.7'], linestyles='solid')
            ax[i].tick_params('both', length=5, width=1, which='major', labelsize=16)
            ax[i].tick_params('both', length=3, width=1, which='minor', labelsize=16)
            ax[i].minorticks_on()
            ax[i].set_ylabel('$G~\mathrm{(mT/m)}$', fontsize=16)
            ax[i].axis([0, ro_gradients[-1].time.m_as('ms') + ro_gradients[-1].dur.m_as('ms'), -1.4*self.Gr_max.m_as('mT/m'), 1.4*self.Gr_max.m_as('mT/m')])
            ax[i].legend(fontsize=14, loc='upper right', ncol=4)
          ax[1].set_xlabel('$t~\mathrm{(ms)}$', fontsize=16)
          plt.tight_layout()
          plt.show()

        # Synchronize all processes
        MPI_comm.Barrier()

      # Time needed to acquire one line
      # It depends on the kspcae bandwidth, the gyromagnetic constant, and
      # the maximun gradient amplitude
      dt = np.linspace(0.0, ro_grad.lenc.m_as('ms'), self.ro_samples)

      # kspace locations
      kx = np.linspace(self.kx_extent[0].m_as('1/m'), self.kx_extent[1].m_as('1/m'), self.ro_samples)
      ky = self.ky_extent[0].m_as('1/m')*np.ones(kx.shape)
      kz = np.linspace(self.kz_extent[0].m_as('1/m'), self.kz_extent[1].m_as('1/m'), self.slices)

      kspace = (np.zeros([self.ro_samples, self.ph_samples, self.slices],     
                dtype=self.dtype),
                np.zeros([self.ro_samples, self.ph_samples, self.slices],dtype=self.dtype),
                np.zeros([self.ro_samples, self.ph_samples, self.slices],dtype=self.dtype))

      # Build shots locations
      for ph in range(self.ph_samples):
        if self.shot_coverage == "partial":
          self.shots[ph // self.lines_per_shot][ph % self.lines_per_shot] = ph
        elif self.shot_coverage == "full":
          self.shots[ph % self.nb_shots][ph // self.nb_shots] = ph

      # kspace times and locations
      t = np.zeros([self.ro_samples, self.ph_samples, self.slices], dtype=self.dtype)
      for shot in self.shots:
        for idx, ph in enumerate(shot):

          # Readout direction
          ro = (-1)**idx

          # Fill locations
          kspace[0][::ro,ph,:] = np.tile(kx[:,None], [1, self.slices])
          kspace[1][::ro,ph,:] = np.tile(ky[:,None] + self.k_spa[1].m_as('1/m')*ph, [1, self.slices])

          # Update timings
          if idx == 0:
            t[::ro,ph,0] = enc_time.m_as('ms') + ro_grad0.dur.m_as('ms') + ro_grad.slope.m_as('ms') + dt
          else:
            t[::ro,ph,0] = t[:,shot[idx-1]].max() + ro_grad.slope.m_as('ms') + ro_grad.slope.m_as('ms') + dt[::ro]

      # Fill kz coordinates
      for s in range(self.slices):
        kspace[2][:,:,s] = kz[s]
        t[:,:,s] = t[:,:,0]

      # Calculate echo time
      self.echo_time = (enc_time + ro_grad0.dur + 0.5*self.lines_per_shot*ro_grad.dur).to('ms')

      return (kspace, Quantity(t, 'ms'))



# Radial trajectory
class RadialStack(Trajectory):
    def __init__(self, *args, 
                 golden_angle: bool = False, 
                 full_spoke: bool = False, 
                 **kwargs):
      super().__init__(*args, **kwargs)
      self.golden_angle = golden_angle
      self.full_spoke = full_spoke
      self.ph_samples = self.check_ph_enc_lines(self.ph_samples)
      self.nb_shots = self.ph_samples // self.lines_per_shot
      (self.points, self.times) = self.kspace_points()

    def kspace_points(self):
      ''' Get kspace points '''
      # k-space positioning gradients
      ph_grad = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
      ph_grad.calculate(0.5*self.k_bw[1].to('1/m'))


      ro_grad0 = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
      blip_grad = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
      if self.golden_angle:
        blip_grad.calculate(-self.k_bw[0].to('1/m')/self.ro_samples)
        ro_grad0 = blip_grad.__copy__()
      else:
        ro_grad0.calculate(-0.5*self.k_bw[0].to('1/m') - 0.5*ro_grad0.scanner.gammabar.to('1/mT/ms')*ro_grad0.strength.to('mT/m')*ro_grad0.slope.to('ms'))
        blip_grad.calculate(-self.k_bw[1].to('1/m')/self.ph_samples)

      # Gradient duration is not used because can be shorter than second half of the slice selection gradient
      enc_time = Quantity(self.t_start.m_as('ms') - np.max([ph_grad.dur.m_as('ms'), ro_grad0.dur.m_as('ms')]), 'ms')
      
      # Update timings
      ph_grad.change_time(enc_time)
      ro_grad0.change_time(enc_time)

      enc_gradients = []
      ro_gradients = [ro_grad0, ]
      ph_gradients = [ph_grad, ]
      for i in range(self.lines_per_shot):
        # Calculate readout gradient
        ro_grad = Gradient(time=ro_gradients[i].timings[-1], scanner=self.scanner)
        ro_grad.calculate((-1)**i*self.k_bw[0].to('1/m'), receiver_bw=self.receiver_bw.to('Hz'), ro_samples=self.ro_samples, ofac=self.oversampling)
        ro_gradients.append(ro_grad)

        # Calculate blip gradient
        if self.lines_per_shot > 1 and i < self.lines_per_shot - 1:
          ref = ro_gradients[-1].time + ro_gradients[-1].dur - 0.5*blip_grad.dur
          blip_grad = Gradient(time=ref, scanner=self.scanner)
          blip_grad.calculate(-self.k_bw[1].to('1/m')/self.ph_samples)
          ph_gradients.append(blip_grad)

      if self.plot_seq:
        if MPI_rank == 0:
          # plt.rcParams['text.usetex'] = True
          plt.rcParams.update({'font.size': 16})

          fig, ax = plt.subplots(2, 1, figsize=(8,4))

          # Phase encoding gradients
          for gr in ph_gradients:
            ax1 = ax[1].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'r-', linewidth=2)

          # Readout gradients
          for gr in ro_gradients:
            ax2 = ax[0].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'b-', linewidth=2)

          # Encoding gradients
          for gr in enc_gradients:
            ax3 = ax[0].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'k--', linewidth=3, zorder=100)
            ax4 = ax[1].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'k--', linewidth=3, zorder=100)

          # Add ADC readout
          for i in range(1, self.lines_per_shot+1):
            a = [(ro_gradients[i].time + ro_gradients[i].slope).m_as('ms'),
                (ro_gradients[i].time + ro_gradients[i].slope + ro_gradients[i].lenc).m_as('ms')]
            ax5 = ax[0].plot(a, [0, 0], 'm-', linewidth=4, zorder=101)

          # Set legend labels
          ax1[0].set_label('PH')
          ax2[0].set_label('RO')
          ax5[0].set_label('ADC')

          # Format plots
          for i in range(len(ax)):
            ax[i].hlines(y=[0], xmin=0, xmax=[100], colors=['0.7'], linestyles='solid')
            ax[i].tick_params('both', length=5, width=1, which='major', labelsize=16)
            ax[i].tick_params('both', length=3, width=1, which='minor', labelsize=16)
            ax[i].minorticks_on()
            ax[i].set_ylabel('$G~\mathrm{(mT/m)}$', fontsize=16)
            ax[i].axis([0, ro_gradients[-1].time.m_as('ms') + ro_gradients[-1].dur.m_as('ms'), -1.4*self.Gr_max.m_as('mT/m'), 1.4*self.Gr_max.m_as('mT/m')])
            ax[i].legend(fontsize=14, loc='upper right', ncol=4)
          ax[1].set_xlabel('$t~\mathrm{(ms)}$', fontsize=16)
          plt.tight_layout()
          plt.show()

        # Synchronize all processes
        MPI_comm.Barrier()

      # Time needed to acquire one line
      # It depends on the kspcae bandwidth, the gyromagnetic constant, and
      # the maximun gradient amplitude
      dt = np.linspace(0.0, ro_grad.lenc.m_as('ms'), self.ro_samples)

      # kspace locations
      # kx = np.linspace(self.kx_extent[0], self.kx_extent[1], self.ro_samples)
      kx = np.linspace(0, self.kx_extent[1].m_as('1/m'), self.ro_samples)
      ky = np.zeros(kx.shape)
      kz = np.linspace(self.kz_extent[0].m_as('1/m'), self.kz_extent[1].m_as('1/m'), self.slices)

      kspace = (np.zeros([self.ro_samples, self.ph_samples, self.slices],     
                dtype=self.dtype),
                np.zeros([self.ro_samples, self.ph_samples, self.slices],dtype=self.dtype),
                np.zeros([self.ro_samples, self.ph_samples, self.slices],dtype=self.dtype))

      # Build shots locations
      for ph in range(self.ph_samples):
        self.shots[ph // self.lines_per_shot][ph % self.lines_per_shot] = ph
        # self.shots[ph % self.nb_shots][ph // self.nb_shots] = ph

      if self.full_spoke:
        # Full-spoke golden-angle radial sampling
        GR = np.deg2rad(111.25)
        theta = np.array([np.mod(np.pi/GR*n, 2*np.pi) for n in range(self.ph_samples)])
      else:
        # Half-spoke golden-angle radial sampling
        GR = 1.61803398875
        theta = np.array([np.mod((2*np.pi - 2*np.pi/GR)*n, 2*np.pi) for n in range(self.ph_samples)])

      # kspace times and locations
      t = np.zeros([self.ro_samples, self.ph_samples, self.slices], dtype=self.dtype)
      for shot in self.shots:
        for idx, ph in enumerate(shot):

          # Readout direction
          ro = (-1)**idx

          # Fill locations
          kspace[0][::ro,ph,:] = np.tile(kx[:,None]*np.cos(theta[ph]) + ky[:,None]*np.sin(theta[ph]), [1, self.slices])
          kspace[1][::ro,ph,:] = np.tile(-kx[:,None]*np.sin(theta[ph]) + ky[:,None]*np.cos(theta[ph]), [1, self.slices])

          # Update timings
          if idx == 0:
            t[::ro,ph,0] = enc_time.m_as('ms') + ro_grad0.dur.m_as('ms') + ro_grad.slope.m_as('ms') + dt
          else:
            t[::ro,ph,0] = t[:,shot[idx-1]].max() + ro_grad.slope.m_as('ms') + ro_grad.slope.m_as('ms') + dt[::ro]

      # Fill kz coordinates
      for s in range(self.slices):
        kspace[2][:,:,s] = kz[s]
        t[:,:,s] = t[:,:,0]

      # Calculate echo time
      self.echo_time = (enc_time + ro_grad0.dur + 0.5*self.lines_per_shot*ro_grad.dur).to('ms')

      return (kspace, Quantity(t, 'ms'))



# SpiralStack trajectory
class SpiralStack(Trajectory):
    """
    SpiralStack defines a realistic 3D stack-of-spirals k-space trajectory,
    constrained by gradient amplitude and slew-rate hardware limits.

    The total acquisition time is determined by the ADC (receiver bandwidth)
    and gradient limits — not by the oversampling factor. Changing the
    oversampling now only affects spatial density in k-space.
    """

    def __init__(self, *args,
                 density_exponent: float = 1.0,
                 safety_margin: float = 0.95,
                 **kwargs):
        """
        Initialize the SpiralStack trajectory.

        Parameters
        ----------
        density_exponent : float, optional
            Power-law exponent controlling radial density.
            p > 1 increases density near the periphery (default = 1.0).
        safety_margin : float, optional
            Fractional margin applied to gradient and slew limits (default = 0.95).
        """
        super().__init__(*args, **kwargs)
        self.ph_samples = self.check_ph_enc_lines(self.ph_samples)
        self.nb_shots = self.ph_samples // self.lines_per_shot
        self.min_samples_per_turn = 32
        self.interleaves = self.ph_samples
        self.density_exponent = float(density_exponent)
        self.safety_margin = float(safety_margin)
        (self.points, self.times) = self.kspace_points()

    def _base_spiral(self, ro_samples: int, k_max: Quantity, turns: float, p: float):
        """
        Generate a variable-density 2D spiral trajectory.

        kr(u)  = k_max * u**p
        phi(u) = 2*pi*turns * u**(1/p)
        """
        u = np.linspace(0.0, 1.0, ro_samples, dtype=self.dtype)
        kr = k_max.m_as('1/m') * (u ** p)
        phi = 2.0 * np.pi * turns * (u ** (1.0 / p))
        K = kr * np.exp(1j * phi)
        return u, K

    def _enforce_hardware_limits(self, u: np.ndarray, K: np.ndarray):
        """
        Enforce gradient amplitude and slew-rate constraints to compute
        the continuous time law t(u).

        Returns
        -------
        t_final : ndarray (s)
            Monotonic time samples corresponding to u.
        T_ro : float
            Total readout duration in seconds.
        """
        # Scanner limits
        gamma = self.gammabar.to('Hz/T').m * 2 * np.pi       # [rad/s/T]
        Gmax = self.Gr_max.to('T/m').m * self.safety_margin
        Smax = self.Gr_sr.to('T/m/s').m * self.safety_margin

        # Derivatives of k(u)
        du = np.gradient(u)
        dK_du = np.gradient(K, u, edge_order=2)
        d2K_du2 = np.gradient(dK_du, u, edge_order=2)

        # Magnitudes
        abs_dK_du = np.abs(dK_du)
        abs_d2K_du2 = np.abs(d2K_du2)

        # Time per unit-u from amplitude and slew constraints
        dt_du_amp = abs_dK_du / (gamma * Gmax)
        dt_du_slew = np.sqrt(np.maximum(abs_d2K_du2, 0.0) / (gamma * Smax))
        dt_du = np.maximum(dt_du_amp, dt_du_slew)

        # Integrate over u to obtain t(u)
        t_final = np.cumsum(0.5 * (dt_du + np.roll(dt_du, 1)) * du)
        t_final[0] = 0.0
        T_ro = float(t_final[-1])
        return t_final.astype(self.dtype), T_ro

    def kspace_points(self):
        """
        Compute the full 3D stack-of-spirals k-space trajectory using
        ADC-based timing (independent of oversampling).

        Returns
        -------
        points : tuple of ndarray
            kx, ky, kz arrays of shape [ro_samples, interleaves, slices].
        times : Quantity
            Time array of shape [ro_samples, interleaves, slices], in ms.
        """

        # k-space positioning gradients
        ro_grad0 = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
        ro_grad0.calculate(-0.5 * self.k_bw[0].to('1/m')
                           - 0.5 * ro_grad0.scanner.gammabar.to('1/mT/ms')
                           * ro_grad0.strength.to('mT/m')
                           * ro_grad0.slope.to('ms'))

        # k-space extent
        k_max = 0.5 * self.k_bw[0]

        # Determine number of turns (independent of oversampling)
        k_spa_base = (1.0 / self.FOV)[0]                       # base grid spacing
        turns_nominal = max(1.0, float((k_max / k_spa_base).m_as('')))
        max_turns_from_sampling = max(1.0, self.res[0] / float(self.min_samples_per_turn))
        turns = min(turns_nominal, max_turns_from_sampling)

        # Generate base 2D spiral
        u, K = self._base_spiral(self.res[0], k_max, turns, self.density_exponent)

        # Enforce gradient limits to get continuous time law
        t_sec_cont, T_ro = self._enforce_hardware_limits(u, K)

        # ADC-based sampling grid (fixed by receiver bandwidth)
        dt_adc = 1.0 / self.receiver_bw.m_as('Hz')              # [s]
        N_adc = int(np.round(T_ro / dt_adc))
        t_adc = np.linspace(0.0, T_ro, N_adc * self.oversampling, endpoint=True)

        # Interpolate spiral onto uniform ADC time base
        K_adc_real = np.interp(t_adc, t_sec_cont, np.real(K))
        K_adc_imag = np.interp(t_adc, t_sec_cont, np.imag(K))
        K_adc = K_adc_real + 1j * K_adc_imag

        # 3D stack dimensions
        self.ro_samples = N_adc                                # ADC defines sample count
        dt_ms = t_adc * 1e3                                    # [ms]
        kz = np.linspace(self.kz_extent[0].m_as('1/m'),
                         self.kz_extent[1].m_as('1/m'),
                         self.slices)

        # Allocate arrays
        kspace = (
            np.zeros([N_adc * self.oversampling, self.ph_samples, self.slices], dtype=self.dtype),
            np.zeros([N_adc * self.oversampling, self.ph_samples, self.slices], dtype=self.dtype),
            np.zeros([N_adc * self.oversampling, self.ph_samples, self.slices], dtype=self.dtype),
        )
        t = np.zeros([N_adc * self.oversampling, self.ph_samples, self.slices], dtype=self.dtype)

        # Interleaf rotation angles
        theta = np.linspace(0.0, 2.0 * np.pi, self.interleaves,
                            endpoint=False, dtype=self.dtype)
        enc_time = Quantity(self.t_start.m_as('ms') - ro_grad0.dur.m_as('ms'), 'ms')

        # Build shots locations and time maps
        for ph in range(self.interleaves):
            self.shots[ph // self.lines_per_shot][ph % self.lines_per_shot] = ph

            # Rotate base spiral
            R = np.exp(1j * theta[ph])
            K_rot = K_adc * R
            kx_ = np.real(K_rot)
            ky_ = np.imag(K_rot)

            # Fill k-space locations and time
            kspace[0][:, ph, :] = np.tile(kx_[:, None], [1, self.slices])
            kspace[1][:, ph, :] = np.tile(ky_[:, None], [1, self.slices])
            t[:, ph, :] = (enc_time.m_as('ms')
                           + ro_grad0.dur.m_as('ms')
                           + dt_ms)[:, None]

        # Fill kz coordinates
        for s in range(self.slices):
            kspace[2][:, :, s] = kz[s]

        # Echo time
        self.echo_time = enc_time + Quantity(0.5 * T_ro * 1e3, 'ms')

        return (kspace, Quantity(t, 'ms'))
