import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from FEelMRI.MPIUtilities import MPI_rank, scatterKspace
from FEelMRI.MRObjects import Gradient, Scanner
from FEelMRI.Units import *

# plt.rcParams['text.usetex'] = True

# Generic tracjectory
class Trajectory:
  def __init__(self, FOV=Quantity(np.array([0.3, 0.3, 0.08]),'m'), res=np.array([100, 100, 1]), oversampling=2, lines_per_shot=7, scanner=Scanner(), t_start=Quantity(0, 'ms'), receiver_bw=Quantity(128.0e+3,'Hz'), plot_seq=False, MPS_ori=np.eye(3), LOC=np.zeros([3,]), dtype=np.float32):
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
    self.t_start = t_start
    self.plot_seq = plot_seq
    self.receiver_bw = receiver_bw          # [Hz]
    self.MPS_ori = MPS_ori.astype(dtype)  # orientation
    self.LOC = LOC.astype(dtype)          # location
    self.dtype = dtype

  def check_ph_enc_lines(self, ph_samples):
    ''' Verify if the number of lines in the phase encoding direction
    satisfies the multishot factor '''
    return int(self.lines_per_shot * (ph_samples // self.lines_per_shot))

# CartesianStack trajectory
class CartesianStack(Trajectory):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.ph_samples = self.check_ph_enc_lines(self.res[1])
      self.nb_shots = self.ph_samples // self.lines_per_shot
      (self.points, self.times) = self.kspace_points()

    def kspace_points(self):
      ''' Get kspace points '''
      # k-space positioning gradients
      ph_grad = Gradient(t_ref=0.0, scanner=self.scanner)
      ph_grad.calculate(0.5*self.k_bw[1].to('1/m'))

      ro_grad0 = Gradient(t_ref=0.0, scanner=self.scanner)
      ro_grad0.calculate(-0.5*self.k_bw[0].to('1/m') - 0.5*ro_grad0.scanner.gammabar.to('1/mT/ms')*ro_grad0.G.to('mT/m')*ro_grad0.slope.to('ms'))

      blip_grad = Gradient(t_ref=0.0, scanner=self.scanner)
      blip_grad.calculate(-self.k_bw[1].to('1/m')/self.ph_samples)

      # Gradient duration is not used because can be shorter than second half of the slice selection gradient
      enc_time = Quantity(self.t_start.m_as('ms') - np.max([ph_grad.dur.m_as('ms'), ro_grad0.dur.m_as('ms')]), 'ms')
      
      # Update timings
      ph_grad.update_reference(enc_time)
      ro_grad0.update_reference(enc_time)

      enc_gradients = []
      ro_gradients = [ro_grad0, ]
      ph_gradients = [ph_grad, ]
      for i in range(self.lines_per_shot):
        # Calculate readout gradient
        ro_grad = Gradient(t_ref=ro_gradients[i].timings[-1], scanner=self.scanner)
        ro_grad.calculate((-1)**i*self.k_bw[0].to('1/m'), receiver_bw=self.receiver_bw.to('Hz'), ro_samples=self.ro_samples, ofac=self.oversampling)
        ro_gradients.append(ro_grad)

        # Calculate blip gradient
        if self.lines_per_shot > 1 and i < self.lines_per_shot - 1:
          ref = ro_gradients[-1].t_ref + ro_gradients[-1].dur - 0.5*blip_grad.dur
          blip_grad = Gradient(t_ref=ref, scanner=self.scanner)
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
            a = [(ro_gradients[i].t_ref + ro_gradients[i].slope).m_as('ms'),
                (ro_gradients[i].t_ref + ro_gradients[i].slope + ro_gradients[i].lenc).m_as('ms')]
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
            ax[i].axis([0, ro_gradients[-1].t_ref.m_as('ms') + ro_gradients[-1].dur.m_as('ms'), -1.4*self.Gr_max.m_as('mT/m'), 1.4*self.Gr_max.m_as('mT/m')])
            ax[i].legend(fontsize=14, loc='upper right', ncol=4)
          ax[1].set_xlabel('$t~\mathrm{(ms)}$', fontsize=16)
          plt.tight_layout()
          plt.show()

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
        self.shots[ph // self.lines_per_shot][ph % self.lines_per_shot] = ph
        # self.shots[ph % self.nb_shots][ph // self.nb_shots] = ph

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

    def plot_trajectory(self):
      ''' Show kspace points and time map'''
      if MPI_rank == 0:
        # plt.rcParams['text.usetex'] = True
        plt.rcParams.update({'font.size': 16})

        # Plot kspace locations and times
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for shot in self.shots:
          kxx = np.concatenate((np.array([0]), self.points[0][:,shot,0].flatten('F')))
          kyy = np.concatenate((np.array([0]), self.points[1][:,shot,0].flatten('F')))
          ax[0].plot(kxx,kyy)
        ax[0].set_xlabel('$k_x ~(1/m)$')
        ax[0].set_ylabel('$k_y ~(1/m)$')

        im = ax[1].scatter(self.points[0][:,:,0],self.points[1][:,:,0],c=self.times,s=2.5,cmap='turbo')
        ax[1].set_xlabel('$k_x ~(1/m)$')
        ax[1].set_yticklabels([])
        fig.tight_layout()
        cbar = fig.colorbar(im, orientation='vertical', ax=ax)
        cbar.ax.set_title('Time [ms]')
        plt.show()


# Radial trajectory
class RadialStack(Trajectory):
    def __init__(self, *args, spokes=20, **kwargs):
      super().__init__(*args, **kwargs)
      self.ph_samples = self.check_ph_enc_lines(self.ph_samples)
      self.nb_shots = self.ph_samples // self.lines_per_shot
      (self.points, self.times) = self.kspace_points()

    def kspace_points(self):
      ''' Get kspace points '''
      # k-space positioning gradients
      ro_grad0 = Gradient(t_ref=0.0, Gr_max=self.Gr_max, Gr_sr=self.Gr_sr)
      ro_grad0.calculate(-0.5*self.k_bw[0] - 0.5*ro_grad0._gammabar*ro_grad0._G*ro_grad0.slope)

      blip_grad = Gradient(t_ref=0.0, Gr_max=self.Gr_max, Gr_sr=self.Gr_sr)
      blip_grad.calculate(-self.k_bw[1]/self.ph_samples)

      # Gradient duration is not used because can be shorter than second half of the slice selection gradient
      enc_time = (self.G_enc.timings[-1] - ro_grad0.dur).to('ms') 

      # Update timings
      ro_grad0.update_reference(enc_time)

      enc_gradients = [self.G_enc, ]
      ro_gradients = [ro_grad0, ]
      ph_gradients = []
      for i in range(self.lines_per_shot):
        # Calculate readout gradient
        ro_grad = Gradient(t_ref=ro_gradients[i].timings[-1], Gr_max=self.Gr_max, Gr_sr=self.Gr_sr)
        ro_grad.calculate((-1)**i*self.k_bw[0], receiver_bw=self.receiver_bw, ro_samples=self.ro_samples, ofac=self.oversampling)
        ro_gradients.append(ro_grad)

        # Calculate blip gradient
        if self.lines_per_shot > 1 and i < self.lines_per_shot - 1:
          ref = ro_gradients[-1].t_ref + ro_gradients[-1].dur - 0.5*blip_grad.dur
          blip_grad = Gradient(t_ref=ref, Gr_max=self.Gr_max, Gr_sr=self.Gr_sr)
          blip_grad.calculate(-self.k_bw[1]/self.ph_samples)
          ph_gradients.append(blip_grad)

      if self.plot_seq:
        if MPI_rank == 0:
          # plt.rcParams['text.usetex'] = True
          plt.rcParams.update({'font.size': 16})

          fig, ax = plt.subplots(2, 1, figsize=(8,4))

          # Phase encoding gradients
          for gr in ph_gradients:
            ax1 = ax[1].plot(gr.timings, gr.amplitudes, 'r-', linewidth=2)

          # Readout gradients
          for gr in ro_gradients:
            ax2 = ax[0].plot(gr.timings, gr.amplitudes, 'b-', linewidth=2)

          # Encoding gradients
          for gr in enc_gradients:
            ax3 = ax[0].plot(gr.timings, gr.amplitudes, 'k--', linewidth=3, zorder=100)
            ax4 = ax[1].plot(gr.timings, gr.amplitudes, 'k--', linewidth=3, zorder=100)

          # Add ADC readout
          for i in range(1, self.lines_per_shot+1):
            a = [ro_gradients[i].t_ref + ro_gradients[i].slope,
                ro_gradients[i].t_ref + ro_gradients[i].slope + ro_gradients[i].lenc]
            ax5 = ax[0].plot(a, [0, 0], 'm-', linewidth=4, zorder=101)

          # Set legend labels
          ax1[0].set_label('PH')
          ax2[0].set_label('RO')
          if self.G_enc.Gr_max != 0:
            ax3[0].set_label('ENC')
            ax4[0].set_label('ENC')
          ax5[0].set_label('ADC')

          # Format plots
          for i in range(len(ax)):
            ax[i].hlines(y=[0], xmin=0, xmax=[100], colors=['0.7'], linestyles='solid')
            ax[i].tick_params('both', length=5, width=1, which='major', labelsize=16)
            ax[i].tick_params('both', length=3, width=1, which='minor', labelsize=16)
            ax[i].minorticks_on()
            ax[i].set_ylabel('$G~\mathrm{(mT/m)}$', fontsize=16)
            ax[i].axis([0, ro_gradients[-1].t_ref + ro_gradients[-1].dur, -1.4*self.Gr_max, 1.4*self.Gr_max])
            ax[i].legend(fontsize=14, loc='upper right', ncol=4)
          ax[1].set_xlabel('$t~\mathrm{(ms)}$', fontsize=16)
          plt.tight_layout()
          plt.show()


      # Time needed to acquire one line
      dt_line = (self.k_bw[0]*2*np.pi)/(self._gamma*self._Gr_max)
      dt = np.linspace(0.0, dt_line, self.ro_samples)

      # kspace locations
      # kx = np.linspace(self.kx_extent[0], self.kx_extent[1], self.ro_samples)
      kx = np.linspace(0, self.kx_extent[1], self.ro_samples)
      ky = np.zeros(kx.shape)
      kz = np.linspace(self.kz_extent[0], self.kz_extent[1], self.slices)

      kspace = (np.zeros([self.ro_samples, self.ph_samples, self.slices],     
                dtype=self.dtype),
                np.zeros([self.ro_samples, self.ph_samples, self.slices],dtype=self.dtype),
                np.zeros([self.ro_samples, self.ph_samples, self.slices],dtype=self.dtype))

      # Build shots locations
      for ph in range(self.ph_samples):
        self.shots[ph // self.lines_per_shot][ph % self.lines_per_shot] = ph
        # self.shots[ph % self.nb_shots][ph // self.nb_shots] = ph

      # Golden ratio
      GR = 1.61803398875

      # Half-spoke golden-angle radial sampling
      theta = np.array([np.mod((2*np.pi - 2*np.pi/GR)*n, 2*np.pi) for n in range(self.ph_samples)])

      # # Full-spoke golden-angle radial sampling
      # GR = np.deg2rad(111.25)
      # theta = np.array([np.mod(np.pi/GR*n, 2*np.pi) for n in range(self.ph_samples)])

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
            t[::ro,ph,0] = enc_time + ro_grad0.dur + ro_grad.slope + dt
          else:
            t[::ro,ph,0] = t[:,shot[idx-1]].max() + ro_grad.slope + ro_grad.slope + dt[::ro]

      # Fill kz coordinates
      for s in range(self.slices):
        kspace[2][:,:,s] = kz[s]
        t[:,:,s] = t[:,:,0]

      # Calculate echo time
      self.echo_time = (enc_time + ro_grad0.dur + 0.5*self.lines_per_shot*ro_grad.dur).to('ms')

      return (kspace, t)

    def plot_trajectory(self):
      ''' Show kspace points and time map'''
      if MPI_rank == 0:
        # plt.rcParams['text.usetex'] = True
        plt.rcParams.update({'font.size': 16})

        # Plot kspace locations and times
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for shot in self.shots:
          kxx = np.concatenate((np.array([0]), self.points[0][:,shot,0].flatten('F')))
          kyy = np.concatenate((np.array([0]), self.points[1][:,shot,0].flatten('F')))
          ax[0].plot(kxx,kyy)
        ax[0].set_xlabel('$k_x ~(1/m)$')
        ax[0].set_ylabel('$k_y ~(1/m)$')

        im = ax[1].scatter(self.points[0][:,:,0],self.points[1][:,:,0],c=self.times,s=2.5,cmap='turbo')
        ax[1].set_xlabel('$k_x ~(1/m)$')
        ax[1].set_yticklabels([])
        fig.tight_layout()
        cbar = fig.colorbar(im, orientation='vertical', ax=ax)
        cbar.ax.set_title('Time [ms]')
        plt.show()


# Spiral trajectory
class SpiralStack(Trajectory):
    def __init__(self, *args, interleaves=20, parameters=[], **kwargs):
      super().__init__(*args, **kwargs)
      self.interleaves = self.check_ph_enc_lines(interleaves)
      self.parameters = parameters
      (self.points, self.times) = self.kspace_points()

    def kspace_points(self):
      ''' Get kspace points '''
      # Spiral parameters
      N = self.interleaves        # Number of interleaves
      f = self.FOV[0]             # Field-of-view
      k0 = self.parameters['k0']
      k1 = self.parameters['k1']
      S = self.parameters['Slew-rate']    # Slew-rate [T/m/s]
      gamma = self.gamma
      r = self.pxsz[0]

      # Radial distance definition
      kr0 = k0*0.5*self.k_bw[0]
      kr1 = k1*0.5*self.k_bw[0]
      kr = np.linspace(0, 1, self.ro_samples)
      kr = kr1*(kr**1)
      phi = 2*np.pi*f*kr/N

      # Complex trajectory
      K = kr*np.exp(1j*phi)

      # Time needed to acquire one interleave
      t_sc = np.sqrt(2)*np.pi*f/(3*N*np.sqrt(1e+6*gamma*1e-3*S)*r**(3/2))
      dt = np.linspace(0.0, t_sc, self.ro_samples)    

      # kspace locations
      kx = np.real(K)
      ky = np.imag(K)
      kspace = (np.zeros([self.ro_samples, self.interleaves]),
                np.zeros([self.ro_samples, self.interleaves]))

      # Angles for each ray
      theta = np.linspace(0, 2*np.pi, self.interleaves+1)
      theta = theta[0:-1]

      # kspace times and locations
      t = np.zeros([self.ro_samples, self.interleaves])
      for sp in range(0, self.interleaves):
        # Rotation matrix
        kspace[0][:,sp] = kx*np.cos(theta[sp]) + ky*np.sin(theta[sp])
        kspace[1][:,sp] = -kx*np.sin(theta[sp]) + ky*np.cos(theta[sp])

        if sp % self.lines_per_shot == 0:
          t[:,sp] = dt
        else:
          t[:,sp] = t[-1,sp-1] + dt

      # Send the information to each process if running in parallel
      kspace, t, local_idx = scatterKspace(kspace, t)
      self.local_idx = local_idx

      return (kspace, t)

    def plot_trajectory(self):
      ''' Show kspace points '''
      # Plot kspace locations and times
      fig, axs = plt.subplots(1, 2, figsize=(10, 5))
      axs[0].plot(self.points[0].flatten('F'),self.points[1].flatten('F'))
      im = axs[1].scatter(self.points[0],self.points[1],c=self.times,s=1.5)
      axs[0].set_xlabel('k_x (1/m)')
      axs[0].set_ylabel('k_y (1/m)')
      axs[1].set_xlabel('k_x (1/m)')
      axs[1].set_ylabel('k_y (1/m)')
      cbar = fig.colorbar(im, ax=axs[1])
      cbar.ax.tick_params(labelsize=8) 
      cbar.ax.set_title('Time [ms]',fontsize=8)
      plt.show()


def SpiralCalculator():
  return True
