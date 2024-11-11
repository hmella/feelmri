import os
import pickle
import time
from pathlib import Path

import numpy as np

from FEelMRI.KSpaceTraj import Cartesian
from FEelMRI.Math import Rx, Ry, Rz
from FEelMRI.MPIUtilities import MPI_print, MPI_rank, gather_image
from FEelMRI.PhaseContrast import PC
from FEelMRI.MRImaging import Gradient, SliceProfile, VelocityEncoding
from FEelMRI.Parameters import ParameterHandler
from FEelMRI.Phantom import FEMPhantom

if __name__ == '__main__':

  # Preview partial results
  preview = False

  # Import imaging parameters
  parameters = ParameterHandler('parameters/aorta_slice.yaml')

  # Imaging orientation paramters
  theta_x = np.deg2rad(parameters.theta_x)
  theta_y = np.deg2rad(parameters.theta_y)
  theta_z = np.deg2rad(parameters.theta_z)
  MPS_ori = Rz(theta_z)@Rx(theta_x)@Ry(theta_y)
  LOC = parameters.LOC

  # Velocity encoding parameters
  venc_dirs = list(parameters.Directions.values())
  enc = VelocityEncoding(parameters.VENC, np.array(venc_dirs))

  # Navier-Stokes simulation data to be used
  phantom_file = 'phantoms/aorta_CFD.xdmf'

  # Create FEM phantom object
  phantom = FEMPhantom(path=phantom_file, velocity_label='velocity', scale_factor=0.01)

  # Translate phantom to obtain the desired slice location
  nodes = phantom.translate(MPS_ori, LOC)

  # Assemble mass matrix for integrals (just once)
  M = phantom.mass_matrix()

  # Slice profile
  Gss = Gradient(Gr_max=parameters.G_max, Gr_sr=parameters.G_sr)
  sp = SliceProfile(delta_z=parameters.FOV[2], flip_angle=np.deg2rad(10), NbLobes=[4,4], RFShape='apodized_sinc', ZPoints=100, dt=1e-5, plot=False, Gss=Gss, bandwidth='maximum', refocusing_area_frac=1)
  # sp.optimize(frac_start=1.38, frac_end=1.52, N=50, ZPoints=50)
  profile = sp.interp_profile(nodes[:,2])
  # profile = np.abs(nodes[:,2]) < parameters.FOV[2]/2

  # Bipolar gradient
  bipolar = Gradient(Gr_max=parameters.G_max, Gr_sr=parameters.G_sr, t_ref=sp.rf_dur2)
  bipolar.make_bipolar(enc.VENC)

  # Field inhomogeneity
  x = phantom.mesh['nodes']
  gammabar = 1.0e+6*42.58 # Hz/T
  delta_B0 = x[:,0]**2 + x[:,1]**2 #+ x[:,2]**2 # spatial distribution
  delta_B0 /= np.abs(delta_B0).max()  # normalization
  delta_B0 *= 2*np.pi * gammabar * (1.5 * 1e-6)
  delta_phi_v = delta_B0 * bipolar.dur_

  # Path to export the generated data
  export_path = Path('MRImages/{:s}_V{:.0f}.pkl'.format(parameters.Sequence, 100.0*parameters.VENC))

  # Make sure the directory exist
  os.makedirs(str(export_path.parent), exist_ok=True)

  # Generate kspace trajectory
  traj = Cartesian(FOV=parameters.FOV, res=parameters.RES, oversampling=parameters.Oversampling, lines_per_shot=parameters.LinesPerShot, G_enc=bipolar, MPS_ori=MPS_ori, LOC=LOC, receiver_bw=parameters.r_BW, Gr_max=parameters.G_max, Gr_sr=parameters.G_sr, plot_seq=False)
  # traj.plot_trajectory()

  # Print echo time
  MPI_print('Echo time = {:.2f} ms'.format(1000.0*traj.echo_time))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, enc.nb_directions, phantom.Nfr], dtype=np.complex64)

  # List to store how much is taking to generate each frame
  times = []

  T2star = (parameters.T2star * np.ones([nodes.shape[0]])).astype(np.float32)

  # Iterate over cardiac phases
  for fr in range(phantom.Nfr):

    # Read velocity data in frame fr
    phantom.read_data(fr)
    velocity = phantom.point_data['velocity'] @ traj.MPS_ori

    # Generate 4D flow image
    MPI_print('Generating frame {:d}'.format(fr))
    t0 = time.time()
    K[traj.local_idx,:,:,:,fr] = PC(MPI_rank, M, traj.local_points, traj.local_times, nodes, velocity, enc.encode(velocity), delta_B0, T2star, profile)
    times.append(time.time()-t0)

    # Show elapsed time from terminal
    MPI_print('Elapsed time: {:.2f} s'.format(times[-1]))

    # Save kspace for debugging purposes
    if preview:
      K_copy = gather_image(K)
      if MPI_rank==0:
        with open(str(export_path), 'wb') as f:
          pickle.dump({'kspace': K_copy, 'MPS_ori': MPS_ori, 'LOC': LOC, 'traj': traj}, f)

    # Synchronize MPI processes
    MPI_print(np.array(times).mean())
    # MPI_comm.Barrier()

  # Show mean time that takes to generate each 3D volume
  MPI_print(np.mean(times))

  # Gather results
  K = gather_image(K)

  # Export generated data
  if MPI_rank==0:
    with open(str(export_path), 'wb') as f:
      pickle.dump({'kspace': K, 'MPS_ori': MPS_ori, 'LOC': LOC, 'traj': traj}, f)
