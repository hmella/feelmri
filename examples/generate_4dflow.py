import os
import pickle
import time
from pathlib import Path
from pint import Quantity as Q_

import numpy as np

from FEelMRI.KSpaceTraj import CartesianStack
from FEelMRI.Math import Rx, Ry, Rz
from FEelMRI.MPIUtilities import MPI_print, MPI_rank, gather_image
from FEelMRI.PhaseContrast import PC
from FEelMRI.MRObjects import Gradient, RF, Scanner
from FEelMRI.MRImaging import SliceProfile, VelocityEncoding
from FEelMRI.Parameters import ParameterHandler
from FEelMRI.Phantom import FEMPhantom

if __name__ == '__main__':

  # Import imaging parameters
  parameters = ParameterHandler('parameters/aorta_volume.yaml')

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
  M = phantom.mass_matrix(lumped=True)

  # Create scanner object
  scanner = Scanner(gradient_strength=Q_(parameters.G_max,'mT/m'), gradient_slew_rate=Q_(parameters.G_sr,'mT/m/ms'))

  # Slice profile
  rf = RF(scanner=scanner, NbLobes=[4,4], alpha=0.46, shape='apodized_sinc', flip_angle=Q_(np.deg2rad(8),'rad') , t_ref=Q_(0.0,'ms'))
  sp = SliceProfile(delta_z=Q_(parameters.FOV[2], 'm'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=True, 
    bandwidth=Q_(15000, 'Hz'), # 'maximum', 
    refocusing_area_frac=1.0242424242424242)
  # sp.optimize(frac_start=1.0, frac_end=1.2, N=100, profile_samples=50)
  profile = sp.interp_profile(nodes[:,2])

  # Bipolar gradient
  bipolar = Gradient(scanner=scanner, t_ref=rf.t_ref)
  bipolar.make_bipolar(Q_(enc.VENC, 'm/s'))

  # Field inhomogeneity
  x = phantom.mesh['nodes']
  gammabar = 1.0e+6*42.58 # Hz/T
  delta_B0 = x[:,0]**2 + x[:,1]**2 #+ x[:,2]**2 # spatial distribution
  delta_B0 /= np.abs(delta_B0).max()  # normalization
  delta_B0 *= 2*np.pi * gammabar * (1.5 * 1e-6) 
  delta_B0 *= 0.0
  # delta_phi_v = delta_B0 * bipolar.dur_
  # delta_phi_v *= 0.0

  # Path to export the generated data
  export_path = Path('MRImages/{:s}_V{:.0f}.pkl'.format(parameters.Sequence, 100.0*parameters.VENC))

  # Make sure the directory exist
  os.makedirs(str(export_path.parent), exist_ok=True)

  # Generate kspace trajectory
  traj = CartesianStack(FOV=Q_(parameters.FOV,'m'), 
    res=parameters.RES, 
    oversampling=parameters.Oversampling, 
    lines_per_shot=parameters.LinesPerShot, 
    G_enc=bipolar, 
    MPS_ori=MPS_ori, 
    LOC=LOC, 
    receiver_bw=Q_(parameters.r_BW,'Hz'), 
    plot_seq=True)
  # traj.plot_trajectory()

  # Print echo time
  MPI_print('Echo time = {:.2f} ms'.format(traj.echo_time.m_as('ms')))

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
    K[traj.local_idx,:,:,:,fr] = PC(MPI_rank, M, traj.local_points, traj.local_times.m_as('s'), nodes, velocity, enc.encode(velocity), delta_B0, T2star, profile)
    times.append(time.time()-t0)

    # Show elapsed time from terminal
    MPI_print('Elapsed time: {:.2f} s'.format(times[-1]))

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
