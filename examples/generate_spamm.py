import os
import pickle
import time
from pathlib import Path

import numpy as np

from FEelMRI.FiniteElements import MassAssemble
from FEelMRI.Tagging import SPAMM
from FEelMRI.KSpaceTraj import CartesianStack, Gradient
from FEelMRI.Math import Rx, Ry, Rz
from FEelMRI.MPIUtilities import MPI_comm, MPI_print, MPI_rank, gather_image
from FEelMRI.MRImaging import SliceProfile, PositionEncoding
from FEelMRI.Phantom import FEMPhantom
from FEelMRI.Parameters import ParameterHandler

def f1(t, T1, alpha, n):
  return np.sin(alpha)*np.cos(alpha)**n*(1.0 - np.exp(-t/T1))

def f2(t, T1, alpha, n):
  return np.sin(alpha)*np.cos(alpha)**n*np.exp(-t/T1)

if __name__ == '__main__':

  # Preview partial results
  preview = True

  # Import imaging parameters
  parameters = ParameterHandler('parameters/PARAMETERS_LV.yaml')

  # Imaging orientation paramters
  theta_x = np.deg2rad(parameters.theta_x)
  theta_y = np.deg2rad(parameters.theta_y)
  theta_z = np.deg2rad(parameters.theta_z)
  MPS_ori = Rz(theta_z)@Rx(theta_x)@Ry(theta_y)
  LOC = parameters.LOC

  # Velocity encoding parameters
  ke_dirs = list(parameters.Directions.values())
  enc = PositionEncoding(parameters.ke, np.array(ke_dirs))

  # Navier-Stokes simulation data to be used
  phantom_file = 'phantoms/LV_displacement.xdmf'

  # Create FEM phantom object
  phantom = FEMPhantom(path=phantom_file, scale_factor=1.0)

  # Translate phantom to obtain the desired slice location
  nodes = phantom.translate(MPS_ori, LOC)

  # Assemble mass matrix for integrals (just once)
  M = phantom.mass_matrix()

  # Calculate position encoding gradient
  G_tag = Gradient(Gr_max=parameters.G_max, Gr_sr=parameters.G_sr)
  G_tag.match_area(parameters.ke/G_tag._gamma)

  # Slice profile
  Gss = Gradient(Gr_max=parameters.G_max, Gr_sr=parameters.G_sr)
  sp = SliceProfile(delta_z=parameters.FOV[2], flip_angle=np.deg2rad(10), NbLobes=[2,2], RFShape='sinc', ZPoints=100, dt=1e-5, plot=True, Gss=Gss, bandwidth='maximum', refocusing_area_frac=1.6116326530612246)
  # sp.optimize(frac_start=1.6, frac_end=1.63, N=50, ZPoints=50)
  # profile = np.abs(nodes[:,2]) < FOV[2]/2

  # Field inhomogeneity
  x =  phantom.mesh['nodes']
  gammabar = 1.0e+6*42.58 # Hz/T
  delta_B0 = (x[:,0] + x[:,1] + x[:,2])**2  # spatial distribution
  delta_B0 /= 2.0*np.abs(delta_B0).max()  # normalization
  delta_B0 *= 1.5*1e-6  # scaling (1 ppm of 1.5T)
  phi_B0 = 2*np.pi*gammabar*delta_B0
  phi_B0_phase = np.stack((phi_B0, phi_B0), axis=1) * G_tag.dur_

  # Path to export the generated data
  export_path = Path('MRImages/{:s}_V{:.0f}.pkl'.format(parameters.Sequence, parameters.ke))

  # Make sure the directory exist
  os.makedirs(str(export_path.parent), exist_ok=True)

  # Generate kspace trajectory
  print(sp.rf_dur2)
  dummy = Gradient(Gr_max=0, slope=0, lenc=0, t_ref=sp.rf_dur2)
  traj = CartesianStack(FOV=parameters.FOV, res=parameters.RES, oversampling=parameters.Oversampling, lines_per_shot=parameters.LinesPerShot, G_enc=dummy, MPS_ori=MPS_ori, LOC=LOC, receiver_bw=parameters.r_BW, Gr_max=parameters.G_max, Gr_sr=parameters.G_sr, plot_seq=True)
  traj.plot_trajectory()

  # Print echo time
  MPI_print('Echo time = {:.2f} ms'.format(1000.0*traj.echo_time))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, enc.nb_directions, phantom.Nfr], dtype=np.complex64)

  # List to store how much is taking to generate one volume
  times = []

  # Encode position
  enc_position = enc.encode(nodes)
  Grid = 0.5*(np.exp(1j*enc_position) + np.exp(-1j*enc_position))
  Grid *= np.exp(1j*phi_B0_phase)

  # Iterate over cardiac phases
  t = 0.0
  for fr in range(phantom.Nfr):

    # Acquisition time
    t += parameters.TimeSpacing

    # In-plane magnetization
    Mxy = f1(t, 0.85, np.deg2rad(10), fr) - f2(t, 0.85, np.deg2rad(10), fr)*Grid

    # Read displacement data in frame fr
    phantom.read_data(fr)
    displacement = phantom.point_data['displacement'] @ traj.MPS_ori
    profile = sp.interp_profile(nodes[:,2] + displacement[:,2])

    # Assemble mass matrix for integrals (just once)
    M = MassAssemble(phantom.mesh['elems'], phantom.mesh['nodes'] + displacement) 

    # Generate 4D flow image
    MPI_print('Generating frame {:d}'.format(fr))
    t0 = time.time()
    K[traj.local_idx,:,:,:,fr] = SPAMM(MPI_rank, M, traj.local_points, traj.local_times, phantom.mesh['nodes'] + displacement, Mxy, phi_B0, parameters.T2star, profile)
    t1 = time.time()
    times.append(t1-t0)

    # Save kspace for debugging purposes
    if preview:
      K_copy = gather_image(K)
      if MPI_rank==0:
        with open(str(export_path), 'wb') as f:
          pickle.dump({'kspace': K_copy, 'MPS_ori': MPS_ori, 'LOC': LOC, 'traj': traj}, f)

    # Synchronize MPI processes
    MPI_print(np.array(times).mean())
    MPI_comm.Barrier()

  # Show mean time that takes to generate each 3D volume
  MPI_print(np.array(times).mean())

  # Gather results
  K = gather_image(K)

  # Export generated data
  if MPI_rank==0:
    with open(str(export_path), 'wb') as f:
      pickle.dump({'kspace': K, 'MPS_ori': MPS_ori, 'LOC': LOC, 'traj': traj}, f)