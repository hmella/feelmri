import os
import pickle
from pathlib import Path

import numpy as np

from FEelMRI.KSpaceTraj import Cartesian
from FEelMRI.Math import Rx, Ry, Rz
from FEelMRI.MPIUtilities import MPI_print, MPI_rank, gather_image
from FEelMRI.MRIEncoding import WaterFat
from FEelMRI.MRImaging import Gradient, SliceProfile
from FEelMRI.Phantom import FEMPhantom
from FEelMRI.Parameters import ParameterHandler

if __name__ == '__main__':

  # Preview partial results
  preview = True

  # Import imaging parameters
  parameters = ParameterHandler('parameters/PARAMETERS_wf.yaml')

  # Imaging orientation paramters
  theta_x = np.deg2rad(parameters.theta_x)
  theta_y = np.deg2rad(parameters.theta_y)
  theta_z = np.deg2rad(parameters.theta_z)
  MPS_ori = Rz(theta_z)@Rx(theta_x)@Ry(theta_y)
  LOC = parameters.LOC

  # Navier-Stokes simulation data to be used
  phantom_file = 'phantoms/water_and_fat.xdmf'

  # Create FEM phantom object
  phantom = FEMPhantom(path=phantom_file, scale_factor=0.01)

  # Translate phantom to obtain the desired slice location
  nodes = phantom.translate(MPS_ori, LOC)

  # Assemble mass matrix for integrals (just once)
  M = phantom.mass_matrix(lumped=True)

  # Slice profile
  Gss = Gradient(Gr_max=parameters.G_max, Gr_sr=parameters.G_sr)
  sp = SliceProfile(delta_z=parameters.FOV[2], flip_angle=np.deg2rad(10), NbLobes=[4,4], RFShape='apodized_sinc', ZPoints=100, dt=1e-5, plot=True, Gss=Gss, bandwidth='maximum', refocusing_area_frac=1.512536443148688)
  # sp.optimize(frac_start=0.9, frac_end=1.2, N=50, ZPoints=50)
  profile = sp.interp_profile(nodes[:,2])
  # profile = np.abs(nodes[:,2]) < parameters.FOV[2]/2

  # Field inhomogeneity
  x = phantom.mesh['nodes']
  gammabar = 1.0e+6*42.58 # Hz/T
  delta_B0 = x[:,0]**2 + x[:,1]**2 #+ x[:,2]**2 # spatial distribution
  delta_B0 /= np.abs(delta_B0).max()  # normalization
  delta_B0 *= 2*np.pi * gammabar * (1.5 * 1e-6)

  # Path to export the generated data
  export_path = Path('MRImages/{:s}_WF.pkl'.format(parameters.Sequence))

  # Make sure the directory exist
  os.makedirs(str(export_path.parent), exist_ok=True)

  # Generate kspace trajectory
  dummy = Gradient(Gr_max=0, slope=0, lenc=0, t_ref=sp.rf_dur2)
  traj = Cartesian(FOV=parameters.FOV, res=parameters.RES, oversampling=parameters.Oversampling, lines_per_shot=parameters.LinesPerShot, G_enc=dummy, MPS_ori=MPS_ori, LOC=LOC, receiver_bw=parameters.r_BW, Gr_max=parameters.G_max, Gr_sr=parameters.G_sr, plot_seq=False)
  traj.plot_trajectory()

  # Print echo time
  MPI_print('Echo time = {:.2f} ms'.format(1000.0*traj.echo_time))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, 1], dtype=np.complex64)

  # List to store how much is taking to generate one volume
  times = []

  # Read velocity data in frame fr
  phantom.read_data(0)

  # Build water and frat fractions
  rho_f = phantom.point_data['point_markers'] - 1.0
  rho_f /= rho_f.max()
  rho_w = 1.0 - rho_f

  M0 = np.ones_like(rho_f)

  T2 = 274.0/1000.0*np.ones_like(rho_f)
  T2[rho_f > 0.0] = 70.0/1000.0

  delta_f = 217
  chemical_shift = 2*np.pi*(rho_f > 0.0)*delta_f

  # Generate 4D flow image
  K[traj.local_idx,:,:,:] = WaterFat(MPI_rank, M, traj.local_points, traj.local_times, nodes, delta_B0, M0, T2, profile, rho_w, rho_f, chemical_shift)

# Gather results
K = gather_image(K)

# Export generated data
if MPI_rank==0:
  with open(str(export_path), 'wb') as f:
    pickle.dump({'kspace': K, 'MPS_ori': MPS_ori, 'LOC': LOC, 'traj': traj}, f)