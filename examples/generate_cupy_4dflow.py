import os
import pickle
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix

from FEelMRI.KSpaceTraj import Cartesian
from FEelMRI.Math import Rx, Ry, Rz
from FEelMRI.MRImaging import Gradient, SliceProfile, VelocityEncoding
from FEelMRI.Parameters import ParameterHandler
from FEelMRI.Phantom import FEMPhantom
from FEelMRI.pyMRIEncoding import PC

if __name__ == '__main__':

  stream = cp.cuda.stream.Stream(non_blocking=False)
  cp.show_config()

  # Preview partial results
  preview = False

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
  cp_nodes = cp.asarray(phantom.translate(MPS_ori, LOC), dtype=cp.float32)

  # Assemble mass matrix for integrals (just once)
  M = cp_csr_matrix(phantom.mass_matrix(), dtype=cp.float32)

  # Slice profile
  Gss = Gradient(Gr_max=parameters.G_max, Gr_sr=parameters.G_sr)
  sp = SliceProfile(delta_z=parameters.FOV[2], flip_angle=np.deg2rad(10), NbLobes=[4,4], RFShape='apodized_sinc', ZPoints=100, dt=1e-5, plot=False, Gss=Gss, bandwidth='maximum', refocusing_area_frac=1.5114285714285716)
  profile = cp.asarray(sp.interp_profile(cp_nodes[:,2].get()), dtype=cp.complex64).reshape(-1, 1)

  # Bipolar gradient
  bipolar = Gradient(Gr_max=parameters.G_max, Gr_sr=parameters.G_sr, t_ref=sp.rf_dur2)
  bipolar.make_bipolar(enc.VENC)

  # Field inhomogeneity
  x = phantom.mesh['nodes']
  gammabar = 1.0e+6*42.58 # Hz/T
  delta_B0 = x[:,0]**2 + x[:,1]**2 #+ x[:,2]**2 # spatial distribution
  delta_B0 /= np.abs(delta_B0).max()  # normalization
  delta_B0 *= 2*np.pi * gammabar * (1.5 * 1e-6)
  delta_B0 = cp.asarray(delta_B0, dtype=cp.float32)
  delta_phi_v = delta_B0 * bipolar.dur_

  # Path to export the generated data
  export_path = Path('MRImages/{:s}_V{:.0f}.pkl'.format(parameters.Sequence, 100.0*parameters.VENC))

  # Make sure the directory exist
  os.makedirs(str(export_path.parent), exist_ok=True)

  # Generate kspace trajectory
  traj = Cartesian(FOV=parameters.FOV, res=parameters.RES, oversampling=parameters.Oversampling, lines_per_shot=parameters.LinesPerShot, G_enc=bipolar, MPS_ori=MPS_ori, LOC=LOC, receiver_bw=parameters.r_BW, Gr_max=parameters.G_max, Gr_sr=parameters.G_sr, plot_seq=False)

  # Convert trajectory numpy arrays to cupy arrays
  cp_traj_points = [cp.asarray(traj.points[i], dtype=cp.float32) for i in range(len(traj.points))] 
  cp_traj_times = cp.asarray(traj.times, dtype=cp.float32)

  # Print echo time
  print('Echo time = {:.1f} ms'.format(1000.0*traj.echo_time))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = cp.zeros([ro_samples, ph_samples, slices, enc.nb_directions, phantom.Nfr], dtype=cp.complex64)

  # List to store how much is taking to generate one frame
  times = []

  # Iterate over cardiac phases
  for fr in range(phantom.Nfr):

    # Read velocity data in frame fr
    phantom.read_data(fr)
    velocity = phantom.point_data['velocity'] @ traj.MPS_ori
    cp_velocity = cp.asarray(velocity, dtype=cp.float32)
    cp_enc_velocity = cp.asarray(enc.encode(velocity), dtype=cp.float32)

    # Generate 4D flow image
    print('Generating frame {:d}'.format(fr))
    t0 = time.time()
    K[traj.local_idx,:,:,:,fr] = PC(M, cp_traj_points, cp_traj_times, cp_nodes, cp_velocity, cp_enc_velocity, delta_B0, parameters.T2star, profile)
    times.append(time.time()-t0)

    # Show elapsed time from terminal
    print('Elapsed time: {:.2f} s'.format(times[-1]))

    # Save kspace for debugging purposes
    if preview:
      with open(str(export_path), 'wb') as f:
        pickle.dump({'kspace': cp.asnumpy(K), 'MPS_ori': cp.asnumpy(MPS_ori), 'LOC': cp.asnumpy(LOC), 'traj': traj}, f)

    # Synchronize MPI processes
    print(np.array(times).mean())

  # Show mean time that takes to generate each 3D volume
  print(np.array(times).mean())

  # Export generated data
  # K_scaled = scale_data(K, mag=False, real=True, imag=True, dtype=np.uint64)
  with open(str(export_path), 'wb') as f:
    pickle.dump({'kspace': cp.asnumpy(K), 'MPS_ori': cp.asnumpy(MPS_ori), 'LOC': cp.asnumpy(LOC), 'traj': traj}, f)

  stream.synchronize()
