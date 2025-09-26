import os
import pickle
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from pint import Quantity as Q_

from feelmri.KSpaceTraj import CartesianStack
from feelmri.Math import Rx, Ry, Rz
from feelmri.MRImaging import SliceProfile, VelocityEncoding
from feelmri.MRObjects import RF, Gradient, Scanner, Sequence, SequenceBlock
from feelmri.Parameters import ParameterHandler
from feelmri.Phantom import FEMPhantom
from feelmri.pyMRIEncoding import PC

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
  phantom.orient(MPS_ori, LOC)

  # # We can use only a submesh to speed up the simulation. The submesh is
  # # created by selecting the elements that are inside the FOV. We can also
  # # refine the submesh to increase the number of nodes and elements
  # midpoints = np.array([np.mean(phantom.mesh['nodes'][e], axis=0) for e in phantom.mesh['elems']])
  # condition = (np.abs(midpoints[:,2]) <= 0.6*parameters.FOV[2]) > 1e-3
  # markers = np.where(condition)[0]
  # phantom.create_submesh(markers, refine=False, element_size=0.0012)

  # Translate phantom to obtain the desired slice location
  cp_nodes = cp.asarray(phantom.mesh['nodes'], dtype=cp.float32)

  # Assemble mass matrix for integrals (just once)
  M = cp_csr_matrix(phantom.mass_matrix(lumped=True, use_submesh=False), dtype=cp.float32)

  # Create scanner object
  scanner = Scanner(gradient_strength=Q_(parameters.G_max,'mT/m'), gradient_slew_rate=Q_(parameters.G_sr,'mT/m/ms'))

  # Slice profile
  rf = RF(scanner=scanner, NbLobes=[8, 8], alpha=0.46, shape='apodized_sinc', flip_angle=Q_(np.deg2rad(8),'rad') , ref=Q_(0.0,'ms'))
  sp = SliceProfile(delta_z=Q_(parameters.FOV[2], 'm'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=True, 
    bandwidth=Q_(10000, 'Hz'), #'maximum',
    refocusing_area_frac=0.975)
  # sp.optimize(frac_start=1.119, frac_end=1.2, N=100, profile_samples=50)
  # profile = cp.asarray(sp.interp_profile(cp_nodes[:,2].get()), dtype=cp.complex64).reshape(-1, 1)
  profile = cp.asarray((np.abs(cp_nodes[:,2].get()) < 0.5*parameters.FOV[2])*sp.interp_profile(cp_nodes[:,2].get()), dtype=cp.complex64).reshape(-1, 1)

  # Bipolar gradient
  bp1 = Gradient(scanner=scanner, ref=sp.rf.ref + sp.rf.dur2, axis=1)
  bp2 = bp1.make_bipolar(Q_(enc.VENC, 'm/s'))

  # Excitation sequence to obtain slice profile
  block = SequenceBlock(gradients=[sp.dephasing, sp.rephasing, bp1, bp2],
                        rf_pulses=[sp.rf], 
                        dt_rf=Q_(1e-2, 'ms'), 
                        dt_gr=Q_(1e-2, 'ms'), 
                        dt=Q_(1, 'ms'))
  seq =  Sequence(blocks=[block])
  seq.plot()

  # Field inhomogeneity
  x = phantom.mesh['nodes']
  gammabar = 1.0e+6*42.58 # Hz/T
  delta_B0 = x[:,0]**2 + x[:,1]**2 #+ x[:,2]**2 # spatial distribution
  delta_B0 /= np.abs(delta_B0).max()  # normalization
  delta_B0 *= 2*np.pi * gammabar * (1.5 * 1e-6)
  delta_B0 = cp.asarray(delta_B0, dtype=cp.float32)
  # delta_B0 *= 0.0
  # delta_phi_v = delta_B0 * bipolar.dur_
  # delta_phi_v *= 0.0

  # Path to export the generated data
  export_path = Path('MRImages/{:s}_V{:.0f}.pkl'.format(parameters.Sequence, 100.0*parameters.VENC))

  # Make sure the directory exist
  os.makedirs(str(export_path.parent), exist_ok=True)

  # Generate kspace trajectory
  traj = CartesianStack(FOV=Q_(parameters.FOV,'m'),
    t_start=block.time_extent[1] - sp.rf.ref,
    res=parameters.RES, 
    oversampling=parameters.Oversampling, 
    lines_per_shot=parameters.LinesPerShot, 
    MPS_ori=MPS_ori, 
    LOC=LOC, 
    receiver_bw=Q_(parameters.r_BW,'Hz'), 
    plot_seq=False)

  # Convert trajectory numpy arrays to cupy arrays
  cp_traj_points = [cp.asarray(traj.points[i], dtype=cp.float32) for i in range(len(traj.points))] 
  cp_traj_times = cp.asarray(traj.times.m_as('s'), dtype=cp.float32)

  # Print echo time
  print('Echo time = {:.1f} ms'.format(traj.echo_time))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = cp.zeros([ro_samples, ph_samples, slices, enc.nb_directions, phantom.Nfr], dtype=cp.complex64)

  # Iterate over cardiac phases
  t0 = time.perf_counter()
  for fr in range(phantom.Nfr):

      t0_fr = time.perf_counter()

      # Read velocity data in frame fr
      phantom.read_data(fr)
      velocity = phantom.point_data['velocity'] @ traj.MPS_ori
      # submesh_velocity = phantom.interpolate_to_submesh(velocity)
      cp_velocity = cp.asarray(velocity, dtype=cp.float32)
      cp_enc_velocity = cp.asarray(enc.encode(velocity), dtype=cp.float32)

      # Generate 4D flow image
      K[traj.local_idx,:,:,:,fr] = PC(M, cp_traj_points, cp_traj_times, cp_nodes, cp_velocity, cp_enc_velocity, delta_B0, parameters.T2star, profile)

      print('Elapsed time frame {:d}: {:.2f} s'.format(fr, time.perf_counter()-t0_fr))
      
      # Save kspace for debugging purposes
      if preview:
        with open(str(export_path), 'wb') as f:
          pickle.dump({'kspace': cp.asnumpy(K), 'MPS_ori': cp.asnumpy(MPS_ori), 'LOC': cp.asnumpy(LOC), 'traj': traj}, f)

  # Store elapsed time
  pc_time = time.perf_counter() - t0

  # Print elapsed times
  print('Elapsed time-per-frame for 4D Flow generation: {:.2f} s'.format(pc_time/phantom.Nfr))
  print('Elapsed time for 4D Flow generation: {:.2f} s'.format(pc_time))

  # Export generated data
  # K_scaled = scale_data(K, mag=False, real=True, imag=True, dtype=np.uint64)
  with open(str(export_path), 'wb') as f:
    pickle.dump({'kspace': cp.asnumpy(K), 'MPS_ori': cp.asnumpy(MPS_ori), 'LOC': cp.asnumpy(LOC), 'traj': traj}, f)

  stream.synchronize()
