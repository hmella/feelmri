import os
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
import pickle
import time
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from FEelMRI.IO import XDMFFile
from FEelMRI.KSpaceTraj import CartesianStack
from FEelMRI.Math import Rx, Ry, Rz
from FEelMRI.Motion import PODTrajectory
from FEelMRI.MPIUtilities import MPI_print, MPI_rank, gather_data
from FEelMRI.MRImaging import SliceProfile, VelocityEncoding
from FEelMRI.MRObjects import (RF, BlochSolver, Gradient, Scanner, Sequence,
                               SequenceBlock)
from FEelMRI.Parameters import ParameterHandler
from FEelMRI.Phantom import FEMPhantom
from FEelMRI.PhaseContrast import PC
from FEelMRI.Plotter import MRIPlotter
from FEelMRI.Recon import CartesianRecon

if __name__ == '__main__':

  # Import imaging parameters
  parameters = ParameterHandler('parameters/phase_contrast.yaml')

  # Imaging orientation paramters
  theta_x = np.deg2rad(parameters.theta_x)
  theta_y = np.deg2rad(parameters.theta_y)
  theta_z = np.deg2rad(parameters.theta_z)
  MPS_ori = Rz(theta_z)@Rx(theta_x)@Ry(theta_y)
  LOC = parameters.LOC

  # Velocity encoding parameters
  venc_dirs = list(parameters.Directions.values())
  enc = VelocityEncoding(parameters.VENC, np.array(venc_dirs))

  # Create FEM phantom object
  phantom = FEMPhantom(path='phantoms/aorta_CFD.xdmf', velocity_label='velocity', scale_factor=0.01)

  # Translate phantom to obtain the desired slice location
  phantom.orient(MPS_ori, LOC)

  # We can use only a submesh to speed up the simulation. The submesh is
  # created by selecting the elements that are inside the FOV. We can also
  # refine the submesh to increase the number of nodes and elements
  midpoints = np.array([np.mean(phantom.global_nodes[e, :], axis=0) for e in phantom.global_elements])
  condition = (np.abs(midpoints[:,2]) <= 0.6*parameters.FOV[2]) > 1e-3
  markers = np.where(condition)[0]
  phantom.create_submesh(markers, refine=False, element_size=0.0012)

  # Distribute the mesh across MPI processes
  phantom.distribute_mesh()

  # Create POD trajectory object
  trajectory = np.zeros([phantom.global_nodes.shape[0], phantom.global_nodes.shape[1], phantom.Nfr], dtype=np.float32)
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr
    phantom.read_data(fr)
    velocity = phantom.point_data['velocity'] @ MPS_ori
    submesh_velocity = phantom.interpolate_to_submesh(velocity, local=False)
    trajectory[..., fr] = submesh_velocity

  # Define POD object
  times = 1000*np.linspace(0, (phantom.Nfr-1)*parameters.TimeSpacing, phantom.Nfr)
  pod_trajectory = PODTrajectory(time_array=times,
                                 data=trajectory,
                                 global_to_local=phantom.global_to_local_nodes,
                                 n_modes=5,
                                 taylor_order=10,
                                 is_periodic=True)

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=Q_(parameters.G_max,'mT/m'), gradient_slew_rate=Q_(parameters.G_sr,'mT/m/ms'))

  # Field inhomogeneity
  def spatial(x):
      return x[:,0] + x[:,1] + x[:,2]
  delta_B0 = spatial(phantom.local_nodes)
  delta_B0 /= np.abs(spatial(phantom.global_nodes).flatten()).max()
  delta_B0 *= 1.5 * 1e-6 * 0            # 1.5 ppm of the main magnetic field
  delta_omega0 = 2.0 * np.pi * scanner.gammabar.m_as('Hz/T') * delta_B0

  # Slice profile
  # The slice profile prepulse is calculated based on a reference RF pulse with
  # user-defined characteristics. The slice profile object allows accessing the calculated adjusted RF pulse and dephasing and rephasing gradients
  rf = RF(scanner=scanner, NbLobes=[4, 4], alpha=0.46, shape='apodized_sinc', flip_angle=Q_(np.deg2rad(8),'rad') , t_ref=Q_(0.0,'ms'))
  sp = SliceProfile(delta_z=Q_(parameters.FOV[2], 'm'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=False, 
    bandwidth=Q_(10000, 'Hz'), #'maximum',
    refocusing_area_frac=0.8)
  # sp.optimize(frac_start=0.5, frac_end=1.5, N=100)

  # Bipolar gradient
  t_ref = sp.rephasing.t_ref + sp.rephasing.dur
  bp1 = Gradient(scanner=scanner, t_ref=t_ref, axis=2)
  bp2 = bp1.make_bipolar(Q_(enc.VENC, 'm/s'))

  # Create sequence object
  seq = Sequence()

  # Imaging block
  imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing, bp1, bp2], rf_pulses=[sp.rf], dt_rf=Q_(1e-2, 'ms'), dt_gr=Q_(1e-2, 'ms'), dt=Q_(1, 'ms'), store_magnetization=True)

  # Add blocks to the sequence
  for fr in range(phantom.Nfr):
    seq.add_block(imaging)
    seq.add_block(Q_(parameters.TimeSpacing, 's'), dt=Q_(1, 'ms'))  # Time spacing between frames
  # seq.plot()

  # Bloch solver
  solver = BlochSolver(seq, phantom, scanner=scanner, M0=1e+9, T1=Q_(parameters.T1, 's'), T2=Q_(parameters.T2star, 's'), delta_B=delta_B0.reshape((-1, 1)), pod_trajectory=pod_trajectory)

  # Solve for x and y directions
  Mxy, Mz = solver.solve() 

  testfile = XDMFFile('test_new.xdmf', nodes=phantom.local_nodes, elements={'tetra': phantom.local_elements})
  for i in range(Mxy.shape[1]):
    testfile.write(pointData={'Mx': np.real(Mxy[:,i]), 'My': np.imag(Mxy[:,i]), 'Mz': Mz[:,i]}, time=i)
  testfile.close()

  # Path to export the generated data
  export_path = Path('MRImages/{:s}_V{:.0f}.pkl'.format(parameters.Sequence, 100.0*parameters.VENC))

  # Make sure the directory exist
  os.makedirs(str(export_path.parent), exist_ok=True)

  # Generate kspace trajectory
  traj = CartesianStack(FOV=Q_(parameters.FOV,'m'),
    t_start=imaging.time_extent[1] - sp.rf.t_ref,
    res=parameters.RES, 
    oversampling=parameters.Oversampling, 
    lines_per_shot=parameters.LinesPerShot, 
    MPS_ori=MPS_ori, 
    LOC=LOC, 
    receiver_bw=Q_(parameters.r_BW,'Hz'), 
    plot_seq=False)

  # Print echo time
  MPI_print('Echo time = {:.2f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, enc.nb_directions, phantom.Nfr], dtype=np.complex64)

  T2star = (parameters.T2star * np.ones([phantom.local_nodes.shape[0]])).astype(np.float32)

  # Assemble mass matrix for integrals (just once to accelerate the simulation)
  M = phantom.mass_matrix(lumped=True)

  # Iterate over cardiac phases
  t0 = time.time()
  for fr in range(phantom.Nfr):
      
      t0_fr = time.time()

      # # Read velocity data in peak-systolic frame
      # phantom.read_data(fr)
      # velocity = phantom.point_data['velocity'] @ traj.MPS_ori
      # submesh_velocity = phantom.interpolate_to_submesh(velocity)

      # Generate 4D flow image
      K[:,:,:,:,fr] = PC(MPI_rank, M, traj.points, traj.times.m_as('s'), phantom.local_nodes, delta_omega0, T2star, Mxy[:, fr])

      MPI_print('Elapsed time frame {:d}: {:.2f} s'.format(fr, time.time()-t0_fr))


  # Store elapsed time
  pc_time = time.time() - t0

  # Print elapsed times
  MPI_print('Elapsed time-per-frame for 4D Flow generation: {:.2f} s'.format(pc_time/phantom.Nfr))
  MPI_print('Elapsed time for 4D Flow generation: {:.2f} s'.format(pc_time))

  # Gather results
  K = gather_data(K)

  # Export generated data
  if MPI_rank==0:
    with open(str(export_path), 'wb') as f:
      pickle.dump({'kspace': K, 'MPS_ori': MPS_ori, 'LOC': LOC, 'traj': traj}, f)

  # Image reconstruction
  I = CartesianRecon(K, traj)

  # Show reconstruction
  m = np.abs(I[...,0,:])
  phi_x = np.angle(I[...,0,:])
  if MPI_rank == 0:
    plotter = MRIPlotter(images=[m, phi_x], title=['Magnitude', 'phi_x'])
    plotter.show()