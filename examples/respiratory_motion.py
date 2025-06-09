import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
import pickle
import time
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from FEelMRI.Bloch import BlochSolver, Sequence, SequenceBlock
from FEelMRI.IO import XDMFFile
from FEelMRI.KSpaceTraj import CartesianStack
from FEelMRI.Math import Rx, Ry, Rz
from FEelMRI.Motion import PODTrajectory, RespiratoryMotion
from FEelMRI.MPIUtilities import MPI_print, MPI_rank, gather_data
from FEelMRI.MRImaging import SliceProfile
from FEelMRI.MRObjects import RF, Scanner
from FEelMRI.Parameters import ParameterHandler
from FEelMRI.Phantom import FEMPhantom
from FEelMRI.Plotter import MRIPlotter
from FEelMRI.Recon import CartesianRecon
from FEelMRI.Tagging import SPAMM

if __name__ == '__main__':

  # Import imaging parameters
  parameters = ParameterHandler('parameters/respiratory_motion.yaml')

  # Imaging orientation paramters
  theta_x = np.deg2rad(parameters.theta_x)
  theta_y = np.deg2rad(parameters.theta_y)
  theta_z = np.deg2rad(parameters.theta_z)
  MPS_ori = Rz(theta_z)@Rx(theta_x)@Ry(theta_y)
  LOC = parameters.LOC

  # Create FEM phantom object
  phantom = FEMPhantom(path='phantoms/beating_heart.xdmf', scale_factor=1.0)

  # Translate phantom to obtain the desired slice location
  phantom.orient(MPS_ori, LOC)

  # We can use only a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV. We can also refine the submesh to increase the number of nodes and elements
  midpoints = np.array([np.mean(phantom.global_nodes[e,:], axis=0) for e in phantom.global_elements])
  condition = (np.abs(midpoints[:,2]) <= 0.6*parameters.FOV[2]) > 1e-3
  markers = np.where(condition)[0]
  phantom.create_submesh(markers, refine=False, element_size=0.003)

  # Create POD trajectory object
  trajectory = np.zeros([phantom.global_nodes.shape[0], phantom.global_nodes.shape[1], phantom.Nfr], dtype=np.float32)
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr
    phantom.read_data(fr)
    displacement = phantom.point_data['displacement'] @ MPS_ori
    submesh_displacement = phantom.interpolate_to_submesh(displacement, local=False)
    trajectory[..., fr] = submesh_displacement

  # Define POD object
  pod_times = np.linspace(0, (phantom.Nfr-1)*parameters.TimeSpacing, phantom.Nfr)
  pod_trajectory = PODTrajectory(time_array=pod_times,
                                 data=trajectory,
                                 global_to_local=phantom.global_to_local_nodes,
                                 n_modes=5,
                                 taylor_order=10,
                                 is_periodic=True)
  
  # Define respiratory motion object
  T = 3.0      # period
  A = 0.008    # amplitude
  times  = np.linspace(0, T, 100)
  motion = A*(1 - np.cos(2*np.pi*times/(2*T))**4)
  pod_resp_motion = RespiratoryMotion(time_array=times, 
                              data=motion, 
                              is_periodic=True,
                              direction=np.array([0, 1, 0]),
                              remove_mean=True)
  
  import matplotlib.pyplot as plt
  motion = []
  times  = np.linspace(0, 2*T, 100)
  for t in times:
    pod_resp_motion.timeshift = t
    motion.append(pod_resp_motion(0).max())
  plt.plot(times, motion)
  plt.show()

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=Q_(parameters.G_max,'mT/m'), gradient_slew_rate=Q_(parameters.G_sr,'mT/m/ms'))

  # Field inhomogeneity
  def spatial(x):
      return x[:,0] + x[:,1] + x[:,2]
  delta_B0 = spatial(phantom.local_nodes)
  delta_B0 /= np.abs(spatial(phantom.local_nodes).flatten()).max()
  delta_B0 *= 1.5 * 1e-6 # 1.5 ppm of the main magnetic field
  delta_omega0 = 2.0 * np.pi * scanner.gammabar.m_as('Hz/T') * delta_B0

  # Slice profile
  # The slice profile prepulse is calculated based on a reference RF pulse with
  # user-defined characteristics. The slice profile object allows accessing the calculated adjusted RF pulse and dephasing and rephasing gradients
  rf = RF(scanner=scanner, NbLobes=[4, 4], alpha=0.46, shape='apodized_sinc', flip_angle=Q_(np.deg2rad(12),'rad') , t_ref=Q_(0.0,'ms'))
  sp = SliceProfile(delta_z=Q_(parameters.FOV[2], 'm'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=False, 
    bandwidth='maximum',
    refocusing_area_frac=0.7758)
  # sp.optimize(frac_start=0.7, frac_end=0.8, N=100, profile_samples=100)

  # Simulate the SPAMM preparation block for each encoding direction
  t0 = time.time()

  # Imaging block
  imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing], 
                          rf_pulses=[sp.rf], 
                          dt_rf=Q_(1e-2, 'ms'), 
                          dt_gr=Q_(1e-2, 'ms'), 
                          dt=Q_(1, 'ms'), 
                          store_magnetization=True)
  dummy = imaging.copy()
  dummy.store_magnetization = False

  # Create and fill sequence object
  seq = Sequence()
  for i in range(80):
    seq.add_block(dummy.copy())  # Add dummy blocks to reach the steady state
    seq.add_block(Q_(parameters.RepetitionTime, 's'))  # Delay between imaging blocks
  for i in range(phantom.Nfr):
    seq.add_block(imaging.copy())
    seq.add_block(Q_(parameters.RepetitionTime, 's'))  # Delay between imaging blocks
  # seq.plot()

  # Bloch solver
  solver = BlochSolver(seq, phantom, scanner=scanner, M0=1e+8, T1=Q_(parameters.T1, 's'), T2=Q_(parameters.T2star, 's'), delta_B=delta_B0.reshape((-1, 1)), pod_trajectory=pod_trajectory)

  # Solve
  Mxy, Mz = solver.solve()

  # Export xdmf for debugging
  file = XDMFFile('magnetization_{:d}.xdmf'.format(MPI_rank), nodes=phantom.local_nodes, elements={'tetra': phantom.local_elements})
  for fr in range(Mxy.shape[1]):
    file.write(pointData={'Mx': np.real(Mxy[:, fr]), 
                          'My': np.imag(Mxy[:, fr]),
                          'Mz': Mz[:, fr]}, time=fr*parameters.RepetitionTime)
  file.close()

  # Store elapsed time
  bloch_time = time.time() - t0

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
  # traj.plot_trajectory()
  MPI_print('Echo time: {:.1f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, 1, 1], dtype=np.complex64)

  # T2star relaxation time
  T2star = np.ones([phantom.global_nodes.shape[0], ], dtype=np.float32)*parameters.T2star

  # Scan time
  scan_time = 0.0

  # Iterate over slices
  t0 = time.time()  
  for s in range(slices):

    # Iterate over shots
    for i, sh in enumerate(traj.shots):

      MPI_print("Generating shot {:d}/{:d} for slice {:d}/{:d}".format(i+1, traj.nb_shots, s+1, K.shape[2]))

      # Update reference time of POD trajectory
      pod_trajectory.timeshift = i * parameters.RepetitionTime
      pod_resp_motion.timeshift = i * parameters.RepetitionTime
      pod_sum = pod_trajectory + pod_resp_motion

      # Get displacement data in frame fr
      displacement = pod_trajectory(i * parameters.RepetitionTime)

      # Assemble mass matrix for integrals (just once)
      M = phantom.mass_matrix_2(phantom.local_nodes + displacement, lumped=True, quadrature_order=2)

      # k-space points per shot
      kspace_points = (traj.points[0][:,sh,s,np.newaxis], 
                      traj.points[1][:,sh,s,np.newaxis], 
                      traj.points[2][:,sh,s,np.newaxis])
      kspace_times = traj.times.m_as('s')[:,sh,s,np.newaxis]

      # Generate 4D flow image
      tmp = SPAMM(MPI_rank, M, kspace_points, kspace_times, phantom.local_nodes, Mxy[:, 0], delta_omega0, T2star, pod_sum)
      K[:,sh,s,:,0] = tmp.swapaxes(0, 1)[:,:,0]

      # Update scan time
      scan_time += parameters.TimeSpacing

  # Store elapsed time
  spamm_time = time.time() - t0

  # Print elapsed times
  MPI_print('Elapsed time for Bloch solver: {:.2f} s'.format(bloch_time))
  MPI_print('Elapsed time-per-frame for SPAMM: {:.2f} s'.format(spamm_time/phantom.Nfr))
  MPI_print('Elapsed time for SPAMM: {:.2f} s'.format(spamm_time))
  MPI_print('Elapsed time for SPAMM + Bloch solver: {:.2f} s'.format(bloch_time + spamm_time))

  # Gather results
  K = gather_data(K)

  # Image reconstruction
  I = CartesianRecon(K, traj)

  # Show reconstruction
  if MPI_rank == 0:
    m = np.abs(I[...,0,:])
    phi = np.angle(I[...,0,:])
    plotter = MRIPlotter(images=[m, phi], title=['Magnitude', 'Phase'], FOV=parameters.FOV)
    plotter.show()

    k = np.abs(K[...,0,:])
    plotter = MRIPlotter(images=[k], title=['k-space'], FOV=parameters.FOV)
    # plotter.export_images('animation_K/')
    plotter.show()