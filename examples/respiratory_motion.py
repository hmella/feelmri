import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
import pickle
import time
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from FEelMRI.IO import XDMFFile
from FEelMRI.KSpaceTraj import CartesianStack, Gradient
from FEelMRI.Math import Rx, Ry, Rz
from FEelMRI.Motion import PODTrajectory
from FEelMRI.MPIUtilities import MPI_print, MPI_rank, gather_data
from FEelMRI.MRImaging import PositionEncoding, SliceProfile
from FEelMRI.MRObjects import (RF, BlochSolver, Gradient, Scanner, Sequence,
                               SequenceBlock)
from FEelMRI.Parameters import ParameterHandler
from FEelMRI.Phantom import FEMPhantom
from FEelMRI.Plotter import MRIPlotter
from FEelMRI.Recon import CartesianRecon
from FEelMRI.Tagging import SPAMM

if __name__ == '__main__':

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

  # Create FEM phantom object
  phantom = FEMPhantom(path='phantoms/beating_heart.xdmf', scale_factor=1.0)

  # Translate phantom to obtain the desired slice location
  phantom.orient(MPS_ori, LOC)

  # We can use only a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV. We can also refine the submesh to increase the number of nodes and elements
  midpoints = np.array([np.mean(phantom.nodes[e,:], axis=0) for e in phantom.elements])
  condition = (np.abs(midpoints[:,2]) <= 0.6*parameters.FOV[2]) > 1e-3
  markers = np.where(condition)[0]
  phantom.create_submesh(markers, refine=False, element_size=0.003)

  # Distribute the mesh to all MPI processes
  phantom.distribute_mesh()

  # Create POD trajectory object
  trajectory = np.zeros([phantom.local_nodes.shape[0], phantom.local_nodes.shape[1], phantom.Nfr], dtype=np.float32)
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr
    phantom.read_data(fr)
    displacement = phantom.point_data['displacement'] @ MPS_ori
    submesh_displacement = phantom.interpolate_to_submesh(displacement)
    trajectory[..., fr] = submesh_displacement

  # Define POD object
  pod_times = np.linspace(0, (phantom.Nfr-1)*parameters.TimeSpacing, phantom.Nfr)
  pod_trajectory = PODTrajectory(time_array=pod_times,
                                 data=trajectory,
                                 n_modes=5,
                                 taylor_order=10,
                                 is_periodic=True)
  
  # Define respiratory motion object
  T = 3.0   # period
  A = 0.008 # amplitude
  times = np.linspace(0, T, 100)
  # def expr(t): return A*(1 - np.cos(2*np.pi*t/(2*T))**4)
  def expr(t): 
    return np.random.uniform(-0.5*A, 0.5*A, size=[1,]).reshape((1,1)) * np.array([[0, 1, 0]]).reshape((1, 3))

  # import matplotlib.pyplot as plt
  # if MPI_rank == 0:
  #   plt.plot(times, expr(times))
  #   plt.show()

  # direction = np.array([[0, 1, 0]]).reshape((1, 3, 1))
  # motion = expr(times).reshape((1, 1, -1)) * direction
  # pod_resp_motion = PODTrajectory(time_array=times, 
  #                             data=motion, 
  #                             n_modes=10, 
  #                             taylor_order=15,
  #                             is_periodic=True)

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

  # SPAMM magnetization
  SPAMM_mag = [None,]*len(enc.directions)

  # Simulate the SPAMM preparation block for each encoding direction
  t0 = time.time()
  for i in range(len(enc.directions)):

      # SPAMM preparation block
      rf1 = RF(scanner=scanner, shape='hard', dur=Q_(0.2, 'ms'), flip_angle=Q_(np.deg2rad(90),'rad'), t_ref=Q_(0.0, 'ms'))

      G_tag = Gradient(scanner=scanner, t_ref=rf1.t_ref + rf1.dur2, axis=i)
      G_tag.match_area(Q_(parameters.ke/scanner.gamma.m_as('1/mT/ms'), 'mT*ms/m'))

      rf2 = RF(scanner=scanner, shape='hard', dur=Q_(0.2, 'ms'), flip_angle=Q_((-1)**i*np.deg2rad(90),'rad'), t_ref=G_tag.t_ref + G_tag.dur + rf1.t_ref + rf1.dur2)

      prep = SequenceBlock(gradients=[G_tag], 
                           rf_pulses=[rf1, rf2], 
                           dt_rf=Q_(1e-2, 'ms'), 
                           dt_gr=Q_(1e-2, 'ms'), 
                           dt=Q_(1, 'ms'), 
                           store_magnetization=False)

      # Imaging block
      sp.rf.update_reference(rf2.t_ref + rf2.dur2 + sp.rf.dur1 + Q_(0.5, 'ms'))
      sp.dephasing.update_reference(sp.rf.t_ref - (sp.dephasing.slope + 0.5*sp.dephasing.lenc))
      sp.rephasing.update_reference(sp.dephasing.t_ref + sp.dephasing.dur)
      imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing], 
                              rf_pulses=[sp.rf], 
                              dt_rf=Q_(1e-2, 'ms'), 
                              dt_gr=Q_(1e-2, 'ms'), 
                              dt=Q_(1, 'ms'), 
                              store_magnetization=True)

      # Create sequence object
      seq = Sequence(prepulse=prep, blocks=[imaging], dt_prep=Q_(0, 'ms'))
      seq.repeat_blocks(nb_times=phantom.Nfr, dt_blocks=Q_(parameters.TimeSpacing, 's'))
      # seq.plot()

      # Bloch solver
      solver = BlochSolver(seq, phantom, scanner=scanner, M0=1e+9, T1=Q_(parameters.T1, 's'), T2=Q_(parameters.T2star, 's'), delta_B=delta_B0.reshape((-1, 1)))

      # Solve
      Mxy, Mz = solver.solve()

      # Assign the magnetization to the corresponding direction
      SPAMM_mag[i] = Mxy

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
  K = np.zeros([ro_samples, ph_samples, slices, enc.nb_directions, phantom.Nfr], dtype=np.complex64)

  # T2star relaxation time
  T2star = np.ones([phantom.nodes.shape[0], ], dtype=np.float32)*parameters.T2star

  # Scan time
  scan_time = 0.0

  # Iterate over slices
  t0 = time.time()  
  for s in range(K.shape[2]):

    # Iterate over shots
    for i, sh in enumerate(traj.shots):

      MPI_print("Generating shot {:d}/{:d} for slice {:d}/{:d}".format(i+1, traj.nb_shots, s+1, K.shape[2]))

      # Iterate over cardiac phases
      for fr in range(phantom.Nfr):

        # Read displacement data in frame fr
        phantom.read_data(fr)
        displacement = phantom.point_data['displacement'] @ traj.MPS_ori
        submesh_displacement = phantom.interpolate_to_submesh(displacement)

        # Assemble mass matrix for integrals (just once)
        M = phantom.mass_matrix_2(phantom.local_nodes + submesh_displacement, lumped=True, quadrature_order=2)

        # SPAMM magnetization for this frame
        M_spamm = np.vstack((SPAMM_mag[0][:, fr], SPAMM_mag[1][:, fr])).T

        # Update reference time of POD trajectory
        pod_trajectory.timeshift = fr * parameters.TimeSpacing
        # pod_resp_motion.timeshift = scan_time
        pod_sum = pod_trajectory + expr

        # k-space points per shot
        kspace_points = (traj.points[0][:,sh,s,np.newaxis], 
                        traj.points[1][:,sh,s,np.newaxis], 
                        traj.points[2][:,sh,s,np.newaxis])
        kspace_times = traj.times.m_as('s')[:,sh,s,np.newaxis]

        # Generate 4D flow image
        tmp = SPAMM(MPI_rank, M, kspace_points, kspace_times, phantom.local_nodes, M_spamm, delta_omega0, T2star, pod_sum)
        K[:,sh,s,:,fr] = tmp.swapaxes(0, 1)[:,:,0,:]

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
    mx = np.abs(I[...,0,:])
    my = np.abs(I[...,1,:])
    mxy = np.abs(I[...,0,:] - I[...,1,:])
    plotter = MRIPlotter(images=[mx, my, mxy], title=['SPAMM X', 'SPAMM Y', 'O-CSPAMM'], FOV=parameters.FOV)
    # plotter.export_images('animation_I{:d}/'.format(parameters.LinesPerShot))
    plotter.show()

    mx = np.abs(K[...,0,:])
    my = np.abs(K[...,1,:])
    mxy = np.abs(K[...,0,:] - K[...,1,:])
    plotter = MRIPlotter(images=[mx, my, mxy], title=['SPAMM X', 'SPAMM Y', 'O-CSPAMM'], FOV=parameters.FOV)
    # plotter.export_images('animation_K/')
    plotter.show()