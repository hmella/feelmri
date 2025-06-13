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
from FEelMRI.Motion import POD, RespiratoryMotion
from FEelMRI.MPIUtilities import MPI_print, MPI_rank, gather_data
from FEelMRI.MRImaging import SliceProfile
from FEelMRI.MRObjects import RF, Scanner
from FEelMRI.Parameters import ParameterHandler, PVSMParser
from FEelMRI.Phantom import FEMPhantom
from FEelMRI.Plotter import MRIPlotter
from FEelMRI.Recon import CartesianRecon
from FEelMRI.Tagging import SPAMM

if __name__ == '__main__':

  # Import imaging parameters
  parameters = ParameterHandler('parameters/respiratory_motion_2.yaml')

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(parameters.Formatting.planning,
                          box_name='Box1',
                          transform_name='Transform1',
                          length_units=parameters.Formatting.units)

  # Create FEM phantom object
  phantom = FEMPhantom(path='phantoms/beating_heart.xdmf', scale_factor=1.0)

  # Translate phantom to obtain the desired slice location
  phantom.orient(planning.MPS, planning.LOC)

  # We can a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV
  mp = phantom.global_nodes[phantom.global_elements].mean(axis=1)
  markers = np.where(np.abs(mp[:, 2]) <= 1.0 * planning.FOV[2].m_as('m'))[0]
  phantom.create_submesh(markers)

  # Create array to store displacements
  u = np.zeros([phantom.global_shape[0], 3, phantom.Nfr], dtype=np.float32)
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr and interpolate to the submesh
    phantom.read_data(fr)
    u[..., fr] = phantom.interpolate_to_submesh(phantom.point_data['displacement'] @ planning.MPS, local=False)

  # Create POD for tissue displacements
  dt = parameters.Imaging.TimeSpacing.to('ms')
  pod_times = np.linspace(0, (phantom.Nfr-1)*dt, phantom.Nfr)
  pod_trajectory = POD(time_array=pod_times.m_as('ms'),
                      data=u,
                      global_to_local=phantom.global_to_local_nodes,
                      n_modes=5,
                      taylor_order=10,
                      is_periodic=True)
  
  # Define respiratory motion object
  T = Q_(3000.0, 'ms')      # period
  A = Q_(0.008, 'm')        # amplitude
  times  = np.linspace(0, T, 100)
  motion = A*(1 - np.cos(2*np.pi*times/(2*T))**4)
  pod_resp_motion = RespiratoryMotion(time_array=times.m_as('ms'), 
                              data=motion.m_as('m'),
                              is_periodic=True,
                              direction=np.array([0, 1, 0]),
                              remove_mean=True)

  # Combine the POD trajectory and the respiratory motion
  pod_sum = pod_resp_motion + pod_trajectory

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

  # Field inhomogeneity
  def spatial(x):
      return x[:,0] + x[:,1] + x[:,2]
  delta_B0 = spatial(phantom.local_nodes)
  delta_B0 /= np.abs(spatial(phantom.global_nodes).flatten()).max()
  delta_B0 *= 15.0 * 1e-6 # 1.5 ppm of the main magnetic field
  delta_omega0 = 2.0 * np.pi * scanner.gammabar.m_as('1/ms/T') * delta_B0

  # Slice profile
  # The slice profile prepulse is calculated based on a reference RF pulse with
  # user-defined characteristics. The slice profile object allows accessing the calculated adjusted RF pulse and dephasing and rephasing gradients
  rf = RF(scanner=scanner, NbLobes=[4, 4], alpha=0.46, shape='apodized_sinc', flip_angle=Q_(np.deg2rad(12),'rad') , t_ref=Q_(0.0,'ms'))
  sp = SliceProfile(delta_z=planning.FOV[2].to('m'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=False, 
    bandwidth='maximum',
    refocusing_area_frac=0.7758)
  # sp.optimize(frac_start=0.7, frac_end=0.8, N=100, profile_samples=100)

  # Simulate the SPAMM preparation block for each encoding direction
  t0 = time.time()

  # Imaging and dummy blocks
  imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing], 
                          rf_pulses=[sp.rf], 
                          dt_rf=Q_(1e-2, 'ms'), 
                          dt_gr=Q_(1e-2, 'ms'), 
                          dt=Q_(1, 'ms'), 
                          store_magnetization=True)
  dummy = imaging.copy()
  dummy.store_magnetization = False

  # Store elapsed time
  bloch_time = time.time() - t0

  # Generate kspace trajectory
  traj = CartesianStack(FOV=planning.FOV.to('m'),
                      t_start=imaging.time_extent[1] - sp.rf.t_ref,
                      res=parameters.Imaging.RES, 
                      oversampling=parameters.Imaging.Oversampling, 
                      lines_per_shot=parameters.Imaging.LinesPerShot, 
                      MPS_ori=planning.MPS,
                      LOC=planning.LOC.m,
                      receiver_bw=parameters.Hardware.r_BW.to('Hz'), 
                      plot_seq=False)
  # traj.plot_trajectory()
  MPI_print('Echo time: {:.1f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, 1, 1], dtype=np.complex64)

  # T2star relaxation time
  T2star = np.ones([phantom.global_nodes.shape[0], ], dtype=np.float32)*parameters.Phantom.T2star

  # Create and fill sequence object
  seq = Sequence()
  time_spacing = parameters.Imaging.TR.to('ms') - (imaging.time_extent[1] - sp.rf.t_ref)
  # for i in range(80):
  #   seq.add_block(dummy)  # Add dummy blocks to reach the steady state
  #   seq.add_block(time_spacing)  # Delay between imaging blocks
  # for i in range(phantom.Nfr):
  #   seq.add_block(imaging)
  #   seq.add_block(time_spacing)  # Delay between imaging blocks
  # seq.plot()

  # Bloch solver
  solver = BlochSolver(seq, phantom, 
                      scanner=scanner, 
                      M0=1e+8, 
                      T1=parameters.Phantom.T1, 
                      T2=parameters.Phantom.T2star, 
                      delta_B=delta_B0.reshape((-1, 1)),
                      pod_trajectory=pod_sum)


  # Create XDMF file for debugging
  file = XDMFFile('magnetization_{:d}.xdmf'.format(MPI_rank), nodes=phantom.local_nodes, elements={'tetra': phantom.local_elements})

  # Assemble mass matrix for integrals (just once)
  M = phantom.mass_matrix(lumped=True, quadrature_order=2)

  # Generate k-space data for each shot and slice
  t0 = time.time()  
  for s in range(slices):

    # Iterate over shots
    for i, sh in enumerate(traj.shots):

      MPI_print("Generating shot {:d}/{:d} for slice {:d}/{:d}".format(i+1, traj.nb_shots, s+1, K.shape[2]))

      # Add imaging and delay blocks to the sequence
      seq.add_block(imaging)
      seq.add_block(time_spacing)  # Delay between imaging blocks
      Mxy, Mz = solver.solve(start=-2)

      # Update reference time of POD trajectory
      pod_sum.update_timeshift(i * parameters.Imaging.TR.m_as('ms'))

      # Get displacement data in frame fr
      displacement = pod_sum(0.0)

      # # Assemble mass matrix for integrals (just once)
      # M = phantom.mass_matrix_2(phantom.local_nodes + displacement, lumped=True, quadrature_order=2)

      # k-space points per shot
      kspace_points = (traj.points[0][:,sh,s,np.newaxis], 
                      traj.points[1][:,sh,s,np.newaxis], 
                      traj.points[2][:,sh,s,np.newaxis])
      kspace_times = traj.times.m_as('ms')[:,sh,s,np.newaxis]

      # Generate 4D flow image
      tmp = SPAMM(MPI_rank, M, kspace_points, kspace_times, phantom.local_nodes + displacement, Mxy, delta_omega0, T2star.m_as('ms'), pod_sum)
      K[:,sh,s,:,0] = tmp.swapaxes(0, 1)[:,:,0]

      # Export magnetization and displacement for debugging
      file.write(pointData={'Mx': np.real(Mxy), 
                            'My': np.imag(Mxy),
                            'Mz': Mz,
                            'displacement': pod_sum(0)},
                            time=i*parameters.Imaging.TR)

  # Store elapsed time
  spamm_time = time.time() - t0

  # Print elapsed times
  MPI_print('Elapsed time for Bloch solver: {:.2f} s'.format(bloch_time))
  MPI_print('Elapsed time-per-frame for SPAMM: {:.2f} s'.format(spamm_time/phantom.Nfr))
  MPI_print('Elapsed time for SPAMM: {:.2f} s'.format(spamm_time))
  MPI_print('Elapsed time for SPAMM + Bloch solver: {:.2f} s'.format(bloch_time + spamm_time))

  # Close the file
  file.close()

  # Gather results
  K = gather_data(K)

  # Image reconstruction
  I = CartesianRecon(K, traj)

  # Show reconstruction
  if MPI_rank == 0:
    mag = np.abs(I[...,0,:])
    phi = np.angle(I[...,0,:])
    plotter = MRIPlotter(images=[mag, phi], title=['Magnitude', 'Phase'], FOV=planning.FOV.m_as('m'))
    plotter.show()

    plotter = MRIPlotter(images=[np.abs(K[...,0,:])], title=['k-space'], FOV=planning.FOV.m_as('m'))
    plotter.show()