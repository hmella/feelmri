import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
import pickle
import time
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.IO import XDMFFile
from feelmri.KSpaceTraj import CartesianStack
from feelmri.Motion import POD, RespiratoryMotion
from feelmri.MPIUtilities import MPI_print, MPI_rank, gather_data
from feelmri.MRImaging import SliceProfile
from feelmri.MRObjects import RF, Scanner
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon
from feelmri.MRI import Signal

if __name__ == '__main__':

  t0_start = time.perf_counter()

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  parameters = ParameterHandler(script_path/'parameters/free_running.yaml')

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(script_path/parameters.Formatting.planning,
                      box_name='Box1',
                      transform_name='Transform1',
                      length_units=parameters.Formatting.units)

  vxsz = planning.FOV.m_as('mm')/parameters.Imaging.RES
  MPI_print('Voxel size: ({:.2f}, {:.2f}, {:.2f}) mm'.format(vxsz[0], vxsz[1], vxsz[2]))

  # Create FEM phantom object
  phantom = FEMPhantom(script_path/'phantoms/beating_heart_P2.xdmf', scale_factor=1.0)

  # Translate phantom to obtain the desired slice location
  phantom.orient(planning.MPS, planning.LOC.to('m'))

  # We can a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV
  mp = phantom.global_nodes[phantom.global_elements].mean(axis=1)
  markers = np.abs(mp[:, 2]) <= 4.0 * planning.FOV[2].m_as('m')
  phantom.create_submesh(markers)

  # Create array to store displacements
  u = np.zeros([phantom.global_shape[0], 3, phantom.Nfr], dtype=np.float32)
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr and interpolate to the submesh
    phantom.read_data(fr)
    u[..., fr] = phantom.to_submesh(phantom.point_data['displacement'] @ planning.MPS, global_mesh=True)

  # Create POD for tissue displacements
  dt = parameters.Imaging.TimeSpacing.to('ms')
  pod_times = np.linspace(0, (phantom.Nfr-1)*dt, phantom.Nfr)
  pod_trajectory = POD(times=pod_times.m_as('ms'),
                      data=u,
                      global_to_local=phantom.local_to_global_nodes,
                      n_modes=10,
                      is_periodic=True,
                      interpolation_method='Pchip')
  
  # Define respiratory motion object
  T = Q_(3000.0, 'ms')      # period
  A = Q_(0.008, 'm')        # amplitude
  times  = np.linspace(0, T, 100)
  motion = A*(1 - np.cos(2*np.pi*times/(2*T))**4)
  pod_resp_motion = RespiratoryMotion(times=times.m_as('ms'), 
                              data=motion.m_as('m'),
                              is_periodic=True,
                              direction=np.array([0, 1, 0]),
                              remove_mean=False,
                              interpolation_method='Pchip')

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
  delta_B0 *= 1.5 * 1e-6    # 1.5 ppm of the main magnetic field
  delta_omega0 = 2.0 * np.pi * scanner.gammabar.m_as('1/ms/T') * delta_B0

  # Slice profile
  # The slice profile prepulse is calculated based on a reference RF pulse with
  # user-defined characteristics. The slice profile object allows accessing the calculated adjusted RF pulse and dephasing and rephasing gradients
  rf = RF(scanner=scanner, 
          NbLobes=[4, 4], 
          alpha=0.46, 
          shape='apodized_sinc', 
          flip_angle=parameters.Imaging.FlipAngle.to('rad'), 
          ref=Q_(0.0,'ms'),
          phase_offset=Q_(-np.pi/2, 'rad'))
  sp = SliceProfile(delta_z=planning.FOV[2].to('m'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=False, 
    bandwidth='maximum')

  # Imaging and dummy blocks
  t0_dummy = time.perf_counter()
  imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing], 
                          rf_pulses=[sp.rf], 
                          dt_rf=Q_(1e-2, 'ms'), 
                          dt_gr=Q_(1e-2, 'ms'), 
                          dt=Q_(1, 'ms'), 
                          store_magnetization=True)
  dummy = imaging.copy()
  dummy.store_magnetization = False

  # Generate kspace trajectory
  traj = CartesianStack(FOV=planning.FOV.to('m'),
                      t_start=imaging.time_extent[1] - sp.rf.time,
                      res=parameters.Imaging.RES, 
                      oversampling=parameters.Imaging.Oversampling, 
                      lines_per_shot=parameters.Imaging.LinesPerShot, 
                      MPS_ori=planning.MPS,
                      LOC=planning.LOC,
                      receiver_bw=parameters.Hardware.r_BW.to('Hz'), 
                      plot_seq=False)
  # traj.plot_trajectory()
  # MPI_print('Echo time: {:.1f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, 1, 1], dtype=np.complex64)

  # T2star relaxation time
  T2star = np.ones([phantom.local_nodes.shape[0], ], dtype=np.float32)*parameters.Phantom.T2star

  # Create and fill sequence object
  seq = Sequence()
  time_spacing = parameters.Imaging.TR.to('ms') - (imaging.time_extent[1] - sp.rf.ref)
  for i in range(80):
    seq.add_block(dummy)  # Add dummy blocks to reach the steady state
    seq.add_block(time_spacing)  # Delay between imaging blocks

  # Bloch solver
  solver = BlochSolver(seq, phantom, 
                      scanner=scanner, 
                      M0=1e+9, 
                      T1=parameters.Phantom.T1, 
                      T2=parameters.Phantom.T2star, 
                      delta_B=delta_B0.reshape((-1, 1)),
                      pod_trajectory=pod_sum)

  # Solve dummy blocks to reach the steady state
  solver.solve()
  dummy_time = time.perf_counter() - t0_dummy

  # # Create XDMF file for debugging
  # file = XDMFFile(script_path/'magnetization_{:d}.xdmf'.format(MPI_rank), nodes=phantom.local_nodes, elements={phantom.cell_type: phantom.local_elements})

  # Assemble mass matrix for integrals (just once)
  M = phantom.mass_matrix(lumped=False, quadrature_order=2)

  # Convert and stripe units
  T2 = T2star.m_as('ms')

  # Generate k-space data for each shot and slice
  imaging_time = 0.0
  kspace_time = 0.0
  t0_loop = time.perf_counter()
  for s in range(slices):

    # Iterate over shots
    for i, sh in enumerate(traj.shots):

      # MPI_print("Generating shot {:d}/{:d} for slice {:d}/{:d}".format(i+1, traj.nb_shots, s+1, K.shape[2]))

      # Add imaging and delay blocks to the sequence
      seq.add_block(imaging)
      seq.add_block(time_spacing)  # Delay between imaging blocks
      t0_imaging = time.perf_counter()
      Mxy, Mz = solver.solve(start=-2)
      imaging_time += time.perf_counter() - t0_imaging

      # k-space points per shot
      kspace_points = (traj.points[0][:,sh,s,np.newaxis], 
                      traj.points[1][:,sh,s,np.newaxis], 
                      traj.points[2][:,sh,s,np.newaxis])
      kspace_times = traj.times.m_as('ms')[:,sh,s,np.newaxis] - traj.t_start.m_as('ms')

      # Generate 4D flow image
      t0_kspace = time.perf_counter()
      tmp = Signal(MPI_rank, M, kspace_points, kspace_times, phantom.local_nodes, delta_omega0, T2, Mxy, pod_sum)
      K[:,sh,s,:,0] = tmp.swapaxes(0, 1)[:,:,0]
      kspace_time += time.perf_counter() - t0_kspace

      # Update reference time of POD trajectory
      pod_sum.update_timeshift(seq.blocks[-2].time_extent[1].m_as('ms'))

      # # Export magnetization and displacement for debugging
      # displacement = pod_sum(0.0)
      # file.write(pointData={'Mx': np.real(Mxy), 
      #                       'My': np.imag(Mxy),
      #                       'Mz': Mz,
      #                       'displacement': displacement},
      #                       time=i*parameters.Imaging.TR)

  # file.close()

  # Store elapsed time
  script_time = time.perf_counter() - t0_start
  sim_time = dummy_time + imaging_time + kspace_time

  # Print elapsed times
  MPI_print('Elapsed time for dummy blocks: {:.2f} s'.format(dummy_time))
  MPI_print('Elapsed time for imaging blocks: {:.2f} s'.format(imaging_time))
  MPI_print('Elapsed time for k-space generation: {:.2f} s'.format(kspace_time))
  MPI_print('Elapsed time for simulation: {:.2f} s'.format(sim_time))
  MPI_print('Elapsed time for script: {:.2f} s'.format(script_time))

  # Check if exp1_times.txt exists and load and append times (dummy_time, imaging_time, kspace_time) if it does or create it if it does not
  if MPI_rank == 0:
    times_file = script_path/'results/free-running/exp1_times.txt'
    try:
      times = np.loadtxt(times_file)
      times = np.vstack((times, (dummy_time, imaging_time, kspace_time, sim_time, script_time)))
      np.savetxt(times_file, times)
    except FileNotFoundError:
      times = np.array([(dummy_time, imaging_time, kspace_time, sim_time, script_time)])
      np.savetxt(times_file, times)

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

  #   plotter = MRIPlotter(images=[np.abs(K[...,0,:])], title=['k-space'], FOV=planning.FOV.m_as('m'))
  #   plotter.show()