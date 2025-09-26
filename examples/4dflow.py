import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
import pickle
import time
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.IO import XDMFFile
from feelmri.KSpaceTraj import CartesianStack, RadialStack
from feelmri.Motion import PODVelocity
from feelmri.MPIUtilities import MPI_print, MPI_rank, gather_data
from feelmri.MRImaging import SliceProfile, VelocityEncoding
from feelmri.MRObjects import RF, Gradient, Scanner
from feelmri.Noise import add_cpx_noise
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.MRI import Signal
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon, reconstruct_nufft

if __name__ == '__main__':

  t0_start = time.perf_counter()

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  pars = ParameterHandler(script_path/'parameters/4dflow.yaml')


  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(script_path/pars.Formatting.planning,
                          box_name='Box1',
                          transform_name='Transform1',
                          length_units=pars.Formatting.units)

  # Velocity encoding parameters
  venc_dirs = list(pars.VelocityEncoding.Directions.values())
  enc = VelocityEncoding(pars.VelocityEncoding.VENC, np.array(venc_dirs))

  # Create FEM phantom object
  phantom = FEMPhantom(script_path/'phantoms/aorta_CFD.xdmf', velocity_label='velocity', scale_factor=0.01)

  # Translate phantom to obtain the desired slice location
  phantom.orient(planning.MPS, planning.LOC.to('m'))

  # We can a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV
  mp = phantom.global_nodes[phantom.global_elements].mean(axis=1)
  markers = np.abs(mp[:, 2]) <= 0.5 * planning.FOV[2].m_as('m')
  phantom.create_submesh(markers)

  # Create array to store displacements
  v = Q_(np.zeros([phantom.global_shape[0], 3, phantom.Nfr], dtype=np.float32), 'm/s')
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr and interpolate to the submesh
    phantom.read_data(fr)
    v[..., fr] = Q_(phantom.to_submesh(phantom.point_data['velocity'] @ planning.MPS, global_mesh=True), 'm/s')

  # Define POD object
  dt = pars.Phantom.TimeSpacing
  times = np.linspace(0, (phantom.Nfr-1)*dt, phantom.Nfr, dtype=np.float32)
  pod_velocity = PODVelocity(times=times.m_as('ms'),
                            data=v.m_as('m/ms'),
                            global_to_local=phantom.local_to_global_nodes,
                            n_modes=25,
                            is_periodic=True,
                            interpolation_method='Pchip')

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=pars.Hardware.G_max,
                    gradient_slew_rate=pars.Hardware.G_sr)

  # Field inhomogeneity
  def spatial(x):
      return x[:,0] + x[:,1] + x[:,2]
  delta_B0 = spatial(phantom.local_nodes)
  delta_B0 /= np.abs(spatial(phantom.global_nodes).flatten()).max()
  delta_B0 *= scanner.field_strength * 1e-6 # 1.5 ppm of the main magnetic field

  # Phase shift in rad/s
  delta_omega0 = (2.0 * np.pi * scanner.gammabar * delta_B0).to('rad/ms')

  # Slice profile
  # The slice profile prepulse is calculated based on a reference RF pulse with
  # user-defined characteristics. The slice profile object allows accessing the calculated adjusted RF pulse and dephasing and rephasing gradients
  rf = RF(scanner=scanner, 
          NbLobes=[4, 4], 
          alpha=0.46, 
          shape='apodized_sinc', 
          flip_angle=pars.Imaging.FlipAngle.to('rad'), 
          phase_offset=Q_(-np.pi/2, 'rad'))
  sp = SliceProfile(delta_z=planning.FOV[2].to('m'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=False, 
    bandwidth=Q_(12, 'kHz')) # 'maximum'
  # sp.optimize(frac_start=0.96, frac_end=0.98, N=100)

  # Create bipolar gradients
  start = sp.rephasing.time + sp.rephasing.dur
  bp1 = Gradient(scanner=scanner, time=start)
  bp2 = bp1.make_bipolar(pars.VelocityEncoding.VENC)

  # Rotate the bipolar gradients to the desired directions
  bp1r = bp1.rotate(enc.directions, normalize_dirs=True)
  bp2r = bp2.rotate(enc.directions, normalize_dirs=True)

  # Number of imaging frames
  Nb_frames = int(times[-1].m_as('ms') // pars.Imaging.TimeSpacing.m_as('ms'))

  # 4D Flow magnetization
  Mxy_PC = np.zeros([phantom.local_nodes.shape[0], Nb_frames, enc.nb_directions], dtype=np.complex64)

  # Create sequence object and solve magnetization
  t0_bloch = time.perf_counter()
  seq = Sequence()
  solver = BlochSolver(seq, phantom, 
                        scanner=scanner, 
                        M0=1e+9, 
                        T1=pars.Phantom.T1.to('ms'),
                        T2=pars.Phantom.T2star.to('ms'), 
                        delta_B=delta_B0.m_as('mT').reshape((-1, 1)),
                        pod_trajectory=pod_velocity)

  # Add dummy blocks to reach steady state
  dummy = SequenceBlock(gradients=[sp.dephasing,sp.rephasing],
                        rf_pulses=[sp.rf], 
                        dt_rf=Q_(1e-2, 'ms'), 
                        dt_gr=Q_(1e-2, 'ms'), 
                        dt=Q_(1, 'ms'), 
                        store_magnetization=False)

  # Add dummy blocks to the sequence to reach steady state
  TR_spacing = pars.Imaging.TR - dummy.dur
  CP_spacing = pars.Imaging.TimeSpacing - 3*pars.Imaging.TR
  for i in range(80):
    seq.add_block(dummy)
    seq.add_block(TR_spacing, dt=Q_(1, 'ms'))
    seq.add_block(dummy)
    seq.add_block(TR_spacing, dt=Q_(1, 'ms'))
    seq.add_block(dummy)
    seq.add_block(TR_spacing, dt=Q_(1, 'ms'))
    seq.add_block(dummy)
    seq.add_block(CP_spacing, dt=Q_(1, 'ms'))

  # Add and additional block to synchronize the sequence with the cardiac cycle
  seq.add_block(times[-1] - seq.blocks[-1].time_extent[1] % times[-1], dt=Q_(1, 'ms'))

  # Add imaging blocks
  for fr in range(Nb_frames):
    for d in range(enc.nb_directions):

      # Update reference time for second lobe (rotate function keep the time reference of the original gradient)
      [g.change_time(bp1r[d][0].time + bp1r[d][0].dur) for g in bp2r[d]]

      # Imaging block
      imaging = SequenceBlock(gradients=[sp.dephasing,sp.rephasing]+bp1r[d]+bp2r[d],
                              rf_pulses=[sp.rf], 
                              dt_rf=Q_(1e-2, 'ms'), 
                              dt_gr=Q_(1e-2, 'ms'), 
                              dt=Q_(1, 'ms'), 
                              store_magnetization=True)
      
      # Spacing between TRs
      TR_spacing = pars.Imaging.TR - imaging.dur

      # Add PC imaging sequence
      seq.add_block(imaging)
      seq.add_block(TR_spacing, dt=Q_(1, 'ms'))  # Time spacing between frames

    # Add spacing to achieve cardiac phases
    CP_spacing = pars.Imaging.TimeSpacing - 4*pars.Imaging.TR
    seq.add_block(CP_spacing, dt=Q_(1, 'ms'))

  # seq.plot(blocks=slice(-9, None), figsize=(10, 6), tight_layout=True, export_to='4dflow_sequence.png')

  # Solve for x and y directions
  Mxy, Mz = solver.solve()
  Mxy_PC[..., 0] = Mxy[..., 0::4]
  Mxy_PC[..., 1] = Mxy[..., 1::4]
  Mxy_PC[..., 2] = Mxy[..., 2::4]
  Mxy_PC[..., 3] = Mxy[..., 3::4]

  # # Export magnetization and displacement for debugging
  # if d == 0:
  #   file = XDMFFile('magnetization_{:d}.xdmf'.format(MPI_rank), nodes=phantom.local_nodes, elements={phantom.cell_type: phantom.local_elements})
  #   for fr in range(Mxy.shape[1]):
  #     # Write magnetization and displacement at each frame
  #     t = fr*pars.Imaging.TimeSpacing.m_as('ms')
  #     pod_velocity.update_timeshift(t)
  #     file.write(pointData={'M': np.stack((np.real(Mxy[:,fr]), np.imag(Mxy[:,fr]), Mz[:,fr]), axis=1), 'displacement': pod_velocity(1.715)}, time=t)
  #   file.close()

  # Measure elapsed time for the dummy sequence
  bloch_time = time.perf_counter() - t0_bloch

  # # Path to export the generated data
  # export_path = Path(script_path/'MRImages/4dflow_{:s}_V{:.0f}.pkl'.format(pars.Imaging.Sequence, pars.VelocityEncoding.VENC.m_as('cm/s')))

  # # Make sure the directory exist
  # os.makedirs(str(export_path.parent), exist_ok=True)

  # Generate kspace trajectory
  # traj = CartesianStack(FOV = planning.FOV.to('m'),
  #   t_start = imaging.time_extent[1] - sp.rf.time,
  #   res = pars.Imaging.RECON_RES, 
  #   oversampling = pars.Imaging.Oversampling, 
  #   lines_per_shot = pars.Imaging.LinesPerShot, 
  #   MPS_ori = planning.MPS, 
  #   LOC = planning.LOC, 
  #   receiver_bw=pars.Hardware.r_BW, 
  #   plot_seq=False)
  traj = RadialStack(FOV = planning.FOV.to('m'),
    t_start = imaging.time_extent[1] - sp.rf.time,
    res = pars.Imaging.ACQ_RES, 
    oversampling = pars.Imaging.Oversampling, 
    lines_per_shot = pars.Imaging.LinesPerShot, 
    MPS_ori = planning.MPS, 
    LOC = planning.LOC, 
    receiver_bw=pars.Hardware.r_BW, 
    plot_seq=False,
    golden_angle=True)
  # traj.plot_trajectory(tight_layout=True, export_to='4dflow_trajectory.png')

  # Print echo time
  MPI_print('Echo time = {:.2f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, enc.nb_directions, Nb_frames], dtype=np.complex64)

  T2star = (pars.Phantom.T2star * np.ones([phantom.local_nodes.shape[0]])).astype(np.float32)

  # Assemble mass matrix for integrals (just once to accelerate the simulation)
  M = phantom.mass_matrix(lumped=True, quadrature_order=2)

  # Convert and stripe units
  traj_points = traj.points
  traj_times  = traj.times.m_as('ms') - traj.t_start.m_as('ms')
  delta_omega = delta_omega0.m_as('rad/ms')
  T2 = T2star.m_as('ms')

  # Iterate over cardiac phases
  kspace_time = 0.0
  for fr in range(Nb_frames):

      # Start time for the frame      
      t0_fr = time.perf_counter()

      # Update timeshift in the POD velocity
      pod_velocity.update_timeshift(fr * pars.Imaging.TimeSpacing.m_as('ms'))

      # Generate 4D flow image
      t0_kspace = time.perf_counter()
      K[:,:,:,:,fr] = Signal(MPI_rank, M, traj_points, traj_times, phantom.local_nodes, delta_omega, T2, Mxy_PC[:, fr, :], pod_velocity)
      kspace_time += time.perf_counter() - t0_kspace

  # Store elapsed time
  script_time = time.perf_counter() - t0_start
  sim_time = bloch_time + kspace_time

  # Print elapsed times
  MPI_print('Elapsed time for bloch simulation: {:.2f} s'.format(bloch_time))
  MPI_print('Elapsed time for k-space generation: {:.2f} s'.format(kspace_time))

  # Check if exp1_times.txt exists and load and append times (dummy_time, imaging_time, kspace_time) if it does or create it if it does not
  if MPI_rank == 0:
    times_file = script_path/f'results/4dflow/exp3_times_{pars.Imaging.Sequence}.txt'
    try:
      times = np.loadtxt(times_file)
      times = np.vstack((times, (bloch_time, kspace_time, sim_time, script_time)))
      np.savetxt(times_file, times)
    except FileNotFoundError:
      times = np.array([(bloch_time, kspace_time, sim_time, script_time)])
      np.savetxt(times_file, times)

  # Gather results
  K = gather_data(K)

  # Add noise to kspace
  K = add_cpx_noise(K, relative_std=0.00001)

  # # Export generated data
  # if MPI_rank==0:
  #   with open(str(export_path), 'wb') as f:
  #     pickle.dump({'kspace': K, 'MPS_ori': planning.MPS, 'LOC': planning.LOC, 'traj': traj}, f)

  # # Image reconstruction
  # I = CartesianRecon(K, traj)

  # kx, ky, kz: (R, L, S) in cycles/FOV
  # kdata: (R, L, S, C)
  RES = pars.Imaging.RECON_RES
  I = reconstruct_nufft(
      kdata=K.reshape((K.shape[0], K.shape[1], K.shape[2], -1)),  # (R, L, S, C)
      ktraj=tuple([k/(2*k.max()) for i, k in enumerate(traj_points)]),
      img_shape=RES,
      auto_dcw="radial-2d",   # fast, good default for radials
      mode="adjoint",
      combine=None
  )
  I = I.transpose((1,2,3,0)).reshape((RES[0], RES[1], RES[2], enc.nb_directions, -1))  # (Nx, Ny, Nz, enc, C)

  # Show reconstruction
  m = np.abs(I[...,0,:])
  mask = m > 0.5*m.max()
  phi = np.angle(I)
  f = enc.VENC[0].m_as('m/s')/np.pi
  phi_ref = f * np.angle(I[...,3,:]) * mask
  phi_vx = f * np.angle(I[...,0,:] * np.conj(I[...,3,:])) * mask
  phi_vy = f * np.angle(I[...,1,:] * np.conj(I[...,3,:])) * mask
  phi_vz = f * np.angle(I[...,2,:] * np.conj(I[...,3,:])) * mask
  if MPI_rank == 0:
    plotter = MRIPlotter(images=[m, phi_vx, phi_vy, phi_vz, phi_ref], title=['Magnitude', '$\\phi_{vx}$ ', '$\\phi_{vy}$', '$\\phi_{vz}$', '$\\phi_{ref}$'], FOV=planning.FOV.m_as('m'), swap_axes=[0, 2])
    plotter.show()
    # plotter.export_images('4dflow_images/plot_1/')

  #   plotter = MRIPlotter(images=[m, phi_vx, phi_vy, phi_vz, phi_ref], title=['Magnitude', '$\\phi_{vx}$ ', '$\\phi_{vy}$', '$\\phi_{vz}$', '$\\phi_{ref}$'], FOV=planning.FOV.m_as('m'))
  #   plotter.show()
  #   plotter.export_images('4dflow_images/plot_2/')