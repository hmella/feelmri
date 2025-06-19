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
from feelmri.KSpaceTraj import CartesianStack
from feelmri.Motion import PODVelocity
from feelmri.MPIUtilities import MPI_print, MPI_rank, gather_data
from feelmri.MRImaging import SliceProfile, VelocityEncoding
from feelmri.MRObjects import RF, Gradient, Scanner
from feelmri.Noise import add_cpx_noise
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.PhaseContrast import PC
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon

if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  parameters = ParameterHandler(script_path/'parameters/phase_contrast.yaml')

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(script_path/parameters.Formatting.planning,
                      box_name='Box1',
                      transform_name='Transform1',
                      length_units=parameters.Formatting.units)

  # Create FEM phantom object
  phantom = FEMPhantom(script_path/'phantoms/aorta_CFD.xdmf', velocity_label='velocity', scale_factor=0.01)

  # Translate phantom to obtain the desired slice location
  phantom.orient(planning.MPS, planning.LOC.to('m'))

  # Velocity encoding parameters
  venc_dirs = list(parameters.VelocityEncoding.Directions.values())
  enc = VelocityEncoding(parameters.VelocityEncoding.VENC, np.array(venc_dirs))

  # We can a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV
  mp = phantom.global_nodes[phantom.global_elements].mean(axis=1)
  markers = np.abs(mp[:, 2]) <= 0.5 * planning.FOV[2].m_as('m')
  markers *= np.abs(mp[:, 1]) <= 0.5 * planning.FOV[1].m_as('m')
  markers *= np.abs(mp[:, 0]) <= 0.5 * planning.FOV[0].m_as('m')
  phantom.create_submesh(markers)

  # Create array to store displacements
  v = Q_(np.zeros([phantom.global_shape[0], 3, phantom.Nfr], dtype=np.float32), 'm/s')
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr and interpolate to the submesh
    phantom.read_data(fr)
    v[..., fr] = Q_(phantom.interpolate_to_submesh(phantom.point_data['velocity'] @ planning.MPS, local=False), 'm/s')

  # Define POD object
  dt = parameters.Imaging.TimeSpacing
  times = np.linspace(0, (phantom.Nfr-1)*dt, phantom.Nfr)
  pod_velocity = PODVelocity(time_array=times.m_as('ms'),
                              data=v.m_as('m/ms'),
                              global_to_local=phantom.global_to_local_nodes,
                              n_modes=5,
                              taylor_order=10,
                              is_periodic=True)

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

  # Field inhomogeneity
  def spatial(x):
      return x[:,0] + x[:,1] + x[:,2]
  delta_B0 = spatial(phantom.local_nodes)
  delta_B0 /= np.abs(spatial(phantom.global_nodes).flatten()).max()
  delta_B0 *= 0.0 * 1.5 * 1e-6           # 1.5 ppm of the main magnetic field
  delta_omega0 = 2.0 * np.pi * scanner.gammabar.m_as('1/ms/T') * delta_B0

  # Slice profile
  # The slice profile prepulse is calculated based on a reference RF pulse with
  # user-defined characteristics. The slice profile object allows accessing the calculated adjusted RF pulse and dephasing and rephasing gradients
  rf = RF(scanner=scanner, 
          NbLobes=[4, 4], 
          alpha=0.46, 
          shape='apodized_sinc', 
          flip_angle=parameters.Imaging.FlipAngle.to('rad'), 
          t_ref=Q_(0.0,'ms'),
          phase_offset=Q_(-90, 'deg'))
  sp = SliceProfile(delta_z=planning.FOV[2].to('m'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=False, 
    bandwidth=Q_(10000, 'Hz'), #'maximum',
    refocusing_area_frac=0.78185)
  # sp.optimize(frac_start=0.79, frac_end=0.81, N=100)

  # Create sequence object and solve magnetization
  Mxy_PC = np.zeros([phantom.local_nodes.shape[0], phantom.Nfr, enc.nb_directions], dtype=np.complex64)
  for d in range(enc.nb_directions):

    # Create sequence object and Bloch solver
    seq = Sequence()
    solver = BlochSolver(seq, phantom, 
                         scanner=scanner, 
                         M0=1e+9, 
                         T1=parameters.Phantom.T1.to('ms'),
                         T2=parameters.Phantom.T2star.to('ms'), 
                         delta_B=delta_B0.reshape((-1, 1)),
                         pod_trajectory=pod_velocity)

    # Bipolar gradient
    t_ref = sp.rephasing.t_ref + sp.rephasing.dur
    bp1 = Gradient(scanner=scanner, t_ref=t_ref)
    bp2 = bp1.make_bipolar(parameters.VelocityEncoding.VENC.to('m/s'))

    # Rotate the bipolar gradients to the desired direction
    bp1_rotated = bp1.rotate(enc.directions[d])
    bp2_rotated = bp2.rotate(enc.directions[d])

    # Update reference time for second lobe
    [g.update_reference(g.t_ref - (bp1.dur - g.dur)) for g in bp2_rotated]

    # Imaging block
    imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing] + bp1_rotated + bp2_rotated,
                            rf_pulses=[sp.rf], 
                            dt_rf=Q_(1e-2, 'ms'), 
                            dt_gr=Q_(1e-2, 'ms'), 
                            dt=Q_(1, 'ms'), 
                            store_magnetization=True)
    dummy = imaging.copy()
    dummy.store_magnetization = False

    # Add dummy blocks to the sequence to reach steady state
    time_spacing = parameters.Imaging.TimeSpacing.to('ms') - (imaging.time_extent[1] - sp.rf.t_ref)
    for i in range(80):
      seq.add_block(dummy)
      seq.add_block(time_spacing, dt=Q_(1, 'ms'))
    
    # Add and additional block to synchronize the sequence with the cardiac cycle
    seq.add_block(times[-1] - seq.blocks[-1].time_extent[1] % times[-1], dt=Q_(1, 'ms'))

    # Add PC imaging sequence
    for fr in range(phantom.Nfr):
      seq.add_block(imaging)
      seq.add_block(time_spacing, dt=Q_(1, 'ms'))  # Time spacing between frames

    # Solve for x and y directions
    Mxy, Mz = solver.solve()
    Mxy_PC[..., d] = Mxy

    # # Export magnetization and displacement for debugging
    # file = XDMFFile('magnetization_{:d}.xdmf'.format(MPI_rank), nodes=phantom.local_nodes, elements={'tetra': phantom.local_elements})
    # for fr in range(Mxy.shape[1]):
    #   # Write magnetization and displacement at each frame
    #   file.write(pointData={'M': np.stack((np.real(Mxy[:,fr]), np.imag(Mxy[:,fr]), Mz[:,fr]), axis=1)}, time=fr*(parameters.Imaging.TimeSpacing.m_as('ms') + time_spacing.m_as('ms')))
    # file.close()

  # Path to export the generated data
  export_path = Path(script_path/'MRImages/phase_contrast_{:s}_V{:.0f}.pkl'.format(parameters.Imaging.Sequence, parameters.VelocityEncoding.VENC.m_as('cm/s')))

  # Make sure the directory exist
  os.makedirs(str(export_path.parent), exist_ok=True)

  # Generate kspace trajectory
  traj = CartesianStack(FOV = planning.FOV.to('m'),
    t_start = imaging.time_extent[1] - sp.rf.t_ref,
    res = parameters.Imaging.RES, 
    oversampling = parameters.Imaging.Oversampling, 
    lines_per_shot = parameters.Imaging.LinesPerShot, 
    MPS_ori = planning.MPS, 
    LOC = planning.LOC, 
    receiver_bw=parameters.Hardware.r_BW, 
    plot_seq=False)

  # Print echo time
  MPI_print('Echo time = {:.2f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, enc.nb_directions, phantom.Nfr], dtype=np.complex64)

  T2star = (parameters.Phantom.T2star * np.ones([phantom.local_nodes.shape[0]])).astype(np.float32)

  # Assemble mass matrix for integrals (just once to accelerate the simulation)
  M = phantom.mass_matrix(lumped=True)

  # Iterate over cardiac phases
  t0 = time.time()
  for fr in range(phantom.Nfr):

      # Start time for the frame      
      t0_fr = time.time()

      # Update timeshift in the POD velocity
      pod_velocity.update_timeshift(fr * parameters.Imaging.TimeSpacing.m_as('ms'))

      # Generate 4D flow image
      K[:,:,:,:,fr] = PC(MPI_rank, M, traj.points, traj.times.m_as('ms'), phantom.local_nodes, delta_omega0, T2star.m_as('ms'), Mxy_PC[:, fr, :], pod_velocity)

      if MPI_rank == 0:
        sys.stdout.flush()
        sys.stdout.write("\r" + 'Elapsed time frame {:d}: {:.2f} s'.format(fr, time.time()-t0_fr))

  # Store elapsed time
  pc_time = time.time() - t0

  # Print elapsed times
  MPI_print('Elapsed time-per-frame for 4D Flow generation: {:.2f} s'.format(pc_time/phantom.Nfr))
  MPI_print('Elapsed time for 4D Flow generation: {:.2f} s'.format(pc_time))

  # Gather results
  K = gather_data(K)

  # Add noise to kspace
  K = add_cpx_noise(K, relative_std=0.0025)

  # Export generated data
  if MPI_rank==0:
    with open(str(export_path), 'wb') as f:
      pickle.dump({'kspace': K, 'MPS_ori': planning.MPS, 'LOC': planning.LOC, 'traj': traj}, f)

  # Image reconstruction
  Im = CartesianRecon(K, traj)

  # Show reconstruction
  mag = np.abs(Im[...,0,:])
  phi_v = np.angle(Im[...,0,:] * np.conj(Im[...,1,:]))
  phi_0 = np.angle(Im[...,1,:])
  phi   = np.angle(Im[...,0,:])
  if MPI_rank == 0:
    plotter = MRIPlotter(images=[mag, phi_v, phi, phi_0], title=['Magnitude', '$\\phi_v$ ', '$\\phi_v + \\phi_0$', '$\\phi_0$'], FOV=planning.FOV.m_as('m'))
    plotter.show()