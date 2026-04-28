import os
from pathlib import Path
import numpy as np
from pint import Quantity as Q_

os.environ["OPENBLAS_NUM_THREADS"] = "1" 

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.IO import VTIFile
from feelmri.KSpaceTraj import CartesianStack, RadialStack, SpiralStack
from feelmri.Motion import PODVelocity
from feelmri.MPIUtilities import MPI_print, gather_data
from feelmri.MRImaging import SliceProfile, VelocityEncoding
from feelmri.MRObjects import RF, Gradient, Scanner
from feelmri.Noise import add_cpx_noise
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon, reconstruct_nufft

# Enable fast mode for testing if the environment variable is set
FAST_MODE = os.getenv("FEELMRI_FAST_TEST", "0") == "1"

if FAST_MODE:
  Nb_frames = 1
  dummy_pulses = 1
  resolution = [30, 15, 1]
else:
  Nb_frames = -1
  dummy_pulses = 80

if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  parameters = ParameterHandler(script_path/'parameters/phase_contrast.yaml')

  # Make resolution lower for CI testing
  if FAST_MODE:
    parameters.Imaging.RES = np.array(resolution)

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(script_path/parameters.Formatting.planning,
                        box_name='Box1',
                        transform_name='Transform1',
                        length_units=parameters.Formatting.units)

  # Create FEM phantom object
  phantom = FEMPhantom(script_path/'phantoms/aorta_P1_tetra.xdmf', velocity_label='velocity', scale_factor=0.01)

  # Translate phantom to obtain the desired slice location
  phantom.orient(planning.MPS, planning.LOC.to('m'))

  # Velocity encoding parameters
  venc_dirs = list(parameters.VelocityEncoding.Directions.values())
  enc = VelocityEncoding(parameters.VelocityEncoding.VENC, np.array(venc_dirs))

  # We can a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV
  mp = phantom.global_nodes[phantom.global_elements].mean(axis=1)
  markers = np.abs(mp[:, 2]) <= 0.5 * planning.FOV[2].m_as('m')
  phantom.create_submesh(markers)

  # Create array to store displacements
  v = Q_(np.zeros([phantom.global_shape[0], 3, phantom.Nfr], dtype=np.float32), 'm/s')
  for fr in range(phantom.Nfr):
    # Read velocity data in frame fr and interpolate to the submesh
    phantom.read_data(fr)
    v[..., fr] = Q_(phantom.to_submesh(phantom.point_data['velocity'] @ planning.MPS, global_mesh=True), 'm/s')

  # Define POD object
  dt = parameters.Imaging.TimeSpacing
  times = np.linspace(0, (phantom.Nfr-1)*dt, phantom.Nfr, dtype=np.float32)
  pod_velocity = PODVelocity(times=times.m_as('ms'),
                             data=v.m_as('m/ms'),
                             global_to_local=phantom.local_to_global_nodes,
                             n_modes=25,
                             is_periodic=True)

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

  # Field inhomogeneity
  def spatial(x):
    return x[:,0] + x[:,1] + x[:,2]
  delta_B0 = spatial(phantom.local_nodes)
  delta_B0 /= np.abs(spatial(phantom.global_nodes).flatten()).max()
  delta_B0 *= scanner.field_strength * 1e-6 # 1.5 ppm of the main magnetic field

  # Phase shift in rad/s
  delta_omega0 = (2.0 * np.pi * scanner.gammabar * delta_B0).to('rad/ms')

  # Slice profile
  rf = RF(scanner=scanner, 
          NbLobes=[4, 4], 
          alpha=0.46, 
          shape='apodized_sinc', 
          flip_angle=parameters.Imaging.FlipAngle.to('rad'), 
          phase_offset=Q_(-90, 'deg'))
  sp = SliceProfile(delta_z=planning.FOV[2].to('m'), 
                    profile_samples=100,
                    rf=rf,
                    dt=Q_(1e-2, 'ms'), 
                    plot=False, 
                    solve_profile=True, 
                    bandwidth=Q_(10, 'kHz'))

  # Create bipolar gradients
  start = sp.rephasing.time + sp.rephasing.dur
  bp1 = Gradient(scanner=scanner, time=start)
  bp2 = bp1.make_bipolar(parameters.VelocityEncoding.VENC)

  # Rotate the bipolar gradients to the desired direction
  bp1r = bp1.rotate(enc.directions)
  bp2r = bp2.rotate(enc.directions)

  # Create sequence object and solve magnetization
  Nb_frames = phantom.Nfr if not FAST_MODE else 1
  Mxy_PC = np.zeros([phantom.local_nodes.shape[0], Nb_frames, enc.nb_directions], dtype=np.complex64)

  # Solve Bloch equations once to use for all trajectory types
  MPI_print('Running Bloch simulation...')
  for d in range(enc.nb_directions):

    # Create sequence object and Bloch solver
    seq = Sequence()
    solver = BlochSolver(seq, phantom, 
                         scanner=scanner, 
                         M0=1e+9, 
                         T1=parameters.Phantom.T1.to('ms'),
                         T2=parameters.Phantom.T2star.to('ms'), 
                         delta_B=delta_B0.m_as('mT').reshape((-1, 1)),
                         pod_trajectory=pod_velocity)

    # Update reference time for second lobe
    [g.change_time(bp1r[d][0].time + bp1r[d][0].dur) for g in bp2r[d]]

    # Imaging block
    imaging = SequenceBlock(gradients=[sp.dephasing,sp.rephasing]+bp1r[d]+bp2r[d],
                            rf_pulses=[sp.rf], 
                            dt_rf=Q_(1e-2, 'ms'), 
                            dt_gr=Q_(1e-2, 'ms'), 
                            dt=Q_(1, 'ms'), 
                            store_magnetization=True)
    dummy = imaging.copy()
    dummy.store_magnetization = False

    # Add dummy blocks to the sequence to reach steady state
    time_spacing = parameters.Imaging.TimeSpacing - imaging.dur
    for i in range(dummy_pulses):
      seq.add_block(dummy)
      seq.add_block(time_spacing, dt=Q_(1, 'ms'))

    # Add and additional block to synchronize the sequence with the cardiac cycle
    seq.add_block(times[-1] - seq.blocks[-1].time_extent[1] % times[-1], dt=Q_(1, 'ms'))

    # Add PC imaging sequence
    for fr in range(Nb_frames):
      seq.add_block(imaging)
      seq.add_block(time_spacing, dt=Q_(1, 'ms'))  # Time spacing between frames

    # Solve for x and y directions
    Mxy, Mz = solver.solve()
    Mxy_PC[..., d] = Mxy

  # Set assembler for MRI signal evaluation using FEM
  vxsz = planning.FOV.m_as('m')/np.array(parameters.Imaging.RES)
  phantom.set_assembler(voxel_size=vxsz[0], lorder=1, horder=6, nodal_approximation=True, lumped=False)

  # Set static fields
  T2star = (parameters.Phantom.T2star * np.ones([phantom.local_nodes.shape[0]])).astype(np.float32)
  phantom.set_static_fields(T2=T2star.m_as('ms'), phi_dB0=delta_omega0.m_as('rad/ms'))

  # Define the trajectory types to test
  trajectories = {
    'Cartesian': CartesianStack,
    'Radial': RadialStack,
    'Spiral': SpiralStack
  }

  # Test all trajectories and reconstruction algorithms
  for traj_name, TrajClass in trajectories.items():
    MPI_print('Testing {:s} trajectory and reconstruction...'.format(traj_name))

    print(traj_name)

    # Generate kspace trajectory
    traj = TrajClass(FOV = planning.FOV.to('m'),
                     t_start = imaging.time_extent[1] - sp.rf.time,
                     res = parameters.Imaging.RES, 
                     oversampling = parameters.Imaging.Oversampling, 
                     lines_per_shot = parameters.Imaging.LinesPerShot, 
                     MPS_ori = planning.MPS, 
                     LOC = planning.LOC, 
                     receiver_bw=parameters.Hardware.r_BW, 
                     plot_seq=False)

    # Echo time
    MPI_print('Echo time = {:.2f} ms'.format(traj.echo_time.m_as('ms')))

    # kspace array
    K = np.zeros([traj.ro_samples, traj.ph_samples, traj.slices, enc.nb_directions, Nb_frames], dtype=np.complex64)

    # Iterate over cardiac phases
    for fr in range(Nb_frames):
      # Update timeshift in the POD velocity
      pod_velocity.update_timeshift(fr * parameters.Imaging.TimeSpacing.m_as('ms'))

      # Update magnetization
      phantom.update_magnetization(Mxy_PC[:, fr, :])

      # Generate 4D flow image
      K[:,:,:,:,fr] = phantom.mri_signal(traj.points, traj.times.m_as('ms'), pod_velocity)

    # Gather results
    K = gather_data(K)

    # Add noise to kspace
    K = add_cpx_noise(K, relative_std=0.01)

    # Image reconstruction based on trajectory topology
    if traj_name == 'Cartesian':
      Im = CartesianRecon(K, traj)
    else:
      # Select density compensation function
      target_dcw = "radial-2d" if traj_name == "Radial" else "speed"
      
      # NUFFT processes one frame at a time
      Im_frames = []
      for fr in range(Nb_frames):
        MPI_print('Running NUFFT recon for frame {:d}/{:d} using {:s} DCF...'.format(fr+1, Nb_frames, target_dcw))
        Im_fr = reconstruct_nufft(kdata=K[..., fr],
                                  ktraj=traj.points,
                                  img_shape=parameters.Imaging.RES,
                                  fov=planning.FOV.m_as('m'),
                                  auto_dcw=target_dcw,
                                  oversamp=1.25,
                                  combine=None)

        # If the trajectory is 2D and Nz=1, NUFFT optimizes by returning (C, X, Y). 
        # We must re-add the singleton Z dimension to make it (C, X, Y, 1) before transposing.
        if Im_fr.ndim == 3:
          Im_fr = np.expand_dims(Im_fr, axis=-1)
        
        # Reshape from (C, X, Y, Z) to (X, Y, Z, C) to match Cartesian output
        Im_fr = np.transpose(Im_fr, (1, 2, 3, 0))
        Im_frames.append(Im_fr)
      
      # Stack frames to yield shape (X, Y, Z, C, F)
      Im = np.stack(Im_frames, axis=-1)

    # Show reconstruction
    mag = np.abs(Im[...,0,:])
    phi_v = np.angle(Im[...,0,:] * np.conj(Im[...,1,:]))
    phi_0 = np.angle(Im[...,1,:])
    phi   = np.angle(Im[...,0,:])
    
    out_dir = script_path / 'phase_contrast_{:s}'.format(traj_name.lower())
    
    plotter = MRIPlotter(images=[mag, phi_v, phi, phi_0], 
                         title=['{:s} M'.format(traj_name), '{:s} $\\phi_v$'.format(traj_name), '$\\phi_v + \\phi_0$', '$\\phi_0$'], 
                         FOV=planning.FOV.m_as('m'))
    plotter.show()

    # Write the velocity field to a VTI file for visualization in Paraview
    spacing = (planning.FOV.m_as('m')/parameters.Imaging.RES).tolist()  
    origin = -0.5*planning.FOV.m_as('m')
    origin  = (planning.MPS@origin + planning.LOC.m_as('m')).tolist()
    direction = planning.MPS.flatten().tolist()

    vti_file = VTIFile(out_dir / 'velocity_{:s}.pvd'.format(traj_name.lower()),
                       origin=origin,
                       spacing=spacing,
                       direction=direction,
                       nbFrames=Nb_frames,
                       dt=parameters.Imaging.TimeSpacing.m_as('ms'))
                       
    vti_file.write(cellData={'magnitude': mag,
                             'phase_v': phi_v, 
                             'phase': phi, 
                             'phase_0': phi_0})

  MPI_print('All reconstruction tests completed.')