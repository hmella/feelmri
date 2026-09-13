import copy
import os

from skimage import data

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.IO import VTIFile
from feelmri.KSpaceTraj import CartesianStack
from feelmri.Motion import PODVelocity
from feelmri.MPIUtilities import MPI_print, gather_data
from feelmri.MRImaging import SliceProfile, VelocityEncoding
from feelmri.MRObjects import RF, Gradient, Scanner
from feelmri.Noise import add_cpx_noise
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.PulseqAdapter import maxwell_moments, maxwell_phase_coefficients
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon

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
  # B0 is passed, not left to the Scanner default: the YAML declares it, and
  # without this editing `Hardware.B0` changed nothing at all. It sets the
  # off-resonance scale below and the concomitant term, which goes as 1/B0.
  scanner = Scanner(field_strength=parameters.Hardware.B0.to('T'),
                    gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

  # Field inhomogeneity
  def spatial(x):
      return x[:,0] + x[:,1] + x[:,2]
  delta_B0 = spatial(phantom.local_nodes)
  delta_B0 /= np.abs(spatial(phantom.global_nodes).flatten()).max()
  delta_B0 = delta_B0 * scanner.field_strength * 1e-6  # 1.0 ppm of the main field

  # Phase shift in rad/s
  delta_omega0 = (2.0 * np.pi * scanner.gammabar * delta_B0).to('rad/ms')

  # Slice profile
  # The slice profile prepulse is calculated based on a reference RF pulse with
  # user-defined characteristics. The slice profile object allows accessing the calculated adjusted RF pulse and dephasing and rephasing gradients
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
    plot=True,          # show slice selection prepulse
    solve_profile=True, # estimate and show slice profile 
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
  conc_coefficients = []
  imaging_blocks = []
  for d in range(enc.nb_directions):

    # Create sequence object and Bloch solver
    seq = Sequence()
    solver = BlochSolver(seq, phantom, 
                         scanner=scanner, 
                         M0=1e+9, 
                         T1=parameters.Phantom.T1.to('ms'),
                         T2=parameters.Phantom.T2star.to('ms'), 
                         delta_B=delta_B0.m_as('mT').reshape((-1, 1)),
                         concomitant_fields=True,
                         pod_trajectory=pod_velocity)

    # Update reference time for second lobe (rotate function keep the time reference of the original gradient)
    [g.change_time(bp1r[d][0].time + bp1r[d][0].dur) for g in bp2r[d]]

    # Imaging block
    imaging = SequenceBlock(gradients=[sp.dephasing,sp.rephasing]+bp1r[d]+bp2r[d],
                            rf_pulses=[sp.rf], 
                            dt_rf=Q_(1e-2, 'ms'), 
                            dt_gr=Q_(1e-2, 'ms'), 
                            dt=Q_(1, 'ms'), 
                            store_magnetization=True)
    # The concomitant phase THIS direction's imaging block accumulates between
    # the excitation and the snapshot, as a quadratic form in imaging-frame
    # position. Same helpers the readout uses, integrated over the block's own
    # gradients from the RF centre -- which is where the transverse
    # magnetization is created and so where the clock starts.
    imaging_blocks.append(imaging)
    conc_coefficients.append(maxwell_phase_coefficients(
        maxwell_moments(imaging.gradients, sp.rf.time.m_as('ms'),
                        np.array([imaging.time_extent[1].m_as('ms')]),
                        rotation=planning.MPS),
        scanner, rotation=planning.MPS)[0])

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

  # Generate kspace trajectory
  traj = CartesianStack(FOV = planning.FOV.to('m'),
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
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, enc.nb_directions, Nb_frames], dtype=np.complex64)

  T2star = (parameters.Phantom.T2star * np.ones([phantom.local_nodes.shape[0]])).astype(np.float32)

  # Set assembler for MRI signal evaluation using FEM
  vxsz = planning.FOV.m_as('m')/np.array(parameters.Imaging.RES)
  phantom.set_assembler(voxel_size=vxsz[0], lorder=1, horder=6, nodal_approximation=True, lumped=False)

  # Set static fields
  phantom.set_static_fields(T2=T2star.m_as('ms'), phi_dB0=delta_omega0.m_as('rad/ms'))

  # Concomitant fields during the readout. The solver carries the term up to
  # the magnetization snapshot -- the slice select and, crucially here, the
  # VENC bipolar, which is exactly the construction Maxwell fields spoil. The
  # readout's own gradients are not in the solver's sequence at all, they live
  # in the trajectory, so their contribution has to be added on the signal
  # side. The two sets are disjoint, so nothing is counted twice.
  #
  # maxwell_coefficients integrates the trajectory's real waveform, prephasers
  # included: on this geometry they carry 52% of the window's whole
  # integral(Gx^2 + Gy^2) dt, and 59% of it has accumulated by the first ADC
  # sample. An estimate built from the sampled k-space sees none of that,
  # because the sampling starts after the prephasers are over. This works here
  # because t_start puts them at positive times, i.e. after the snapshot.
  # It also applies both rotations implied by the oblique MPS orientation --
  # the gradients are along logical axes while Bc is B0-aligned, and the nodes
  # are in the imaging frame.
  #
  # `carried` is what makes the split exact. `Bc` is QUADRATIC in G, so
  # `Bc(G_a + G_b) != Bc(G_a) + Bc(G_b)`, and the readout prephasers overlap
  # the tail of the VENC bipolar by 0.45 ms here -- computing the two halves
  # independently drops their cross term. Handing the solver's own gradients
  # over integrates the whole field once and subtracts back what the solver
  # already applied. Measured: the cross term is 15.8% of the concomitant
  # readout phase on the encoding direction and exactly 0% on the reference,
  # so it does NOT cancel in phi_v -- it lands on the quantity this example is
  # about. It is also constant across the readout (to 2.8e-13), because the
  # overlap ends at the snapshot.
  #
  # The gradients are shifted into the trajectory's own time frame, which runs
  # from the RF centre.
  maxwell = []
  for d in range(enc.nb_directions):
      carried = []
      for g in imaging_blocks[d].gradients:
          g_shifted = copy.deepcopy(g)
          g_shifted.change_time(g.time - sp.rf.time)
          carried.append(g_shifted)
      maxwell.append(traj.maxwell_coefficients(scanner, carried=carried))
  # The part that does not cancel between the two encodings, as a constant
  # quadratic form: what the panel below has to include alongside the solver's.
  readout_bias = [m[0] - traj.maxwell_coefficients(scanner)[0]
                  for m in maxwell]

  # Iterate over cardiac phases
  for fr in range(Nb_frames):

      # Print progress
      MPI_print('Frame {:d}/{:d}'.format(fr+1, Nb_frames))

      # Update timeshift in the POD velocity
      pod_velocity.update_timeshift(fr * parameters.Imaging.TimeSpacing.m_as('ms'))

      # One readout per encoding direction, because `maxwell` now differs
      # between them: the overlap cross term above is a property of the VENC
      # bipolar, which is what the two directions differ by.

      # Generate 4D flow image
      # Elapsed time since the MAGNETIZATION SNAPSHOT, not since the
      # trajectory's own origin. `mri_signal` applies exp(-t/T2*) and
      # exp(-i*phi*t) continuing from the instant the magnetization was
      # captured, and on a CartesianStack that instant is `t_start` -- the
      # timeline runs from the RF centre and the readout begins where the
      # imaging block ends. Feeding absolute times applies a spurious
      # exp(-t_start/T2*) and, worse, a SPATIALLY VARYING phi*t_start:
      # measured 1.688 rad peak-to-peak across the object here.
      for d in range(enc.nb_directions):
          phantom.update_magnetization(Mxy_PC[:, fr, d])
          K[:,:,:,d:d+1,fr] = phantom.mri_signal(
              traj.points,
              traj.times.m_as('ms') - traj.t_start.m_as('ms'),
              pod_velocity,
              maxwell=maxwell[d])

  # Gather results
  K = gather_data(K)

  # Add noise to kspace
  K = add_cpx_noise(K, relative_std=0.01)

  # Image reconstruction
  Im = CartesianRecon(K, traj)

  # Show reconstruction
  mag = np.abs(Im[...,0,:])
  phi_v = np.angle(Im[...,0,:] * np.conj(Im[...,1,:]))
  phi_0 = np.angle(Im[...,1,:])
  phi   = np.angle(Im[...,0,:])

  # Concomitant field: the part that SURVIVES the velocity subtraction.
  #
  # phi_v is direction 0 minus the reference, so any phase the two share
  # cancels -- and the READOUT term does share: `maxwell` above is one array
  # handed to every direction, because the readout gradients do not change
  # with the encoding. What does not cancel is the phase the SOLVER
  # accumulates from the VENC bipolar, which direction 0 plays and the
  # reference does not. That difference is a systematic velocity error, fixed
  # in space and quadratic in position -- it does not average out over frames
  # and it is indistinguishable from flow.
  # Voxel centres, sized from the RECONSTRUCTED shape rather than from
  # `Imaging.RES`: `CartesianRecon` undoes the readout oversampling, and if the
  # two ever disagreed the map would be evaluated on the wrong grid.
  vox = planning.FOV.m_as('m') / np.array(Im.shape[:3])
  grid = [(np.arange(n) - 0.5 * (n - 1)) * h
          for n, h in zip(Im.shape[:3], vox)]
  gx, gy, gz = np.meshgrid(*grid, indexing='ij')
  def _quadratic(c):
      return (c[0]*gx*gx + c[1]*gy*gy + c[2]*gz*gz
              + c[3]*gx*gy + c[4]*gx*gz + c[5]*gy*gz)
  phi_c = (_quadratic(conc_coefficients[0] + readout_bias[0])
           - _quadratic(conc_coefficients[1] + readout_bias[1]))
  phi_c = np.repeat(phi_c[..., np.newaxis], Im.shape[-1], axis=-1)
  v_err = parameters.VelocityEncoding.VENC.m_as('m/s') / np.pi * phi_c
  MPI_print(f'[concomitant] bias in phi_v over the FOV: '
            f'{phi_c.min():+.4f} to {phi_c.max():+.4f} rad, i.e. an apparent '
            f'{v_err.min():+.4f} to {v_err.max():+.4f} m/s')

  plotter = MRIPlotter(images=[mag, phi_v, phi, phi_0, phi_c],
                        title=['M', '$\\phi_v$ ', '$\\phi_v + \\phi_0$', '$\\phi_0$',
                               '$\\phi_c$ (concomitant bias in $\\phi_v$)'],
                        FOV=planning.FOV.m_as('m'))
  plotter.show()
  plotter.export_images(script_path/'phase_contrast/')

  # Write the velocity field to a VTI file for visualization in Paraview
  spacing = (planning.FOV.m_as('m')/parameters.Imaging.RES).tolist()  
  origin = -0.5*planning.FOV.m_as('m')
  origin  = (planning.MPS@origin + planning.LOC.m_as('m')).tolist()
  direction = planning.MPS.flatten().tolist()

  vti_file = VTIFile(script_path/'phase_contrast/velocity.pvd',
                    origin=origin,
                    spacing=spacing,
                    direction=direction,
                    nbFrames=Nb_frames,
                    dt=parameters.Imaging.TimeSpacing.m_as('ms'))
  vti_file.write(cellData={'magnitude': mag,
                            'phase_v': phi_v, 
                            'phase': phi, 
                            'phase_0': phi_0})