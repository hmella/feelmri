import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.IO import XDMFFile
from feelmri.KSpaceTraj import CartesianStack
from feelmri.Motion import POD
from feelmri.MPIUtilities import MPI_print, MPI_rank, gather_data
from feelmri.MRI import Signal
from feelmri.MRImaging import PositionEncoding, SliceProfile
from feelmri.MRObjects import RF, Gradient, Scanner
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon

# Enable fast mode for testing if the environment variable is set
FAST_MODE = os.getenv("FEELMRI_FAST_TEST", "0") == "1"

if FAST_MODE:
    Nb_frames = 1
    dummy_pulses = 1
else:
    Nb_frames = int(...)
    dummy_pulses = 80

if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  parameters = ParameterHandler(script_path/'parameters/spamm.yaml')

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(script_path/parameters.Formatting.planning,
                          box_name='Box1',
                          transform_name='Transform1',
                          length_units=parameters.Formatting.units)

  # Create FEM phantom object
  phantom = FEMPhantom(path=script_path/'phantoms/heart_P2_tetra.xdmf', scale_factor=1.0)

  # Translate phantom to obtain the desired slice location
  phantom.orient(planning.MPS, planning.LOC)

  # Velocity encoding parameters
  ke_dirs = list(parameters.PositionEncoding.Directions.values())
  enc = PositionEncoding(parameters.PositionEncoding.ke.m_as('1/m'), np.array(ke_dirs))

  # We can a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV
  mp = phantom.global_nodes[phantom.global_elements].mean(axis=1)
  markers = np.abs(mp[:, 2]) <= 4.0*planning.FOV[2].m_as('m')
  phantom.create_submesh(markers)

  # Create array to store displacements
  u = np.zeros([phantom.global_shape[0], 3, phantom.Nfr], dtype=np.float32)
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr and interpolate to the submesh
    phantom.read_data(fr)
    u[..., fr] = phantom.to_submesh(phantom.point_data['displacement'] @ planning.MPS, global_mesh=True)

  # Create POD for tissue displacements
  dt = parameters.Imaging.TimeSpacing.to('ms')
  u_times = np.linspace(0, (phantom.Nfr-1)*dt, phantom.Nfr, dtype=np.float32)
  pod_trajectory = POD(times=u_times.m_as('ms'),
                      data=u,
                      global_to_local=phantom.local_to_global_nodes,
                      n_modes=10,
                      is_periodic=True,
                      interpolation_method='Pchip')
  
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
          flip_angle=parameters.Imaging.FlipAngle.to('rad'))
  sp = SliceProfile(delta_z=planning.FOV[2].to('m'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=False, 
    bandwidth=Q_(10, 'kHz'))
  # sp.optimize(frac_start=0.7, frac_end=0.8, N=100, profile_samples=100)

  # SPAMM magnetization
  Nb_frames = phantom.Nfr if not FAST_MODE else 1
  Mxy_spamm = np.zeros((phantom.local_nodes.shape[0], Nb_frames, enc.nb_directions), dtype=np.complex64)

  # Simulate the SPAMM preparation block for each encoding direction
  for d in range(len(enc.directions)):

    # Create sequence object
    seq = Sequence()

    # SPAMM preparation block
    rf1 = RF(scanner=scanner, shape='hard', dur=Q_(0.2, 'ms'), flip_angle=Q_(np.deg2rad(90),'rad'), time=Q_(0.0, 'ms'))

    G_tag = Gradient(scanner=scanner, time=rf1.time + rf1.dur)
    G_tag.match_area((parameters.PositionEncoding.ke/scanner.gamma).to('mT*ms/m'))
    G_tag_list = G_tag.rotate(enc.directions[d])

    rf2 = RF(scanner=scanner, shape='hard', dur=Q_(0.2, 'ms'), flip_angle=Q_(np.deg2rad(90),'rad'), phase_offset=Q_(np.deg2rad(180)*d,'rad'), time=G_tag.time + G_tag.dur)

    prep = SequenceBlock(gradients=G_tag_list, rf_pulses=[rf1, rf2], dt_rf=Q_(1e-2, 'ms'), dt_gr=Q_(1e-2, 'ms'), dt=Q_(1, 'ms'), store_magnetization=False)

    # Imaging block
    imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing], rf_pulses=[sp.rf], dt_rf=Q_(1e-2, 'ms'), dt_gr=Q_(1e-2, 'ms'), dt=Q_(1, 'ms'), store_magnetization=True)

    # Create dummy block to reach steady state
    dummy = imaging.copy()
    dummy.store_magnetization = False

    # Add dummy blocks to the sequence to reach steady state
    time_spacing = parameters.Imaging.TimeSpacing.to('ms') - (imaging.time_extent[1] - sp.rf.ref)
    for i in range(dummy_pulses):
      seq.add_block(dummy)
      seq.add_block(time_spacing, dt=Q_(1, 'ms'))

    # Add and additional block to synchronize the sequence with the cardiac cycle
    seq.add_block(u_times[-1] - seq.blocks[-1].time_extent[1] % u_times[-1], dt=Q_(1, 'ms'))

    # Add blocks to the sequence
    seq.add_block(prep)
    seq.add_block(Q_(0.25, 'ms'))
    for fr in range(Nb_frames):
      seq.add_block(imaging)
      # seq.plot(blocks=slice(-3, None), figsize=(4, 6), tight_layout=True, export_to='ocspamm_sequence_{:d}.png'.format(d))
      seq.add_block(time_spacing, dt=Q_(1, 'ms'))  # Time spacing between frames

    # Bloch solver
    solver = BlochSolver(seq, phantom, 
                         scanner=scanner, 
                         M0=1e+9, 
                         T1=parameters.Phantom.T1, 
                         T2=parameters.Phantom.T2, 
                         delta_B=delta_B0.reshape((-1, 1)),
                         pod_trajectory=pod_trajectory)

    # Solve for x and y directions
    Mxy, Mz = solver.solve() 
    # # Export magnetization and displacement for debugging
    # file = XDMFFile('magnetization_{:d}.xdmf'.format(MPI_rank), nodes=phantom.local_nodes, elements={phantom.cell_type: phantom.local_elements})
    # for fr in range(Mxy.shape[1]):
    #   # Write magnetization and displacement at each frame
    #   t = fr*parameters.Imaging.TimeSpacing.m_as('ms') + imaging.rf_pulses[0].time.m_as('ms')
    #   file.write(pointData={'M': np.stack((np.real(Mxy[:,fr]), np.imag(Mxy[:,fr]), Mz[:,fr]), axis=1), 'displacement': pod_trajectory(t)}, time=t)
    # file.close()

    # Assign the magnetization to the corresponding direction
    Mxy_spamm[..., d] = Mxy

  # Generate kspace trajectory
  traj = CartesianStack(FOV=planning.FOV.to('m'),
                      t_start=imaging.time_extent[1] - sp.rf.time,
                      res=parameters.Imaging.RES, 
                      oversampling=parameters.Imaging.Oversampling, 
                      lines_per_shot=parameters.Imaging.LinesPerShot, 
                      MPS_ori=planning.MPS,
                      LOC=planning.LOC.to('m'),
                      receiver_bw=parameters.Hardware.r_BW.to('Hz'), 
                      plot_seq=False)

  # Print echo time
  MPI_print('Echo time: {:.2f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, enc.nb_directions, Nb_frames], dtype=np.complex64)

  # T2 relaxation time
  T2 = np.ones([phantom.local_nodes.shape[0], ], dtype=np.float32)*parameters.Phantom.T2

  # Assemble mass matrix for integrals (just once)
  M = phantom.mass_matrix(lumped=True, quadrature_order=2)

  # Iterate over cardiac phases
  kspace_points = traj.points
  kspace_times = traj.times.m_as('ms') - traj.t_start.m_as('ms')
  for fr in range(Nb_frames):

    # Print progress
    MPI_print("Generating frame {:d}/{:d}".format(fr+1, Nb_frames))

    # Update reference time of POD trajectory
    pod_trajectory.update_timeshift(fr * parameters.Imaging.TimeSpacing.m_as('ms'))

    # Generate 4D flow image
    K[:,:,:,:,fr] = Signal(MPI_rank, M, kspace_points, kspace_times, phantom.local_nodes, delta_omega0, T2.m_as('ms'), Mxy_spamm[:, fr, :], pod_trajectory)

  # Gather results
  K = gather_data(K)

  # Image reconstruction
  I = CartesianRecon(K, traj)

  # Show reconstruction
  if MPI_rank == 0:
    mx = np.abs(I[...,0,:])
    my = np.abs(I[...,1,:])
    mxy = np.abs(I[...,0,:] - I[...,1,:])
    plotter = MRIPlotter(images=[mx, my, mxy], 
                        title=['SPAMM X', 'SPAMM Y', 'O-CSPAMM'], 
                        FOV=planning.FOV.m_as('m'), caxis=[0, 13])
    # plotter.export_images('spamm_images_semi_bloch/')
    plotter.show()