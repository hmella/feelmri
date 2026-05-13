import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.KSpaceTraj import CartesianStack
from feelmri.IO import XDMFFile
from feelmri.Motion import POD
from feelmri.MPIUtilities import MPI_print, gather_data
from feelmri.MRImaging import PositionEncoding, SliceProfile
from feelmri.MRObjects import RF, Gradient, Scanner
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon, reconstruct_nufft
from feelmri.PulseqAdapter import (
    import_pulseq,
    kspace_to_signal_inputs,
    kspace_trajectory,
)
import matplotlib.pyplot as plt


# Enable fast mode for testing if the environment variable is set
FAST_MODE = os.getenv("FEELMRI_FAST_TEST", "0") == "1"

if FAST_MODE:
    Nb_frames = 1
    dummy_pulses = 1
else:
    Nb_frames = -1
    dummy_pulses = 80

if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  parameters = ParameterHandler(script_path/'parameters/spamm_pulseq.yaml')

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(script_path/parameters.Formatting.planning,
                          box_name='Box1',
                          transform_name='Transform1',
                          length_units=parameters.Formatting.units)

  # Create FEM phantom object
  phantom = FEMPhantom(path=script_path/'phantoms/heart_P1_hex.xdmf', scale_factor=1.0)

  # Translate phantom to obtain the desired slice location
  phantom.orient(planning.MPS, planning.LOC)

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
  dt = parameters.Phantom.TimeSpacing.to('ms')
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
  delta_B0 *= scanner.field_strength * 1e-6 * 0.0 # 1.5 ppm of the main magnetic field

  # Phase shift in rad/s
  delta_omega0 = (2.0 * np.pi * scanner.gammabar * delta_B0).to('rad/ms')

  # SPAMM magnetization
  Nb_frames = np.floor(u_times.m_as('ms').max()/parameters.Imaging.TimeSpacing.m_as('ms')).astype(np.int32) if not FAST_MODE else 1
  Mxy_spamm = np.zeros((phantom.local_nodes.shape[0], Nb_frames, 1), dtype=np.complex64)

  # Import the Pulseq sequence and extract the k-space trajectory
  seq_path = script_path / 'pulseq/epi_pypulseq.seq'
  imp  = import_pulseq(seq_path)
  traj = kspace_trajectory(imp.pulseq_seq)

  # Diagnostic: report which blocks each SET category covers.
  for s, name in [(0, 'prepx'), (1, 'prepy'), (100, 'spoiler'), (2, 'excitation'), (3, 'readout')]:
    idx = imp.filter_blocks(SET=s)
    n = len(imp.filter_blocks(SET=s))
    MPI_print(f"  SET={s} ({name}): {n} block(s)")

  # Create sequence object
  seq = Sequence()

  # Excitation block
  tmp1 = imp.feelmri_seq.blocks[imp.filter_blocks(SET=2)[0]]
  tmp2 = imp.feelmri_seq.blocks[imp.filter_blocks(SET=2)[1]]
  ex = SequenceBlock(gradients=tmp1.S_gradients+tmp2.S_gradients, rf_pulses=tmp1.rf_pulses, store_magnetization=True)

  # Readout block
  ro_grads = []
  for i in imp.filter_blocks(SET=3):
    ro_grads += imp.feelmri_seq.blocks[i].gradients
  ro = SequenceBlock(gradients=ro_grads)

  # Spoiler block
  spoiler = imp.feelmri_seq.blocks[imp.filter_blocks(SET=100)[0]]
  spoiler._spoiler = True

  # Create dummy block to reach steady state
  dummy = ex.copy()
  dummy.store_magnetization = False

  # Add dummy blocks to the sequence to reach steady state
  time_spacing = (parameters.Imaging.TimeSpacing - ex.dur - ro.dur - spoiler.dur).to('ms')
  print("Time spacing between frames: {:.2f} ms".format(time_spacing.m_as('ms')))
  for i in range(dummy_pulses):
    seq.add_block(dummy)
    seq.add_block(ro.dur.to('ms'), dt=Q_(1, 'ms'))
    seq.add_block(spoiler.dur.to('ms'), dt=Q_(1, 'ms'))
    seq.add_block(time_spacing, dt=Q_(1, 'ms'))
    # seq.plot(figsize=(4, 6), tight_layout=True)

  # Add and additional block to synchronize the sequence with the cardiac cycle
  seq.add_block(u_times[-1] - seq.blocks[-1].time_extent[1] % u_times[-1], dt=Q_(1, 'ms'))

  # Build sequence by concatenating blocks from the imported Pulseq sequence, using the SET label to identify the block categories. The time spacing between frames is set to match the SPAMM time spacing in the original Pulseq sequence.
  # Tagging prepulses
  [seq.add_block(imp.feelmri_seq.blocks[i], dt=Q_(1e-2, 'ms')) for i in imp.filter_blocks(SET=0)]
  seq.add_block(spoiler, dt=Q_(1e-2, 'ms')) # Spoiler after prepulses
  [seq.add_block(imp.feelmri_seq.blocks[i], dt=Q_(1e-2, 'ms')) for i in imp.filter_blocks(SET=1)]
  seq.add_block(spoiler, dt=Q_(1e-2, 'ms')) # Spoiler after prepulses

  # Add imaging blocks to the sequence
  for fr in range(Nb_frames):
    seq.add_block(ex, dt=Q_(1e-2, 'ms'))
    seq.add_block(ro.dur.to('ms'), dt=Q_(1e-2, 'ms'))
    seq.add_block(spoiler, dt=Q_(1e-2, 'ms'))
    # seq.plot(blocks=slice(-4, None), figsize=(4, 6), tight_layout=True)
    seq.add_block(time_spacing, dt=Q_(1, 'ms'))  # Time spacing between frames

  # Bloch solver.
  # Note: perfect_spoiling=False is required here. The script marks
  # store_magnetization=True on the rephaser/encoder block (the last
  # block of `ex`, which is RF-free per write_epi_tagging.py:160-161),
  # so the transverse magnetization created by the preceding RF block
  # must survive across the block boundary. With perfect_spoiling=True
  # (BlochSolver's default, Bloch.py:495) the solver zeros initial_Mxy
  # on every non-empty block (Bloch.py:642-645), and the captured Mxy
  # comes out identically zero while Mz still shows a credible
  # slice-selective profile.
  solver = BlochSolver(seq, phantom,
                       scanner=scanner,
                       M0=1e+9,
                       T1=parameters.Phantom.T1,
                       T2=parameters.Phantom.T2,
                       delta_B=delta_B0.m_as('mT').reshape((-1, 1)),
                       pod_trajectory=pod_trajectory,
                       perfect_spoiling=False,
                       isochromat_K=50)

  # Solve for x and y directions
  Mxy, Mz = solver.solve()
  Mxy_spamm[:, :, 0] = Mxy

  # Create XDMF file to store the POD velocity for comparison with the original velocity field
  file = XDMFFile(script_path/'pulseq_tagging.xdmf', nodes=phantom.global_nodes, elements={phantom.cell_type: phantom.global_elements})

  # Write the POD velocity and original velocity field to the XDMF file for each frame
  for fr in range(Nb_frames):

    # Current time
    time = fr * parameters.Imaging.TimeSpacing.m_as('ms')
    pod_trajectory.update_timeshift(time)

    # Pack local results into dictionaries
    local_p_data = {
        'Mx': np.real(Mxy[:, fr]),
        'My': np.imag(Mxy[:, fr]),
        'Mz': Mz[:, fr],
        'u': pod_trajectory(0.0)
    }
    
    # Stitch to Global! (Rank 0 gets the dict, other ranks get None)
    global_p_data, _ = phantom.gather_to_global(local_point_data=local_p_data)

    # Write
    file.write(pointData=global_p_data, time=time)

  # Close the XDMF file
  file.close()

  # Trajectory and timings
  kspace_points = (traj['kx'].reshape((-1, 1, 1)).astype(np.float32),
                  traj['ky'].reshape((-1, 1, 1)).astype(np.float32),
                  traj['kz'].reshape((-1, 1, 1)).astype(np.float32))
  kspace_times = 1e-3 * traj['times'].reshape((-1, 1, 1)).astype(np.float32)

  # k-space buffer matches the (N, 1, 1, 1) shape that mri_signal returns
  # for the default as_signal_inputs layout; an additional leading axis
  # holds the cardiac frame index. EPI gridding back to (Nx, Ny, Nz) is
  # a follow-up scope (see TODO near the reconstruction block).
  N_samples = kspace_times.size
  K = np.zeros([N_samples, 1, 1, 1, Nb_frames], dtype=np.complex64)

  # T2 relaxation time
  T2 = np.ones([phantom.local_nodes.shape[0], ], dtype=np.float32)*parameters.Phantom.T2

  # Set assembler for MRI signal evaluation using FEM
  vxsz = 2*planning.FOV.m_as('m')/np.array(parameters.Imaging.RES)
  phantom.set_assembler(voxel_size=vxsz[0], lorder=1, nodal_approximation=True, lumped=True)

  # Set static fields
  phantom.set_static_fields(T2=T2.m_as('ms'), phi_dB0=delta_omega0.m_as('rad/ms'))

  # Iterate over cardiac phases
  for fr in range(Nb_frames):

    # Print progress
    MPI_print("Generating frame {:d}/{:d}".format(fr+1, Nb_frames))

    # Update reference time of POD trajectory
    pod_trajectory.update_timeshift(fr * parameters.Imaging.TimeSpacing.m_as('ms'))

    # Update magnetization
    phantom.update_magnetization(Mxy_spamm[:, fr, :])

    # Generate the signal for this cardiac phase.
    K[..., fr] = phantom.mri_signal(kspace_points, kspace_times, pod_trajectory)

  # Gather results
  K = gather_data(K)

  # ------------------------------------------------------------------
  # Reshape the flat (N, 1, 1, 1, Nb_frames) k-space buffer onto the
  # EPI grid (Nx readout x Ny phase encodes x Nz slices).
  #
  # write_epi_tagging.py orders the readout sequence as
  #   for slice in 0..Nz-1:
  #     for line  in 0..Ny-1:
  #       for sample in 0..Nx-1: <one ADC sample>
  #       (Gy blip, alternating Gx polarity)
  # so the flat arrays from kspace_trajectory reshape cleanly to
  # (Nz, Ny, Nx) and transpose to (Nx, Ny, Nz). The Gx-polarity
  # alternation is reflected in the kx values themselves; NUFFT honours
  # them directly, and CartesianRecon flips alternate lines internally
  # via the CartesianStack lines_per_shot machinery.
  # ------------------------------------------------------------------
  Nx, Ny, Nz = (int(x) for x in parameters.Imaging.RES)

  def _to_3d(flat):
    arr = np.ascontiguousarray(flat, dtype=np.float32)
    return arr.reshape(Nz, Ny, Nx).transpose(2, 1, 0)

  kx_3d = _to_3d(traj['kx'])
  ky_3d = _to_3d(traj['ky'])
  kz_3d = _to_3d(traj['kz'])

  K_grid = np.zeros((Nx, Ny, Nz, 1, Nb_frames), dtype=np.complex64)
  for fr in range(Nb_frames):
    K_grid[..., fr] = K[..., fr].reshape(Nz, Ny, Nx, 1).transpose(2, 1, 0, 3)

  fov = tuple(2*planning.FOV.m_as('m'))

  # ------------------------------------------------------------------
  # (2) Cartesian reconstruction via a CartesianStack rebuilt to match
  # the EPI fixture written by examples/write_epi_tagging.py. Setting
  # lines_per_shot=Ny declares a single-shot EPI acquisition; with
  # that, CartesianRecon.line-flip pass (Recon.py:49-53) automatically
  # reverses K along the readout axis on every odd phase-encode line
  # before the inverse FFT.
  # ------------------------------------------------------------------
  cart_traj = CartesianStack(
    FOV=2*planning.FOV.to('m'),
    res=np.array([Nx, Ny, Nz], dtype=np.int32),
    oversampling=1,
    lines_per_shot=Ny,
    scanner=scanner,
    receiver_bw=parameters.Hardware.r_BW,
    MPS_ori=planning.MPS,
    LOC=planning.LOC.m_as('m'),
  )
  I_cart = CartesianRecon(K_grid, cart_traj)

  # ------------------------------------------------------------------
  # Display magnitude images from both reconstructions side-by-side.
  # ------------------------------------------------------------------
  mag_cart  = np.abs(np.squeeze(I_cart, axis=3))
  plotter = MRIPlotter(
    images=[mag_cart],
    title=['EPI Cartesian'],
    FOV=2*planning.FOV.m_as('m'),
  )
  plotter.show()