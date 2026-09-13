import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

# This example evaluates k-space trajectories through pypulseq's
# Sequence.calculate_kspace, which is an optional dependency. Fail fast
# and cleanly here so users don't burn time on phantom setup before
# hitting the missing dependency.
try:
  import pypulseq as _pp_probe  # noqa: F401
except ImportError:
  print(
    "[pulseq_run_epi_tagging] Skipped: this example requires the optional "
    "'pypulseq' package (used by feelmri.PulseqAdapter.kspace_trajectory / "
    "kspace_to_signal_inputs). Install it with 'pip install pypulseq' (or "
    "'pip install feelmri[pulseq]') to run this example.",
    file=sys.stderr,
  )
  sys.exit(0)

import numpy as np
from pint import Quantity as Q_

from feelmri.Bloch import BlochSolver, Sequence
from feelmri.IO import XDMFFile
from feelmri.Motion import POD
from feelmri.MPIUtilities import MPI_print, MPI_rank, gather_data
from feelmri.MRImaging import PositionEncoding, SliceProfile
from feelmri.MRObjects import RF, Gradient, Scanner
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import reconstruct_nufft
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
  # Off-resonance is DISABLED here (the trailing 0.0), not set to a small
  # value -- the tagging image is read against a pure gradient encoding.
  delta_B0 = delta_B0 * scanner.field_strength * 1e-6 * 0.0

  # Phase shift in rad/s
  delta_omega0 = (2.0 * np.pi * scanner.gammabar * delta_B0).to('rad/ms')

  # SPAMM magnetization
  Nb_frames = np.floor(u_times.m_as('ms').max()/parameters.Imaging.TimeSpacing.m_as('ms')).astype(np.int32) if not FAST_MODE else 1
  Mxy_spamm = np.zeros((phantom.local_nodes.shape[0], Nb_frames, 1), dtype=np.complex64)

  # Import the Pulseq sequence and extract the k-space trajectory.
  # `readout_set_values=(3,)` instructs the adapter to collapse every
  # block carrying SET=3 (the writer convention for the EPI readout
  # train) into an empty delay of identical duration on
  # `imp.feelmri_sim_seq`. Block indices and storage flags survive the
  # substitution, so SET-based lookups stay valid on the simulation
  # sequence.
  seq_path = script_path / 'pulseq/epi_pypulseq.seq'
  imp  = import_pulseq(seq_path, readout_set_values=(3,))
  traj = kspace_trajectory(imp.pulseq_seq)
  sim  = imp.feelmri_sim_seq

  # Diagnostic: report which blocks each SET category covers.
  for s, name in [(0, 'prep x'), (1, 'prep y'), (100, 'spoiler'),
                  (2, 'excitation'), (4, 'prephaser'), (3, 'readout')]:
    n = len(imp.filter_blocks(SET=s))
    MPI_print(f"  SET={s} ({name}): {n} block(s)")

  # Block-index groups, sourced from the running LABELSET state.
  prep_x_idx  = imp.filter_blocks(SET=0)
  prep_y_idx  = imp.filter_blocks(SET=1)
  excite_idx  = imp.filter_blocks(SET=2)
  prephas_idx = imp.filter_blocks(SET=4)
  readout_idx = imp.filter_blocks(SET=3)
  spoiler_idx = imp.filter_blocks(SET=100)

  # The writer emits one excitation + one rephaser as two adjacent SET=2
  # blocks per slice, so the first contiguous run is slice 0's pair and the
  # slice count never has to be known here.
  ex_group_idx    = imp.contiguous_groups(excite_idx)[0]
  pre_group_idx   = imp.contiguous_groups(prephas_idx)[0] if prephas_idx else []
  ro_group_idx    = imp.contiguous_groups(readout_idx)[0]
  sp_template_idx = spoiler_idx[0] if spoiler_idx else None

  ex_dur  = imp.duration_of(ex_group_idx)
  pre_dur = imp.duration_of(pre_group_idx)
  ro_dur  = imp.duration_of(ro_group_idx)
  sp_dur = imp.duration_of([sp_template_idx] if sp_template_idx is not None else [])

  # Create sequence object
  seq    = Sequence()
  dt_seq = Q_(1e-2, 'ms')  # Time step for sequence blocks (10 us)

  def _spoiler_copy():
    # All SET=100 blocks come from the same gradient events, so the first is
    # a valid template for every spoiler in the sequence.
    if sp_template_idx is None:
      return None
    return imp.copy_block(sp_template_idx, spoiler=True)

  # Time spacing between frames, computed from sim-seq block durations.
  time_spacing = (parameters.Imaging.TimeSpacing
                  - ex_dur - pre_dur - ro_dur - sp_dur).to('ms')
  print("Time spacing between frames: {:.2f} ms".format(time_spacing.m_as('ms')))

  # Dummy steady-state pulses: real excitation, then duration-only
  # placeholders for readout and spoiler. This mirrors the original
  # runner's performance shortcut (no need to evolve the spoiler or
  # readout physics during steady-state convergence).
  for _ in range(dummy_pulses):
    for j in ex_group_idx + pre_group_idx:
      seq.add_block(imp.copy_block(j))
    seq.add_block(ro_dur, dt=Q_(1, 'ms'))
    seq.add_block(sp_dur, dt=dt_seq)
    seq.add_block(time_spacing, dt=Q_(1, 'ms'))

  # Sync the sequence to the cardiac-cycle boundary.
  # (the cycle is N*dt, one sampling interval longer than u_times[-1])
  cycle = Q_(pod_trajectory.period, 'ms')
  seq.add_block(cycle - seq.blocks[-1].time_extent[1] % cycle, dt=Q_(1, 'ms'))

  # Tagging preparation: one SPAMM module along x, then one along y, each
  # followed by a spoiler. Together they give a tag grid. Prep blocks come
  # straight from the simulation skeleton; they carry no readout content.
  for j in prep_x_idx:
    seq.add_block(imp.copy_block(j))
  seq.add_block(_spoiler_copy())
  for j in prep_y_idx:
    seq.add_block(imp.copy_block(j))
  seq.add_block(_spoiler_copy())

  # Imaging frames: real excitation (snapshot Mxy at the end of the
  # excitation group), then readout-as-delay (already collapsed on
  # `sim`), then real spoiler with multi-isochromat dephasing, then
  # the per-frame timing gap.
  for fr in range(Nb_frames):
    # Snapshot at the end of the SET=2 group. The writer ends that group on
    # the slice rephaser, where the gradient moment measured from the
    # excitation is zero on all three axes -- the only instant at which the
    # magnetization can be captured for the k-space integral, since
    # calculate_kspace resets k=0 at the RF and the assembler then applies the
    # whole trajectory itself. The in-plane prephasers are SET=4 and follow
    # the snapshot, so their winding reaches the signal only through k.
    ex_blks = [imp.copy_block(j) for j in ex_group_idx]
    if ex_blks:
      ex_blks[-1].store_magnetization = True
    for b in ex_blks:
      seq.add_block(b)
    for j in pre_group_idx:
      seq.add_block(imp.copy_block(j))
    seq.add_block(ro_dur, dt=Q_(1, 'ms'))
    seq.add_block(_spoiler_copy())
    seq.add_block(time_spacing, dt=Q_(1, 'ms'))

  # Bloch solver.
  # perfect_spoiling=False is passed explicitly because `seq` is rebuilt here
  # from individual blocks, so it does not carry the explicit_spoiling flag
  # import_pulseq sets on the sequences it returns. It is required either
  # way: the script marks store_magnetization=True on the rephaser/encoder
  # block, which is RF-free, so the transverse magnetization created by the
  # preceding RF block must survive the block boundary. Zeroing it leaves the
  # captured Mxy identically zero while Mz still looks credible.
  solver = BlochSolver(seq, phantom,
                       scanner=scanner,
                       M0=1e+9,
                       T1=parameters.Phantom.T1,
                       T2=parameters.Phantom.T2,
                       delta_B=delta_B0.m_as('mT').reshape((-1, 1)),
                       pod_trajectory=pod_trajectory,
                       perfect_spoiling=False,
                       isochromat_K=200,
                       method='cayley_klein')

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
  # traj['times'] is in ms, which is what mri_signal expects (T2 is passed as
  # ms and phi_dB0 as rad/ms), but it is measured from the start of the .seq
  # file. The exponentials exp(-t/T2) and exp(i*phi*t) are applied to the
  # magnetization captured at the excitation, so t has to be measured from the
  # start of the readout. Same convention as gradient_spoiling.py, which
  # subtracts traj.t_start.
  kspace_times = (traj['times'] - traj['times'].min())
  kspace_times = kspace_times.reshape((-1, 1, 1)).astype(np.float32)

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

  # Reconstruct the image
  RES = parameters.Imaging.RES
  Im = reconstruct_nufft(
    kdata=K.reshape((K.shape[0], K.shape[1], K.shape[2], -1)),  # (R, L, S, C)
    ktraj=kspace_points,
    img_shape=RES,
    fov=2*planning.FOV.m_as('m'),
    auto_dcw='pipe-menon',
    oversamp=1.25,
    kernel_size=6,
    mode='adjoint',
    combine=None,
  )
  # reconstruct_nufft drops the channel axis when there is only one, which is
  # the single-frame case. Put it back before moving it to the end.
  Im = np.asarray(Im)
  if Im.ndim == len(RES):
    Im = Im[np.newaxis, ...]
  Im = Im.transpose((1,2,3,0)).reshape((RES[0], RES[1], RES[2], 1, -1))  # (Nx, Ny, Nz, enc, C)
  print(Im.shape)

  # Show the image
  mag = np.abs(Im[...,0,:])
  MRIPlotter(
    images=[mag],
    title=['EPI NUFFT'],
    FOV=2*planning.FOV.m_as('m'),
  ).show()