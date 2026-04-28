import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.KSpaceTraj import CartesianStack
from feelmri.MPIUtilities import MPI_print, gather_data
from feelmri.MRImaging import SliceProfile
from feelmri.MRObjects import RF, Gradient, Scanner
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon

# Enable fast mode for testing if the environment variable is set
FAST_MODE = os.getenv("FEELMRI_FAST_TEST", "0") == "1"

if FAST_MODE:
    slices = 1
    dummy_pulses = 1
    resolution = [50, 40, 1]
else:
    slices = -1
    dummy_pulses = 80

if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  parameters = ParameterHandler(script_path/'parameters/gradient_spoiling.yaml')

  # Make resolution lower for CI testing
  if FAST_MODE:
    parameters.Imaging.RES = np.array(resolution)

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(script_path/parameters.Formatting.planning,
                      box_name='Box1',
                      transform_name='Transform1',
                      length_units=parameters.Formatting.units)

  vxsz = planning.FOV.m_as('mm')/parameters.Imaging.RES
  MPI_print('Voxel size: ({:.2f}, {:.2f}, {:.2f}) mm'.format(vxsz[0], vxsz[1], vxsz[2]))

  # Create FEM phantom object
  phantom = FEMPhantom(script_path/'phantoms/heart_P1_hex.xdmf', scale_factor=1.0)

  # Translate phantom to obtain the desired slice location
  phantom.orient(planning.MPS, planning.LOC.to('m'))

  # We can a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV
  mp = phantom.global_nodes[phantom.global_elements].mean(axis=1)
  markers = np.abs(mp[:, 2]) <= 0.5 * planning.FOV[2].m_as('m')
  phantom.create_submesh(markers)

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

  # Field inhomogeneity
  def spatial(x):
      return x[:,0] + x[:,1] + x[:,2]
  delta_B0 = spatial(phantom.local_nodes)
  delta_B0 /= np.abs(spatial(phantom.global_nodes).flatten()).max()
  delta_B0 *= 10.5 * 1e-6    # 1.5 ppm of the main magnetic field
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
  imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing], 
                          rf_pulses=[sp.rf], 
                          dt_rf=Q_(1e-2, 'ms'), 
                          dt_gr=Q_(1e-2, 'ms'), 
                          dt=Q_(1, 'ms'), 
                          store_magnetization=True)
  
  # Spoiler block
  nb_cycles = 10.0
  spoiler_area = nb_cycles / (scanner.gammabar.to('1/ms/mT') * planning.FOV[2].to('m'))
  spoil_gr = Gradient(axis=2)
  spoil_gr.match_area(spoiler_area)
  print('Spoiler gradient area: {:.2f} mT/m·ms'.format(spoil_gr.area().m_as('mT/m*ms')))
  spoiler = SequenceBlock(gradients=[spoil_gr], 
                          rf_pulses=[], 
                          dt_gr=Q_(1e-2, 'ms'), 
                          store_magnetization=False)
  spoiler._spoiler = True

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
  print(planning.FOV.to('m')/traj.res)
  
  # Echo time
  MPI_print('Echo time: {:.1f} ms'.format(traj.echo_time.m_as('ms')))

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

  # Bloch solver
  solver = BlochSolver(seq, phantom, 
                      scanner=scanner, 
                      M0=1e+9, 
                      T1=parameters.Phantom.T1, 
                      T2=parameters.Phantom.T2star, 
                      delta_B=delta_B0.reshape((-1, 1)),
                      pod_trajectory=None,
                      perfect_spoiling=False)

  # Convert and stripe units
  T2 = T2star.m_as('ms')

  # Set assembler for MRI signal evaluation using FEM
  vxsz = planning.FOV.m_as('m')/np.array(parameters.Imaging.RES)
  phantom.set_assembler(voxel_size=vxsz[0], lorder=1, horder=6, nodal_approximation=True, lumped=False)

  # Set static fields
  phantom.set_static_fields(T2=T2, phi_dB0=delta_omega0)

  # Fast mode for CI testing
  if FAST_MODE:
     traj.shots = [traj.shots[0]]

  # Generate k-space data for each shot and slice
  for s in range(slices):

    # Iterate over shots
    for i, sh in enumerate(traj.shots):

      # Print progress
      MPI_print("Generating shot {:d}/{:d} for slice {:d}/{:d}".format(i+1, traj.nb_shots, s+1, K.shape[2]))

      # Add imaging and delay blocks to the sequence
      seq.add_block(imaging)
      seq.add_block(2*traj.echo_time - traj.t_start)  # Delay between imaging and spoiler
      seq.add_block(spoiler)
      seq.add_block(time_spacing)  # Delay between imaging blocks
      # seq.plot()
      Mxy, Mz = solver.solve(start=-4)

      # Update magnetization
      phantom.update_magnetization(Mxy)

      # k-space points per shot
      kspace_points = (traj.points[0][:,sh,s,np.newaxis], 
                      traj.points[1][:,sh,s,np.newaxis], 
                      traj.points[2][:,sh,s,np.newaxis])
      kspace_times = traj.times.m_as('ms')[:,sh,s,np.newaxis] - traj.t_start.m_as('ms')

      # Generate 4D flow image
      tmp = phantom.mri_signal(kspace_points, kspace_times)
      K[:,sh,s,:,0] = tmp.swapaxes(0, 1)[:,:,0]

  # file.close()

  # Gather results
  K = gather_data(K)

  # Image reconstruction
  I = CartesianRecon(K, traj)

  # Show reconstruction
  mag = np.abs(I[...,0,:])
  phi = np.angle(I[...,0,:])
  plotter = MRIPlotter(images=[mag, phi], title=['Magnitude', 'Phase'], FOV=planning.FOV.m_as('m'))
  # plotter.export_images('free_running/im')
  plotter.show()

  #   plotter = MRIPlotter(images=[np.abs(K[...,0,:])], title=['k-space'], FOV=planning.FOV.m_as('m'))
  #   plotter.show()