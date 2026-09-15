import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.KSpaceTraj import CartesianStack
from feelmri.MPIUtilities import MPI_print, gather_data
from feelmri.MRImaging import SliceProfile
from feelmri.MRObjects import RF, B0Field, Scanner
from feelmri.Noise import add_cpx_noise
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon

# Enable fast mode for testing if the environment variable is set
FAST_MODE = os.getenv("FEELMRI_FAST_TEST", "0") == "1"


# Field inhomogeneity
def spatial(x):
    return x[:,0] + x[:,1] + x[:,2]


if __name__ == '__main__':

  # Number of chemical species
  Nb_species = 2

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  parameters = ParameterHandler(script_path/'parameters/water_and_fat_abdomen.yaml')

  # A 150 x 110 acquisition at one line per shot is 110 solver passes per
  # species; the test runner only needs the pipeline to execute end to end.
  if FAST_MODE:
    parameters.Imaging.RES = [60, 44, 1]
    parameters.Imaging.LinesPerShot = 44

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(script_path/parameters.Formatting.planning,
                          box_name='Box1',
                          transform_name='Transform1',
                          length_units=parameters.Formatting.units)

  # Create a different phantom object per chemical specie
  phantoms = []
  rho = []
  for cs in range(Nb_species):

    # Create FEM phantom object
    phantoms.append(FEMPhantom(path=script_path/'phantoms/abdomen_P1_tetra.xdmf', scale_factor=0.001))

    # Translate phantom to obtain the desired slice location
    phantoms[cs].orient(planning.MPS, planning.LOC.to('m'))

    # We can a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV
    mp = phantoms[cs].global_nodes[phantoms[cs].global_elements, :].mean(axis=1)
    m1 = np.abs(mp[:, 2]) <= 0.5 * planning.FOV[2].m_as('m')
    print("Submesh contains {:d} elements out of {:d} total elements".format(m1.sum(), m1.shape[0]) )
    phantoms[cs].create_submesh(m1)

    # Calculate fat fraction
    if cs == 0:
      fat_fraction = phantoms[cs].point_data['fat']
      fat_fraction /= fat_fraction.max()
      fat_fraction = phantoms[cs].to_submesh(fat_fraction)
      rho.append(fat_fraction)
    else:
      water_fraction = phantoms[cs].point_data['water']
      water_fraction /= water_fraction.max()
      water_fraction = phantoms[cs].to_submesh(water_fraction)
      rho.append(water_fraction)

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

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

  # Fat and water dependent parameters
  df = []   # Frequency offset
  T1 = []   # Longitudinal relaxation time
  T2 = []   # Transverse relaxation time
  for cs in range(Nb_species):

    # Fat frequency offset
    if cs == 0:
      df.append(parameters.Phantom.dfat * np.ones(rho[cs].shape, dtype=np.float32))
    else:
      df.append(Q_(0.0, 'Hz') * np.ones(rho[cs].shape, dtype=np.float32))

    # T1 and T2 relaxation times
    if cs == 0:
      T1.append(parameters.Phantom.T1_fat * np.ones(rho[cs].shape, dtype=np.float32))
      T2.append(parameters.Phantom.T2_fat * np.ones(rho[cs].shape, dtype=np.float32))
    else:
      T1.append(parameters.Phantom.T1_water * np.ones(rho[cs].shape, dtype=np.float32))
      T2.append(parameters.Phantom.T2_water * np.ones(rho[cs].shape, dtype=np.float32))

  # Main-field inhomogeneity, written in SCANNER coordinates because it belongs
  # to the bore and not to the tissue: a spin that moves samples it wherever it
  # has moved to, which is what B0Field gives both the solver and the readout.
  # The chemical shift below stays on delta_B, frozen onto the node as a tissue
  # property is. One field for every species, normalised over the whole imaged
  # slab, not per submesh, or each species would see a different bore.
  lab_nodes = (phantoms[-1].global_nodes @ planning.MPS.T
               + planning.LOC.m_as('m'))
  peak = np.abs(spatial(lab_nodes).flatten()).max()
  b0_field = B0Field.on_phantom(
      lambda x: Q_(20.5e-6 * spatial(x) / peak, 'T'), phantoms[-1])

  # Create sequence object
  seq = Sequence()

  # Imaging block
  imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing], 
                          rf_pulses=[sp.rf], 
                          dt_rf=Q_(1e-2, 'ms'), 
                          dt_gr=Q_(1e-2, 'ms'), 
                          dt=Q_(1, 'ms'), 
                          store_magnetization=True)

  # Create a different solver for each chemical specie
  solvers  = []
  delta_B0 = []
  for cs in range(Nb_species):
    # Chemical shift only: this one really is a property of the material point.
    delta_B0.append((df[cs].to('1/ms')/scanner.gammabar.to('1/ms/mT')).reshape((-1,)))

    solvers.append(BlochSolver(seq, phantoms[cs], 
                        scanner=scanner, 
                        T1=T1[cs].reshape((-1, 1)),
                        T2=T2[cs].reshape((-1, 1)), 
                        b0_field=b0_field,
                        initial_Mz=1e+10*rho[cs].reshape((-1, 1)), delta_B=delta_B0[cs].m_as('mT').reshape((-1, 1))))

  # Generate kspace trajectory
  start = imaging.time_extent[1] - sp.rf.time
  traj = CartesianStack(FOV=planning.FOV.to('m'),
                      t_start=start,
                      res=parameters.Imaging.RES, 
                      oversampling=parameters.Imaging.Oversampling, 
                      lines_per_shot=parameters.Imaging.LinesPerShot, 
                      MPS_ori=planning.MPS,
                      LOC=planning.LOC.to('m'),
                      receiver_bw=parameters.Hardware.r_BW.to('Hz'), 
                      plot_seq=False)
  
  # Echo time
  MPI_print('Echo time: {:.2f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, 1, 1], dtype=np.complex64)

  # Field inhomogeneities. The uniform part of the scanner field is spatially
  # constant, so it rides here; the rest of it reaches the readout as the shift
  # of the sample's k computed below.
  delta_phi = [scanner.gammabar.to('1/mT/ms') * delta_B0[cs].to('mT') for cs in range(Nb_species)]
  delta_omega = [2 * np.pi * delta_phi[cs].m_as('1/ms')
                 + b0_field.uniform_phi_rate(scanner, location=traj.LOC)
                 for cs in range(Nb_species)]

  # The nominal trajectory is what the reconstruction grids on, so the shift is
  # kept separate: the difference between the two is the geometric distortion.
  b0_points = traj.b0_shifted_points(b0_field, scanner)

  # Set assembler for MRI signal evaluation using FEM
  vxsz = planning.FOV.m_as('m')/np.array(parameters.Imaging.RES)
  [phantoms[cs].set_assembler(voxel_size=vxsz[0], horder=6, lumped=True, nodal_approximation=True) for cs in range(Nb_species)]

  # Set static fields
  [phantoms[cs].set_static_fields(T2=T2[cs].m_as('ms'), phi_dB0=delta_omega[cs]) for cs in range(Nb_species)]

  # Time spacing needed to achieve the desired TR
  time_spacing = parameters.Imaging.TR - (imaging.time_extent[1] - sp.rf.ref)

  # Generate k-space data for chemical specie, shot and slice
  for s in range(slices):
    for i, sh in enumerate(traj.shots):     

      # k-space points per shot
      kspace_points = (b0_points[0][:,sh,s,np.newaxis], 
                      b0_points[1][:,sh,s,np.newaxis], 
                      b0_points[2][:,sh,s,np.newaxis])
      kspace_times = (traj.times.m_as('ms')[:,sh,s,np.newaxis] - traj.t_start.m_as('ms'))

      # Add imaging and delay blocks to the sequence
      seq.add_block(imaging)
      seq.add_block(time_spacing, dt=Q_(1, 'ms'))  # Delay between imaging blocks

      for cs in range(Nb_species):

        MPI_print("Generating shot {:d}/{:d} for slice {:d}/{:d}".format(i+1, traj.nb_shots, s+1, K.shape[2]))

        # Solve new blocks
        Mxy, Mz = solvers[cs].solve(start=-2)
        phantoms[cs].update_magnetization(Mxy)

        # Generate 4D flow image
        tmp = phantoms[cs].mri_signal(kspace_points, kspace_times)
        K[:,sh,s,:,0] += tmp.swapaxes(0, 1)[:,:,0]

  # Gather results
  K = gather_data(K)
  K = add_cpx_noise(K, relative_std=1e-5)

  # Image reconstruction
  I = CartesianRecon(K, traj)

  # Show reconstruction
  mag = np.abs(I[...,0,:])
  phi = np.angle(I[...,0,:])
  plotter = MRIPlotter(images=[mag, phi], title=['Magnitude', 'Phase'], FOV=planning.FOV.m_as('m'))
  plotter.show()