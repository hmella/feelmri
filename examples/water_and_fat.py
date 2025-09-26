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
from feelmri.MPIUtilities import MPI_print, MPI_rank, gather_data
from feelmri.MRI import Signal
from feelmri.MRImaging import SliceProfile
from feelmri.MRObjects import RF, Scanner
from feelmri.Noise import add_cpx_noise
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon


# Field inhomogeneity
def spatial(x):
    return x[:,0] + x[:,1] + x[:,2]

if __name__ == '__main__':

  # Number of chemical species
  Nb_species = 2

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  parameters = ParameterHandler(script_path/'parameters/water_and_fat.yaml')

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
    phantoms.append(FEMPhantom(path=script_path/'phantoms/water_and_fat.xdmf', scale_factor=0.01))
    phantoms[cs].read_data(0)

    # Translate phantom to obtain the desired slice location
    phantoms[cs].orient(planning.MPS, planning.LOC)

    # We can a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV
    mp = phantoms[cs].global_nodes[phantoms[cs].global_elements, :].mean(axis=1)
    m1 = (np.abs(mp[:, 2]) <= 0.5 * planning.FOV[2].m_as('m'))
    if cs == 0:
      m2 = phantoms[cs].cell_data['cell_markers'][0] >= 2 - 1e-6  # Fat
      phantoms[cs].create_submesh(np.logical_and(m1, m2))
    else:
      phantoms[cs].create_submesh(m1)

    # Calculate fat fraction
    fat_fraction = phantoms[cs].point_data['point_markers'] - 1.0
    fat_fraction /= fat_fraction.max()
    fat_fraction = phantoms[cs].to_submesh(fat_fraction)
    if cs == 0:
      # fat_fraction *= np.abs(fat_fraction - 1.0) <= 1e-6  # Fat is marked with a 9 in the point data
      rho.append(fat_fraction)
    else:
      # fat_fraction *= np.abs(fat_fraction - 1.0) <= 1e-6  # Fat is marked with a 9 in the point data
      rho.append(1.0 - fat_fraction)


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
  inhomogeneity = []  # Field inhomogeneity
  for cs in range(Nb_species):

    # Fat frequency offset
    if cs == 0:
      df.append(parameters.Phantom.dfat * np.ones(rho[cs].shape, dtype=np.float32))
    else:
      df.append(Q_(0.0, 'Hz') * np.ones(rho[cs].shape, dtype=np.float32))

    # file = XDMFFile('test_wf_{:d}_{:d}.xdmf'.format(cs, MPI_rank), nodes=phantoms[cs].local_nodes, elements={phantoms[cs].cell_type: phantoms[cs].local_elements})
    # file.write(pointData={'rho': rho[cs], 'df': df[cs]}, time=0.0)
    # file.close()

    # T1 and T2 relaxation times
    if cs == 0:
      T1.append(parameters.Phantom.T1_fat * np.ones(rho[cs].shape, dtype=np.float32))
      T2.append(parameters.Phantom.T2_fat * np.ones(rho[cs].shape, dtype=np.float32))
    else:
      T1.append(parameters.Phantom.T1_water * np.ones(rho[cs].shape, dtype=np.float32))
      T2.append(parameters.Phantom.T2_water * np.ones(rho[cs].shape, dtype=np.float32))

    # Field inhomogeneity (e.g., due to susceptibility effects)
    inhomogeneity.append(spatial(phantoms[cs].local_nodes))
    inhomogeneity[cs] /= np.abs(spatial(phantoms[cs].global_nodes).flatten()).max()
    inhomogeneity[cs] *= Q_(0.0 * 1.5 * 1e-6, 'T')  # Scale to a reasonable value (e.g., 1.5 ppm)

  # Create sequence object
  seq = Sequence()

  # Imaging block
  imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing], 
                          rf_pulses=[sp.rf], 
                          dt_rf=Q_(1e-2, 'ms'), 
                          dt_gr=Q_(1e-2, 'ms'), 
                          dt=Q_(1, 'ms'), 
                          store_magnetization=True)

  # Add dummy blocks to achieve the steady state
  dummy = imaging.copy()
  dummy.store_magnetization = False
  time_spacing = parameters.Imaging.TR - (imaging.time_extent[1] - sp.rf.ref)
  # for _ in range(80):
  #   seq.add_block(dummy)
  #   seq.add_block(time_spacing, dt=Q_(1, 'ms'))  # Time spacing between TRs

  # Create a different solver for each chemical specie
  solvers  = []
  delta_B0 = []
  for cs in range(Nb_species):
    # Define field inhomogeneity for this chemical specie (including chemical shift)
    delta_B0.append(inhomogeneity[cs].to('mT') + (df[cs].to('1/ms')/scanner.gammabar.to('1/ms/mT')).reshape((-1,)))

    solvers.append(BlochSolver(seq, phantoms[cs], 
                        scanner=scanner, 
                        T1=T1[cs].reshape((-1, 1)),
                        T2=T2[cs].reshape((-1, 1)), 
                        initial_Mz=1e+10*rho[cs].reshape((-1, 1)), delta_B=delta_B0[cs].m_as('mT').reshape((-1, 1))))

    # Solve dummy blocks
    solvers[cs].solve()

  # Generate kspace trajectory
  print(type(imaging.time_extent[1].m_as('ms')), type(sp.rf.time.m_as('ms')))
  start = imaging.time_extent[1] - sp.rf.time
  print(start)
  # imaging.plot()
  traj = CartesianStack(FOV=planning.FOV.to('m'),
                      t_start=start,
                      res=parameters.Imaging.RES, 
                      oversampling=parameters.Imaging.Oversampling, 
                      lines_per_shot=parameters.Imaging.LinesPerShot, 
                      MPS_ori=planning.MPS,
                      LOC=planning.LOC.to('m'),
                      receiver_bw=parameters.Hardware.r_BW.to('Hz'), 
                      plot_seq=False)
  # traj.plot_trajectory()
  MPI_print('Echo time: {:.2f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, 1, 1], dtype=np.complex64)

  # # Create XDMF file for debugging
  # file = XDMFFile(script_path/'magnetization_{:d}.xdmf'.format(MPI_rank), nodes=phantoms[0].local_nodes, elements={phantoms[0].cell_type: phantoms[0].local_elements})

  # Assemble mass matrices for integrals (just once)
  M = [phantoms[cs].mass_matrix(lumped=True, quadrature_order=2) for cs in range(Nb_species)]

  # Generate k-space data for chemical specie, shot and slice
  t0 = time.perf_counter()  
  for s in range(slices):
    for i, sh in enumerate(traj.shots):     

      # k-space points per shot
      kspace_points = (traj.points[0][:,sh,s,np.newaxis], 
                      traj.points[1][:,sh,s,np.newaxis], 
                      traj.points[2][:,sh,s,np.newaxis])
      kspace_times = (traj.times.m_as('ms')[:,sh,s,np.newaxis] - traj.t_start.m_as('ms'))
      print(kspace_times.dtype, start.dtype)

      # Add imaging and delay blocks to the sequence
      seq.add_block(imaging)
      seq.add_block(time_spacing, dt=Q_(1, 'ms'))  # Delay between imaging blocks

      for cs in range(Nb_species):

        MPI_print("Generating shot {:d}/{:d} for slice {:d}/{:d}".format(i+1, traj.nb_shots, s+1, K.shape[2]))

        # Solve new blocks
        Mxy, Mz = solvers[cs].solve(start=-2)
        # if cs == 0:
        #   # Export magnetization and displacement for debugging
        #   file.write(pointData={'Mx': np.real(Mxy), 
        #                         'My': np.imag(Mxy),
        #                         'Mz': Mz},
        #                         time=i*parameters.Imaging.TR)

        # Define field inhomogeneity for this frame
        delta_phi = scanner.gammabar.to('1/mT/ms') * delta_B0[cs].to('mT')
        delta_omega = 2 * np.pi * delta_phi.m_as('1/ms')

        # Generate 4D flow image
        tmp = Signal(MPI_rank, M[cs], kspace_points, kspace_times, phantoms[cs].local_nodes, delta_omega, T2[cs].m_as('ms'), Mxy, None)
        K[:,sh,s,:,0] += tmp.swapaxes(0, 1)[:,:,0]

  # Store elapsed time
  wf_time = time.perf_counter() - t0

  # # Print elapsed times
  # MPI_print('Elapsed time for Bloch solver: {:.2f} s'.format(bloch_time))
  # MPI_print('Elapsed for k-space generation: {:.2f} s'.format(wf_time))
  # MPI_print('Elapsed time for k-space + Bloch solver: {:.2f} s'.format(bloch_time + wf_time))

  # # Close the file
  # file.close()

  # Gather results
  K = gather_data(K)
  K = add_cpx_noise(K, relative_std=1e-5)

  # Image reconstruction
  I = CartesianRecon(K, traj)

  # Show reconstruction
  if MPI_rank == 0:
    mag = np.abs(I[...,0,:])
    phi = np.angle(I[...,0,:])
    plotter = MRIPlotter(images=[mag, phi], title=['Magnitude', 'Phase'], FOV=planning.FOV.m_as('m'))
    plotter.show()

    # plotter = MRIPlotter(images=[np.abs(K[...,0,:])], title=['k-space'], FOV=planning.FOV.m_as('m'))
    # plotter.show()