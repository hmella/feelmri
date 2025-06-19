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
from feelmri.MRImaging import SliceProfile
from feelmri.MRObjects import RF, Scanner
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon
from feelmri.MRIEncoding import WaterFat

if __name__ == '__main__':

  # Import imaging parameters
  parameters = ParameterHandler('parameters/water_and_fat.yaml')

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(parameters.Formatting.planning,
                          box_name='Box1',
                          transform_name='Transform1',
                          length_units=parameters.Formatting.units)

  # Create FEM phantom object
  phantom = FEMPhantom(path='phantoms/water_and_fat.xdmf', scale_factor=0.01)

  # Translate phantom to obtain the desired slice location
  phantom.orient(planning.MPS, planning.LOC)

  # We can a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV
  mp = phantom.global_nodes[phantom.global_elements].mean(axis=1)
  markers = np.where(np.abs(mp[:, 2]) <= 0.5 * planning.FOV[2].m_as('m'))[0]
  phantom.create_submesh(markers)

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

  # Fat frequency offset
  phantom.read_data(0)
  tubes = phantom.point_data['point_markers'] - 1
  fraction = phantom.interpolate_to_submesh(tubes/tubes.max())
  df = parameters.Phantom.dfat * fraction

  # T1 and T2 relaxation times
  T1 = np.ones(fraction.shape, dtype=np.float32) * parameters.Phantom.T1_water
  T2 = np.ones(fraction.shape, dtype=np.float32) * parameters.Phantom.T2_water
  T1 = (1 - fraction) * T1 + fraction * parameters.Phantom.T1_fat
  T2 = (1 - fraction) * T2 + fraction * parameters.Phantom.T2_fat

  # Field inhomogeneity
  def spatial(x):
      return x[:,0] + x[:,1] + x[:,2]
  delta_B0 = spatial(phantom.local_nodes)
  delta_B0 /= np.abs(spatial(phantom.global_nodes).flatten()).max()
  delta_B0 *= 1.5 * 1e-2
  delta_B0 += df.m_as('1/ms').reshape((-1,))
  delta_omega0 = 2.0 * np.pi * delta_B0

  # Slice profile
  # The slice profile prepulse is calculated based on a reference RF pulse with
  # user-defined characteristics. The slice profile object allows accessing the calculated adjusted RF pulse and dephasing and rephasing gradients
  rf = RF(scanner=scanner, 
          NbLobes=[4, 4], 
          alpha=0.46, shape='apodized_sinc', 
          flip_angle=parameters.Imaging.FlipAngle.to('rad'), 
          t_ref=Q_(0.0,'ms'),
          phase_offset=Q_(-np.pi/2,'rad'))
  sp = SliceProfile(delta_z=planning.FOV[2].to('m'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=False, 
    bandwidth=Q_(10000, 'Hz'), #'maximum',
    refocusing_area_frac=0.7424)
  # sp.optimize(frac_start=0.5, frac_end=1.5, N=100, profile_samples=100)

  # Simulate the SPAMM preparation block for each encoding direction
  t0 = time.time()

  # Imaging and dummy blocks
  imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing], 
                          rf_pulses=[sp.rf], 
                          dt_rf=Q_(1e-2, 'ms'), 
                          dt_gr=Q_(1e-2, 'ms'), 
                          dt=Q_(1, 'ms'), 
                          store_magnetization=True)
  dummy = imaging.copy()
  dummy.store_magnetization = False


  # Create and fill sequence object with dummy blocks
  seq = Sequence()
  time_spacing = parameters.Imaging.TR.to('ms') - (imaging.time_extent[1] - sp.rf.t_ref)
  for i in range(80):
    seq.add_block(dummy)         # Add dummy blocks to reach the steady state
    seq.add_block(time_spacing)  # Delay between imaging blocks

  # Bloch solver
  t0 = time.time()  # Start timer
  solver = BlochSolver(seq, phantom, 
                      scanner=scanner, 
                      M0=1e+9, 
                      T1=T1, 
                      T2=T2,
                      delta_B=delta_B0.reshape((-1, 1)))

  # Solve dummy blocks to reach the steady state
  solver.solve()
  bloch_time = time.time() - t0

  # Generate kspace trajectory
  traj = CartesianStack(FOV=planning.FOV.to('m'),
                      t_start=imaging.time_extent[1] - sp.rf.t_ref,
                      res=parameters.Imaging.RES, 
                      oversampling=parameters.Imaging.Oversampling, 
                      lines_per_shot=parameters.Imaging.LinesPerShot, 
                      MPS_ori=planning.MPS,
                      LOC=planning.LOC.m,
                      receiver_bw=parameters.Hardware.r_BW.to('Hz'), 
                      plot_seq=False)
  # traj.plot_trajectory()
  MPI_print('Echo time: {:.1f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, 1, 1], dtype=np.complex64)

  # Create XDMF file for debugging
  file = XDMFFile('magnetization_{:d}.xdmf'.format(MPI_rank), nodes=phantom.local_nodes, elements={phantom.cell_type: phantom.local_elements})

  # Assemble mass matrix for integrals (just once)
  M = phantom.mass_matrix(lumped=True, quadrature_order=2)

  # Generate k-space data for each shot and slice
  t0 = time.time()  
  for s in range(slices):

    # Iterate over shots
    for i, sh in enumerate(traj.shots):

      MPI_print("Generating shot {:d}/{:d} for slice {:d}/{:d}".format(i+1, traj.nb_shots, s+1, K.shape[2]))

      # Add imaging and delay blocks to the sequence
      seq.add_block(imaging)
      seq.add_block(time_spacing)  # Delay between imaging blocks
      Mxy, Mz = solver.solve(start=-2)

      # k-space points per shot
      kspace_points = (traj.points[0][:,sh,s,np.newaxis], 
                      traj.points[1][:,sh,s,np.newaxis], 
                      traj.points[2][:,sh,s,np.newaxis])
      kspace_times = traj.times.m_as('ms')[:,sh,s,np.newaxis]

      # Generate 4D flow image
      tmp = WaterFat(MPI_rank, M, kspace_points, kspace_times, phantom.local_nodes, delta_omega0, T2.m_as('ms'), Mxy)
      K[:,sh,s,:,0] = tmp.swapaxes(0, 1)[:,:,0]

      # Export magnetization and displacement for debugging
      file.write(pointData={'Mx': np.real(Mxy), 
                            'My': np.imag(Mxy),
                            'Mz': Mz},
                            time=i*parameters.Imaging.TR)

  # Store elapsed time
  wf_time = time.time() - t0

  # Print elapsed times
  MPI_print('Elapsed time for Bloch solver: {:.2f} s'.format(bloch_time))
  MPI_print('Elapsed for k-space generation: {:.2f} s'.format(wf_time))
  MPI_print('Elapsed time for k-space + Bloch solver: {:.2f} s'.format(bloch_time + wf_time))

  # Close the file
  file.close()

  # Gather results
  K = gather_data(K)

  # Image reconstruction
  I = CartesianRecon(K, traj)

  # Show reconstruction
  if MPI_rank == 0:
    mag = np.abs(I[...,0,:])
    phi = np.angle(I[...,0,:])
    plotter = MRIPlotter(images=[mag, phi], title=['Magnitude', 'Phase'], FOV=planning.FOV.m_as('m'))
    plotter.show()

    plotter = MRIPlotter(images=[np.abs(K[...,0,:])], title=['k-space'], FOV=planning.FOV.m_as('m'))
    plotter.show()