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
from feelmri.MRImaging import PositionEncoding, SliceProfile
from feelmri.MRObjects import RF, Gradient, Scanner
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon
from feelmri.Tagging import SPAMM

if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  parameters = ParameterHandler(script_path/'parameters/spamm_fat.yaml')

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(script_path/parameters.Formatting.planning,
                          box_name='Box1',
                          transform_name='Transform1',
                          length_units=parameters.Formatting.units)

  # Create FEM phantom object
  phantom = FEMPhantom(path=script_path/'phantoms/water_and_fat.xdmf', scale_factor=0.01)

  # Translate phantom to obtain the desired slice location
  phantom.orient(planning.MPS, planning.LOC)

  # Velocity encoding parameters
  ke_dirs = list(parameters.PositionEncoding.Directions.values())
  enc = PositionEncoding(parameters.PositionEncoding.ke.m_as('1/m'), np.array(ke_dirs))

  # We can a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV
  mp = phantom.global_nodes[phantom.global_elements].mean(axis=1)
  markers = np.abs(mp[:, 2]) <= 0.5 * planning.FOV[2].m_as('m')
  phantom.create_submesh(markers)

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

  # Number of chemical species
  Nb_species = 2

  # Fat and water fraction
  phantom.read_data(0)
  tubes = (phantom.point_data['point_markers'] - 1)
  tubes /= tubes.max()  # Normalize to [0, 1]
  # tubes *= 0.2
  # tubes = (tubes > 1e-3).astype(np.float32)  # Threshold to create a binary mask
  rho_f = phantom.interpolate_to_submesh(tubes)
  rho_w = 1.0 - rho_f
  rho = [rho_f, rho_w]

  # file = XDMFFile('test_wf.xdmf', nodes=phantom.local_nodes, elements={phantom.cell_type: phantom.local_elements})
  # file.write(pointData={'tubes': tubes, 'rho_fat': rho_f, 'rho_water': rho_w}, time=0.0)
  # file.close()

  # Fat frequency offset
  df_f = parameters.Phantom.dfat * np.ones_like(rho_f)
  df_w = Q_(0.0, 'Hz') * np.ones_like(rho_w)
  df = [df_f, df_w]

  # T1 and T2 relaxation times
  T1_f = parameters.Phantom.T1_fat * np.ones_like(rho_f)
  T2_f = parameters.Phantom.T2_fat * np.ones_like(rho_f)
  T1_w = parameters.Phantom.T1_water * np.ones_like(rho_w)
  T2_w = parameters.Phantom.T2_water * np.ones_like(rho_w)
  T1 = [T1_f, T1_w]
  T2 = [T2_f, T2_w]

  # Field inhomogeneity
  def spatial(x):
      return x[:,0] + x[:,1] + x[:,2]
  delta_B0 = spatial(phantom.local_nodes)
  delta_B0 /= np.abs(spatial(phantom.global_nodes).flatten()).max()
  delta_B0 *= Q_(1.5 * 1e-6, 'T')  # Scale to a reasonable value (e.g., 1.5 ppm)

  # Slice profile
  # The slice profile prepulse is calculated based on a reference RF pulse with
  # user-defined characteristics. The slice profile object allows accessing the calculated adjusted RF pulse and dephasing and rephasing gradients
  rf = RF(scanner=scanner, 
          NbLobes=[4, 4], 
          alpha=0.46, 
          shape='apodized_sinc', 
          flip_angle=parameters.Imaging.FlipAngle , 
          t_ref=Q_(0.0,'ms'),
          phase_offset=Q_(-np.pi/2, 'rad'))
  sp = SliceProfile(delta_z=planning.FOV[2].to('m'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=False, 
    bandwidth='maximum',
    refocusing_area_frac=0.7758)
  # sp.optimize(frac_start=0.7, frac_end=0.8, N=100, profile_samples=100)

  # SPAMM magnetization
  Mxy_spamm = np.zeros((phantom.local_nodes.shape[0], enc.nb_directions, Nb_species), dtype=np.complex64)

  # Simulate the SPAMM preparation block for each encoding direction
  t0 = time.perf_counter()
  for cs in range(Nb_species):
    for i in range(len(enc.directions)):

      # Create sequence object
      seq = Sequence()

      # SPAMM preparation block
      rf1 = RF(scanner=scanner, shape='hard', dur=Q_(0.2, 'ms'), flip_angle=Q_(np.deg2rad(90),'rad'), t_ref=Q_(0.0, 'ms'))

      G_tag = Gradient(scanner=scanner, t_ref=rf1.t_ref + rf1.dur2, axis=i)
      G_tag.match_area(Q_(parameters.PositionEncoding.ke.m_as('1/m')/scanner.gamma.m_as('1/mT/ms'), 'mT*ms/m'))

      # Adjust gradient duration to control chemical shift artifact
      G_tag.change_dur(G_tag.dur)

      rf2 = RF(scanner=scanner, shape='hard', dur=Q_(0.2, 'ms'), flip_angle=Q_((-1)**i*np.deg2rad(90),'rad'), t_ref=G_tag.t_ref + G_tag.dur + rf1.t_ref + rf1.dur2)

      prep = SequenceBlock(gradients=[G_tag], 
                           rf_pulses=[rf1, rf2], 
                           dt_rf=Q_(1e-2, 'ms'), 
                           dt_gr=Q_(1e-2, 'ms'), 
                           dt=Q_(1, 'ms'), 
                           store_magnetization=False)

      # Imaging block
      imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing], 
                              rf_pulses=[sp.rf], 
                              dt_rf=Q_(1e-2, 'ms'), 
                              dt_gr=Q_(1e-2, 'ms'), 
                              dt=Q_(1, 'ms'), 
                              store_magnetization=True)

      # Add blocks to the sequence
      time_spacing = parameters.Imaging.TR - (imaging.time_extent[1] - sp.rf.t_ref)
      seq.add_block(prep)
      seq.add_block(Q_(1, 'ms'), dt=Q_(0.1, 'ms'))  # Delay after SPAMM preparation
      seq.add_block(imaging)
      seq.add_block(Q_(time_spacing, 's'), dt=Q_(1, 'ms'))  # Time spacing between TRs

      # seq.plot()

      # Define field inhomogeneity for this frame
      dB0 = delta_B0.m_as('mT') + (df[cs].m_as('1/ms')/scanner.gammabar.m_as('1/ms/mT')).reshape((-1,))

      # Bloch solver
      solver = BlochSolver(seq, phantom, 
                          scanner=scanner, 
                          # M0=1e+9, 
                          T1=T1[cs], 
                          T2=T2[cs], 
                          initial_Mz=1e+9*rho[cs],
                          delta_B=dB0.reshape((-1, 1)))

      # Solve for x and y directions
      Mxy, Mz = solver.solve() 

      # Assign the magnetization to the corresponding direction
      Mxy_spamm[:, i, cs] = Mxy.flatten()

  # Store elapsed time
  bloch_time = time.perf_counter() - t0

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
  MPI_print('Echo time: {:.2f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, enc.nb_directions, 1], dtype=np.complex64)

  # Assemble mass matrix for integrals (just once)
  M = phantom.mass_matrix(lumped=True, quadrature_order=2)

  # Iterate over cardiac phases
  t0 = time.perf_counter()
  for cs in range(Nb_species):

    # Define field inhomogeneity for this frame
    dB0 = scanner.gammabar.to('1/mT/ms') * delta_B0.to('mT') 
    dB0 += df[cs].reshape((-1,))
    delta_omega0 = 2 * np.pi * dB0.m_as('1/ms')

    # Generate 4D flow image
    MPI_print('Generating chemical specie {:d}'.format(cs))
    K[:,:,:,:,0] += SPAMM(MPI_rank, M, traj.points, traj.times.m_as('ms'), phantom.local_nodes, Mxy_spamm[:,:,cs], delta_omega0, T2[cs].m_as('ms'))

  # Store elapsed time
  spamm_time = time.perf_counter() - t0

  # Print elapsed times
  MPI_print('Elapsed time for Bloch solver: {:.2f} s'.format(bloch_time))
  MPI_print('Elapsed time-per-frame for SPAMM: {:.2f} s'.format(spamm_time/phantom.Nfr))
  MPI_print('Elapsed time for SPAMM: {:.2f} s'.format(spamm_time))
  MPI_print('Elapsed time for SPAMM + Bloch solver: {:.2f} s'.format(bloch_time + spamm_time))

  # Gather results
  K = gather_data(K)

  # # Export generated data
  # if MPI_rank==0:
  #   with open(str(export_path), 'wb') as f:
  #     pickle.dump({'kspace': K, 'MPS_ori': MPS_ori, 'LOC': LOC, 'traj': traj}, f)

  # Image reconstruction
  I = CartesianRecon(K, traj)

  # Show reconstruction
  if MPI_rank == 0:
    mx = np.abs(I[...,0,:])
    my = np.abs(I[...,1,:])
    mxy = np.abs(I[...,0,:] - I[...,1,:])
    plotter = MRIPlotter(images=[mx, my, mxy], title=['SPAMM X', 'SPAMM Y', 'O-CSPAMM'], FOV=planning.FOV.m_as('m'))
    plotter.show()