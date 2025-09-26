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
from feelmri.Motion import POD, RespiratoryMotion
from feelmri.MPIUtilities import MPI_print, MPI_rank, gather_data
from feelmri.MRImaging import PositionEncoding, SliceProfile
from feelmri.MRObjects import RF, Gradient, Scanner
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon
from feelmri.Tagging import SPAMM

def f(tn, n, T1 ,M0=1e+9, alpha=np.deg2rad(15)):
  f1 = M0*np.sin(alpha)*(np.cos(alpha)**n)*(1.0 - np.exp(-tn/T1))
  f2 = M0*np.sin(alpha)*(np.cos(alpha)**n)*np.exp(-tn/T1)
  return f1, f2


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
  phantom = FEMPhantom(path=script_path/'phantoms/beating_heart.xdmf', scale_factor=1.0)

  # Translate phantom to obtain the desired slice location
  phantom.orient(planning.MPS, planning.LOC)

  # Velocity encoding parameters
  ke_dirs = list(parameters.PositionEncoding.Directions.values())
  enc = PositionEncoding(parameters.PositionEncoding.ke.m_as('1/m'), np.array(ke_dirs))

  # We can a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV
  mp = phantom.global_nodes[phantom.global_elements].mean(axis=1)
  markers = np.where(np.abs(mp[:, 2]) <= 4.0*planning.FOV[2].m_as('m'))[0]
  phantom.create_submesh(markers)

  # Create array to store displacements
  u = np.zeros([phantom.global_shape[0], 3, phantom.Nfr], dtype=np.float32)
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr and interpolate to the submesh
    phantom.read_data(fr)
    u[..., fr] = phantom.interpolate_to_submesh(phantom.point_data['displacement'] @ planning.MPS, local=False)
    # u[..., fr] = phantom.point_data['displacement'] @ planning.MPS

  # Create POD for tissue displacements
  dt = parameters.Imaging.TimeSpacing.to('ms')
  pod_times = np.linspace(0, (phantom.Nfr-1)*dt, phantom.Nfr)
  pod_trajectory = POD(time_array=pod_times.m_as('ms'),
                      data=u,
                      global_to_local=phantom.global_to_local_nodes,
                      n_modes=5,
                      is_periodic=True)
  
  # Define respiratory motion object
  T = Q_(3000.0, 'ms')      # period
  A = Q_(0.008, 'm')        # amplitude
  times  = np.linspace(0, T, 100)
  motion = A*(1 - np.cos(2*np.pi*times/(2*T))**4)
  pod_resp_motion = RespiratoryMotion(time_array=times.m_as('ms'), 
                              data=motion.m_as('m'),
                              is_periodic=True,
                              direction=np.array([0, 1, 0]),
                              remove_mean=True)

  # Combine the POD trajectory and the respiratory motion
  pod_sum = pod_trajectory + pod_resp_motion

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

  # Field inhomogeneity
  def spatial(x):
      return x[:,0] + x[:,1] + x[:,2]
  delta_B0 = spatial(phantom.local_nodes)
  delta_B0 /= np.abs(spatial(phantom.global_nodes).flatten()).max()
  delta_B0 *= 15.0 * 1e-6 # 1.5 ppm of the main magnetic field
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
    bandwidth='maximum',
    refocusing_area_frac=0.7758)
  # sp.optimize(frac_start=0.7, frac_end=0.8, N=100, profile_samples=100)

  # Create dummy imaging block for timings
  imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing], rf_pulses=[sp.rf], dt_rf=Q_(1e-2, 'ms'), dt_gr=Q_(1e-2, 'ms'), dt=Q_(1, 'ms'))

  # Generate kspace trajectory
  traj = CartesianStack(FOV=planning.FOV.to('m'),
                      t_start=imaging.time_extent[1] - sp.rf.ref,
                      res=parameters.Imaging.RES, 
                      oversampling=parameters.Imaging.Oversampling, 
                      lines_per_shot=parameters.Imaging.LinesPerShot, 
                      MPS_ori=planning.MPS,
                      LOC=planning.LOC.m,
                      receiver_bw=parameters.Hardware.r_BW.to('Hz'), 
                      plot_seq=False)
  # traj.plot_trajectory()
  print('Echo time: {:.2f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, enc.nb_directions, phantom.Nfr], dtype=np.complex64)

  # T2star relaxation time
  T2star = np.ones([phantom.local_nodes.shape[0], ], dtype=np.float32)*parameters.Phantom.T2star

  # Assemble mass matrix for integrals (just once)
  M = phantom.mass_matrix(lumped=True, quadrature_order=2)

  # SPAMM grid
  grid = np.stack([np.sin(enc.ke[i]*phantom.local_nodes[:, i]) for i in range(2)], axis=1)

  # Iterate over cardiac phases
  t0 = time.perf_counter()  
  for fr in range(phantom.Nfr):

    # Current time
    tn = fr * parameters.Imaging.TimeSpacing.m_as('ms')

    # Update reference time of POD trajectory
    pod_sum.update_timeshift(tn)

    # Displacement
    u = pod_sum(0)

    # Current position
    current_pos = phantom.local_nodes + u

    # SPAMM magnetization
    f1, f2 = f(tn, fr, parameters.Phantom.T1.m_as('ms'), alpha=parameters.Imaging.FlipAngle.m_as('rad'))
    Mxy = (f1 + f2*grid) * sp.interp_profile(current_pos[:, 2])[:, np.newaxis]

    # Generate 4D flow image
    MPI_print('Generating frame {:d}'.format(fr))
    # K[:,:,:,:,fr] = SPAMM(MPI_rank, M, traj.points, traj.times.m_as('s'), phantom.local_nodes + submesh_displacement, M_spamm, delta_omega0, T2star, profile)
    K[:,:,:,:,fr] = SPAMM(MPI_rank, M, traj.points, traj.times.m_as('ms'), current_pos, Mxy, delta_omega0, T2star.m_as('ms'))

  # Store elapsed time
  spamm_time = time.perf_counter() - t0

  # Print elapsed times
  MPI_print('Elapsed time-per-frame for SPAMM: {:.2f} s'.format(spamm_time/phantom.Nfr))
  MPI_print('Elapsed time for SPAMM: {:.2f} s'.format(spamm_time))

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

    # mx = np.abs(K[...,0,:])
    # my = np.abs(K[...,1,:])
    # mxy = np.abs(K[...,0,:] - K[...,1,:])
    # plotter = MRIPlotter(images=[mx, my, mxy], title=['SPAMM X', 'SPAMM Y', 'O-CSPAMM'], FOV=parameters.FOV)
    # plotter.export_images('animation_K/')
    # plotter.show()