import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
import pickle
import time
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from FEelMRI.Bloch import BlochSolver, Sequence, SequenceBlock
from FEelMRI.KSpaceTraj import CartesianStack
from FEelMRI.Math import Rx, Ry, Rz
from FEelMRI.Motion import PODTrajectory
from FEelMRI.MPIUtilities import MPI_print, MPI_rank, gather_data
from FEelMRI.MRImaging import SliceProfile, VelocityEncoding
from FEelMRI.MRObjects import RF, Gradient, Scanner
from FEelMRI.Noise import add_cpx_noise
from FEelMRI.Parameters import ParameterHandler2
from FEelMRI.Phantom import FEMPhantom
from FEelMRI.PhaseContrast import PC
from FEelMRI.Plotter import MRIPlotter
from FEelMRI.Recon import CartesianRecon

if __name__ == '__main__':

  # Import imaging parameters
  parameters = ParameterHandler2('parameters/phase_contrast.yaml')

  # Create FEM phantom object
  phantom = FEMPhantom(path='phantoms/aorta_CFD.xdmf', velocity_label='velocity', scale_factor=0.01)

  # Imaging orientation paramters
  theta_x = np.deg2rad(parameters.Formatting.theta_x)
  theta_y = np.deg2rad(parameters.Formatting.theta_y)
  theta_z = np.deg2rad(parameters.Formatting.theta_z)
  MPS_ori = Rz(theta_z)@Rx(theta_x)@Ry(theta_y)
  LOC = parameters.Formatting.LOC

  # Translate phantom to obtain the desired slice location
  phantom.orient(MPS_ori, LOC)

  # Velocity encoding parameters
  venc_dirs = list(parameters.VelocityEncoding.Directions.values())
  enc = VelocityEncoding(parameters.VelocityEncoding.VENC, np.array(venc_dirs))

  # We can use only a submesh to speed up the simulation. The submesh is created by selecting the elements that are inside the FOV. We can also refine the submesh to increase the number of nodes and elements  
  midpoints = phantom.global_nodes[phantom.global_elements].mean(axis=1)
  markers = np.where(np.abs(midpoints[:, 2]) <= 0.6 * parameters.Imaging.FOV[2].m_as('m'))[0]
  phantom.create_submesh(markers, refine=False, element_size=0.0012)

  # Create POD trajectory object for velocity
  shape = [phantom.global_nodes.shape[0], phantom.global_nodes.shape[1], phantom.Nfr]
  trajectory = Q_(np.zeros(shape, dtype=np.float32), 'm/s')
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr
    phantom.read_data(fr)
    velocity = phantom.point_data['velocity'] @ MPS_ori
    submesh_velocity = phantom.interpolate_to_submesh(velocity, local=False)
    trajectory[..., fr] = Q_(submesh_velocity, 'm/s')

  # Define POD object
  dt = parameters.Imaging.TimeSpacing
  times = np.linspace(0, (phantom.Nfr-1)*dt, phantom.Nfr)
  pod_trajectory = PODTrajectory(time_array=times.m_as('ms'),
                                 data=trajectory.m_as('m/ms'),
                                 global_to_local=phantom.global_to_local_nodes,
                                 n_modes=5,
                                 taylor_order=10,
                                 is_periodic=True)

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

  # Field inhomogeneity
  def spatial(x):
      return x[:,0] + x[:,1] + x[:,2]
  delta_B0 = spatial(phantom.local_nodes)
  delta_B0 /= np.abs(spatial(phantom.global_nodes).flatten()).max()
  delta_B0 *= 1.5 * 1e-6 * 0.0           # 1.5 ppm of the main magnetic field
  delta_omega0 = 2.0 * np.pi * scanner.gammabar.m_as('Hz/T') * delta_B0

  # Slice profile
  # The slice profile prepulse is calculated based on a reference RF pulse with
  # user-defined characteristics. The slice profile object allows accessing the calculated adjusted RF pulse and dephasing and rephasing gradients
  rf = RF(scanner=scanner, 
          NbLobes=[4, 4], 
          alpha=0.46, 
          shape='apodized_sinc', 
          flip_angle=Q_(np.deg2rad(8),'rad'), 
          t_ref=Q_(0.0,'ms'))
  sp = SliceProfile(delta_z=parameters.Imaging.FOV[2].to('m'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=False, 
    bandwidth=Q_(10000, 'Hz'), #'maximum',
    refocusing_area_frac=0.7961)
  # sp.optimize(frac_start=0.79, frac_end=0.81, N=100)

  # Create sequence object and solve magnetization
  Mxy_PC = np.zeros([phantom.local_nodes.shape[0], phantom.Nfr, enc.nb_directions], dtype=np.complex64)
  for d in range(enc.nb_directions):
    # Create sequence object
    seq = Sequence()

    # Bipolar gradient
    t_ref = sp.rephasing.t_ref + sp.rephasing.dur
    bp1 = Gradient(scanner=scanner, t_ref=t_ref, axis=2)
    bp2 = bp1.make_bipolar(parameters.VelocityEncoding.VENC.to('m/s'))
    bp1 *= d + (-1)**d
    bp2 *= d + (-1)**d

    # Imaging block
    imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing, bp1, bp2],
                            rf_pulses=[sp.rf], 
                            dt_rf=Q_(1e-2, 'ms'), 
                            dt_gr=Q_(1e-2, 'ms'), 
                            dt=Q_(1, 'ms'), 
                            store_magnetization=True)
    dummy = imaging.copy()
    dummy.store_magnetization = False

    # Add blocks to the sequence
    time_spacing = parameters.Imaging.TimeSpacing.to('ms') - (imaging.time_extent[1] - sp.rf.t_ref)
    for i in range(80):
      seq.add_block(dummy.copy())  # Add dummy blocks to reach the steady state
      seq.add_block(time_spacing, dt=Q_(1, 'ms'))  # Delay between imaging blocks
    for fr in range(phantom.Nfr):
      seq.add_block(imaging)
      seq.add_block(time_spacing, dt=Q_(1, 'ms'))  # Time spacing between frames

    # Bloch solver
    solver = BlochSolver(seq, phantom, 
                         scanner=scanner, 
                         M0=1e+9, 
                         T1=parameters.Phantom.T1.to('ms'),
                         T2=parameters.Phantom.T2star.to('ms'), 
                         delta_B=delta_B0.reshape((-1, 1)),
                         pod_trajectory=pod_trajectory)

    # Solve for x and y directions
    Mxy, Mz = solver.solve()
    Mxy_PC[..., d] = Mxy

  # Path to export the generated data
  export_path = Path('MRImages/{:s}_V{:.0f}.pkl'.format(parameters.Imaging.Sequence, parameters.VelocityEncoding.VENC.m_as('cm/s')))

  # Make sure the directory exist
  os.makedirs(str(export_path.parent), exist_ok=True)

  # Generate kspace trajectory
  traj = CartesianStack(FOV = parameters.Imaging.FOV.to('m'),
    t_start = imaging.time_extent[1] - sp.rf.t_ref,
    res = parameters.Imaging.RES, 
    oversampling = parameters.Imaging.Oversampling, 
    lines_per_shot = parameters.Imaging.LinesPerShot, 
    MPS_ori = MPS_ori, 
    LOC = LOC, 
    receiver_bw=parameters.Hardware.r_BW, 
    plot_seq=False)

  # Print echo time
  MPI_print('Echo time = {:.2f} ms'.format(traj.echo_time.m_as('ms')))

  # kspace array
  ro_samples = traj.ro_samples
  ph_samples = traj.ph_samples
  slices = traj.slices
  K = np.zeros([ro_samples, ph_samples, slices, enc.nb_directions, phantom.Nfr], dtype=np.complex64)

  T2star = (parameters.Phantom.T2star * np.ones([phantom.local_nodes.shape[0]])).astype(np.float32)

  # Assemble mass matrix for integrals (just once to accelerate the simulation)
  M = phantom.mass_matrix(lumped=True)

  # Iterate over cardiac phases
  t0 = time.time()
  for fr in range(phantom.Nfr):
      
      t0_fr = time.time()

      # Generate 4D flow image
      K[:,:,:,:,fr] = PC(MPI_rank, M, traj.points, traj.times.m_as('ms'), phantom.local_nodes, delta_omega0, T2star.m_as('ms'), Mxy_PC[:, fr, :])

      sys.stdout.flush()
      sys.stdout.write("\r" + 'Elapsed time frame {:d}: {:.2f} s'.format(fr, time.time()-t0_fr))

  # Store elapsed time
  pc_time = time.time() - t0

  # Print elapsed times
  MPI_print('Elapsed time-per-frame for 4D Flow generation: {:.2f} s'.format(pc_time/phantom.Nfr))
  MPI_print('Elapsed time for 4D Flow generation: {:.2f} s'.format(pc_time))

  # Gather results
  K = gather_data(K)

  # Add noise to kspace
  K = add_cpx_noise(K, relative_std=0.01)

  # Export generated data
  if MPI_rank==0:
    with open(str(export_path), 'wb') as f:
      pickle.dump({'kspace': K, 'MPS_ori': MPS_ori, 'LOC': LOC, 'traj': traj}, f)

  # Image reconstruction
  I = CartesianRecon(K, traj)

  # Show reconstruction
  m = np.abs(I[...,0,:])
  phi = np.angle(I[...,0,:])
  phi_ref = np.angle(I[...,1,:])
  phi_v = np.angle(I[...,0,:] * np.conj(I[...,1,:]))
  if MPI_rank == 0:
    plotter = MRIPlotter(images=[m, phi_v, phi, phi_ref], title=['Magnitude', '$\\phi_v$ ', '$\\phi + \\phi_0$', '$\\phi_{ref}$'],)
    plotter.show()