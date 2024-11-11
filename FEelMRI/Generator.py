import time

import matplotlib.pyplot as plt
import numpy as np

from FEelMRI.Connectivity import getConnectivity3, update_p2s
from FEelMRI.Helpers import (cropping_ranges, m_dirs, order,
                                restore_resolution)
from FEelMRI.ImageBuilding import (CSPAMM_magnetizations,
                                      DENSE_magnetizations,
                                      EXACT_magnetizations,
                                      ORI_O_CSPAMM_magnetizations, get_images)
from FEelMRI.KSpace import kspace
from FEelMRI.Math import itok, ktoi
from FEelMRI.MPIUtilities import MPI_print, MPI_rank, MPI_size, gather_image
from FEelMRI.pySpinBasedUtils import (check_kspace_bw, check_nb_slices,
                                         update_s2p)
from FEelMRI.Spins import Function


# Complementary SPAMM images
def get_oriocspamm_image(image, epi, phantom, parameters, debug=False):
  """ Generate tagging images
  """
  # Time-steping parameters
  t     = parameters.t
  dt    = parameters.dt
  n_t   = parameters.time_steps

 # Sequence parameters
  T1    = image.T1                     # relaxation
  alpha = image.flip_angle             # flip angle
  beta  = image.encoding_angle         # tip angle
  ke    = image.encoding_frequency     # encoding frequency
  M0    = image.M0                     # thermal equilibrium magnetization

  # Regional M0
  M0r = np.zeros([phantom.spins.Nb_samples,1])
  if isinstance(M0,np.ndarray) or isinstance(M0,list):
      M0r[phantom.spins.regions[:,0]] = M0[0]  # ventricle
      M0r[phantom.spins.regions[:,1]] = M0[1]  # static
  else:
      M0r[phantom.spins.regions[:,0]] = M0  # ventricle

  # T1 parameters across regions
  T1r = np.zeros([phantom.spins.Nb_samples,1])
  if isinstance(T1,np.ndarray) or isinstance(T1,list):
      T1r[phantom.spins.regions[:,0]] = T1[0]  # ventricle
      T1r[phantom.spins.regions[:,1]] = T1[1]  # static
  else:
      T1r[phantom.spins.regions[:,0]] = T1  # ventricle

  # Determine if the image and phantom geometry are 2D or 3D
  di = image.type_dim()                      # image geometric dimension
  dp = phantom.x.shape[-1]                   # phantom (fem) geometric dimension
  dk = np.sum(int(k/k) for k in ke if k!= 0) # Number of encoding directions

  # Output image
  size = np.append(image.resolution, [1, n_t])

  # Output kspaces
  k_nsa_1 = kspace(size, image, epi)
  k_nsa_2 = kspace(size, image, epi)
  k_mask = kspace(size, image, epi)
  k_T1 = kspace(size, image, epi)  

  # Flip angles
  if isinstance(alpha,float) or isinstance(alpha,int):
      alpha = alpha*np.ones([n_t],dtype=np.float)
  print(alpha)

  # Spins positions
  x = phantom.x

  # Check k space bandwidth to avoid folding artifacts
  res, incr_bw, D = check_kspace_bw(image, x)

  # Off resonance
  if not image.off_resonance:
    phi = np.zeros(image.resolution)
  else:
    phi = image.off_resonance(D["grid"][0],D["grid"][1])

  # Grid, voxel width, image resolution and number of voxels
  Xf = D['grid']
  width = D['voxel_size']
  resolution = D['resolution']
  nr_voxels = Xf[0].size

  # Check if the number of slices needs to be increased
  # for the generation of the connectivity when generating
  # Slice-Following (SF) images
  # if image.slice_following:
  #     Xf, SL = check_nb_slices(Xf, x, width, res)

  # Connectivity (this is done just once)
  voxel_coords = [X.flatten('F') for X in Xf]
  (s2p, excited_spins) = getConnectivity3(x, voxel_coords, width)
  s2p = np.array(s2p)

  # Spins positions with respect to its containing voxel center
  # Obs: the option -order='F'- is included to make the grid of the
  # Z coordinate at the end of the flattened array
  corners = np.array([Xf[j].flatten('F')[s2p]-0.5*width[j] for j in range(di)]).T
  x_rel = x[:, 0:dp] - corners

  # List of spins inside the excited slice
  # if image.slice_following:
  #     voxel_coords  = [X.flatten('F') for X in D['grid']] # reset voxel coords

  # Magnetization images and spins magnetizations
  m0_image = np.zeros(np.append(resolution, 1), dtype=np.complex128)
  m1_image = np.zeros(np.append(resolution, 1), dtype=np.complex128)
  m2_image = np.zeros(np.append(resolution, 1), dtype=np.complex128)
  m3_image = np.zeros(np.append(resolution, 1), dtype=np.complex128)  

  # Grid to evaluate magnetizations
  X = D['grid']

  # Resolutions and cropping ranges
  ovrs_fac = image.oversampling_factor
  r, c, dr, dc = cropping_ranges(image.resolution, resolution, ovrs_fac)

  # Spins inside the image slice
  exc_slice  = (x[:,2] < image.center[2] + image.slice_offset + 0.5*image.slice_thickness)
  exc_slice *= (x[:,2] > image.center[2] + image.slice_offset - 0.5*image.slice_thickness)

  # Time stepping
  prod = 1
  upre = np.zeros([x.shape[0], dp])
  for time_step in range(n_t):

    # Update time
    t += dt

    # Get displacements in the reference frame and deform mesh
    u = phantom.displacement(time_step)
    reshaped_u = u.vector()

    # Displacement in terms of pixels
    x_new = x_rel + reshaped_u - upre
    pixel_u = np.floor(np.divide(x_new, width))
    subpixel_u = x_new - np.multiply(pixel_u, width)

    # Change spins connectivity according to the new positions
    update_s2p(s2p, pixel_u, resolution)

    # Update pixel-to-spins connectivity
    # if image.slice_following:
    #   p2s = update_p2s(s2p-SL, excited_spins, nr_voxels)
    # else:
      # p2s = update_p2s(s2p, excited_spins, nr_voxels)
    p2s = update_p2s(s2p, excited_spins, nr_voxels)

    # Update relative spins positions
    x_rel[:,:] = subpixel_u

    # Updated spins positions
    x_upd = x + reshaped_u

    # Copy previous displcement field
    upre = np.copy(reshaped_u)

    # Get magnetization on each spin
    mags = ORI_O_CSPAMM_magnetizations(alpha[time_step], beta, prod, M0r,
                                 np.exp(-t/T1r), T1r, ke[0:dk], x[:,0:dk])
                                 
    for i in range(len(mags)):
        mags[i][(~exc_slice),:] = 0
    mags[-2][~phantom.spins.regions[:,2]] = 0
    # mags = list(mags)
    # mags.append(T1r)
    # mags = tuple(mags)

    # # Debug
    # if MPI_rank==0:
    #     from FEelMRI.IO import write_vtk
    #     from FEelMRI.Math import wrap
    #     S2P = Function(u.spins, dim=1)  # spins-to-pixel connectivity
    #     EXC = Function(u.spins, dim=1)  # excited slice (spins)
    #     rot = Function(u.spins, dim=1)
    #     theta = np.arctan(x[:,1]/x[:,0])
    #     rot.vector()[:] = wrap(theta.reshape((-1,1)), np.pi/8)
    #     EXC.vector()[exc_slice] = 1
    #     S2P.vector()[excited_spins] = s2p.reshape((-1,1))
    #     write_vtk([u,S2P,EXC,rot], path='output/Normal_{:04d}.vtu'.format(time_step), name=['displacement','s2p_connectivity','slice','rot'])

    # Fill images
    # Obs: the option -order='F'- is included because the grid was flattened
    # using this option. Therefore the reshape must be performed accordingly
    (I, m) = get_images(mags, x_upd, voxel_coords, width, p2s)
    if debug:
        MPI_print('Time step {:d}. Average nb. of spins inside a voxel: {:.0f}'.format(time_step, MPI_size*0.5*(m.max()-m.min())))
    else:
        MPI_print('Time step {:d}'.format(time_step))

    # Gather results
    m0_image[...] = gather_image(I[0].reshape(m0_image.shape,order='F'))
    m1_image[...] = gather_image(I[1].reshape(m1_image.shape,order='F'))
    m2_image[...] = gather_image(I[2].reshape(m2_image.shape,order='F'))
    m3_image[...] = gather_image(I[3].reshape(m3_image.shape,order='F'))    

    # Iterates over slices
    for slice in range(resolution[2]):

      # Complex magnetization data
      for enc_dir in range(1):

        # Magnetization expressions
        tmp0 = m0_image[...,slice,enc_dir]
        tmp1 = m1_image[...,slice,enc_dir]
        tmp2 = m2_image[...,slice,enc_dir]
        tmp3 = m3_image[...,slice,enc_dir]        

        # Uncorrected kspaces
        k0 = restore_resolution(itok(tmp0), r, c, dr, dc, enc_dir, image.resolution, ovrs_fac)
        k1 = restore_resolution(itok(tmp1), r, c, dr, dc, enc_dir, image.resolution, ovrs_fac)
        k2 = restore_resolution(itok(tmp2), r, c, dr, dc, enc_dir, image.resolution, ovrs_fac)
        k3 = restore_resolution(itok(tmp3), r, c, dr, dc, enc_dir, image.resolution, ovrs_fac)        

        # kspace resizing and epi artifacts generation
        delta_ph = image.FOV[m_dirs[enc_dir][1]]/image.phase_profiles
        k_nsa_1.gen_to_acq(k0, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_nsa_2.gen_to_acq(k1, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_mask.gen_to_acq(k2, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_T1.gen_to_acq(k3, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)

    # Flip angles product
    prod = prod*np.cos(alpha[time_step])

  return k_nsa_1, k_nsa_2, k_mask, k_T1


# Complementary SPAMM images
def get_cspamm_image(image, epi, phantom, parameters, debug=False):
  """ Generate tagging images
  """
  # Time-steping parameters
  t     = parameters.t
  dt    = parameters.dt
  n_t   = parameters.time_steps

 # Sequence parameters
  T1    = image.T1                     # relaxation
  alpha = image.flip_angle             # flip angle
  beta  = image.encoding_angle         # tip angle
  ke    = image.encoding_frequency     # encoding frequency
  M0    = image.M0                     # thermal equilibrium magnetization

  # Regional M0
  M0r = np.zeros([phantom.spins.Nb_samples,1])
  if isinstance(M0,np.ndarray) or isinstance(M0,list):
      M0r[phantom.spins.regions[:,0]] = M0[0]  # ventricle
      M0r[phantom.spins.regions[:,1]] = M0[1]  # static
  else:
      M0r[phantom.spins.regions[:,0]] = M0  # ventricle

  # T1 parameters across regions
  T1r = np.zeros([phantom.spins.Nb_samples,1])
  if isinstance(T1,np.ndarray) or isinstance(T1,list):
      T1r[phantom.spins.regions[:,0]] = T1[0]  # ventricle
      T1r[phantom.spins.regions[:,1]] = T1[1]  # static
  else:
      T1r[phantom.spins.regions[:,0]] = T1  # ventricle

  # Determine if the image and phantom geometry are 2D or 3D
  di = image.type_dim()                      # image geometric dimension
  dp = phantom.x.shape[-1]                   # phantom (fem) geometric dimension
  dk = np.sum(int(k/k) for k in ke if k!= 0) # Number of encoding directions

  # Output image
  size = np.append(image.resolution, [dk, n_t])

  # Output kspaces
  k_nsa_1 = kspace(size, image, epi)
  k_nsa_2 = kspace(size, image, epi)
  k_mask = kspace(size, image, epi)

  # Flip angles
  if isinstance(alpha,float) or isinstance(alpha,int):
      alpha = alpha*np.ones([n_t],dtype=np.float)

  # Spins positions
  x = phantom.x

  # Check k space bandwidth to avoid folding artifacts
  res, incr_bw, D = check_kspace_bw(image, x)

  # Off resonance
  if not image.off_resonance:
    phi = np.zeros(image.resolution)
  else:
    phi = image.off_resonance(D["grid"][0],D["grid"][1])

  # Grid, voxel width, image resolution and number of voxels
  Xf = D['grid']
  width = D['voxel_size']
  resolution = D['resolution']
  nr_voxels = Xf[0].size

  # Check if the number of slices needs to be increased
  # for the generation of the connectivity when generating
  # Slice-Following (SF) images
  # if image.slice_following:
  #     Xf, SL = check_nb_slices(Xf, x, width, res)

  # Connectivity (this is done just once)
  voxel_coords = [X.flatten('F') for X in Xf]
  (s2p, excited_spins) = getConnectivity3(x, voxel_coords, width)
  s2p = np.array(s2p)

  # Spins positions with respect to its containing voxel center
  # Obs: the option -order='F'- is included to make the grid of the
  # Z coordinate at the end of the flattened array
  corners = np.array([Xf[j].flatten('F')[s2p]-0.5*width[j] for j in range(di)]).T
  x_rel = x[:, 0:dp] - corners

  # List of spins inside the excited slice
  # if image.slice_following:
  #     voxel_coords  = [X.flatten('F') for X in D['grid']] # reset voxel coords

  # Magnetization images and spins magnetizations
  m0_image = np.zeros(np.append(resolution, dk), dtype=np.complex128)
  m1_image = np.zeros(np.append(resolution, dk), dtype=np.complex128)
  m2_image = np.zeros(np.append(resolution, dk), dtype=np.complex128)

  # Grid to evaluate magnetizations
  X = D['grid']

  # Resolutions and cropping ranges
  ovrs_fac = image.oversampling_factor
  r, c, dr, dc = cropping_ranges(image.resolution, resolution, ovrs_fac)

  # Spins inside the ventricle
  exc_slice  = (x[:,2] < image.center[2] + image.slice_offset + 0.5*image.slice_thickness)
  exc_slice *= (x[:,2] > image.center[2] + image.slice_offset - 0.5*image.slice_thickness)

  # Time stepping
  prod = 1
  upre = np.zeros([x.shape[0], dp])
  for time_step in range(n_t):

    # Update time
    t += dt

    # Get displacements in the reference frame and deform mesh
    u = phantom.displacement(time_step)
    reshaped_u = u.vector()

    # Displacement in terms of pixels
    x_new = x_rel + reshaped_u - upre
    pixel_u = np.floor(np.divide(x_new, width))
    subpixel_u = x_new - np.multiply(pixel_u, width)

    # Change spins connectivity according to the new positions
    update_s2p(s2p, pixel_u, resolution)

    # Update pixel-to-spins connectivity
    # if image.slice_following:
    #   p2s = update_p2s(s2p-SL, excited_spins, nr_voxels)
    # else:
      # p2s = update_p2s(s2p, excited_spins, nr_voxels)
    p2s = update_p2s(s2p, excited_spins, nr_voxels)

    # Update relative spins positions
    x_rel[:,:] = subpixel_u

    # Updated spins positions
    x_upd = x + reshaped_u

    # Copy previous displcement field
    upre = np.copy(reshaped_u)

    # Get magnetization on each spin
    mags = CSPAMM_magnetizations(alpha[time_step], beta, prod, M0r,
                                 np.exp(-t/T1r), ke[0:dk], x[:,0:dk])
    for i in range(len(mags)):
        mags[i][(~exc_slice),:] = 0
    mags[-1][~phantom.spins.regions[:,0]] = 0

    # # Debug
    # if MPI_rank==0:
    #     from FEelMRI.IO import write_vtk
    #     from FEelMRI.Math import wrap
    #     S2P = Function(u.spins, dim=1)  # spins-to-pixel connectivity
    #     EXC = Function(u.spins, dim=1)  # excited slice (spins)
    #     rot = Function(u.spins, dim=1)
    #     theta = np.arctan(x[:,1]/x[:,0])
    #     rot.vector()[:] = wrap(theta.reshape((-1,1)), np.pi/8)
    #     EXC.vector()[exc_slice] = 1
    #     S2P.vector()[excited_spins] = s2p.reshape((-1,1))
    #     write_vtk([u,S2P,EXC,rot], path='output/Normal_{:04d}.vtu'.format(time_step), name=['displacement','s2p_connectivity','slice','rot'])

    # Fill images
    # Obs: the option -order='F'- is included because the grid was flattened
    # using this option. Therefore the reshape must be performed accordingly
    (I, m) = get_images(mags, x_upd, voxel_coords, width, p2s)
    if debug:
        MPI_print('Time step {:d}. Average nb. of spins inside a voxel: {:.0f}'.format(time_step, MPI_size*0.5*(m.max()-m.min())))
    else:
        MPI_print('Time step {:d}'.format(time_step))

    # Gather results
    m0_image[...] = gather_image(I[0].reshape(m0_image.shape,order='F'))
    m1_image[...] = gather_image(I[1].reshape(m1_image.shape,order='F'))
    m2_image[...] = gather_image(I[2].reshape(m2_image.shape,order='F'))

    # Iterates over slices
    for slice in range(resolution[2]):

      # Complex magnetization data
      for enc_dir in range(dk):

        # Magnetization expressions
        tmp0 = m0_image[...,slice,enc_dir]
        tmp1 = m1_image[...,slice,enc_dir]
        tmp2 = m2_image[...,slice,enc_dir]

        # Uncorrected kspaces
        k0 = restore_resolution(itok(tmp0), r, c, dr, dc, enc_dir, image.resolution, ovrs_fac)
        k1 = restore_resolution(itok(tmp1), r, c, dr, dc, enc_dir, image.resolution, ovrs_fac)
        k2 = restore_resolution(itok(tmp2), r, c, dr, dc, enc_dir, image.resolution, ovrs_fac)

        # kspace resizing and epi artifacts generation
        delta_ph = image.FOV[m_dirs[enc_dir][1]]/image.phase_profiles
        k_nsa_1.gen_to_acq(k0, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_nsa_2.gen_to_acq(k1, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_mask.gen_to_acq(k2, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)

    # Flip angles product
    prod = prod*np.cos(alpha[time_step])

  return k_nsa_1, k_nsa_2, k_mask


# Complementary DENSE images
def get_cdense_image(image, epi, phantom, parameters, debug=False):
  """ Generate DENSE images
  """
  # Time-steping parameters
  t     = parameters.t
  dt    = parameters.dt
  n_t   = parameters.time_steps

 # Sequence parameters
  T1    = image.T1                     # relaxation
  alpha = image.flip_angle             # flip angle
  ke    = image.encoding_frequency     # encoding frequency
  M0    = image.M0                     # thermal equilibrium magnetization

  # Regional M0
  M0r = np.zeros([phantom.spins.Nb_samples,1])
  if isinstance(M0,np.ndarray) or isinstance(M0,list):
      M0r[phantom.spins.regions[:,0]] = M0[0]  # ventricle
      M0r[phantom.spins.regions[:,1]] = M0[1]  # static
  else:
      M0r[phantom.spins.regions[:,0]] = M0  # ventricle

  # Regional T1
  T1r = np.zeros([phantom.spins.Nb_samples,1])
  if isinstance(T1,np.ndarray) or isinstance(T1,list):
      T1r[phantom.spins.regions[:,0]] = T1[0]  # ventricle
      T1r[phantom.spins.regions[:,1]] = T1[1]  # static
  else:
      T1r[phantom.spins.regions[:,0]] = T1  # ventricle

  # Determine if the image and phantom geometry are 2D or 3D
  di = image.type_dim()                      # image geometric dimension
  dp = phantom.x.shape[-1]                   # phantom (fem) geometric dimension
  dk = np.sum(int(k/k) for k in ke if k!= 0) # Number of encoding directions

  # Output image
  size = np.append(image.resolution, [dk, n_t])

  # Output kspaces
  k_nsa_1 = kspace(size, image, epi)
  k_nsa_2 = kspace(size, image, epi)
  k_mask = kspace(size, image, epi)

  # Flip angles
  if isinstance(alpha,float) or isinstance(alpha,int):
      alpha = alpha*np.ones([n_t],dtype=np.float)

  # Spins positions
  x = phantom.x

  # Check k space bandwidth to avoid folding artifacts
  res, incr_bw, D = check_kspace_bw(image, x)

  # Off resonance
  if not image.off_resonance:
    phi = np.zeros(image.resolution)
  else:
    phi = image.off_resonance(D["grid"][0],D["grid"][1])

  # Grid, voxel width, image resolution and number of voxels
  Xf = D['grid']
  width = D['voxel_size']
  resolution = D['resolution']
  nr_voxels = Xf[0].size

  # Check if the number of slices needs to be increased
  # for the generation of the connectivity when generating
  # Slice-Following (SF) images
  # if image.slice_following:
  #     Xf, SL = check_nb_slices(Xf, x, width, res)

  # Connectivity (this is done just once)
  voxel_coords = [X.flatten('F') for X in Xf]
  (s2p, excited_spins) = getConnectivity3(x, voxel_coords, width)
  s2p = np.array(s2p)

  # Spins positions with respect to its containing voxel center
  # Obs: the option -order='F'- is included to make the grid of the
  # Z coordinate at the end of the flattened array
  corners = np.array([Xf[j].flatten('F')[s2p]-0.5*width[j] for j in range(di)]).T
  x_rel = x[:, 0:dp] - corners

  # List of spins inside the excited slice
  # if image.slice_following:
  #     voxel_coords  = [X.flatten('F') for X in D['grid']] # reset voxel coords

  # Magnetization images and spins magnetizations
  m0_image = np.zeros(np.append(resolution, dk), dtype=np.complex128)
  m1_image = np.zeros(np.append(resolution, dk), dtype=np.complex128)
  m2_image = np.zeros(np.append(resolution, dk), dtype=np.complex128)

  # Grid to evaluate magnetizations
  X = D['grid']

  # Resolutions and cropping ranges
  ovrs_fac = image.oversampling_factor
  r, c, dr, dc = cropping_ranges(image.resolution, resolution, ovrs_fac)

  # Spins inside the ventricle
  exc_slice  = (x[:,2] < image.center[2] + image.slice_offset + 0.5*image.slice_thickness)
  exc_slice *= (x[:,2] > image.center[2] + image.slice_offset - 0.5*image.slice_thickness)

  # Time stepping
  prod = 1
  upre = np.zeros([x.shape[0], dp])
  for time_step in range(n_t):

    # Update time
    t += dt

    # Get displacements in the reference frame and deform mesh
    u = phantom.displacement(time_step)
    reshaped_u = u.vector()

    # Displacement in terms of pixels
    x_new = x_rel + reshaped_u - upre
    pixel_u = np.floor(np.divide(x_new, width))
    subpixel_u = x_new - np.multiply(pixel_u, width)

    # Change spins connectivity according to the new positions
    update_s2p(s2p, pixel_u, resolution)

    # Update pixel-to-spins connectivity
    # if image.slice_following:
    #   p2s = update_p2s(s2p-SL, excited_spins, nr_voxels)
    # else:
    #   p2s = update_p2s(s2p, excited_spins, nr_voxels)
    p2s = update_p2s(s2p, excited_spins, nr_voxels)

    # Update relative spins positions
    x_rel[:,:] = subpixel_u

    # Updated spins positions
    x_upd = x + reshaped_u

    # Field inhomogeneity
    phi_in = image.off_resonance(x_upd[:,0],x_upd[:,1]) 

    # Copy previous displcement field
    upre = np.copy(reshaped_u)

    # Get magnetization on each spin
    mags = DENSE_magnetizations(alpha[time_step], prod, M0r, np.exp(-t/T1r),
                                ke[0:dk], x[:,0:dk], reshaped_u[:,0:dk], phi_in)
    for i in range(len(mags)):
        mags[i][(~exc_slice),:] = 0
    mags[-1][~phantom.spins.regions[:,0]] = 0

    # # Debug
    # if MPI_rank==0:
    #     from FEelMRI.IO import write_vtk
    #     from FEelMRI.Math import wrap
    #     S2P = Function(u.spins, dim=1)  # spins-to-pixel connectivity
    #     EXC = Function(u.spins, dim=1)  # excited slice (spins)
    #     rot = Function(u.spins, dim=1)
    #     theta = np.arctan(x[:,1]/x[:,0])
    #     rot.vector()[:] = wrap(theta.reshape((-1,1)), np.pi/8)
    #     EXC.vector()[exc_slice] = 1
    #     S2P.vector()[excited_spins] = s2p.reshape((-1,1))
    #     write_vtk([u,S2P,EXC,rot], path='output/SF_{:04d}.vtu'.format(time_step), name=['displacement','s2p_connectivity','slice','rot'])

    # Fill images
    # Obs: the option -order='F'- is included because the grid was flattened
    # using this option. Therefore the reshape must be performed accordingly
    (I, m) = get_images(mags, x_upd, voxel_coords, width, p2s)
    if debug:
        MPI_print('Time step {:d}. Average nb. of spins inside a voxel: {:.0f}'.format(time_step, MPI_size*0.5*(m.max()-m.min())))
    else:
        MPI_print('Time step {:d}'.format(time_step))

    # Gather results
    m0_image[...] = gather_image(I[0].reshape(m0_image.shape,order='F'))
    m1_image[...] = gather_image(I[1].reshape(m1_image.shape,order='F'))
    m2_image[...] = gather_image(I[2].reshape(m2_image.shape,order='F'))

    # Iterates over slices
    for slice in range(resolution[2]):

      # Complex magnetization data
      for enc_dir in range(dk):

        # Magnetization expressions
        tmp0 = m0_image[...,slice,enc_dir]
        tmp1 = m1_image[...,slice,enc_dir]
        tmp2 = m2_image[...,slice,enc_dir]

        # Uncorrected kspaces
        k0 = restore_resolution(itok(tmp0), r, c, dr, dc, enc_dir, image.resolution, ovrs_fac)
        k1 = restore_resolution(itok(tmp1), r, c, dr, dc, enc_dir, image.resolution, ovrs_fac)
        k2 = restore_resolution(itok(tmp2), r, c, dr, dc, enc_dir, image.resolution, ovrs_fac)

        # kspace resizing and epi artifacts generation
        delta_ph = image.FOV[m_dirs[enc_dir][1]]/image.phase_profiles
        k_nsa_1.gen_to_acq(k0, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_nsa_2.gen_to_acq(k1, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_mask.gen_to_acq(k2, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)

    # Flip angles product
    prod = prod*np.cos(alpha[time_step])

  return k_nsa_1, k_nsa_2, k_mask


# Exact image
def get_exact_image(image, epi, phantom, parameters, debug=False):
  """ Generate EXACT images
  """
  # Time-steping parameters
  t     = parameters.t
  dt    = parameters.dt
  n_t   = parameters.time_steps

 # Sequence parameters
  ke    = image.encoding_frequency     # encoding frequency
  M0    = image.M0                     # thermal equilibrium magnetization

  # Regional M0
  M0r = np.zeros([phantom.spins.Nb_samples,1])
  if isinstance(M0,np.ndarray) or isinstance(M0,list):
      M0r[phantom.spins.regions[:,0]] = M0[0]  # ventricle
      M0r[phantom.spins.regions[:,1]] = M0[1]  # static
  else:
      M0r[phantom.spins.regions[:,0]] = M0  # ventricle

  # Determine if the image and phantom geometry are 2D or 3D
  di = image.type_dim()                      # image geometric dimension
  dp = phantom.x.shape[-1]                   # phantom (fem) geometric dimension
  dk = np.sum(int(k/k) for k in ke if k!= 0) # Number of encoding directions

  # Output image
  size = np.append(image.resolution, [dk, n_t])
  mask = np.zeros(np.append(image.resolution, n_t), dtype=np.float32)

  # Output kspaces
  k_nsa_1 = kspace(size, image, epi)
  k_mask  = kspace(size, image, epi)

  # Spins positions
  x = phantom.x

  # Check k space bandwidth to avoid folding artifacts
  res, incr_bw, D = check_kspace_bw(image, x)

  # Grid, voxel width, image resolution and number of voxels
  Xf         = D['grid']
  width      = D['voxel_size']
  resolution = D['resolution']
  nr_voxels  = Xf[0].size

  # Check if the number of slices needs to be increased
  # for the generation of the connectivity when generating
  # Slice-Following (SF) images
  # if image.slice_following:
  #     Xf, SL = check_nb_slices(Xf, x, width, res)

  # Connectivity (this is done just once)
  voxel_coords = [X.flatten('F') for X in Xf]
  (s2p, excited_spins) = getConnectivity3(x, voxel_coords, width)
  s2p = np.array(s2p)

  # Spins positions with respect to its containing voxel center
  # Obs: the option -order='F'- is included to make the grid of the
  # Z coordinate at the end of the flattened array
  corners = np.array([Xf[j].flatten('F')[s2p]-0.5*width[j] for j in range(di)]).T
  x_rel = x[:, 0:dp] - corners

  # List of spins inside the excited slice
  # if image.slice_following:
  #     voxel_coords  = [X.flatten('F') for X in D['grid']] # reset voxel coords

  # Magnetization images and spins magnetizations
  m0_image = np.zeros(np.append(resolution, dk), dtype=np.complex128)
  m1_image = np.zeros(np.append(resolution, dk), dtype=np.complex128)

  # Grid to evaluate magnetizations
  X = D['grid']

  # Resolutions and cropping ranges
  ovrs_fac = image.oversampling_factor
  r, c, dr, dc = cropping_ranges(image.resolution, resolution, ovrs_fac)

  # Spins inside the ventricle
  exc_slice  = (x[:,2] < image.center[2] + image.slice_offset + 0.5*image.slice_thickness)
  exc_slice *= (x[:,2] > image.center[2] + image.slice_offset - 0.5*image.slice_thickness)

  # Time stepping
  for time_step in range(n_t):

    # Update time
    t += dt

    # Get displacements in the reference frame and deform mesh
    u = phantom.displacement(time_step)
    reshaped_u = u.vector()

    # Displacement in terms of pixels
    x_new = x_rel #+ reshaped_u - upre
    pixel_u = np.floor(np.divide(x_new, width))
    subpixel_u = x_new - np.multiply(pixel_u, width)

    # Change spins connectivity according to the new positions
    update_s2p(s2p, pixel_u, resolution)

    # Update pixel-to-spins connectivity
    # if image.slice_following:
    #   p2s = update_p2s(s2p-SL, excited_spins, nr_voxels)
    # else:
    #   p2s = update_p2s(s2p, excited_spins, nr_voxels)
    p2s = update_p2s(s2p, excited_spins, nr_voxels)

    # Update relative spins positions
    x_rel[:,:] = subpixel_u

    # Updated spins positions
    x_upd = x

    # Get magnetization on each spin
    mags = EXACT_magnetizations(M0r, ke[0:dk], reshaped_u[:,0:dk])
    for i in range(len(mags)):
        mags[i][(~exc_slice),:] = 0
    mags[-1][~phantom.spins.regions[:,0]] = 0

    # # Debug
    # if MPI_rank==0:
    #     from FEelMRI.IO import write_vtk
    #     from FEelMRI.Math import wrap
    #     S2P = Function(u.spins, dim=1)  # spins-to-pixel connectivity
    #     EXC = Function(u.spins, dim=1)  # excited slice (spins)
    #     rot = Function(u.spins, dim=1)
    #     theta = np.arctan(x[:,1]/x[:,0])
    #     rot.vector()[:] = wrap(theta.reshape((-1,1)), np.pi/8)
    #     EXC.vector()[exc_slice] = 1
    #     S2P.vector()[excited_spins] = s2p.reshape((-1,1))
    #     write_vtk([u,S2P,EXC,rot], path='output/Normal_{:04d}.vtu'.format(time_step), name=['displacement','s2p_connectivity','slice','rot'])

    # Fill images
    # Obs: the option -order='F'- is included because the grid was flattened
    # using this option. Therefore the reshape must be performed accordingly
    (I, m) = get_images(mags, x_upd, voxel_coords, width, p2s)
    if debug:
        MPI_print('Time step {:d}. Average nb. of spins inside a voxel: {:.0f}'.format(time_step, MPI_size*0.5*(m.max()-m.min())))
    else:
        MPI_print('Time step {:d}'.format(time_step))

    # Gather results
    m0_image[...] = gather_image(I[0].reshape(m0_image.shape,order='F'))
    m1_image[...] = gather_image(I[1].reshape(m1_image.shape,order='F'))

    # Iterates over slices
    for slice in range(resolution[2]):

      # Complex magnetization data
      for enc_dir in range(dk):

        # Magnetization expressions
        tmp0 = m0_image[...,slice,enc_dir]
        tmp1 = m1_image[...,slice,enc_dir]

        # Uncorrected kspaces
        k0 = restore_resolution(itok(tmp0), r, c, dr, dc, enc_dir, image.resolution, ovrs_fac)
        k1 = restore_resolution(itok(tmp1), r, c, dr, dc, enc_dir, image.resolution, ovrs_fac)

        # import matplotlib.pyplot as plt
        # from FEelMRI.Math import ktoi
        # fig,ax = plt.subplots(1,2)
        # ax[0].imshow(np.angle(ktoi(itok(tmp0)[1::ovrs_fac,:])))
        # ax[1].imshow(np.angle(ktoi(k0)))
        # plt.show(block=False)
        # plt.pause(5)

        # kspace resizing and epi artifacts generation
        delta_ph = image.FOV[m_dirs[enc_dir][1]]/image.phase_profiles
        k_nsa_1.gen_to_acq(k0, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)
        k_mask.gen_to_acq(k1, delta_ph, m_dirs[enc_dir], slice, enc_dir, time_step)

  return k_nsa_1, k_mask


# PC-SPAMM images
def get_PCSPAMM_image(image, phantom, parameters, debug=False):
  """ Generate tagging images
  """
  return True, True, True
