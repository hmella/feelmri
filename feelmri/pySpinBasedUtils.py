import numpy as np

from feelmri.Helpers import iseven, isodd, round_to_even
from feelmri.MPIUtilities import MPI_print


# Three dimensional images with number of slices greater than 1
def update_s2p(s2p, pixel_u, resolution):
    s2p += (resolution[0]*pixel_u[:,1]             # jump betwen rows
        + resolution[0]*resolution[1]*pixel_u[:,2] # jump betwen slices
        + pixel_u[:,0]).astype(np.int64)           # jump between columns


# Check if the kspace bandwidth needs to be modified
# to avoid artifacts during the generation and to
# reproduce the behavior of the scanner
def check_kspace_bw(image, x):

  # Encoding frequency, voxelsize, FOV, kspace factor,
  # and oversampling_factor
  ke = image.encoding_frequency
  kf = image.kspace_factor
  pxsz = image.voxel_size()
  FOV = np.copy(image.FOV)
  ofac = image.oversampling_factor

  # Lower bound for the new encoding frequencies
  kl = 2*np.pi/pxsz

  # Field of view in the measurement and phase directions including the
  # oversampling factor
  FOV[:2]  = ofac*FOV[:2]
  if ofac != 1:
      dk = 1.0/FOV[:2]
      BW = dk*(ofac*image.resolution[:2]+1)
      FOV[:2] = ofac*image.resolution[:2]*(1/BW)
      pxsz[:2] = 1/(kf*BW)
  else:
      pxsz[:2] *= 1/kf

  # Virtual resolution for the generation process
  virtual_resolution = np.divide(FOV, pxsz).astype(np.int64)
  incr_bw = np.any([image.resolution[i] < virtual_resolution[i] for i in range(ke.size)])  

  # If the resolution needs modification then check if the new
  # one has even or odd components to make the cropping process
  # easier
  if incr_bw:

      # Create a new image object with the new FOV and resolution
      new_image = image.__class__(FOV=FOV,
              center=image.center,
              resolution=virtual_resolution,
              encoding_frequency=ke,
              T1=image.T1,
              off_resonance=image.off_resonance)
  else:
      new_image = image

  # Output dict
  D = {"voxel_size": new_image.voxel_size(),
       "grid":       new_image.grid,
       "resolution":  new_image.resolution,
       "FOV": new_image.FOV}

  # Debug printings
  MPI_print(' Acq. matrix: ({:d},{:d})'.format(image.acq_matrix[0],
      image.acq_matrix[1]))
  MPI_print(' Generation matrix: ({:d},{:d})'.format(virtual_resolution[0],
      virtual_resolution[1]))

  return virtual_resolution, incr_bw, D


#
def check_nb_slices(grid, x, vsz, res):

  # Flatten grid
  Xf = [p.flatten('F') for p in grid]

  # Min and max position in the slice direction
  Z = np.array([Xf[2].min(), Xf[2].max()])
  z = np.array([x[:,2].min(), x[:,2].max()])
  borders = np.array([-0.5*vsz[2], 0.5*vsz[2]])
  N = (z - (Z+borders))/vsz[2]
  sign = [-1, 1]
  N = [int(np.ceil(np.abs(N[i]))) if sign[i]*N[i] > 0 else 0 for i in range(len(N))]

  # Number of additional slices
  r_slice = res[0]*res[1]
  for i in range(len(N)):
    for j in range(N[i]):
      for k in range(3):
        if k < 2:
          Xf[k] = np.append(Xf[k], Xf[k][0:r_slice], axis=0)
        else:
          Xf[k] = np.insert(Xf[k], i*Xf[k].size, Z[i]*np.ones([r_slice,]) + (j+1)*vsz[k]*(-1)**(i+1), axis=0)

  # Slice location factor and number of voxels
  SL = r_slice*N[0]

  # Reshape output
  resolution = [res[0], res[1], res[2]+sum(N)]
  Xf = [p.reshape(resolution, order='F') for p in Xf]

  return Xf, SL
