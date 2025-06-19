import warnings

import numpy as np
import scipy
from skimage.transform import resize

from feelmri.Filters import Riesz, Tukey
from feelmri.Math import ktoi
from feelmri.MPIUtilities import MPI_print
from feelmri.KSpaceTraj import CartesianStack, RadialStack, SpiralStack


def CartesianRecon(K, trajectory, filter={'type': 'Tukey', 'width': 0.9, 'lift': 0.3}):

  # Fix the direction of kspace lines measured in the opposite direction
  if isinstance(trajectory, CartesianStack) and trajectory.lines_per_shot > 1:   
    # Reorder lines depending of their readout direction
    for shot in trajectory.shots:
      for idx, ph in enumerate(shot):

        # Readout direction
        ro = (-1)**idx

        # Reverse orientations (only when ro=-1)
        K[::ro,ph,...] = K[::1,ph,...]

  # Zero padding in the dimensions with even measurements to avoid shifts in 
  # the image domain
  if trajectory.res[0] % 2 == 0:
    pad_width = ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0))
    K = np.pad(K, pad_width, mode='constant')
    # trajectory.res[0] += 1
  if trajectory.res[1] % 2 == 0:
    pad_width = ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0))
    K = np.pad(K, pad_width, mode='constant')
    # trajectory.res[1] += 1
  if trajectory.res[2] % 2 == 0:
    pad_width = ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0))
    K = np.pad(K, pad_width, mode='constant')
    # trajectory.res[2] += 1

  # Kspace filtering (as the scanner would do)
  if filter['type'] == 'Tukey':
    h_meas = Tukey(K.shape[0], width=0.9, lift=0.3)
    h_pha  = Tukey(K.shape[1], width=0.9, lift=0.3)
  elif filter['type'] == 'Riesz':
    h_meas = Riesz(K.shape[0], width=0.9, lift=0.3)
    h_pha  = Riesz(K.shape[1], width=0.9, lift=0.3)
  else:
    h_meas = 1.0
    h_pha = 1.0
    warnings.warn("Unknown filter type. No filtering applied.")

  h = np.outer(h_meas, h_pha)
  H = np.tile(h[:,:,np.newaxis, np.newaxis, np.newaxis], (1, 1, K.shape[2], K.shape[3], K.shape[4]))
  K_fil = H*K

  # Apply the inverse Fourier transform to obtain the image
  I = ktoi(K_fil[::1,...], [0,1,2])

  # The final image can resized to achieve the desired resolution
  resized_shape = np.hstack((trajectory.oversampling_arr*trajectory.res, I.shape[3:]))  
  I = resize(np.real(I), resized_shape) + 1j*resize(np.imag(I), resized_shape)

  # Chop if needed
  enc_Nx = K.shape[0]
  rec_Nx = trajectory.res[0]
  if (enc_Nx == rec_Nx):
      I = I
  else:
      ind1 = (enc_Nx - rec_Nx) // 2 #+ (trajectory.res[0]-1 % 2 != 0)
      ind2 = (enc_Nx - rec_Nx) // 2 + rec_Nx #+ (trajectory.res[0]-1 % 2 != 0)
      I = I[ind1:ind2,...]
  MPI_print("Image shape after correcting oversampling: ", I.shape)

  return I


def NUFFTRecon(K, trajectory, Jd=(6, 6, 6), iterative=False):
  """
  Perform Non-Uniform Fast Fourier Transform (NUFFT) reconstruction.

  Parameters:
  K (np.ndarray): The k-space data with dimensions (kx, ky, kz, coils, timepoints).
  trajectory (object): An object containing the trajectory information with attributes:
    - res (np.ndarray): The resolution of the reconstructed image.
    - oversampling_arr (np.ndarray): The oversampling factors for each dimension.
    - points (list of np.ndarray): The trajectory points for each dimension.
  Jd (tuple of int, optional): The interpolation size for each dimension. Default is (6, 6, 6).

  Returns:
  np.ndarray: The reconstructed image with dimensions (x, y, z, coils, timepoints).

  Raises:
  ImportError: If the 'pynufft' library is not installed.
  """

  try:
    from pynufft import NUFFT
  except ImportError:
    raise ImportError("pynufft is required for NUFFT reconstruction.")

  # Calculate pixelsize based on kspace extent
  pxsz = np.min(trajectory.pxsz)
  print(trajectory.pxsz)

  # Reconstructed image dimensions
  Nd = (trajectory.FOV/pxsz).astype(int)
  print("trjectory.FOV/pxsz = ", trajectory.FOV/pxsz)
  print("Nd = ", Nd)
  if isinstance(trajectory, CartesianStack) or isinstance(trajectory, SpiralStack) or isinstance(trajectory, RadialStack):
    Nd[-1] = K.shape[2]
  Nd = tuple(Nd.tolist())

  # Kspace dimensions
  Kd = tuple((trajectory.oversampling_arr*trajectory.res).astype(int).tolist())

  print("NUFFTRecon: Nd = ", Nd)
  print("NUFFTRecon: Kd = ", Kd)

  # Trajectory points
  om = np.hstack([p.flatten().reshape((-1, 1)) for p in trajectory.points])
  om *= np.pi/om.max()

  # Density compensation function
  if isinstance(trajectory, CartesianStack):
    dcf = 1.0
  else:
    dcf = [om[:,i]**2 for i in range(om.shape[1])]
    dcf = np.sum(dcf, axis=0)**0.5

  # NUFFT object initialization
  N = NUFFT()
  N.plan(om, Nd, Kd, Jd)

  # Array for reconstructed images
  im_shape = list(Nd)
  im_shape.append(K.shape[3])
  im_shape.append(K.shape[4])
  I = np.zeros(im_shape, dtype=np.complex64)

  if not iterative:
    for i in range(K.shape[3]):
      for j in range(K.shape[4]):
        x0 = K[...,i,j].flatten().reshape((-1,))
        I[...,i,j] = N.adjoint(x0*dcf)

  else:
    m, n, prodk = om.shape[0],np.prod(Nd), np.prod(Kd)
    print("NUFFTRecon: m = ", m)
    def mv(v):
        out = np.zeros((m+m+prodk, ),dtype=np.complex64)
        out[:m] = 1*N.forward(v[:n].reshape(Nd))
        out[m:2*m] = N.k2y(v[n:].reshape(Kd))
        out[2*m:] =  N.xx2k(N.x2xx(v[:n].reshape(Nd))).flatten() - v[n:]
        return out

    def mvh(u):
        out =  np.zeros((n+prodk, ),dtype=np.complex64)
        out[:n] =1* N.adjoint(u[:m]).flatten() + ((1/N.sn)*N.k2xx(u[2*m:].reshape(Kd))).flatten()
        out[n:] = N.y2k(u[m:2*m]).flatten() - u[2*m:]
        return out

    A = scipy.sparse.linalg.LinearOperator((m+m+prodk, n+prodk), matvec=mv, rmatvec=mvh)
    y2 = np.zeros((2*m+prodk, ),dtype=np.complex64)
    for i in range(K.shape[3]):
      for j in range(K.shape[4]):
        y2[:m] = dcf*K[...,i,j].flatten().reshape((-1,))
        y2[m:2*m] = K[...,i,j].flatten().reshape((-1,))
        z = scipy.sparse.linalg.lsqr(A, y2, iter_lim=100, atol=1e-5)[0]
        # x2 = z[:n].reshape(Nd)
        I[...,i,j] = z[:n].reshape(Nd)
        k = z[n:].reshape(Kd)

  return I