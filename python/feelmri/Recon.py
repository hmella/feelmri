import warnings

import numpy as np
from numpy.fft import fftshift, ifft, ifftshift
from pynufft import NUFFT
from skimage.transform import resize

from feelmri.Filters import Riesz, Tukey
from feelmri.KSpaceTraj import CartesianStack, RadialStack, SpiralStack
from feelmri.Math import ktoi
from feelmri.MPIUtilities import MPI_print


def CartesianRecon(K, trajectory, filter={'type': 'Tukey', 'width': 0.9, 'lift': 0.3}):
  '''Reconstruct an image from Cartesian k-space data.
  Parameters
  ----------
  K: np.ndarray
      The k-space data to be reconstructed. Shape: (num_measurements, num_phases, num_slices, ...)
  trajectory: KSpaceTrajectory object
      The k-space trajectory object.
  filter: dict, optional
      The filter to be applied in k-space. The default is {'type': 'Tukey', 'width': 0.9, 'lift': 0.3}.
  Returns
  -------
  I: np.ndarray
      The reconstructed image. Shape: (num_measurements, num_phases, num_slices, ...)
  '''

  # Fix the direction of kspace lines measured in the opposite direction
  if isinstance(trajectory, CartesianStack) and trajectory.lines_per_shot > 1:   
    for shot in trajectory.shots:
      for idx, ph in enumerate(shot):
        ro = (-1)**idx
        K[::ro,ph,...] = K[::1,ph,...]

  # Zero padding in the dimensions with even measurements to avoid shifts
  if trajectory.res[0] % 2 == 0:
    pad_width = ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0))
    K = np.pad(K, pad_width, mode='constant')
  if trajectory.res[1] % 2 == 0:
    pad_width = ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0))
    K = np.pad(K, pad_width, mode='constant')
  if trajectory.res[2] % 2 == 0:
    pad_width = ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0))
    K = np.pad(K, pad_width, mode='constant')

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
    ind1 = (enc_Nx - rec_Nx) // 2 
    ind2 = (enc_Nx - rec_Nx) // 2 + rec_Nx 
    I = I[ind1:ind2,...]
  MPI_print("Image shape after correcting oversampling: ", I.shape)

  return I


# --------------------------- DCF helpers ---------------------------

def dcf_pipe_menon(nufft: NUFFT, M: int, n_iter: int = 20, eps: float = 1e-8) -> np.ndarray:
  """
  Pipe–Menon iterative density compensation: w <- w / (A A^H w).
  """
  w = np.ones(M, dtype=np.complex64)
  for _ in range(n_iter):
    g = nufft.forward(nufft.adjoint(w))
    w = w / (np.real(g) + eps)
  w = np.real(w)
  return (w / (w.mean() + eps)).astype(np.float32)


def dcf_radial_stack(kx: np.ndarray, ky: np.ndarray, eps: float = 1e-6,
                     per_slice_normalize: bool = True) -> np.ndarray:
  """
  Analytic ramp for (stack-of-)radials: w ∝ r_inplane = sqrt(kx^2 + ky^2).
  """
  R, L, S = kx.shape
  r = np.sqrt(kx**2 + ky**2)
  if per_slice_normalize:
    w = r / (r.reshape(R, L, S).mean(axis=(0,1), keepdims=True) + eps)
  else:
    w = r / (r.mean() + eps)
  return w.astype(np.float32).ravel(order="C")


def dcf_local_speed_readout(kx: np.ndarray, ky: np.ndarray, kz: np.ndarray | None,
                            eps: float = 1e-8) -> np.ndarray:
  """
  'Speed' DCF along the acquisition path, computed PER (line, slice).
  """
  R, L, S = kx.shape
  if kz is None:
    dk = np.sqrt(np.diff(kx, axis=0)**2 + np.diff(ky, axis=0)**2) 
  else:
    dk = np.sqrt(np.diff(kx, axis=0)**2 + np.diff(ky, axis=0)**2 + np.diff(kz, axis=0)**2)

  w = np.empty((R, L, S), dtype=np.float64)
  w[0, :, :]  = dk[0, :, :]
  w[-1, :, :] = dk[-1, :, :]
  w[1:-1, :, :] = 0.5 * (dk[:-1, :, :] + dk[1:, :, :])
  
  denom = w.mean(axis=0, keepdims=True) + eps  
  w = (w / denom).astype(np.float32)
  return w.ravel(order="C")


# --------------------------- NUFFT recon ---------------------------

def reconstruct_nufft(
    kdata: np.ndarray,
    ktraj: tuple,
    img_shape: tuple,
    fov: tuple = None,
    *,
    dcw: np.ndarray | None = None,
    auto_dcw: str | None = "pipe-menon",
    oversamp: float = 1.25,
    kernel_size: int = 6,
    mode: str = "adjoint",
    maxiter: int = 30,
    tol: float = 1e-6,
    combine: str | None = None
) -> np.ndarray:
  """
  Intelligent NUFFT reconstruction that automatically detects trajectory topology.
  """
  kx, ky, kz = ktraj
  if kz is None:
    kz = np.zeros_like(kx)
      
  kx = np.asarray(kx, dtype=np.float64)
  ky = np.asarray(ky, dtype=np.float64)
  kz = np.asarray(kz, dtype=np.float64)

  # Safely scale spatial frequencies to dimensionless cycles [-0.5, 0.5]
  if fov is not None:
    vxsz_x = fov[0] / img_shape[0]
    vxsz_y = fov[1] / img_shape[1]
    vxsz_z = fov[2] / img_shape[2] if len(img_shape) > 2 else 0.0

    kx = kx * vxsz_x
    ky = ky * vxsz_y
    kz = kz * vxsz_z

  R, L, S = kx.shape
  is_2d = np.allclose(kz, 0.0)

  # --- Auto-Detect Trajectory Topology ---
  is_kz_cartesian = (S > 1) and np.allclose(kz.max(axis=(0,1)), kz.min(axis=(0,1)))
  is_kxy_uniform = (S > 1) and np.allclose(kx, kx[:, :, 0:1]) and np.allclose(ky, ky[:, :, 0:1])
  
  is_uniform_stack = is_kz_cartesian and is_kxy_uniform and not is_2d

  if is_uniform_stack:
    return _recon_hybrid_stack(
      kdata, kx, ky, img_shape, dcw, auto_dcw, oversamp, kernel_size, mode, maxiter, tol, combine
    )
  else:
    return _recon_full_3d(
      kdata, (kx, ky, kz), img_shape, dcw, auto_dcw, oversamp, kernel_size, mode, maxiter, tol, combine
    )


def _recon_hybrid_stack(
    kdata, kx, ky, img_shape, dcw, auto_dcw, oversamp, kernel_size, mode, maxiter, tol, combine
):
  R, L, S, C = kdata.shape
  Nx, Ny = img_shape[0], img_shape[1]
  Nz = img_shape[2] if len(img_shape) == 3 else S
  
  if S != Nz:
    if S < Nz:
      pad_diff = Nz - S
      pad_b = pad_diff // 2
      pad_a = pad_diff - pad_b
      kdata_z = np.pad(kdata, ((0,0), (0,0), (pad_b, pad_a), (0,0)), mode='constant')
    else:
      crop_b = (S - Nz) // 2
      crop_a = (S - Nz) - crop_b
      kdata_z = kdata[:, :, crop_b:S - crop_a, :]
  else:
    kdata_z = kdata.copy()

  # 1D Cartesian IFFT along Z
  spatial_kdata = fftshift(ifft(ifftshift(kdata_z, axes=2), axis=2), axes=2)

  kx_2d = kx[:, :, 0]
  ky_2d = ky[:, :, 0]
  om_cycles = np.stack([kx_2d.ravel(order="C"), ky_2d.ravel(order="C")], axis=1)
  om_radians = 2.0 * np.pi * om_cycles
  
  Nd = (Nx, Ny)
  Kd = tuple(int(np.ceil(n * oversamp)) for n in Nd)
  Jd = (kernel_size, kernel_size)
  
  nufft = NUFFT()
  nufft.plan(om_radians, Nd=Nd, Kd=Kd, Jd=Jd)
  
  dcw_2d = None
  if dcw is not None:
    dcw_2d = dcw.reshape(R, L, S)[:, :, 0].ravel(order="C")
  elif auto_dcw is not None:
    method = auto_dcw.lower()
    if method == "pipe-menon":
      dcw_2d = dcf_pipe_menon(nufft, M=om_radians.shape[0], n_iter=20)
    elif method in ["radial-2d", "speed"]:
      dcw_2d = dcf_radial_stack(np.expand_dims(kx_2d, 2), np.expand_dims(ky_2d, 2), per_slice_normalize=False)
  
  if dcw_2d is not None:
    dcw_2d = np.asarray(dcw_2d, dtype=np.float32)

  img_3d = np.zeros((C, Nx, Ny, Nz), dtype=np.complex64)
  
  for z_idx in range(Nz):
    for c in range(C):
      y = spatial_kdata[:, :, z_idx, c].ravel(order="C").astype(np.complex64, copy=False)
      if dcw_2d is not None:
        y = y * dcw_2d
          
      if mode.lower() == "adjoint":
        img_3d[c, :, :, z_idx] = nufft.adjoint(y)
      else:
        try:
          img_3d[c, :, :, z_idx] = nufft.solve(y, solver='cg', maxiter=maxiter, tol=tol)
        except Exception:
          img_3d[c, :, :, z_idx] = nufft.adjoint(y)

  if C == 1:
    return img_3d[0]
  elif combine == "rss":
    return np.sqrt(np.sum(np.abs(img_3d)**2, axis=0)).astype(np.complex64)
  return img_3d


def _recon_full_3d(kdata, ktraj, img_shape, dcw, auto_dcw, oversamp, kernel_size, mode, maxiter, tol, combine):
  kx, ky, kz = ktraj
  R, L, S = kx.shape
  C = kdata.shape[3]
  M = R * L * S

  om_cycles = np.stack([kx.ravel(order="C"), ky.ravel(order="C"), kz.ravel(order="C")], axis=1)
  ksamples = kdata.reshape(M, C)

  is_2d = np.allclose(om_cycles[:, 2], 0.0)
  if is_2d:
    om_cycles = om_cycles[:, :2]

  Nd_user = tuple(int(n) for n in img_shape)
  Nd = Nd_user[:2] if (is_2d and len(Nd_user) == 3 and Nd_user[2] == 1) else Nd_user
  D = len(Nd)

  Kd = tuple(int(np.ceil(n * oversamp)) for n in Nd)
  Jd = tuple([kernel_size] * D)
  om_radians = 2.0 * np.pi * om_cycles

  nufft = NUFFT()
  nufft.plan(om_radians, Nd=Nd, Kd=Kd, Jd=Jd)

  if dcw is None and auto_dcw is not None:
    method = auto_dcw.lower()
    if method == "pipe-menon":
      dcw = dcf_pipe_menon(nufft, M=om_radians.shape[0], n_iter=20)
    elif method == "radial-2d":
      dcw = dcf_radial_stack(kx, ky)
    elif method == "speed":
      dcw = dcf_local_speed_readout(kx, ky, (None if is_2d else kz))
        
  if dcw is not None:
    dcw = np.asarray(dcw, dtype=np.float32)

  imgs = []
  for c in range(C):
    y = ksamples[:, c].astype(np.complex64, copy=False)
    if dcw is not None:
      y = y * dcw
        
    if mode.lower() == "adjoint":
      x_recon = nufft.adjoint(y)
    else:
      try:
        x_recon = nufft.solve(y, solver='cg', maxiter=maxiter, tol=tol)
      except Exception:
        x_recon = nufft.adjoint(y)
        
    imgs.append(x_recon.astype(np.complex64, copy=False))

  img = np.stack(imgs, axis=0)

  if C == 1:
    return img[0]
  elif combine == "rss":
    return np.sqrt(np.sum(np.abs(img)**2, axis=0)).astype(np.complex64)
  return img