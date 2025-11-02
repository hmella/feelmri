import warnings

import numpy as np
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
  K : np.ndarray
      The k-space data to be reconstructed. Shape: (num_measurements, num_phases, num_slices, ...)
  trajectory : KSpaceTrajectory object
      The k-space trajectory object.
  filter : dict, optional
      The filter to be applied in k-space. The default is {'type': 'Tukey', 'width': 0.9, 'lift': 0.3}.
  Returns
  -------
  I : np.ndarray
      The reconstructed image. Shape: (num_measurements, num_phases, num_slices, ...)
  '''

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


# --------------------------- DCF helpers ---------------------------

def dcf_pipe_menon(nufft: NUFFT, n_iter: int = 20, eps: float = 1e-8) -> np.ndarray:
    """
    Pipe–Menon iterative density compensation: w <- w / (A A^H w).
    Works for any trajectory dimensionality supported by the NUFFT plan.
    """
    M = nufft.om.shape[0]
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
    kx, ky: (R, L, S). Returns w flattened to shape (R*L*S,).
    """
    R, L, S = kx.shape
    r = np.sqrt(kx**2 + ky**2)  # (R, L, S)
    if per_slice_normalize:
        w = r / (r.reshape(R, L, S).mean(axis=(0,1), keepdims=True) + eps)
    else:
        w = r / (r.mean() + eps)
    return w.astype(np.float32).ravel(order="C")


def dcf_local_speed_readout(kx: np.ndarray, ky: np.ndarray, kz: np.ndarray | None,
                            eps: float = 1e-8) -> np.ndarray:
    """
    'Speed' DCF along the acquisition path, computed PER (line, slice)
    along the readout dimension to avoid crossing discontinuities.
    kx, ky, kz: (R, L, S). Returns flattened (R*L*S,).
    """
    R, L, S = kx.shape
    if kz is None:
        # 2D
        dk = np.sqrt(np.diff(kx, axis=0)**2 + np.diff(ky, axis=0)**2)  # (R-1, L, S)
    else:
        dk = np.sqrt(np.diff(kx, axis=0)**2 + np.diff(ky, axis=0)**2 + np.diff(kz, axis=0)**2)

    w = np.empty((R, L, S), dtype=np.float64)
    # endpoints for each (line, slice)
    w[0, :, :]  = dk[0, :, :]
    w[-1, :, :] = dk[-1, :, :]
    # interior
    w[1:-1, :, :] = 0.5 * (dk[:-1, :, :] + dk[1:, :, :])
    # normalize per (line, slice) to avoid inter-line bias
    denom = w.mean(axis=0, keepdims=True) + eps  # (1, L, S)
    w = (w / denom).astype(np.float32)
    return w.ravel(order="C")

# --------------------------- NUFFT recon ---------------------------

def reconstruct_nufft(
    kdata: np.ndarray,
    ktraj: tuple,
    img_shape: tuple,
    *,
    dcw: np.ndarray | None = None,
    auto_dcw: str | None = "pipe-menon",   # None|"pipe-menon"|"radial-2d"|"speed"
    oversamp: float = 1.25,
    kernel_size: int = 6,
    mode: str = "adjoint",                 # "adjoint" or "cg" (fallback to adjoint)
    maxiter: int = 30,
    tol: float = 1e-6,
    combine: str | None = None             # None|"rss"
) -> np.ndarray:
    """
    NUFFT reconstruction for stack trajectories with inputs:
      - ktraj = (kx, ky, kz) each shaped (R, L, S) in cycles/FOV
      - kdata shaped (R, L, S, C) complex

    Parameters
    ----------
    kdata : np.ndarray
        (nb_readout=R, nb_lines=L, nb_slices=S, nb_channels=C), complex.
    ktraj : tuple[np.ndarray, np.ndarray, np.ndarray | None]
        (kx, ky, kz) each shaped (R, L, S) in cycles/FOV. For 2D stacks, kz can be zeros.
    img_shape : tuple
        (Nx, Ny, Nz) for 3D, or (Nx, Ny) for 2D (if all kz are zero/None).
    dcw : np.ndarray | None
        Optional density compensation weights, flattened length M=R*L*S.
    auto_dcw : str | None
        "pipe-menon" (default, robust), "radial-2d" (fast for stack-of-radials),
        or "speed" (good for time-ordered spirals). Ignored if dcw provided.
    oversamp, kernel_size : NUFFT plan parameters.
    mode : "adjoint" or "cg" (if available in your pynufft).
    combine : None keeps coil axis; "rss" does root-sum-of-squares.

    Returns
    -------
    np.ndarray
        Complex image with shape:
         - (*img_shape,) if single-channel or combine="rss"
         - (C, *img_shape) if multi-channel and combine=None
    """
    # ---- Unpack & validate shapes ----
    kx, ky, kz = ktraj
    if kz is None:
        kz = np.zeros_like(kx)
    kx = np.asarray(kx, dtype=np.float64)
    ky = np.asarray(ky, dtype=np.float64)
    kz = np.asarray(kz, dtype=np.float64)

    if kx.shape != ky.shape or kx.shape != kz.shape:
        raise ValueError("kx, ky, kz must share the same shape (R, L, S).")
    if kdata.shape[:3] != kx.shape:
        raise ValueError("kdata (R,L,S, C) must match ktraj shapes on first 3 dims.")
    R, L, S = kx.shape
    C = kdata.shape[3]
    M = R * L * S

    # ---- Flatten to (M, D) and (M, C) consistently (C-order: readout fastest) ----
    # This preserves the natural (readout, line, slice) ordering in memory.
    om_cycles = np.stack([kx.ravel(order="C"),
                          ky.ravel(order="C"),
                          kz.ravel(order="C")], axis=1)  # (M, 3)
    ksamples = kdata.reshape(M, C)  # (M, C)

    # Determine dimensionality: 2D if kz is (near) zero everywhere and Nz==1 in img_shape
    is_2d = np.allclose(om_cycles[:, 2], 0.0)
    if is_2d:
        om_cycles = om_cycles[:, :2]  # (M, 2)

    # ---- NUFFT plan ----
    Nd_user = tuple(int(n) for n in img_shape)
    if is_2d and len(Nd_user) == 3 and Nd_user[2] == 1:
        Nd = Nd_user[:2]
    else:
        Nd = Nd_user
    D = len(Nd)
    if D not in (2, 3):
        raise ValueError("img_shape must be 2D or 3D.")

    Kd = tuple(int(np.ceil(n * oversamp)) for n in Nd)
    Jd = tuple([kernel_size] * D)
    om_radians = 2.0 * np.pi * om_cycles  # cycles/FOV -> radians

    nufft = NUFFT()
    nufft.plan(om_radians, Nd=Nd, Kd=Kd, Jd=Jd)

    # ---- Density compensation ----
    if dcw is None and auto_dcw is not None:
        method = auto_dcw.lower()
        if method == "pipe-menon":
            dcw = dcf_pipe_menon(nufft, n_iter=20)
        elif method == "radial-2d":
            if D == 3:
                # Still OK: ramp uses in-plane radius only
                dcw = dcf_radial_stack(kx, ky)
            else:
                dcw = dcf_radial_stack(kx, ky)  # 2D case degenerates naturally
        elif method == "speed":
            dcw = dcf_local_speed_readout(kx, ky, (None if is_2d else kz))
        else:
            raise ValueError("auto_dcw ∈ {None, 'pipe-menon','radial-2d','speed'}")
    if dcw is not None:
        dcw = np.asarray(dcw, dtype=np.float32)
        if dcw.shape != (M,):
            raise ValueError(f"dcw must be shape ({M},), got {dcw.shape}")

    # ---- Recon per channel ----
    imgs = []
    for c in range(C):
        y = ksamples[:, c].astype(np.complex64, copy=False)
        if dcw is not None:
            y = y * dcw
        if mode.lower() == "adjoint":
            x = nufft.adjoint(y)
        elif mode.lower() == "cg":
            try:
                x = nufft.solve(y, solver='cg', maxiter=maxiter, tol=tol)
            except Exception:
                x = nufft.adjoint(y)
        else:
            raise ValueError("mode must be 'adjoint' or 'cg'.")
        imgs.append(x.astype(np.complex64, copy=False))

    img = np.stack(imgs, axis=0)  # (C, *Nd)

    # ---- Coil combine / output ----
    if C == 1:
        img = img[0]
    elif combine is None:
        pass
    elif combine.lower() == "rss":
        img = np.sqrt(np.sum(np.abs(img)**2, axis=0)).astype(np.complex64)
    else:
        raise ValueError("combine must be None or 'rss'")

    # If "2D but S>1" (stack of 2D slices with kz=0), caller should pass Nd=(Nx,Ny,S)
    # so D==3 and recon is 3D. If they passed (Nx,Ny) and S>1, return (C, Nx, Ny) or (Nx, Ny).
    return img

