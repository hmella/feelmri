"""A finished run's k-space, and the image it reconstructs to.

A run is a subprocess writing `kspace.npz`, so a result arrives as four
arrays: `kspace`, `times`, `points` and `plan_fov`. This module turns those into
what a panel draws, and holds no figure.

**Reconstruction is the library's own**, `Recon.reconstruct_nufft`, not a
second implementation. The trajectory is non-Cartesian in general -- an EPI
train ramp-samples, so its samples are not evenly spaced in k -- and writing
a gridder here would be a second thing to keep correct.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: A result needs these to reconstruct. `times` is carried for display only.
REQUIRED = ('kspace', 'points')


def describe(result: Dict[str, np.ndarray]) -> list:
  """`(label, value)` rows summarising a result, ready to display."""
  rows = []
  kspace = np.asarray(result.get('kspace', np.zeros(0)))
  rows.append(('samples', str(kspace.shape[0] if kspace.size else 0)))
  rows.append(('k-space shape', str(kspace.shape)))
  if kspace.size:
    rows.append(('|k| peak', f'{float(np.abs(kspace).max()):.6e}'))
    rows.append(('|k| at centre', f'{float(np.abs(kspace).ravel()[0]):.6e}'))
  times = np.asarray(result.get('times', np.zeros(0)))
  if times.size:
    rows.append(('readout', f'{float(times.min()):.3f} to '
                            f'{float(times.max()):.3f} ms'))
  points = np.asarray(result.get('points', np.zeros((0, 3))))
  if points.size:
    extent = np.abs(points).max(axis=0)
    rows.append(('k extent (1/m)',
                 ' '.join(f'{float(v):.1f}' for v in extent)))
  encoded = encoded_fov(result)
  if any(v > 0 for v in encoded):
    rows.append(('FOV encoded (m)',
                 ' '.join(f'{v:.4g}' if v > 0 else '-' for v in encoded)))
    rows.append(('matrix acquired', ' '.join(str(v)
                                             for v in default_matrix(result))))
  plan = np.asarray(result.get('plan_fov', result.get('fov', np.zeros(3))))
  if plan.size == 3 and float(np.max(plan)) > 0:
    # Shown BESIDE the encoded one on purpose: they are different quantities
    # and look alike, which is exactly how they get confused.
    rows.append(('FOV of the plan', ' '.join(f'{float(v):.4g}' for v in plan)))
  return rows


def trajectory(result: Dict[str, np.ndarray]) -> Tuple[np.ndarray, ...]:
  """`(kx, ky, kz)` in 1/m, as flat arrays, for an overlay."""
  points = np.asarray(result.get('points', np.zeros((0, 3))), dtype=float)
  if points.ndim != 2 or points.shape[1] != 3:
    raise ValueError(
      f'results.trajectory: expected an (N, 3) points array, got '
      f'{points.shape}')
  return points[:, 0], points[:, 1], points[:, 2]


def _spacing(k: np.ndarray) -> float:
  """Median gap between distinct sample positions on one axis, or NaN.

  The median rather than the minimum: a ramp-sampled readout puts samples
  closer together on the ramps, and the minimum would read the FOV off those
  instead of off the phase-encode step.
  """
  unique = np.unique(np.round(np.asarray(k, dtype=float), 6))
  if unique.size < 2:
    return float('nan')
  gaps = np.diff(unique)
  gaps = gaps[gaps > 1e-9]
  return float(np.median(gaps)) if gaps.size else float('nan')


def encoded_fov(result: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
  """The field of view the SEQUENCE encodes, from the sample spacing.

  **This is not the plan's field of view**, and confusing the two is how this
  panel first drew pure aliasing texture and called it an image. The plan's
  FOV is the slab chosen for simulation; the reconstruction FOV is a property
  of the trajectory, `1/dk`. On `epi_v142.seq` over a 0.30 x 0.22 m slab they
  read 0.32 x 0.24 against 0.30 x 0.22 -- close enough to look like the same
  number and not be it.

  Measured on that pair: reconstructing with the plan's FOV doubled, which is
  what a first version did, gives peak/mean **2.0** and no object; with the
  encoded FOV, **15.0** and the phantom's vials.

  An axis with no spread -- a single slice -- has no encoded extent, and
  reports 0.0 for the caller to fill in.
  """
  points = np.asarray(result.get('points', np.zeros((0, 3))), dtype=float)
  out = []
  for axis in range(3):
    column = points[:, axis] if points.size else np.zeros(0)
    step = _spacing(column)
    out.append(0.0 if not np.isfinite(step) or step <= 0 else 1.0 / step)
  return tuple(out)


def default_matrix(result: Dict[str, np.ndarray]) -> Tuple[int, int, int]:
  """The matrix the trajectory supports, `2 * k_max / dk` per axis.

  Derived from the data rather than guessed, so the default reconstruction is
  at the resolution that was actually acquired. An axis with no spread gives
  one slice.
  """
  points = np.asarray(result.get('points', np.zeros((0, 3))), dtype=float)
  if points.size == 0:
    return (64, 64, 1)
  out = []
  for axis in range(3):
    column = points[:, axis]
    step = _spacing(column)
    if not np.isfinite(step) or step <= 0:
      out.append(1)
      continue
    out.append(max(1, int(round(2.0 * float(np.abs(column).max()) / step))))
  return tuple(out)


def reconstruct(result: Dict[str, np.ndarray],
                matrix: Optional[Sequence[int]] = None,
                *, fov: Optional[Sequence[float]] = None,
                auto_dcw: str = 'pipe-menon') -> np.ndarray:
  """Reconstruct the magnitude-and-phase image, via the library's NUFFT.

  `fov` defaults to what the SEQUENCE encodes, `1/dk` from the trajectory --
  NOT the plan's, which is the slab chosen for simulation and a different
  quantity. See `encoded_fov`; using the plan's instead produced aliasing
  texture with peak/mean 2.0 where the encoded one gives 15.0 and an object.

  Returns a complex array, so a panel can show magnitude and phase from one
  reconstruction rather than doing it twice.
  """
  missing = [name for name in REQUIRED if name not in result]
  if missing:
    raise ValueError(
      f'results.reconstruct: the result has no {missing} -- it was written '
      f'by a run older than the trajectory was saved, so it can be plotted '
      f'but not reconstructed')

  from ...Recon import reconstruct_nufft

  kspace = np.asarray(result['kspace'])
  kx, ky, kz = trajectory(result)
  if kx.shape[0] != kspace.shape[0]:
    raise ValueError(
      f'results.reconstruct: {kx.shape[0]} trajectory points for '
      f'{kspace.shape[0]} samples')

  matrix = tuple(int(v) for v in (matrix if matrix is not None
                                  else default_matrix(result)))
  if len(matrix) != 3 or any(v < 1 for v in matrix):
    raise ValueError(
      f'results.reconstruct: matrix must be three positive integers, got '
      f'{matrix}')

  if fov is None:
    fov = encoded_fov(result)
    if not any(v > 0 for v in fov):
      raise ValueError(
        'results.reconstruct: the trajectory has no sample spacing to read a '
        'field of view from; pass fov= explicitly')
    # A single-slice axis has no encoded extent. Its value only has to be
    # non-zero for the normalisation, and the slice is not resolved either
    # way, so the plan's thickness is used where it is known.
    plan = np.asarray(result.get('plan_fov', result.get('fov', np.zeros(3))),
                      dtype=float)
    fov = tuple(v if v > 0 else (float(plan[i]) if plan.size == 3
                                 and plan[i] > 0 else 1.0)
                for i, v in enumerate(fov))

  shape = (kspace.shape[0], 1, 1)
  image = reconstruct_nufft(
    kdata=np.asarray(kspace).reshape(shape + (-1,)),
    ktraj=(kx.reshape(shape), ky.reshape(shape), kz.reshape(shape)),
    img_shape=matrix,
    fov=tuple(float(v) for v in fov),
    auto_dcw=auto_dcw, oversamp=1.25, kernel_size=6, mode='adjoint',
    combine=None)
  return np.asarray(image)
