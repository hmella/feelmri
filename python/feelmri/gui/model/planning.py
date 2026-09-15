"""Scan geometry: the FOV box, the M/P/S triad, and the Euler convention.

`PVSMParser` builds its rotation as `MPS = Rz(tz) @ Rx(tx) @ Ry(ty)`
(`Parameters.py`), so a plan the GUI writes back to a ParaView state file has
to be expressed in that exact Z-X-Y order. Going from a matrix the user has
rotated interactively to those three angles is the load-bearing piece here, and
`mps_to_euler` is its inverse, gimbal lock included.

The axis convention is shared with the rest of the library and is worth stating
once: **column 0 of MPS is M (measurement, readout), column 1 is P (phase),
column 2 is S (slice)**. That is the same ordering as `Trajectory`'s
`FOV`/`res` triples and as `SequenceBlock.M_gradients` / `P_gradients` /
`S_gradients`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from feelmri.Math import Rx, Ry, Rz

#: Column index of each imaging direction in an MPS matrix.
M_AXIS, P_AXIS, S_AXIS = 0, 1, 2
AXIS_NAMES = ('M (readout)', 'P (phase)', 'S (slice)')

# Below this, `cos(tx)` is small enough that `ty` and `tz` stop being
# separately determined and only their sum or difference survives. 1e-7 on the
# cosine is around 1.6e-4 rad away from the pole, comfortably outside the range
# where the atan2 branches lose precision.
_LOCK_TOL = 1e-7


def euler_to_mps(tx: float, ty: float, tz: float) -> np.ndarray:
  """`Rz(tz) @ Rx(tx) @ Ry(ty)`, the composition `PVSMParser` uses.

  Angles in radians. This is a thin wrapper so that every site agrees on the
  order; getting it wrong is silent, since any order produces a valid rotation.
  """
  return Rz(tz) @ Rx(tx) @ Ry(ty)


def mps_to_euler(mps: np.ndarray) -> Tuple[float, float, float]:
  """Recover `(tx, ty, tz)` in radians from an MPS matrix.

  Writing out the product gives

      M[2,1] =  sin(tx)
      M[2,0] = -cos(tx) sin(ty)      M[2,2] = cos(tx) cos(ty)
      M[0,1] = -sin(tz) cos(tx)      M[1,1] = cos(tz) cos(tx)

  so `tx` comes from one entry and the other two from an `atan2` each.

  **At `tx = +-pi/2` the decomposition is not unique**: `cos(tx)` is zero, the
  four entries above vanish, and only `tz +- ty` is determined. The convention
  taken here is `ty = 0`, which is the usual one and keeps the result
  continuous in `tz`.

  **The accuracy contract, measured.** Outside the lock tolerance the matrix
  round trip is exact to about 1e-16. Inside it the branch pins `ty = 0` and so
  discards terms of size `cos(tx)`, which bounds the error by `_LOCK_TOL`
  rather than by machine epsilon: 2.0e-08 at `cos(tx) = 1e-8`, 2.0e-09 at 1e-9.
  That is the price of returning a unique answer where the parametrisation has
  none, and it is under a millionth of a degree of orientation.

  The angles themselves are unique only away from the poles; at lock they will
  differ from whatever produced the matrix, while the matrix does not move.
  """
  mps = np.asarray(mps, dtype=np.float64)
  if mps.shape != (3, 3):
    raise ValueError(f"mps_to_euler: expected a 3x3 matrix, got {mps.shape}")

  # `tx` comes from atan2, not arcsin. `arcsin` has an infinite derivative at
  # +-1, so near the pole a rounding error in M[2,1] becomes a large error in
  # the angle: measured 2.4e-10 worst matrix round trip with arcsin against
  # 1e-16 with this form. Since M[2,0] = -cos(tx) sin(ty) and
  # M[2,2] = cos(tx) cos(ty), their hypotenuse IS cos(tx), which is
  # non-negative over the principal range, so no sign recovery is needed.
  sx = float(mps[2, 1])
  cx = float(np.hypot(mps[2, 0], mps[2, 2]))
  tx = float(np.arctan2(sx, cx))

  if cx > _LOCK_TOL:
    ty = float(np.arctan2(-mps[2, 0], mps[2, 2]))
    tz = float(np.arctan2(-mps[0, 1], mps[1, 1]))
  elif sx > 0:
    # tx = +pi/2: M[0,0] = cos(tz+ty), M[0,2] = sin(tz+ty). Pin ty = 0.
    ty = 0.0
    tz = float(np.arctan2(mps[0, 2], mps[0, 0]))
  else:
    # tx = -pi/2: M[0,0] = cos(tz-ty), M[0,2] = -sin(tz-ty). Pin ty = 0.
    ty = 0.0
    tz = float(np.arctan2(-mps[0, 2], mps[0, 0]))
  return tx, ty, tz


def is_rotation(m: np.ndarray, tol: float = 1e-8) -> bool:
  """Whether `m` is a proper rotation: orthonormal with determinant +1."""
  m = np.asarray(m, dtype=np.float64)
  if m.shape != (3, 3):
    return False
  return bool(np.allclose(m.T @ m, np.eye(3), atol=tol)
              and abs(np.linalg.det(m) - 1.0) < tol)


@dataclass
class FOVBox:
  """A field of view: extent, centre and orientation.

  `fov` and `loc` are in metres and `angles` in radians, all plain floats, so
  this class carries no unit dependency. Convert at the boundary with pint, the
  way `PVSMParser` and the example YAML files do.
  """

  fov: np.ndarray                 # (3,) M, P, S extent in metres
  loc: np.ndarray                 # (3,) centre in metres, lab frame
  angles: np.ndarray              # (3,) tx, ty, tz in radians

  def __post_init__(self):
    self.fov = np.asarray(self.fov, dtype=np.float64).reshape(3)
    self.loc = np.asarray(self.loc, dtype=np.float64).reshape(3)
    self.angles = np.asarray(self.angles, dtype=np.float64).reshape(3)
    if np.any(self.fov < 0) or not np.all(np.isfinite(self.fov)):
      raise ValueError(
        f"FOVBox: fov must be finite and non-negative, got {tuple(self.fov)}")
    if not np.all(np.isfinite(self.loc)) or not np.all(np.isfinite(self.angles)):
      raise ValueError("FOVBox: loc and angles must be finite")

  @classmethod
  def from_mps(cls, fov, loc, mps) -> 'FOVBox':
    """Build from a rotation matrix, decomposing it to the Z-X-Y angles."""
    if not is_rotation(mps):
      raise ValueError(
        "FOVBox.from_mps: mps is not a proper rotation (orthonormal, det +1)")
    return cls(fov, loc, np.array(mps_to_euler(mps)))

  @property
  def mps(self) -> np.ndarray:
    """The 3x3 orientation, columns M, P, S in lab coordinates."""
    return euler_to_mps(*self.angles)

  @property
  def voxel_size(self):
    """`fov / res` is the caller's to compute; this is here as a reminder that
    voxel size is DERIVED, never stored. See `examples/free_running.py`."""
    raise AttributeError(
      "FOVBox has no voxel_size: it is fov / resolution, and the resolution "
      "lives in the imaging parameters, not in the box")

  def corners(self) -> np.ndarray:
    """The eight corners in lab coordinates, `(8, 3)`."""
    h = 0.5 * self.fov
    signs = np.array([[sx, sy, sz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=np.float64)
    return (signs * h) @ self.mps.T + self.loc

  def axis_arrows(self, length: Optional[float] = None
                  ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """The M/P/S triad as `(origin, direction, label)`, ready to draw.

    `direction` is scaled to `length`, defaulting to 60% of the corresponding
    FOV extent so each arrow sits inside its own half-width and the three stay
    visually comparable to the box they annotate.
    """
    mps = self.mps
    out = []
    for i, name in enumerate(AXIS_NAMES):
      scale = 0.6 * self.fov[i] if length is None else length
      out.append((self.loc.copy(), mps[:, i] * scale, name))
    return out

  def to_imaging(self, x: np.ndarray) -> np.ndarray:
    """Lab coordinates to imaging coordinates.

    This is byte-for-byte the arithmetic `FEMPhantom.orient` applies,
    `(x - LOC) @ MPS`, which for row vectors is `R^T (x - LOC)`. The GUI's
    preview and the simulation must agree, so it is written once.
    """
    return (np.asarray(x, dtype=np.float64) - self.loc) @ self.mps

  def to_lab(self, x: np.ndarray) -> np.ndarray:
    """The inverse of `to_imaging`, matching `FEMPhantom.reorient`."""
    return np.asarray(x, dtype=np.float64) @ self.mps.T + self.loc

  def contains(self, x: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Boolean mask of the lab-frame points inside the box.

    The comparison carries a relative tolerance because a strict `<=` does not
    even admit the box's OWN corners: `corners()` rotates out and `to_imaging`
    rotates back, and the round trip lands a few ulp either side of the half
    width. At `rtol = 1e-12` on a 0.3 m field of view that is 0.3 pm, far below
    anything geometric, and it makes the two functions agree with each other.
    """
    local = self.to_imaging(x)
    half = 0.5 * self.fov
    return np.all(np.abs(local) <= half + rtol * np.maximum(half, 1.0), axis=1)


def parse_triplet(text: str, name: str = 'value') -> np.ndarray:
  """Three numbers from a line of text, separated by spaces or commas.

  The entry fields of a planning panel are free text, so this is where a
  typo becomes a message rather than a traceback or, worse, a plan quietly
  built from two numbers and a default.
  """
  parts = str(text).replace(',', ' ').split()
  if len(parts) != 3:
    raise ValueError(f'{name}: expected three numbers, got {len(parts)}')
  try:
    return np.array([float(p) for p in parts], dtype=float)
  except ValueError as exc:
    raise ValueError(f'{name}: {exc}') from exc


def format_triplet(values) -> str:
  """Three numbers as a line of text, the inverse of `parse_triplet`.

  `%g` with six significant digits: enough that a round trip through the
  entry does not move the plan, short enough to read. A plain `str` would put
  `0.30000000000000004` in front of the user.
  """
  return ' '.join(f'{float(v):.6g}' for v in np.asarray(values).ravel())
