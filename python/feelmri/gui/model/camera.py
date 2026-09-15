"""A hand-rolled trackball, so the blitting viewport has a camera.

Blitting means VTK renders offscreen and never sees a mouse, so the interactor
that would normally orbit, pan and dolly does not exist. This is that
interactor, written as pure arithmetic on a camera triple.

Keeping it here rather than in the view has one concrete benefit: it is the
part most likely to be subtly wrong, and here it can be tested without a
display. The view layer is then only glue.

Every operation returns a NEW `Camera`. Nothing mutates, so an interaction can
be replayed, undone or compared against its starting point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

#: How close to the pole an orbit may take the view direction, in radians.
#: At exactly the pole the up vector and the view direction are parallel and
#: the right vector collapses, which makes the next drag jump.
_POLE_GUARD = 1e-3

#: Dolly limits, as a multiple of the framing distance. Without them a scroll
#: run either puts the camera inside the mesh or loses it at infinity, and
#: neither is recoverable by scrolling back.
_MIN_DISTANCE_FACTOR = 1e-3
_MAX_DISTANCE_FACTOR = 1e4


def _unit(v: np.ndarray) -> np.ndarray:
  n = float(np.linalg.norm(v))
  if n == 0.0:
    raise ValueError('cannot normalise a zero-length vector')
  return v / n


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
  """Rodrigues rotation of `v` about a unit `axis` by `angle` radians."""
  c, s = math.cos(angle), math.sin(angle)
  return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


@dataclass(frozen=True)
class Camera:
  """Position, what it looks at, and which way is up.

  The same triple VTK uses, so handing it to a render window is a direct
  assignment with no conversion to get wrong.
  """

  position: np.ndarray
  focal_point: np.ndarray
  view_up: np.ndarray
  #: The distance the framing chose, kept so dolly limits mean something.
  reference_distance: float = 1.0

  def __post_init__(self):
    object.__setattr__(self, 'position',
                       np.asarray(self.position, dtype=np.float64).reshape(3))
    object.__setattr__(self, 'focal_point',
                       np.asarray(self.focal_point, dtype=np.float64).reshape(3))
    object.__setattr__(self, 'view_up',
                       _unit(np.asarray(self.view_up, dtype=np.float64).reshape(3)))
    if self.distance == 0.0:
      raise ValueError('Camera: position and focal_point coincide')

  # -- derived frame --------------------------------------------------------

  @property
  def distance(self) -> float:
    return float(np.linalg.norm(self.position - self.focal_point))

  @property
  def forward(self) -> np.ndarray:
    """Unit vector from the camera towards what it is looking at."""
    return _unit(self.focal_point - self.position)

  @property
  def right(self) -> np.ndarray:
    """Screen-right in world coordinates."""
    return _unit(np.cross(self.forward, self.view_up))

  @property
  def up(self) -> np.ndarray:
    """Screen-up, re-orthogonalised against the view direction.

    `view_up` is whatever was last set and need not be perpendicular to the
    view; this is the vector that actually points up on screen.
    """
    return _unit(np.cross(self.right, self.forward))

  # -- framing --------------------------------------------------------------

  @classmethod
  def frame(cls, lo: Sequence[float], hi: Sequence[float],
            view_up: Sequence[float] = (0.0, 0.0, 1.0),
            direction: Sequence[float] = (1.0, -1.0, 0.5),
            margin: float = 2.0) -> 'Camera':
    """Place a camera so an axis-aligned box fills the view.

    A degenerate box, which a planar mesh produces, still has to give a usable
    distance, so the diagonal falls back to the largest non-zero extent and
    then to 1.
    """
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    centre = 0.5 * (lo + hi)
    diagonal = float(np.linalg.norm(hi - lo))
    if diagonal == 0.0:
      extent = float(np.max(np.abs(hi - lo)))
      diagonal = extent if extent > 0 else 1.0
    distance = margin * diagonal

    d = _unit(np.asarray(direction, dtype=np.float64).reshape(3))
    up = np.asarray(view_up, dtype=np.float64).reshape(3)
    # A view direction parallel to up has no right vector; nudge the up.
    if abs(float(np.dot(_unit(up), d))) > 1.0 - 1e-6:
      up = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return cls(position=centre - d * distance, focal_point=centre,
               view_up=up, reference_distance=distance)

  # -- interaction ----------------------------------------------------------

  def orbit(self, d_azimuth: float, d_elevation: float) -> 'Camera':
    """Rotate about the focal point. Angles in radians.

    This is a TURNTABLE, not a free trackball: `view_up` is a world axis that
    never moves, so the horizon stays level however far the view is dragged.
    Azimuth turns about that axis and elevation about screen-right.

    Because the axis is fixed, each drag axis reverses exactly: `orbit(a, 0)`
    then `orbit(-a, 0)` returns to the start, and likewise for elevation. A
    COMBINED orbit does not reverse by negating both, since the two rotations
    do not commute; undo it in the opposite order.

    Elevation is clamped short of the pole. There the up axis and the view
    direction are parallel, `right` collapses, and the next drag would jump to
    an arbitrary orientation.
    """
    offset = self.position - self.focal_point
    # The STORED axis, not the re-orthogonalised `self.up`. Re-deriving it
    # every call lets the azimuth axis drift with the view, which stops the
    # rotation being reversible and slowly tilts the horizon.
    axis = _unit(self.view_up)

    if d_azimuth:
      offset = _rotate(offset, axis, d_azimuth)

    if d_elevation:
      forward = _unit(-offset)
      right = np.cross(forward, axis)
      norm = float(np.linalg.norm(right))
      if norm > 1e-12:
        right = right / norm
        # Angle between the view direction and the up axis, before and after.
        current = math.acos(float(np.clip(np.dot(forward, axis), -1.0, 1.0)))
        lo, hi = _POLE_GUARD, math.pi - _POLE_GUARD
        d_elevation = float(np.clip(current + d_elevation, lo, hi)) - current
        if d_elevation:
          offset = _rotate(offset, right, d_elevation)

    return Camera(self.focal_point + offset, self.focal_point, self.view_up,
                  self.reference_distance)

  def pan(self, dx: float, dy: float) -> 'Camera':
    """Slide the view. `dx`/`dy` are fractions of the viewport.

    The world distance moved scales with how far away the camera is, so a drag
    covers the same fraction of the screen whatever the zoom. Without that,
    panning is unusably fast when zoomed out and unusably slow when zoomed in.
    """
    shift = (-dx * self.right + dy * self.up) * self.distance
    return Camera(self.position + shift, self.focal_point + shift,
                  self.view_up, self.reference_distance)

  def dolly(self, factor: float) -> 'Camera':
    """Move towards (`factor` < 1) or away from (`> 1`) the focal point.

    Clamped either side of the framing distance, because a scroll run that
    lands the camera inside the mesh or at infinity cannot be scrolled back.
    """
    if not factor > 0:
      raise ValueError(f'Camera.dolly: factor must be positive, got {factor}')
    offset = self.position - self.focal_point
    new = float(np.linalg.norm(offset)) * factor
    new = float(np.clip(new,
                        _MIN_DISTANCE_FACTOR * self.reference_distance,
                        _MAX_DISTANCE_FACTOR * self.reference_distance))
    return Camera(self.focal_point + _unit(offset) * new, self.focal_point,
                  self.view_up, self.reference_distance)

  def roll(self, angle: float) -> 'Camera':
    """Spin about the view direction."""
    return Camera(self.position, self.focal_point,
                  _rotate(self.up, self.forward, angle),
                  self.reference_distance)

  def look_along(self, direction: Sequence[float],
                 view_up: Sequence[float] = (0.0, 0.0, 1.0)) -> 'Camera':
    """Jump to a named view, keeping the focal point and the distance."""
    d = _unit(np.asarray(direction, dtype=np.float64).reshape(3))
    up = np.asarray(view_up, dtype=np.float64).reshape(3)
    if abs(float(np.dot(_unit(up), d))) > 1.0 - 1e-6:
      up = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return Camera(self.focal_point - d * self.distance, self.focal_point, up,
                  self.reference_distance)

  # -- handing it to VTK ----------------------------------------------------

  def as_vtk(self) -> Tuple[Tuple[float, float, float],
                            Tuple[float, float, float],
                            Tuple[float, float, float]]:
    """`(position, focal_point, view_up)`, the shape `camera_position` takes."""
    return (tuple(self.position), tuple(self.focal_point), tuple(self.up))


#: Named views, as (direction the camera looks along, up).
STANDARD_VIEWS = {
  'axial': ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
  'coronal': ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
  'sagittal': ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}
