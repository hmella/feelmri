"""The trackball behind the blitting viewport.

Blitting means VTK renders offscreen and never sees a mouse, so the interactor
that would orbit, pan and dolly has to be written by hand. This is the part of
the viewport most likely to be subtly wrong and the only part testable without
a display, which is why it lives in the model layer.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from feelmri.gui.model.camera import STANDARD_VIEWS, Camera

LO, HI = np.array([-1.0, -2.0, -0.5]), np.array([3.0, 2.0, 1.5])


@pytest.fixture
def cam():
  return Camera.frame(LO, HI)


def test_framing_centres_on_the_box_and_stands_off_it(cam):
  centre = 0.5 * (LO + HI)
  np.testing.assert_allclose(cam.focal_point, centre, atol=1e-12)
  diagonal = float(np.linalg.norm(HI - LO))
  assert cam.distance == pytest.approx(2.0 * diagonal)
  assert cam.reference_distance == pytest.approx(2.0 * diagonal)


def test_framing_survives_a_degenerate_box():
  """A planar mesh has a zero extent on one axis and must still frame."""
  flat = Camera.frame([0, 0, 0], [1, 1, 0])
  assert flat.distance > 0
  point = Camera.frame([2, 2, 2], [2, 2, 2])      # a single point
  assert point.distance > 0 and np.all(np.isfinite(point.position))


def test_the_frame_vectors_are_orthonormal(cam):
  f, r, u = cam.forward, cam.right, cam.up
  for v in (f, r, u):
    assert np.linalg.norm(v) == pytest.approx(1.0)
  assert float(np.dot(f, r)) == pytest.approx(0.0, abs=1e-12)
  assert float(np.dot(f, u)) == pytest.approx(0.0, abs=1e-12)
  assert float(np.dot(r, u)) == pytest.approx(0.0, abs=1e-12)
  # Right-handed: right x up points back towards the camera.
  np.testing.assert_allclose(np.cross(r, u), -f, atol=1e-12)


def test_orbit_keeps_the_focal_point_and_the_distance(cam):
  """An orbit is a rotation about the target, so only the direction moves."""
  moved = cam.orbit(0.4, -0.25)
  np.testing.assert_allclose(moved.focal_point, cam.focal_point, atol=1e-12)
  assert moved.distance == pytest.approx(cam.distance, rel=1e-12)
  assert not np.allclose(moved.position, cam.position), 'the orbit did nothing'


def test_each_orbit_axis_reverses_exactly(cam):
  """Dragging back must return, or an interaction drifts.

  Per axis, because the two rotations do not commute: negating both components
  of a combined orbit is not its inverse, and asserting that it is, as a first
  version of this test did, only measures the non-commutation.
  """
  np.testing.assert_allclose(cam.orbit(0.37, 0).orbit(-0.37, 0).position,
                             cam.position, atol=1e-12)
  np.testing.assert_allclose(cam.orbit(0, 0.21).orbit(0, -0.21).position,
                             cam.position, atol=1e-12)
  # And in the opposite order, a combined orbit does reverse.
  there = cam.orbit(0.37, 0).orbit(0, 0.21)
  back = there.orbit(0, -0.21).orbit(-0.37, 0)
  np.testing.assert_allclose(back.position, cam.position, atol=1e-12)


def test_the_horizon_stays_level_through_a_long_drag(cam):
  """A turntable keeps the world up axis fixed; a free trackball would not."""
  spun = cam
  for _ in range(40):
    spun = spun.orbit(0.3, 0.13)
  np.testing.assert_allclose(spun.view_up, cam.view_up, atol=1e-12)
  assert float(np.dot(spun.up, cam.view_up)) > 0, 'the view rolled over'


def test_a_full_turn_in_azimuth_returns(cam):
  turned = cam
  for _ in range(8):
    turned = turned.orbit(math.pi / 4, 0.0)
    assert turned.distance == pytest.approx(cam.distance, rel=1e-12)
  np.testing.assert_allclose(turned.position, cam.position, atol=1e-9)


def test_elevation_is_clamped_short_of_the_pole(cam):
  """At the pole `right` collapses and the next drag jumps.

  Driving far past the pole in one step and then many steps must both leave a
  usable frame, which is what the guard exists for.
  """
  for push in (10.0, -10.0):
    extreme = cam.orbit(0.0, push)
    angle = math.acos(float(np.clip(np.dot(extreme.forward, -extreme.up), -1, 1)))
    assert 0.0 < angle < math.pi, 'the view direction reached the pole'
    # The frame is still well-conditioned: right must not have collapsed.
    assert np.linalg.norm(np.cross(extreme.forward, extreme.up)) > 1e-3

  spun = cam
  for _ in range(50):
    spun = spun.orbit(0.0, 0.5)
  assert np.all(np.isfinite(spun.position))
  assert np.linalg.norm(np.cross(spun.forward, spun.up)) > 1e-3


def test_pan_moves_both_ends_so_the_view_direction_is_unchanged(cam):
  moved = cam.pan(0.2, -0.1)
  np.testing.assert_allclose(moved.forward, cam.forward, atol=1e-12)
  assert moved.distance == pytest.approx(cam.distance, rel=1e-12)
  shift = moved.position - cam.position
  np.testing.assert_allclose(moved.focal_point - cam.focal_point, shift,
                             atol=1e-12)
  # It moves in the screen plane, not along the view.
  assert float(np.dot(shift, cam.forward)) == pytest.approx(0.0, abs=1e-12)


def test_pan_scales_with_distance_so_a_drag_covers_the_same_screen_fraction(cam):
  """Otherwise panning is unusable zoomed out and glacial zoomed in."""
  near = cam.dolly(0.25)
  far = cam.dolly(4.0)
  d_near = np.linalg.norm(near.pan(0.1, 0).position - near.position)
  d_far = np.linalg.norm(far.pan(0.1, 0).position - far.position)
  assert d_far > d_near
  assert d_far / d_near == pytest.approx(far.distance / near.distance, rel=1e-9)


def test_pan_is_reversible(cam):
  back = cam.pan(0.3, 0.15).pan(-0.3, -0.15)
  np.testing.assert_allclose(back.position, cam.position, atol=1e-9)
  np.testing.assert_allclose(back.focal_point, cam.focal_point, atol=1e-9)


def test_dolly_moves_along_the_view_and_is_clamped(cam):
  closer = cam.dolly(0.5)
  assert closer.distance == pytest.approx(0.5 * cam.distance)
  np.testing.assert_allclose(closer.focal_point, cam.focal_point, atol=1e-12)
  np.testing.assert_allclose(closer.forward, cam.forward, atol=1e-12)

  # A scroll run must not land inside the mesh or at infinity.
  deep = cam
  for _ in range(200):
    deep = deep.dolly(0.5)
  assert deep.distance > 0 and np.all(np.isfinite(deep.position))
  far = cam
  for _ in range(200):
    far = far.dolly(2.0)
  assert far.distance <= 1e4 * cam.reference_distance * (1 + 1e-9)

  with pytest.raises(ValueError, match='must be positive'):
    cam.dolly(0.0)
  with pytest.raises(ValueError, match='must be positive'):
    cam.dolly(-1.0)


def test_roll_spins_about_the_view_without_moving_the_camera(cam):
  rolled = cam.roll(0.7)
  np.testing.assert_allclose(rolled.position, cam.position, atol=1e-12)
  np.testing.assert_allclose(rolled.forward, cam.forward, atol=1e-12)
  assert not np.allclose(rolled.up, cam.up), 'the roll did nothing'
  # A full turn returns.
  full = cam
  for _ in range(4):
    full = full.roll(math.pi / 2)
  np.testing.assert_allclose(full.up, cam.up, atol=1e-9)


@pytest.mark.parametrize('name', sorted(STANDARD_VIEWS))
def test_the_standard_views_look_along_their_axis(cam, name):
  direction, up = STANDARD_VIEWS[name]
  view = cam.look_along(direction, up)
  np.testing.assert_allclose(view.forward, np.asarray(direction, float),
                             atol=1e-12)
  assert view.distance == pytest.approx(cam.distance, rel=1e-12)
  np.testing.assert_allclose(view.focal_point, cam.focal_point, atol=1e-12)
  # The frame stays usable, which is the thing a parallel up would break.
  assert np.linalg.norm(np.cross(view.forward, view.up)) > 0.5


def test_a_view_direction_parallel_to_up_still_gives_a_frame(cam):
  """Looking straight down the up axis has no natural right vector."""
  view = cam.look_along((0, 0, -1), (0, 0, 1))
  assert np.all(np.isfinite(view.right))
  assert np.linalg.norm(np.cross(view.forward, view.up)) > 0.5

  framed = Camera.frame(LO, HI, view_up=(0, 0, 1), direction=(0, 0, 1))
  assert np.all(np.isfinite(framed.right))


def test_as_vtk_is_the_triple_the_render_window_takes(cam):
  pos, foc, up = cam.as_vtk()
  assert len(pos) == len(foc) == len(up) == 3
  np.testing.assert_allclose(pos, cam.position, atol=1e-12)
  np.testing.assert_allclose(foc, cam.focal_point, atol=1e-12)
  # The up handed over is the re-orthogonalised one, not the stored vector.
  np.testing.assert_allclose(up, cam.up, atol=1e-12)
  assert float(np.dot(np.asarray(up) - cam.focal_point * 0, cam.forward)) \
      == pytest.approx(0.0, abs=1e-12)


def test_a_degenerate_camera_is_refused():
  with pytest.raises(ValueError, match='coincide'):
    Camera(position=[1, 1, 1], focal_point=[1, 1, 1], view_up=[0, 0, 1])
  with pytest.raises(ValueError, match='zero-length'):
    Camera(position=[0, 0, 0], focal_point=[1, 0, 0], view_up=[0, 0, 0])


def test_operations_do_not_mutate(cam):
  """Every operation returns a new camera, so an interaction can be replayed."""
  before = cam.position.copy()
  cam.orbit(0.3, 0.2)
  cam.pan(0.1, 0.1)
  cam.dolly(0.5)
  cam.roll(0.4)
  np.testing.assert_array_equal(cam.position, before)
