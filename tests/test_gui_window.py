"""The box-widget transform, which is the reason the native window exists.

`vtkBoxWidget` reports what a drag did as a 4x4 transform relative to where the
widget was placed. Turning that back into a `FOVBox` is the only real
arithmetic in the window backend, and it is testable without ever showing a
window: the decomposition is a static method on plain matrices.

Dragging itself is not tested. Driving a VTK interactor from pytest is brittle
and the panel is deliberately thin so that the untested surface is small.
"""
from __future__ import annotations

import numpy as np
import pytest

from feelmri.gui.model.planning import FOVBox, euler_to_mps, mps_to_euler


def decompose(matrix, centre0, extent0, rotation0):
  """The arithmetic `Window3D._on_box` performs, isolated.

  Kept here rather than imported so the test states the contract it is
  checking. `Window3D` needs PyVista to import, and this needs nothing.
  """
  linear = np.asarray(matrix, dtype=float)[:3, :3]
  translation = np.asarray(matrix, dtype=float)[:3, 3]
  scale = np.linalg.norm(linear, axis=0)
  if np.any(scale <= 0):
    return None
  rotation = linear / scale
  if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
    return None                       # sheared: refuse rather than guess
  return FOVBox(fov=extent0 * scale,
                loc=linear @ centre0 + translation,
                angles=np.array(mps_to_euler(rotation)))


def homogeneous(linear=None, translation=(0, 0, 0)):
  m = np.eye(4)
  if linear is not None:
    m[:3, :3] = linear
  m[:3, 3] = translation
  return m


CENTRE0 = np.array([0.04, 0.0, 0.02])
EXTENT0 = np.array([0.30, 0.22, 0.008])
ROT0 = euler_to_mps(0.0, 0.0, np.radians(30.0))


def test_an_untouched_widget_leaves_the_plan_exactly_as_it_was():
  """The identity must be a no-op.

  This is not a formality: PyVista fires the widget callback once while
  PLACING it, with an identity transform. Anything that loses information here
  silently discards the plan the moment the widget appears, which is exactly
  what happened before `_sync_box_widget` suppressed that first call.
  """
  box = decompose(homogeneous(ROT0), CENTRE0, EXTENT0, ROT0)
  np.testing.assert_allclose(box.fov, EXTENT0, atol=1e-12)
  np.testing.assert_allclose(box.loc, ROT0 @ CENTRE0, atol=1e-12)
  np.testing.assert_allclose(np.degrees(box.angles), [0, 0, 30], atol=1e-9)


def test_a_translation_moves_the_centre_and_nothing_else():
  shift = np.array([0.05, -0.02, 0.01])
  box = decompose(homogeneous(ROT0, shift), CENTRE0, EXTENT0, ROT0)
  np.testing.assert_allclose(box.loc, ROT0 @ CENTRE0 + shift, atol=1e-12)
  np.testing.assert_allclose(box.fov, EXTENT0, atol=1e-12)
  np.testing.assert_allclose(np.degrees(box.angles), [0, 0, 30], atol=1e-9)


def test_scaling_a_face_changes_only_that_extent():
  """A face handle scales along the box's OWN axis, so the linear part is
  `R @ diag(s)`, whose column norms are the scale."""
  box = decompose(homogeneous(ROT0 @ np.diag([1.5, 1.0, 1.0])),
                  CENTRE0, EXTENT0, ROT0)
  np.testing.assert_allclose(box.fov, EXTENT0 * [1.5, 1, 1], atol=1e-12)
  np.testing.assert_allclose(np.degrees(box.angles), [0, 0, 30], atol=1e-9)


def test_rotating_the_widget_composes_with_the_plan():
  extra = euler_to_mps(0.0, 0.0, np.radians(20.0))
  box = decompose(homogeneous(extra @ ROT0), CENTRE0, EXTENT0, ROT0)
  np.testing.assert_allclose(np.degrees(box.angles), [0, 0, 50], atol=1e-6)
  np.testing.assert_allclose(box.fov, EXTENT0, atol=1e-12)


def test_a_sheared_transform_is_refused_rather_than_decomposed():
  """Scaling along WORLD axes after a rotation is not rotation times scale.

  The widget cannot produce it, but feeding one through anyway yields a
  plausible and wrong plan: measured, a world-axis 1.5x on a box rotated 30
  degrees came out as 40.89 degrees with a skewed field of view. Refusing is
  the only safe answer, since there is no correct FOVBox to report.
  """
  sheared = np.diag([1.5, 1.0, 1.0]) @ ROT0
  assert decompose(homogeneous(sheared), CENTRE0, EXTENT0, ROT0) is None

  # And the control: the same scale applied in the box frame IS accepted.
  assert decompose(homogeneous(ROT0 @ np.diag([1.5, 1.0, 1.0])),
                   CENTRE0, EXTENT0, ROT0) is not None


def test_a_degenerate_scale_is_refused():
  """A face dragged through itself gives a zero or negative extent."""
  assert decompose(homogeneous(ROT0 @ np.diag([0.0, 1.0, 1.0])),
                   CENTRE0, EXTENT0, ROT0) is None


@pytest.mark.parametrize('tx,ty,tz', [(0.0, 0.0, 0.0), (0.3, -0.7, 1.1),
                                      (np.pi / 2, 0.0, 0.4), (-1.2, 2.5, -0.3)])
def test_any_orientation_round_trips_through_the_widget_transform(tx, ty, tz):
  """Place at one orientation, report it back unchanged.

  Covers the pole, where the Euler decomposition pins `ty = 0` and the angles
  legitimately differ from the input while the MATRIX must not.
  """
  rot = euler_to_mps(tx, ty, tz)
  box = decompose(homogeneous(rot), CENTRE0, EXTENT0, rot)
  np.testing.assert_allclose(box.mps, rot, atol=1e-9)


class _FakeWindow:
  def __init__(self, handle):
    self._handle = handle

  def GetGenericWindowId(self):
    return self._handle


class _FakePlotter:
  """Just enough of a plotter for the liveness check, with no real window."""

  def __init__(self, closed=False, has_window=True, has_handle=True):
    self._closed = closed
    self.render_window = (_FakeWindow(object() if has_handle else None)
                          if has_window else None)


def _is_open(plotter, user_closed=False):
  """The check `Window3D._is_open` performs, isolated from the class.

  Stated here rather than imported because `Window3D` needs PyVista to import
  and this needs nothing.
  """
  if plotter is None or user_closed:
    return False
  if getattr(plotter, '_closed', False):
    return False
  window = getattr(plotter, 'render_window', None)
  if window is None:
    return False
  try:
    return window.GetGenericWindowId() is not None
  except Exception:
    return False


@pytest.mark.parametrize('plotter, user_closed, expected, why', [
  (None, False, False, 'no plotter at all'),
  (_FakePlotter(), False, True, 'healthy window'),
  (_FakePlotter(closed=True), False, False, 'plotter.close() sets _closed'),
  (_FakePlotter(has_window=False), False, False, 'plotter.close() nulls it'),
  (_FakePlotter(has_handle=False), False, False, 'WINDOW MANAGER destroyed it'),
  (_FakePlotter(), True, False, 'the interactor fired ExitEvent'),
])
def test_every_way_the_3d_window_can_close_is_detected(plotter, user_closed,
                                                       expected, why):
  """The ways of closing look completely different, and one was missed twice.

  A first attempt caught exceptions around `plotter.update()`, assuming a dead
  window would raise. It does not: after `close()`, three successive `update()`
  calls returned normally.

  A second attempt checked `_closed` and `render_window is None`. Both are set
  by `plotter.close()`, which is what the test drove, and **neither is set when
  the window manager destroys the window**, which is what a user does.
  Measured after the window manager took it: `_closed` still False,
  `render_window` and `iren` both live, `update()` and `render()` both
  returning normally. The application still broke on the next plan change.

  What does change is the native handle: `GetGenericWindowId()` goes from a
  pointer to None. That is polled, with an `ExitEvent` observer for the
  title-bar button. Each row below names which of the three paths it stands
  for, so the one that was missed cannot be dropped as a duplicate of one that
  was not.
  """
  assert _is_open(plotter, user_closed) is expected, why
