"""The box-widget transform, which is the reason the native window exists.

`vtkBoxWidget` reports what a drag did as a 4x4 transform relative to where the
widget was placed. Turning that back into a `FOVBox` is the only real
arithmetic in the window backend, and it is testable without ever showing a
window: the decomposition is a static method on plain matrices.

Dragging itself is not tested. Driving a VTK interactor from pytest is brittle
and the panel is deliberately thin so that the untested surface is small.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from feelmri.gui.model.planning import FOVBox, euler_to_mps, mps_to_euler
from feelmri.gui.view.window3d import Window3D


def decompose(matrix, centre0, extent0, rotation0):
  """The arithmetic `Window3D._on_box` performs, isolated.

  Kept here rather than imported because `_on_box` is an instance method
  wired into a live session and a widget; this is the arithmetic inside it.
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
  """Drive the REAL `Window3D._is_open` against a stub holding no window.

  A copy of a liveness check cannot fail when the real one is wrong, which is
  exactly the defect class this test exists for -- and the check has already
  been wrong twice. `Window3D` imports with pyvista, vtk and tkinter all
  blocked, because the window is built lazily, so no display is involved.
  """
  stub = Window3D.__new__(Window3D)
  stub._plotter = plotter
  stub._user_closed = user_closed
  return stub._is_open()


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


class _Note:
  def __init__(self):
    self.text = ''

  def config(self, text):
    self.text = text


class _Interactor:
  """The raw VTK interactor, which is what flushes the destroy request."""

  def __init__(self):
    self.processed = 0

  def ProcessEvents(self):
    self.processed += 1


class _Iren:
  def __init__(self):
    self.interactor = _Interactor()


class _ClosablePlotter:
  """Records the teardown, and drops `iren` on close exactly as PyVista does.

  That last detail is the whole point: `Plotter.close()` sets `self.iren` to
  None, so anything reaching for the interactor afterwards gets
  `AttributeError` on a NoneType.
  """

  def __init__(self, raises=False):
    self.closed = 0
    self._raises = raises
    self.iren = _Iren()
    self.order = []

  def close(self):
    self.closed += 1
    self.order.append('close')
    self.iren = None            # PyVista does this, and it is the trap
    if self._raises:
      raise RuntimeError('already gone')


def _viewport(plotter):
  """A `Window3D` with only the attributes `_went_away` touches.

  The real method is driven, not a copy of it, so this fails if the teardown
  is removed. `Window3D` imports with pyvista, vtk and tkinter all blocked --
  the window is built lazily -- so no display is involved.
  """
  stub = Window3D.__new__(Window3D)
  stub.alive = True
  stub._plotter = plotter
  stub._box_widget = object()
  stub._actor = object()
  stub._overlay = [object()]
  stub._note = _Note()
  stub.on_status = lambda _msg: None
  return stub


def test_a_closed_window_is_destroyed_and_not_merely_flagged():
  """Marking it closed is not enough; the window has to go.

  `ExitEvent` signals intent and nothing more. With `interactive_update=True`
  there is no interactor loop for it to terminate, so a title-bar click used
  to leave a window that was still mapped while the pump had stopped
  servicing it -- a frozen window, which is worse than either a live one or
  none at all.

  Measured through the real shell with the `close()` call removed: after the
  title-bar path the render window still reported a live native handle, so
  the window stayed on screen unserviced. With it, the handle is gone.
  """
  plotter = _ClosablePlotter()
  stub = _viewport(plotter)
  stub._went_away()

  assert plotter.closed == 1, 'the window must actually be destroyed'
  assert stub.alive is False
  assert stub._box_widget is None and stub._actor is None
  assert stub._overlay == []
  assert 'closed' in stub._note.text.lower()


def test_the_teardown_runs_once_however_many_signals_arrive():
  """Three paths watch for a close and any of them can fire first.

  The poll, the `ExitEvent` observer and an explicit close all funnel here, so
  without the guard a title-bar click would call `close()` again on every
  subsequent pump.
  """
  plotter = _ClosablePlotter()
  stub = _viewport(plotter)
  for _ in range(5):
    stub._went_away()
  assert plotter.closed == 1


def test_a_plotter_that_refuses_to_close_still_leaves_the_viewport_closed():
  """`close()` raising is the normal case when VTK got there first.

  It must not propagate: `_went_away` is called from a Tk timer callback, and
  an exception there kills the pump, which is the failure it exists to
  prevent.
  """
  stub = _viewport(_ClosablePlotter(raises=True))
  stub._went_away()
  assert stub.alive is False
  assert 'closed' in stub._note.text.lower()


def test_a_viewport_with_no_plotter_closes_cleanly():
  """`reopen` can fail before a plotter exists, and the poll still runs."""
  stub = _viewport(None)
  stub._went_away()
  assert stub.alive is False


def test_the_destroy_request_is_flushed_or_the_window_stays_on_screen():
  """`close()` does not remove the window. The flush after it does.

  VTK destroys a window by queueing an `XDestroyWindow` on its own Xlib
  connection, and that request only reaches the server when something services
  that connection -- which is precisely what the pump has just stopped doing.

  Measured on a bare `pyvista.Plotter`, no tkinter involved: after `close()`
  the window reports `Map State: IsViewable` indefinitely. `Finalize()`,
  `terminate_app()`, `SetShowWindow(False)`, switching to offscreen rendering
  and dropping the last Python reference all leave it on screen. One
  `ProcessEvents()` removes it. Confirmed end to end by sending a real
  `WM_DELETE_WINDOW` -- what the title-bar button sends -- to the running
  shell: window still mapped before, gone after.
  """
  plotter = _ClosablePlotter()
  interactor = plotter.iren.interactor
  stub = _viewport(plotter)
  stub._went_away()

  assert plotter.closed == 1
  assert interactor.processed == 1, 'the queued destroy was never flushed'


def test_the_interactor_is_captured_before_close_drops_it():
  """Order is load-bearing, and getting it wrong looks like a VTK bug.

  `Plotter.close()` sets `plotter.iren` to None, so reading the interactor
  after closing raises `AttributeError: 'NoneType' object has no attribute
  ...` and the flush silently never happens -- which is how this was first
  misread as VTK being unable to destroy its own window.
  """
  plotter = _ClosablePlotter()
  stub = _viewport(plotter)
  stub._went_away()

  assert plotter.iren is None, 'the fake must drop iren the way PyVista does'
  # It still got flushed, so it cannot have been read after the close.
  assert stub._plotter.closed == 1


def test_a_plotter_that_refuses_to_close_is_still_flushed():
  """`close()` raising must not skip the flush.

  VTK having got there first is the normal case, and the window can still be
  mapped when it does.
  """
  plotter = _ClosablePlotter(raises=True)
  interactor = plotter.iren.interactor
  stub = _viewport(plotter)
  stub._went_away()

  assert interactor.processed == 1
  assert stub.alive is False


def test_a_flush_that_raises_does_not_kill_the_pump():
  """This runs from a Tk timer callback, where an exception stops the pump."""
  plotter = _ClosablePlotter()
  plotter.iren.interactor.ProcessEvents = lambda: (_ for _ in ()).throw(
    RuntimeError('interactor gone'))
  stub = _viewport(plotter)
  stub._went_away()                     # must not raise
  assert stub.alive is False


def test_quitting_destroys_the_window_too():
  """`close()` is the quit path and needs the same flush as a user close.

  Without it the render window outlives the shell whenever the process does
  not exit immediately afterwards -- measured: after `Shell.close()` the X
  window was still on screen. Both paths go through one destroy so they
  cannot drift apart again.
  """
  plotter = _ClosablePlotter()
  interactor = plotter.iren.interactor
  stub = _viewport(plotter)
  Window3D.close(stub)

  assert plotter.closed == 1
  assert interactor.processed == 1, 'the quit path skipped the flush'
  assert stub.alive is False


# -- the drawn outline, which is what a user actually places ----------------

WIDGET_GEOMETRY_PROBE = '''
import sys
import numpy as np
import tkinter as tk
import vtk

from feelmri.gui.model.planning import FOVBox
from feelmri.gui.model.session import Session
from feelmri.gui.view.window3d import Window3D

FOV = np.array([0.180, 0.140, 0.030])
root = tk.Tk()
root.withdraw()
session = Session()
view = Window3D(root, session, embed=False)
try:
  for angles in ([0.0, 0.0, 0.0], [25.0, 0.0, 15.0]):
    box = FOVBox(fov=FOV, loc=np.array([0.01, -0.02, 0.03]),
                 angles=np.radians(angles))
    session.box = box
    polydata = vtk.vtkPolyData()
    view._box_widget.GetPolyData(polydata)
    points = np.array([polydata.GetPoint(i)
                       for i in range(polydata.GetNumberOfPoints())])
    corners = points[:8]
    local = (corners - corners.mean(axis=0)) @ box.mps
    drawn = local.max(axis=0) - local.min(axis=0)
    print(' '.join(f'{v:.9f}' for v in drawn / FOV),
          ' '.join(f'{v:.9f}' for v in corners.mean(axis=0) - box.loc))
finally:
  view.close()
  root.destroy()
'''


@pytest.mark.slow
def test_the_drawn_box_is_the_planned_box(tmp_path):
  """The outline the user drags must BE the field of view, not a proxy for it.

  PyVista's `add_box_widget` defaults `factor=1.25` and hands it to
  `SetPlaceFactor`, which inflates the placed box about its centre. Measured
  before this was pinned: a plan of 0.180 x 0.140 x 0.030 m was drawn as
  0.225 x 0.175 x 0.0375, exactly 1.25 on every axis and at every orientation.

  **Nothing else could see it.** The decomposition reads the widget's
  transform RELATIVE to where it was placed, so the plan round-tripped
  perfectly and every test of that arithmetic passed; what was wrong was only
  the handle. Dragging it onto an anatomical edge therefore produced a field
  of view 20% smaller per axis than the one drawn.

  Run in a subprocess, like the other tests that need a real window: this
  project has already measured that building more than one Tk+VTK pair in one
  process segfaults inside `ProcessEvents`.
  """
  if not os.environ.get('DISPLAY'):
    pytest.skip('needs a display: the widget geometry only exists once VTK '
                'has a render window')
  pytest.importorskip('pyvista')

  script = tmp_path / 'widget_geometry.py'
  script.write_text(WIDGET_GEOMETRY_PROBE)
  done = subprocess.run([sys.executable, str(script)], capture_output=True,
                        text=True, timeout=120)
  assert done.returncode == 0, done.stderr[-2000:]

  rows = [line.split() for line in done.stdout.strip().splitlines() if line]
  assert len(rows) == 2, f'expected one row per orientation, got {done.stdout!r}'
  for row in rows:
    ratio = np.array([float(v) for v in row[:3]])
    offset = np.array([float(v) for v in row[3:]])
    assert np.allclose(ratio, 1.0, atol=1e-6), (
      f'the outline is {ratio} times the plan, not the plan')
    assert np.allclose(offset, 0.0, atol=1e-9), (
      f'the outline is centred {offset} away from the plan')


REPLACEMENT_PROBE = '''
import sys
import numpy as np
import tkinter as tk

from feelmri.gui.model.planning import FOVBox
from feelmri.gui.model.session import Session
from feelmri.gui.view.window3d import Window3D

# A real cine, because a session with NO MESH returns early from `rebuild`
# and never reaches the widget at all -- which is how the first version of
# this test read 0 whether the defect was present or not.
root = tk.Tk()
root.withdraw()
session = Session()
session.load_mesh(sys.argv[1])
assert session.n_frames > 6, session.n_frames

view = Window3D(root, session, embed=False)
try:
  session.box = FOVBox(fov=np.array([0.18, 0.14, 0.03]),
                       loc=np.array([0.0, 0.0, 0.0]),
                       angles=np.radians([30.0, 0.0, 20.0]))

  placements = []
  original = view._sync_box_widget
  view._sync_box_widget = lambda: (placements.append(1), original())[1]

  # Six frame steps, which is what the play button does. The plan does not
  # move, so the widget has no reason to be rebuilt.
  for _ in range(6):
    session.step_frame(1)
  steps = len(placements)

  # A real plan change must still reach it.
  session.box = FOVBox(fov=np.array([0.10, 0.20, 0.05]),
                       loc=np.array([0.01, 0.0, 0.0]),
                       angles=np.radians([10.0, 0.0, 0.0]))
  print(steps, len(placements) - steps)
finally:
  view.close()
  root.destroy()
'''


@pytest.mark.slow
def test_playing_a_cine_does_not_rebuild_the_field_of_view(tmp_path):
  """Stepping the frame must leave the box widget alone.

  `refresh_overlay` ran `_sync_box_widget` on every redraw, and that call
  DESTROYS the widget and builds a new one -- so playing a cine did it once
  per frame. Measured on the 30-frame aorta before this: 6 frame steps, 6
  re-placements, and the outline missing from exactly one render per tick,
  which reads as the field of view strobing.

  The box never actually MOVED -- 0 of 90 renders had it away from its
  resting pose -- so this asserts the rebuild count rather than the pose.

  **The probe needs a real time series.** A bare `Session` with `n_frames`
  set by hand has no mesh, `rebuild` returns early, and a frame step never
  reaches the widget: the first version of this test read 0 either way and
  would have passed with the defect in place.
  """
  if not os.environ.get('DISPLAY'):
    pytest.skip('needs a display: the box widget is a VTK object')
  pytest.importorskip('pyvista')
  phantom = (pathlib.Path(__file__).resolve().parent.parent / 'examples'
             / 'phantoms' / 'heart_P1_hex.xdmf')
  if not phantom.exists():
    pytest.skip(f'{phantom.name} not present')

  script = tmp_path / 'replacement.py'
  script.write_text(REPLACEMENT_PROBE)
  done = subprocess.run([sys.executable, str(script), str(phantom)],
                        capture_output=True, text=True, timeout=120)
  assert done.returncode == 0, done.stderr[-2000:]

  on_frames, on_plan_change = (int(v) for v in done.stdout.split())
  assert on_frames == 0, (
    f'six frame steps rebuilt the field of view {on_frames} times')
  assert on_plan_change >= 1, (
    'the widget stopped following a plan change, which is what it is for')
