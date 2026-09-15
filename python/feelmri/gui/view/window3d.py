"""The 3D view as a native VTK window, driven from the Tk event loop.

This is the default viewport. The alternative, `canvas3d.Canvas3D`, renders
offscreen and blits the frame into a Tk canvas; both expose the same methods so
the shell can hold either.

**Why this one.** Measured on an RTX 4070 Ti at 880x731, blitting runs at 45
fps and a native window at 179. Both are interactive, so speed is not the
reason. The reason is that every 3D widget drives through a real
`vtkRenderWindowInteractor`: `add_box_widget`, `add_plane_widget`,
`enable_cell_picking`. A blitted canvas has no interactor, so a draggable field
of view has to be hand written, and the field of view is the planning module.

**The cost** is that the render window is a separate top-level window rather
than a panel inside the shell. It can be moved behind, and closing it must not
take the application with it, so `pump` watches for that and the shell offers
to reopen.

**Two event loops.** Tk owns `mainloop`; the plotter is shown with
`interactive_update=True`, which returns immediately, and `pump` calls
`plotter.update()` from a Tk timer. Never call `plotter.show()` without that
flag here: it blocks, and the shell stops responding.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..model.camera import STANDARD_VIEWS, Camera
from ..model.planning import FOVBox, mps_to_euler

#: Tk timer interval for pumping VTK, in ms. 16 is about 60 Hz.
PUMP_MS = 16


class Window3D:
  """A native PyVista window showing the session's mesh and plan."""

  def __init__(self, parent, session, on_status: Optional[Callable[[str], None]] = None):
    import tkinter as tk
    from tkinter import ttk

    try:
      import pyvista as pv
    except ImportError as exc:
      raise RuntimeError(
        f'The 3D view needs PyVista and VTK, which are an optional extra: '
        f'pip install "feelmri[gui]" ({exc})') from exc

    self.session = session
    self.on_status = on_status or (lambda _: None)
    self.alive = False
    self._actor = None
    self._overlay = []
    self._box_widget = None
    self._box_reference = None      # (centre, extent) the widget was placed at
    self._user_closed = False       # set by the ExitEvent observer
    self._suppress = False          # guard against a feedback loop

    # The shell still needs something to pack, so the panel is a placeholder
    # that reports state and can bring the window back.
    self.widget = ttk.Frame(parent, padding=16)
    ttk.Label(self.widget, text='3D view',
              font=('TkDefaultFont', 11, 'bold')).pack(anchor='w')
    self._note = ttk.Label(self.widget, wraplength=420, justify='left', text=(
      'The 3D view is a separate window. Drag the yellow box to place the '
      'field of view; the plan and the element count follow it.'))
    self._note.pack(anchor='w', pady=(6, 12))
    ttk.Button(self.widget, text='Reopen 3D window',
               command=self.reopen).pack(anchor='w')

    self._pv = pv
    self._plotter = None
    self._open()

    session.mesh_changed.connect(lambda *_: self.rebuild())
    session.plan_changed.connect(lambda *_: self.refresh_overlay())
    session.view_changed.connect(lambda *_: self.rebuild(keep_camera=True))

  # -- window lifetime ------------------------------------------------------

  def _open(self) -> None:
    self._plotter = self._pv.Plotter(window_size=(900, 750),
                                     title='feelmri  3D view')
    self._plotter.set_background('#1a1a1a')
    self._plotter.add_axes()
    # Returns immediately. Without interactive_update this blocks and the Tk
    # shell stops responding.
    self._plotter.show(interactive_update=True, auto_close=False)

    # The title-bar button routes through the interactor's ExitEvent. Polling
    # the native handle catches the rest, so both are wired.
    self._user_closed = False
    try:
      self._plotter.iren.interactor.AddObserver(
        'ExitEvent', lambda *_: setattr(self, '_user_closed', True))
    except Exception:
      pass                      # polling alone still covers it
    self.alive = True

  def reopen(self) -> None:
    if self.alive and self._is_open():
      return
    self.alive = False
    self._user_closed = False
    self._actor = None
    self._overlay = []
    self._box_widget = None
    self._open()
    self.rebuild()
    self.on_status('3D window reopened')

  def _is_open(self) -> bool:
    """Whether the render window still exists.

    **Three signals, because the two ways of closing look different.**

    `plotter.close()` nulls `render_window` and sets `_closed`. Closing the
    window from its title bar does NEITHER: measured after the window manager
    destroyed it, `_closed` was still False, `render_window` and `iren` were
    both live, and `update()` and `render()` both returned normally. A check
    built on those alone is blind to the case a user actually hits.

    What does change is the native handle: `GetGenericWindowId()` goes from a
    pointer to None and `IsCurrent()` from True to False. That is polled here,
    alongside an `ExitEvent` observer, which is the event the interactor fires
    when the title-bar button is used.
    """
    if self._plotter is None or self._user_closed:
      return False
    if getattr(self._plotter, '_closed', False):
      return False
    window = getattr(self._plotter, 'render_window', None)
    if window is None:
      return False
    try:
      return window.GetGenericWindowId() is not None
    except Exception:
      return False

  def _went_away(self) -> None:
    """Tear the window down, once, and record that it is gone.

    **Marking it closed is not enough, and neither is `close()`.** `ExitEvent`
    only signals intent, and with `interactive_update=True` there is no
    interactor loop for it to terminate, so a title-bar click left a window
    still mapped while the pump had stopped servicing it.

    `close()` alone does not remove it either. VTK destroys the window by
    queueing an `XDestroyWindow` on its OWN Xlib connection, and that request
    only reaches the server when something services that connection -- which
    is exactly what the pump has just stopped doing. Measured on a bare
    `pyvista.Plotter` with no tkinter involved: after `close()` the window
    reports `Map State: IsViewable` indefinitely, and `Finalize`,
    `terminate_app`, `SetShowWindow(False)`, offscreen rendering and dropping
    the last Python reference all leave it there. One `ProcessEvents()` on the
    interactor removes it.

    So the interactor is captured BEFORE `close()`, because `close()` sets
    `plotter.iren` to None and the flush would otherwise raise `AttributeError`
    on a `NoneType` -- which is what made this look like a VTK bug rather than
    an unflushed queue.

    Every step is guarded: VTK having got there first is the normal case, and
    this runs from a Tk timer callback, where an exception kills the pump it
    exists to protect.
    """
    if not self.alive:
      return
    self.alive = False
    self._box_widget = None
    self._actor = None
    self._overlay = []
    interactor = None
    try:
      interactor = self._plotter.iren.interactor
    except Exception:
      pass                        # already torn down, which is the normal case
    try:
      if self._plotter is not None:
        self._plotter.close()
    except Exception:
      pass
    try:
      if interactor is not None:
        interactor.ProcessEvents()   # flush the queued XDestroyWindow
    except Exception:
      pass
    self._note.config(text='The 3D window was closed. Reopen it below.')
    self.on_status('3D window closed')

  def pump(self) -> None:
    """Service VTK from the Tk timer. Never raises; a closed window is normal."""
    if not self.alive:
      return
    if not self._is_open():
      self._went_away()
      return
    try:
      self._plotter.update()
    except Exception:
      self._went_away()

  def close(self) -> None:
    self.alive = False
    try:
      if self._plotter is not None:
        self._plotter.close()
    except Exception:
      pass

  # -- scene ----------------------------------------------------------------

  def rebuild(self, keep_camera: bool = False) -> None:
    if not self.alive or not self.session.has_mesh:
      return
    if not self._is_open():
      self._went_away()
      return
    pv = self._pv

    points = self.session.points
    warp = self._warp_vectors()
    if warp is not None:
      points = points + self.session.warp_scale * warp

    tris = self.session.surface
    faces = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).ravel()
    surface = pv.PolyData(np.ascontiguousarray(points, dtype=float), faces)

    scalars = self._scalar_field()
    if scalars is not None:
      surface['field'] = scalars[:len(points)]

    if self._actor is not None:
      self._plotter.remove_actor(self._actor, render=False)
    self._actor = self._plotter.add_mesh(
      surface, scalars='field' if scalars is not None else None,
      cmap='viridis', show_scalar_bar=scalars is not None,
      color=None if scalars is not None else '#c8c8c8')

    if not keep_camera:
      self._plotter.reset_camera()
    self.refresh_overlay()

  def _scalar_field(self) -> Optional[np.ndarray]:
    name = self.session.field
    if not name:
      return None
    _, point_data, _ = self.session.read_frame(self.session.frame)
    values = point_data.get(name)
    if values is None:
      return None
    values = np.asarray(values)
    return values if values.ndim == 1 else np.linalg.norm(values, axis=1)

  def _warp_vectors(self) -> Optional[np.ndarray]:
    name = self.session.warp_field
    if not name:
      return None
    _, point_data, _ = self.session.read_frame(self.session.frame)
    values = point_data.get(name)
    if values is None:
      return None
    values = np.asarray(values)
    return values if values.ndim == 2 and values.shape[1] == 3 else None

  def refresh_overlay(self) -> None:
    """The M/P/S arrows. The box itself is the widget, not an actor."""
    if not self.alive:
      return
    if not self._is_open():
      self._went_away()
      return
    pv = self._pv
    for actor in self._overlay:
      self._plotter.remove_actor(actor, render=False)
    self._overlay = []

    box = self.session.box
    if box is None:
      return
    for (origin, direction, _name), colour in zip(
        box.axis_arrows(), ('#ff4d4d', '#4dff88', '#4d9cff')):
      arrow = pv.Arrow(start=origin, direction=direction,
                       scale=float(np.linalg.norm(direction)))
      self._overlay.append(self._plotter.add_mesh(arrow, color=colour,
                                                  render=False))

    # A plan can appear or move without the widget knowing: the first
    # `apply_plan` creates it, and a later numeric edit must drag it. Skip
    # while `_suppress` is set, or the widget's own callback re-places it
    # underneath the drag.
    if not self._suppress:
      self._sync_box_widget()
    self._plotter.render()

  # -- the box widget, which is the reason for this backend -----------------

  def _sync_box_widget(self) -> None:
    """Place the draggable field of view, or move it onto the current plan.

    The widget carries its own transform relative to where it was placed, so
    moving it means replacing it and resetting the reference. That is cheap and
    it is the only way to keep the numeric entries and the handle agreeing.
    """
    if self.session.box is None or not self._is_open():
      return
    if self._box_widget is not None:
      try:
        self._plotter.clear_box_widgets()
      except Exception:
        pass
      self._box_widget = None
    box = self.session.box
    centre = box.loc.astype(float)
    extent = box.fov.astype(float)
    self._box_reference = (centre.copy(), extent.copy())
    bounds = [centre[0] - extent[0] / 2, centre[0] + extent[0] / 2,
              centre[1] - extent[1] / 2, centre[1] + extent[1] / 2,
              centre[2] - extent[2] / 2, centre[2] + extent[2] / 2]

    # `add_box_widget` fires the callback once while placing, with an identity
    # transform. Left unguarded that writes the reference box straight back
    # over the plan, and since identity decomposes to zero rotation it SILENTLY
    # DISCARDS the orientation: a plan set to 30 degrees came back as 0.
    self._suppress = True
    try:
      self._plotter.add_box_widget(self._on_box, bounds=bounds,
                                   rotation_enabled=True, color='#ffd24a',
                                   pass_widget=True, interaction_event='end')
      # `PlaceWidget` takes AXIS-ALIGNED bounds only, so the handle would be
      # drawn unrotated while the plan is oriented: the user would drag a box
      # that is not where the plan says it is. Rotating the widget itself
      # fixes both halves at once, since `GetTransform` then reports the full
      # orientation and the decomposition below needs no correction term.
      if self._box_widget is not None and np.any(box.angles):
        self._orient_widget(self._box_widget, box.mps, centre)
    finally:
      self._suppress = False

  @staticmethod
  def _orient_widget(widget, rotation: np.ndarray, centre: np.ndarray) -> None:
    """Rotate a placed box widget about its own centre."""
    import vtk

    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    for r in range(3):
      for c in range(3):
        matrix.SetElement(r, c, float(rotation[r, c]))
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Translate(*(-np.asarray(centre, float)))
    transform.Concatenate(matrix)
    transform.Translate(*np.asarray(centre, float))
    transform.Update()
    widget.SetTransform(transform)

  def _on_box(self, _polydata, widget) -> None:
    """Translate the widget's transform back into a `FOVBox`.

    `vtkBoxWidget.GetTransform` gives the transform from the box AS PLACED to
    the box as it now stands, so the reference bounds are what the scale and
    the translation are relative to. The upper-left 3x3 is rotation times
    scale: the column norms are the scale and the normalised columns are the
    rotation.
    """
    # The widget arrives only through this callback, so capture it even on the
    # suppressed placement call: `_orient_widget` needs it immediately after.
    self._box_widget = widget
    if self._suppress or self._box_reference is None:
      return
    import vtk

    transform = vtk.vtkTransform()
    widget.GetTransform(transform)
    m = np.array([[transform.GetMatrix().GetElement(r, c) for c in range(4)]
                  for r in range(4)])

    linear = m[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    if np.any(scale <= 0):
      return
    rotation = linear / scale

    # The decomposition assumes the linear part is a rotation times a per-axis
    # scale IN THE BOX'S OWN FRAME, which is what dragging a face handle
    # produces: the columns of `R @ diag(s)` are `s_i * R[:,i]`, so the column
    # norms are the scale and the normalised columns are the rotation. A
    # transform that scales along WORLD axes after a rotation is not of that
    # form, and feeding it through anyway yields a plausible, wrong plan: a
    # world-axis 1.5x on a box rotated 30 degrees came out as 40.89 degrees
    # with a skewed field of view. Refuse it instead.
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
      self.on_status('the box was sheared; the plan was left unchanged')
      return

    centre0, extent0 = self._box_reference
    centre = linear @ centre0 + m[:3, 3]
    extent = extent0 * scale

    try:
      angles = np.array(mps_to_euler(rotation))
    except Exception:
      return                       # a degenerate drag; keep the previous plan

    self._suppress = True
    try:
      self.session.box = FOVBox(fov=extent, loc=centre, angles=angles)
    finally:
      self._suppress = False

    if self.session.has_mesh:
      markers = self.session.submesh_markers()
      self.on_status(f'{int(markers.sum())} of {markers.size} elements '
                     f'in the field of view')

  # -- camera ---------------------------------------------------------------

  def look(self, name: str) -> None:
    if not (self.alive and self._is_open()):
      return
    if name not in STANDARD_VIEWS:
      raise ValueError(f'look: unknown view {name!r}, '
                       f'expected one of {sorted(STANDARD_VIEWS)}')
    direction, up = STANDARD_VIEWS[name]
    focus = np.array(self._plotter.camera_position[1])
    distance = float(np.linalg.norm(
      np.array(self._plotter.camera_position[0]) - focus))
    camera = Camera(focus - np.asarray(direction, float) * distance, focus, up)
    self._plotter.camera_position = camera.as_vtk()
    self._plotter.render()

  def reset_view(self) -> None:
    if self.alive and self._is_open():
      self._plotter.reset_camera()
      self._plotter.render()

  def draw(self, scale: float = 1.0) -> None:
    """Present for interface parity with the blit backend; VTK draws itself."""
    if self.alive and self._is_open():
      self._plotter.render()
