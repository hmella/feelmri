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
from ..model.planning import FOVBox, mps_to_euler, same_box
from .theme import COLOUR_MAP, PALETTE

#: M, P and S, in the order `FOVBox.axis_arrows` returns them.
AXIS_COLOURS = ('#ff6b6b', '#3fd07a', '#5b9cf8')

#: Tk timer interval for pumping VTK, in ms. 16 is about 60 Hz.
PUMP_MS = 16


class Window3D:
  """A native PyVista window showing the session's mesh and plan."""

  def __init__(self, parent, session,
               on_status: Optional[Callable[[str], None]] = None,
               embed: bool = True):
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
    self._glyph_actor = None
    self._overlay = []
    self._box_widget = None
    self._box_reference = None      # (centre, extent) the widget was placed at
    self._widget_for = None         # the plan the widget currently draws
    self._user_closed = False       # set by the ExitEvent observer
    self._suppress = False          # guard against a feedback loop

    # Embedded, the widget IS the 3D view: a black host frame whose X window
    # the render window is reparented into. As a separate top-level window it
    # is a placeholder that reports state and can bring the window back.
    self.widget = ttk.Frame(parent)
    self._host = None
    if embed:
      self._host = tk.Frame(self.widget, bg=PALETTE['window'],
                            width=640, height=480)
      self._host.pack(fill='both', expand=True)
      self._host.bind('<Configure>', self._on_host_resize)
      self._note = ttk.Label(self.widget, text='')
    else:
      inner = ttk.Frame(self.widget, padding=16)
      inner.pack(fill='both', expand=True)
      ttk.Label(inner, text='3D view',
                font=('TkDefaultFont', 11, 'bold')).pack(anchor='w')
      self._note = ttk.Label(inner, wraplength=420, justify='left', text=(
        'The 3D view is a separate window. Drag the yellow box to place the '
        'field of view; the plan and the element count follow it.'))
      self._note.pack(anchor='w', pady=(6, 12))
      ttk.Button(inner, text='Reopen 3D window',
                 command=self.reopen).pack(anchor='w')

    self._pv = pv
    self._plotter = None
    if self._host is None:
      self._open()
    else:
      # **Wait for the frame to be mapped.** A Tk widget has no X window until
      # it is, and handing VTK the id of one that does not exist yet kills the
      # process outright: `BadWindow` on `X_CreateWindow`, not an exception.
      # The shell packs a view only when its tab is selected, so this can be
      # a long wait, and opening early is not an option.
      self._host.bind('<Map>', self._open_when_mapped)

    session.mesh_changed.connect(lambda *_: self.rebuild())
    session.plan_changed.connect(lambda *_: self.refresh_overlay())
    session.view_changed.connect(lambda *_: self.rebuild(keep_camera=True))

  # -- window lifetime ------------------------------------------------------

  def _open(self) -> None:
    size = (900, 750)
    if self._host is not None:
      # The host must have a real X window before its id can be handed to
      # VTK, and a freshly packed frame does not until Tk has processed the
      # geometry.
      self._host.update_idletasks()
      size = (max(self._host.winfo_width(), 32),
              max(self._host.winfo_height(), 32))
    self._plotter = self._pv.Plotter(window_size=size,
                                     title='feelmri  3D view')
    self._plotter.set_background(PALETTE['window'])
    # `color` here is the LABEL colour, and its default is black -- invisible
    # on a dark background, which is what it looked like.
    self._plotter.add_axes(color=PALETTE['text'])

    if self._host is not None:
      try:
        self._reparent()
      except Exception as exc:
        # Not X11, or a VTK build that will not take a parent id. A separate
        # top-level window is the documented alternative, not a failure.
        self._host = None
        self.on_status(f'3D view could not be embedded ({exc}); '
                       f'opening its own window')

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

  def _open_when_mapped(self, _event=None) -> None:
    """Open once, the first time the host frame actually has a window."""
    if self._plotter is not None or self._host is None:
      return
    if not self._host.winfo_ismapped():
      return
    self._open()
    self.rebuild()

  def _reparent(self) -> None:
    """Make the render window a CHILD of the Tk frame, before it is shown.

    **This is what keeps the 3D view in one window without losing anything.**
    PyVista has no tkinter embedding, so the alternative was either a separate
    top-level window or the blitting canvas -- and the blit has no
    `vtkRenderWindowInteractor`, hence no `add_box_widget`, which is the
    draggable field of view. Reparenting keeps the real interactor and puts it
    inside the shell.

    `SetParentId` takes a void pointer, and the Python binding spells one the
    way `GetGenericWindowId` hands it back: `_<16 hex digits>_p_void`. Passing
    the integer raises "object does not have a readable buffer".

    X11 only. It is attempted and, if it fails, the window opens on its own as
    before -- which is why every caller still treats it as a separate window
    that can go away.
    """
    window = self._plotter.render_window
    window.SetParentId(f'_{self._host.winfo_id():016x}_p_void')
    window.SetPosition(0, 0)

  def _on_host_resize(self, event) -> None:
    """Follow the Tk frame. VTK does not learn its parent's new size."""
    if not self.alive or self._plotter is None:
      return
    try:
      self._plotter.render_window.SetSize(max(event.width, 32),
                                          max(event.height, 32))
    except Exception:
      pass                      # the window went away between the two

  def reopen(self) -> None:
    if self.alive and self._is_open():
      return
    self.alive = False
    self._user_closed = False
    self._actor = None
    self._glyph_actor = None
    self._overlay = []
    self._box_widget = None
    self._widget_for = None
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

  def _destroy_plotter(self) -> None:
    """Actually remove the render window. Never raises.

    `close()` alone does not remove it: VTK queues an `XDestroyWindow` on its
    own Xlib connection, and that request reaches the server only when
    something services that connection. `ProcessEvents()` is what does.

    The interactor is captured FIRST, because `close()` sets `plotter.iren` to
    None and reaching for it afterwards raises on a NoneType, leaving the flush
    silently undone.
    """
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
    self._widget_for = None
    self._actor = None
    self._glyph_actor = None
    self._overlay = []
    self._destroy_plotter()
    self._note.config(text='The 3D window was closed. Reopen it below.')
    self.on_status('3D window closed')

  def pump(self) -> None:
    """Service VTK from the Tk timer. Never raises; a closed window is normal."""
    if not self.alive:
      # Embedded, the open is deferred until the host frame is mapped, and a
      # `<Map>` that arrived before the binding existed would otherwise never
      # be noticed. Cheap to re-check, and it only fires once.
      if self._plotter is None and self._host is not None:
        self._open_when_mapped()
      return
    if not self._is_open():
      self._went_away()
      return
    try:
      self._plotter.update()
    except Exception:
      self._went_away()

  def close(self) -> None:
    """Tear the window down on quit.

    Goes through the same destroy as a user close: `plotter.close()` alone
    leaves the window mapped, which outlives the shell whenever the process
    does not exit immediately afterwards.
    """
    self.alive = False
    self._destroy_plotter()

  # -- scene ----------------------------------------------------------------

  def rebuild(self, keep_camera: bool = False) -> None:
    if not self.alive or not self.session.has_mesh:
      return
    if not self._is_open():
      self._went_away()
      return
    pv = self._pv

    points = self.session.points
    warp = self.session.warp_vectors()
    if warp is not None and warp.shape == points.shape:
      points = points + self.session.warp_scale * warp

    tris = self.session.surface
    faces = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).ravel()
    surface = pv.PolyData(np.ascontiguousarray(points, dtype=float), faces)

    # A CELL field carries one value per element and has already been indexed
    # through the triangle-to-element map, so it goes on `cell_data` and is
    # drawn flat per facet -- which is what a tissue map like `cell_markers`
    # means. `preference` is passed because a point and a cell field may share
    # a name, and PyVista would otherwise pick for us.
    resolved = self.session.surface_colour_values()
    if resolved is not None:
      values, association = resolved
      if association == 'cell':
        surface.cell_data['field'] = np.asarray(values)[:surface.n_cells]
      else:
        surface.point_data['field'] = np.asarray(values)[:len(points)]

    if self._actor is not None:
      self._plotter.remove_actor(self._actor, render=False)
    if not self.session.is_visible('phantom'):
      self._actor = None
      self._rebuild_glyphs()
      self.refresh_overlay()
      return
    # `Surface With Edges` is one flag on the surface style, not a fourth
    # style, which is why this is a lookup rather than a pass-through.
    representation = self.session.representation
    self._actor = self._plotter.add_mesh(
      surface,
      style={'Surface': 'surface', 'Surface With Edges': 'surface',
             'Wireframe': 'wireframe', 'Points': 'points'}[representation],
      show_edges=representation == 'Surface With Edges',
      edge_color=PALETTE['border'], point_size=3,
      scalars='field' if resolved is not None else None,
      preference='cell' if resolved and resolved[1] == 'cell' else 'point',
      cmap=COLOUR_MAP, show_scalar_bar=resolved is not None,
      # The bar carries the CHOSEN label, so a component and a magnitude of
      # the same field are told apart on the picture rather than only in the
      # combobox the user has since looked away from.
      #
      # **`color` is the TEXT colour and PyVista defaults it to black**, which
      # on this ground is 828 pure-black pixels against (27, 27, 31) -- a
      # title and a set of tick labels that are there and cannot be read. Same
      # trap as `add_axes`, whose label colour defaults the same way.
      scalar_bar_args={'title': str(self.session.field or ''),
                       'color': PALETTE['text']},
      opacity=float(self.session.opacity),
      color=None if resolved is not None else '#c8c8c8')

    self._rebuild_glyphs()
    if not keep_camera:
      self._plotter.reset_camera()
    self.refresh_overlay()

  def _rebuild_glyphs(self) -> None:
    """Arrows oriented and scaled by a vector field, coloured by its length.

    The sampling and the scale factor are the session's, so this is placement
    only. **Subsampling is not optional**: `heart_P2_tetra` has 191 576 nodes,
    and an arrow on each is a solid block of colour long before it is slow.
    """
    pv = self._pv
    if self._glyph_actor is not None:
      self._plotter.remove_actor(self._glyph_actor, render=False)
      self._glyph_actor = None
    if not self.session.is_visible('glyphs'):
      return
    arrows = self.session.glyph_arrows()
    if arrows is None:
      return
    points, vectors, factor = arrows
    cloud = pv.PolyData(np.ascontiguousarray(points, dtype=float))
    cloud['vectors'] = np.ascontiguousarray(vectors, dtype=float)
    cloud['magnitude'] = np.linalg.norm(vectors, axis=1)
    # `factor` already carries the automatic scale and the user's multiplier,
    # so `scale` names the array and nothing is multiplied twice.
    glyphs = cloud.glyph(orient='vectors', scale='magnitude', factor=factor,
                         geom=pv.Arrow())
    self._glyph_actor = self._plotter.add_mesh(
      glyphs, scalars='magnitude', cmap=COLOUR_MAP, show_scalar_bar=False,
      render=False)

  def _add_plan_solid(self, box) -> None:
    """Draw the planned volume as a translucent solid.

    The plan was a wireframe widget and nothing else, so **where it cuts the
    phantom was invisible** -- the one thing a planner is looking at. A
    translucent solid tints the intersection instead.

    Two details it depends on. It is built in the box's own frame and then
    transformed, because `pv.Box` takes axis-aligned bounds only and an
    oblique plan would otherwise be drawn square. And it is NOT pickable: it
    sits exactly where the draggable widget does, and a pickable actor there
    would swallow the drags the widget exists for.
    """
    pv = self._pv
    opacity = float(self.session.plan_opacity)
    if opacity <= 0 or not np.all(box.fov > 0):
      return
    half = 0.5 * box.fov
    solid = pv.Box(bounds=(-half[0], half[0], -half[1], half[1],
                           -half[2], half[2]))
    transform = np.eye(4)
    transform[:3, :3] = box.mps
    transform[:3, 3] = box.loc
    solid.transform(transform, inplace=True)
    self._overlay.append(self._plotter.add_mesh(
      solid, color='#ffd24a', opacity=opacity, pickable=False,
      show_scalar_bar=False, render=False))

  def refresh_overlay(self, box=None) -> None:
    """The M/P/S arrows and the translucent plan. The box outline is the
    widget, not an actor.

    `box` overrides the session's, which is what a drag in progress passes:
    the widget reports its pose continuously but the plan is only written on
    release, so without this the outline moves and the solid inside it stays
    where it was until the mouse comes up.
    """
    if not self.alive:
      return
    if not self._is_open():
      self._went_away()
      return
    pv = self._pv
    for actor in self._overlay:
      self._plotter.remove_actor(actor, render=False)
    self._overlay = []

    box = self.session.box if box is None else box
    if box is None:
      return
    if self.session.is_visible('plan'):
      self._add_plan_solid(box)
    tips, names, colours = [], [], AXIS_COLOURS
    for (origin, direction, name), colour in (
        zip(box.axis_arrows(), colours) if self.session.is_visible('arrows')
        else ()):
      arrow = pv.Arrow(start=origin, direction=direction,
                       scale=float(np.linalg.norm(direction)))
      self._overlay.append(self._plotter.add_mesh(arrow, color=colour,
                                                  render=False))
      # A little past the tip, so the text clears the cone rather than
      # sitting inside it.
      tips.append(np.asarray(origin) + np.asarray(direction) * 1.12)
      names.append(name)

    if tips:
      # `always_visible` keeps a label readable when its arrow points away
      # from the camera; without it the depth test hides exactly the one the
      # user turned the volume to read.
      self._overlay.append(self._plotter.add_point_labels(
        np.asarray(tips), names, text_color=PALETTE['text'], font_size=13,
        bold=True, show_points=False, always_visible=True, render=False,
        # A dark backing, because a label lands wherever its arrow points --
        # the slice arrow is short on a thin slab and its text sits on the
        # bright phantom, where unbacked light text is unreadable.
        shape='rect', shape_color=PALETTE['window'], shape_opacity=0.55,
        fill_shape=True, margin=3))

    # A plan can appear or move without the widget knowing: the first
    # `apply_plan` creates it, and a later numeric edit must drag it. Skip
    # while `_suppress` is set, or the widget's own callback re-places it
    # underneath the drag.
    #
    # **Only when the plan has actually MOVED.** This runs on every
    # `view_changed` too, and re-placing destroys the widget and builds a new
    # one -- so playing a cine did it on every frame, and the outline was
    # missing from exactly one render per tick. Measured on the 30-frame
    # aorta: 6 frame steps, 6 re-placements, 6 renders with no box. The box
    # itself never moved, so what a viewer sees is the outline strobing, not
    # the plan changing.
    if not self._suppress and not same_box(self._widget_for,
                                           self.session.box):
      self._sync_box_widget()
    self._plotter.render()

  # -- the box widget, which is the reason for this backend -----------------

  def _sync_box_widget(self) -> None:
    """Place the draggable field of view, or move it onto the current plan.

    The widget carries its own transform relative to where it was placed, so
    moving it means replacing it and resetting the reference. That is cheap and
    it is the only way to keep the numeric entries and the handle agreeing.
    """
    self._widget_for = None
    if self.session.box is None or not self._is_open():
      return
    if self._box_widget is not None:
      try:
        self._plotter.clear_box_widgets()
      except Exception:
        pass
      self._box_widget = None
    # Hiding the field of view hides its HANDLE too. The outline and the
    # translucent solid are one object to a user, and leaving a draggable
    # outline behind after switching the layer off would be the kind of
    # disagreement between two renderings this viewer has already paid for.
    if not self.session.is_visible('plan'):
      return
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
      # **`factor` must be 1.0.** PyVista defaults it to 1.25 and passes it to
      # `SetPlaceFactor`, which inflates the placed box about its centre -- so
      # the outline was drawn at 1.25x the plan on every axis, measured
      # 0.225 x 0.175 x 0.0375 for a plan of 0.180 x 0.140 x 0.030. The
      # decomposition is relative to the placement and so stayed
      # self-consistent, which is why nothing caught it: what was wrong was
      # only the handle the user drags. Dragging it onto an anatomical edge
      # therefore produced a field of view 20% smaller per axis than the one
      # drawn.
      self._plotter.add_box_widget(self._on_box, bounds=bounds, factor=1.0,
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
    self._widget_for = box

    # `interaction_event='end'` above is what keeps a drag from writing the
    # plan on every motion event. This second observer is the other half: it
    # moves the drawn volume with the outline in the meantime, without
    # touching the session.
    if self._box_widget is not None:
      try:
        self._box_widget.AddObserver('InteractionEvent', self._on_box_moving)
      except Exception:
        pass                # the solid then follows on release, as before

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

  def _decompose(self, widget):
    """The widget's current pose as a `FOVBox`, or None if it is not one.

    `vtkBoxWidget.GetTransform` gives the transform from the box AS PLACED to
    the box as it now stands, so the reference bounds are what the scale and
    the translation are relative to. The upper-left 3x3 is rotation times
    scale: the column norms are the scale and the normalised columns are the
    rotation.
    """
    if self._box_reference is None:
      return None
    import vtk

    transform = vtk.vtkTransform()
    widget.GetTransform(transform)
    m = np.array([[transform.GetMatrix().GetElement(r, c) for c in range(4)]
                  for r in range(4)])

    linear = m[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    if np.any(scale <= 0):
      return None
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
      return None

    centre0, extent0 = self._box_reference
    try:
      return FOVBox(fov=extent0 * scale, loc=linear @ centre0 + m[:3, 3],
                    angles=np.array(mps_to_euler(rotation)))
    except Exception:
      return None                  # a degenerate drag

  def _on_box_moving(self, widget, _event=None) -> None:
    """Move the overlay WITH the outline, while the mouse is still down.

    The plan itself is written on release, because each write pushes an undo
    entry and recounts the submesh over every element. But the translucent
    volume and the arrows are a handful of quads, so redrawing them per motion
    event is free -- and without it the yellow outline moves while the solid
    inside it stays where the plan last was, which reads as the two disagreeing
    rather than as one of them lagging.

    Nothing here touches the session, and `_suppress` stops the redraw
    re-placing the widget underneath the drag.
    """
    box = self._decompose(widget)
    if box is None or not self.alive:
      return
    self._suppress = True
    try:
      self.refresh_overlay(box)
    finally:
      self._suppress = False

  def _on_box(self, _polydata, widget) -> None:
    """Write the widget's pose into the plan, at the end of a drag."""
    # The widget arrives only through this callback, so capture it even on the
    # suppressed placement call: `_orient_widget` needs it immediately after.
    self._box_widget = widget
    if self._suppress:
      return

    box = self._decompose(widget)
    if box is None:
      # The plan is unchanged, so the OUTLINE has to go back to it -- left
      # where the drag put it, the widget and the solid inside it show two
      # different fields of view and neither is the plan. Deferred to the Tk
      # loop because re-placing means destroying this widget, and it is the
      # one dispatching this callback.
      self.on_status('that drag was not a box; the plan was left unchanged')
      self.widget.after(0, self._restore_widget)
      return

    # The widget is already drawing this pose -- it is what the user dragged
    # it to -- so record it as placed, or the redraw below would tear it down
    # and build an identical one.
    self._widget_for = box
    self._suppress = True
    try:
      self.session.box = box
    finally:
      self._suppress = False

    if self.session.has_mesh:
      markers = self.session.submesh_markers()
      self.on_status(f'{int(markers.sum())} of {markers.size} elements '
                     f'in the field of view')

  def _restore_widget(self) -> None:
    """Put the outline back on the plan after a drag that was refused."""
    if not (self.alive and self._is_open()):
      return
    self._sync_box_widget()
    self.refresh_overlay()

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
