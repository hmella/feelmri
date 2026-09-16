"""The strip over the 3D view: ParaView's toolbars, with what applies here.

Four of them, in ParaView's own order -- camera, representation, active
variable, and the VCR controls for a time series. They exist because a menu is
not discoverable: the standard views were under `View` and nobody looking at a
phantom thinks to open a menu to turn it.

**Colour by is TWO comboboxes**, the field and its component, which is how
ParaView presents it and not how this viewer did. One flat list grows as the
product -- `velocity` alone contributes four entries -- and the component stops
sticking, so comparing two fields on the same axis means re-picking it each
time.

Every control writes to the `Session` and reads back from its signals, so the
toolbar and the control column cannot disagree about what is shown.
"""
from __future__ import annotations

from typing import Callable, Optional

from ..model.camera import AXIS_VIEWS
from ..model.session import REPRESENTATIONS

#: The VCR buttons, as `(glyph, frame delta or None)`. None is the play toggle.
VCR_BUTTONS = (('⏮', 'first'), ('◀', -1), ('▶', None),
               ('▸', 1), ('⏭', 'last'))

#: Milliseconds between frames while playing. 20 frames at 120 ms is a 2.4 s
#: loop, which reads as a heartbeat rather than a flicker.
PLAY_MS = 120


class Toolbar:
  """Camera, representation, colouring and playback, above the view."""

  def __init__(self, parent, session, viewport,
               on_status: Optional[Callable[[str], None]] = None):
    import tkinter as tk
    from tkinter import ttk

    self.session = session
    self.viewport = viewport
    self.on_status = on_status or (lambda _: None)
    self._playing = False
    self._play_job = None
    self._updating = False

    # **Two rows, not one.** Packed left to right on a single row these run
    # off the right edge of the window and the last controls are simply not
    # there -- measured at 1300 px wide, the component box and every playback
    # button were past the edge. ParaView wraps its toolbars for the same
    # reason. The row is chosen per group, not by measuring: a layout that
    # reflows as the window resizes moves buttons under the pointer.
    self.widget = ttk.Frame(parent, style='Toolbar.TFrame', padding=(6, 3))
    self._row = ttk.Frame(self.widget, style='Toolbar.TFrame')
    self._row.pack(fill='x')

    # -- camera ------------------------------------------------------------
    self._group('View')
    for name in AXIS_VIEWS:
      self._button(name, lambda n=name: self._look(n), width=3)
    self._button('Reset', self._reset, width=6)

    # -- representation ----------------------------------------------------
    self._separator()
    self.representation = tk.StringVar(value=session.representation)
    box = ttk.Combobox(self._row, textvariable=self.representation,
                       state='readonly', values=list(REPRESENTATIONS),
                       width=17)
    box.pack(side='left', padx=(0, 4))
    box.bind('<<ComboboxSelected>>', lambda _e: self._set_representation())

    # -- the active variable ----------------------------------------------
    self._new_row()
    self._group('Colour by')
    self.field = tk.StringVar(value='')
    self._field_box = ttk.Combobox(self._row, textvariable=self.field,
                                   state='readonly', values=[''], width=18)
    self._field_box.pack(side='left', padx=(0, 4))
    self._field_box.bind('<<ComboboxSelected>>', lambda _e: self._on_field())

    self.component = tk.StringVar(value='')
    self._component_box = ttk.Combobox(
      self._row, textvariable=self.component, state='readonly',
      values=[''], width=11)
    self._component_box.pack(side='left', padx=(0, 4))
    self._component_box.bind('<<ComboboxSelected>>',
                             lambda _e: self._apply_field())

    # -- playback ----------------------------------------------------------
    self._separator()
    for glyph, action in VCR_BUTTONS:
      if action is None:
        self._play_button = self._button(glyph, self.toggle_play, width=3)
      else:
        self._button(glyph, lambda a=action: self._step(a), width=3)
    self._frame_label = ttk.Label(self._row, text='-- / --', width=9,
                                  style='Toolbar.TLabel', anchor='e')
    self._frame_label.pack(side='left', padx=(4, 0))

    self._groups = None
    session.mesh_changed.connect(lambda *_: self.refresh_fields())
    session.view_changed.connect(lambda *_: self.refresh())
    self.refresh_fields()

  # -- construction helpers -------------------------------------------------

  def _new_row(self):
    from tkinter import ttk
    self._row = ttk.Frame(self.widget, style='Toolbar.TFrame')
    self._row.pack(fill='x', pady=(3, 0))

  def _group(self, text: str):
    from tkinter import ttk
    ttk.Label(self._row, text=text,
              style='Toolbar.TLabel').pack(side='left', padx=(0, 4))

  def _separator(self):
    from tkinter import ttk
    ttk.Separator(self._row, orient='vertical').pack(
      side='left', fill='y', padx=6)

  def _button(self, text: str, command, width: int):
    from tkinter import ttk
    button = ttk.Button(self._row, text=text, command=command,
                        style='Tool.TButton', width=width)
    button.pack(side='left', padx=1)
    return button

  # -- camera ---------------------------------------------------------------

  def _look(self, name: str) -> None:
    try:
      self.viewport.look(name)
    except Exception as exc:                        # a closed 3D window
      self.on_status(f'could not set the view: {exc}')
      return
    self.on_status(f'looking from {name}')

  def _reset(self) -> None:
    try:
      self.viewport.reset_view()
    except Exception:
      pass

  # -- representation and colouring ----------------------------------------

  def _set_representation(self) -> None:
    self.session.representation = self.representation.get()

  def refresh_fields(self) -> None:
    """Re-read what the loaded phantom offers.

    Done on `mesh_changed` only: the field LIST is a property of the file, not
    of the frame, so rebuilding it on every frame of a playing cine would
    reset the user's choice sixty times a loop.
    """
    self._groups = dict(self.session.field_group_choices(0))
    self._field_box.config(values=[''] + list(self._groups))
    if self.field.get() not in self._groups:
      self.field.set('')
    self._refresh_components()
    self.refresh()

  def _on_field(self) -> None:
    self._refresh_components()
    self._apply_field()

  def _refresh_components(self) -> None:
    """Offer the components of the chosen field, keeping the current one.

    **Keeping it is the point of splitting the two boxes.** Moving from
    `velocity` to another vector while looking at `(Y)` should stay on `(Y)`.
    """
    components = [name for name, _ in (self._groups or {}).get(
      self.field.get(), [])]
    self._component_box.config(values=components or [''])
    if self.component.get() not in components:
      self.component.set(components[0] if components else '')
    # A scalar has one nameless component, so the box would be an empty
    # control that does nothing. Disabled rather than hidden, or the toolbar
    # reflows every time the field changes.
    self._component_box.config(
      state='readonly' if len(components) > 1 else 'disabled')

  def _apply_field(self) -> None:
    label = ''
    for component, full in (self._groups or {}).get(self.field.get(), []):
      if component == self.component.get():
        label = full
        break
    self.session.field = label or None

  # -- playback -------------------------------------------------------------

  def _step(self, action) -> None:
    if action == 'first':
      self.session.frame = 0
    elif action == 'last':
      self.session.frame = max(0, self.session.n_frames - 1)
    else:
      self.session.step_frame(action)

  def toggle_play(self) -> None:
    """Start or stop the cine.

    Stopping is not optional: the loop is a chain of `after` callbacks, and one
    left running after the window closes calls into a dead viewport.
    """
    self._playing = not self._playing and self.session.n_frames > 1
    self._play_button.config(text='⏸' if self._playing else '▶')
    if self._playing:
      self._tick()
    elif self._play_job is not None:
      try:
        self.widget.after_cancel(self._play_job)
      except Exception:
        pass
      self._play_job = None

  def _tick(self) -> None:
    if not self._playing:
      return
    self.session.step_frame(1)
    self._play_job = self.widget.after(PLAY_MS, self._tick)

  def stop(self) -> None:
    """Called on teardown, so no timer outlives the window."""
    if self._playing:
      self.toggle_play()

  # -- keeping in step ------------------------------------------------------

  def refresh(self) -> None:
    """Follow the session, whoever changed it."""
    if self._updating:
      return
    self._updating = True
    try:
      self.representation.set(self.session.representation)
      total = self.session.n_frames
      self._frame_label.config(
        text=f'{self.session.frame + 1} / {total}' if total else '-- / --')
    finally:
      self._updating = False
