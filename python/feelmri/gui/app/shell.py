"""The window: a control column, a 3D panel and a status bar.

Panels do not talk to each other. Each reads the `Session` and subscribes to
the signals it cares about, so this file is wiring and nothing else. Anything
that computes belongs in `gui.model`, where it can be tested without a display.

The 3D panel is chosen at construction: the blitting canvas when PyVista
imports, the fallback otherwise. Both expose the same handful of methods so
the menu wiring below does not branch.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..model.planning import FOVBox
from ..model.session import Session


class Shell:
  """The application window."""

  def __init__(self, session: Optional[Session] = None):
    import tkinter as tk
    from tkinter import ttk

    self.session = session or Session()
    self.root = tk.Tk()
    self.root.title('feelmri')
    self.root.geometry('1180x760')

    self._status = tk.StringVar(value='ready')
    self._updating_plan = False

    # The status bar is packed BEFORE the body: Tk allocates in pack order, so
    # an expanding body packed first takes the whole window and squeezes the
    # bar off the bottom entirely.
    ttk.Label(self.root, textvariable=self._status, anchor='w',
              relief='sunken', padding=(6, 2)).pack(fill='x', side='bottom')

    body = ttk.Frame(self.root)
    body.pack(fill='both', expand=True)

    self._tabs = ttk.Notebook(body, width=320)
    self._tabs.pack(side='left', fill='y')
    self._tabs.pack_propagate(False)
    self.controls = ttk.Frame(self._tabs, padding=10)
    self._tabs.add(self.controls, text='Plan')

    # A draggable split rather than a fixed layout: with the native 3D window
    # the top panel is only a note and the sequence wants the room, while the
    # blit backend needs the opposite. The user decides.
    self._split = ttk.PanedWindow(body, orient='vertical')
    self._split.pack(side='right', fill='both', expand=True)

    self.viewport = self._make_viewport(self._split)
    self._split.add(self.viewport.widget, weight=0)

    self.sequence_panel = self._make_sequence_panel(self._split)
    if self.sequence_panel is not None:
      self._split.add(self.sequence_panel.widget, weight=1)

    self.label_panel = self._make_label_panel(self._tabs)
    if self.label_panel is not None:
      self._tabs.add(self.label_panel.widget, text='Labels')

    # After the viewport: the View menu binds straight to its methods.
    self._build_menu()
    self._build_controls()
    self.session.mesh_changed.connect(lambda *_: self._on_mesh())
    # The plan travels BOTH ways. Without this the entries write to the
    # session and never read back, so dragging the box in the 3D window
    # changed the plan while the numbers beside it went stale.
    self.session.plan_changed.connect(lambda *_: self._on_plan())
    self.root.protocol('WM_DELETE_WINDOW', self.close)

  # -- construction ---------------------------------------------------------

  def _make_viewport(self, parent):
    """Pick a 3D backend: native window, blit, or the no-VTK panel.

    The native window is the default because the 3D widgets, and therefore the
    draggable field of view, need a real `vtkRenderWindowInteractor`. Set
    `FEELMRI_GUI_VIEWPORT=blit` to get the offscreen canvas embedded in this
    window instead; it is about four times slower and has no box widget, but
    it keeps everything in one window.
    """
    import os

    from ..view.fallback3d import Fallback3D

    choice = os.environ.get('FEELMRI_GUI_VIEWPORT', 'window').lower()
    if choice not in ('window', 'blit'):
      raise ValueError(
        f'FEELMRI_GUI_VIEWPORT must be "window" or "blit", got {choice!r}')
    try:
      if choice == 'blit':
        from ..view.canvas3d import Canvas3D
        return Canvas3D(parent, self.session, on_status=self.status)
      from ..view.window3d import Window3D
      viewport = Window3D(parent, self.session, on_status=self.status)
      self._start_pump(viewport)
      return viewport
    except Exception as exc:
      self.status('3D view unavailable, see the panel')
      return Fallback3D(parent, self.session, reason=str(exc),
                        on_status=self.status)

  def _make_label_panel(self, parent):
    """The labels tab, or nothing if it cannot be built."""
    try:
      from ..view.label_panel import LabelPanel
      return LabelPanel(parent, self.session, on_status=self.status)
    except Exception as exc:
      self.status(f'label panel unavailable: {exc}')
      return None

  def _on_pick(self, block, obj=None) -> None:
    """Route a pick from the sequence panel to the labels tab.

    Looked up rather than bound, because the sequence panel is built first --
    it is what the labels tab listens to.
    """
    panel = getattr(self, 'label_panel', None)
    if panel is not None:
      panel.set_target(block, obj)

  def _make_sequence_panel(self, parent):
    """The sequence rows, or nothing if matplotlib is missing.

    Optional the same way the 3D view is: the shell is still useful for
    planning without it, so a missing dependency costs a panel rather than
    the application.
    """
    try:
      from ..view.sequence_panel import SequencePanel
      return SequencePanel(parent, self.session, on_status=self.status,
                           on_pick=self._on_pick)
    except Exception as exc:
      self.status(f'sequence panel unavailable: {exc}')
      return None

  def _start_pump(self, viewport) -> None:
    """Service VTK from the Tk timer, so the two event loops coexist.

    Tk owns `mainloop`; the plotter was shown with `interactive_update=True`
    and needs `update()` called for it to respond. The timer keeps running
    after the window is closed so that reopening it starts working again.
    """
    from ..view.window3d import PUMP_MS

    def tick():
      viewport.pump()
      self._pump_id = self.root.after(PUMP_MS, tick)

    self._pump_id = self.root.after(PUMP_MS, tick)

  def _build_menu(self) -> None:
    import tkinter as tk

    bar = tk.Menu(self.root)

    file_menu = tk.Menu(bar, tearoff=0)
    file_menu.add_command(label='Open phantom...', command=self.open_phantom)
    file_menu.add_command(label='Open sequence (.seq)...',
                          command=self.open_sequence)
    file_menu.add_separator()
    file_menu.add_command(label='Import plan (.pvsm)...', command=self.import_plan)
    file_menu.add_command(label='Export plan (.pvsm)...', command=self.export_plan)
    file_menu.add_separator()
    file_menu.add_command(label='Quit', command=self.close)
    bar.add_cascade(label='File', menu=file_menu)

    view_menu = tk.Menu(bar, tearoff=0)
    for name in ('axial', 'coronal', 'sagittal'):
      view_menu.add_command(label=name.capitalize(),
                            command=lambda n=name: self.viewport.look(n))
    view_menu.add_separator()
    view_menu.add_command(label='Reset', command=self.viewport.reset_view)
    bar.add_cascade(label='View', menu=view_menu)

    self.root.config(menu=bar)

  def _build_controls(self) -> None:
    import tkinter as tk
    from tkinter import ttk

    ttk.Label(self.controls, text='Phantom',
              font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
    self._mesh_label = ttk.Label(self.controls, text='none loaded',
                                 wraplength=270, justify='left')
    self._mesh_label.pack(anchor='w', pady=(0, 10))

    ttk.Label(self.controls, text='Colour by').pack(anchor='w')
    self._field = tk.StringVar(value='')
    self._field_box = ttk.Combobox(self.controls, textvariable=self._field,
                                   state='readonly', values=[''])
    self._field_box.pack(fill='x')
    self._field_box.bind('<<ComboboxSelected>>',
                         lambda _e: setattr(self.session, 'field',
                                            self._field.get() or None))

    ttk.Label(self.controls, text='Warp by').pack(anchor='w', pady=(10, 0))
    self._warp = tk.StringVar(value='')
    self._warp_box = ttk.Combobox(self.controls, textvariable=self._warp,
                                  state='readonly', values=[''])
    self._warp_box.pack(fill='x')
    self._warp_box.bind('<<ComboboxSelected>>',
                        lambda _e: setattr(self.session, 'warp_field',
                                           self._warp.get() or None))

    ttk.Label(self.controls, text='Warp scale').pack(anchor='w', pady=(10, 0))
    self._warp_scale = tk.DoubleVar(value=1.0)
    ttk.Scale(self.controls, from_=0.0, to=10.0, variable=self._warp_scale,
              command=lambda _v: setattr(self.session, 'warp_scale',
                                         float(self._warp_scale.get()))
              ).pack(fill='x')

    ttk.Label(self.controls, text='Frame').pack(anchor='w', pady=(10, 0))
    self._frame = tk.IntVar(value=0)
    self._frame_scale = ttk.Scale(self.controls, from_=0, to=0,
                                  variable=self._frame,
                                  command=lambda _v: self._set_frame())
    self._frame_scale.pack(fill='x')

    ttk.Separator(self.controls).pack(fill='x', pady=12)
    ttk.Label(self.controls, text='Field of view',
              font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')

    self._plan_vars = {}
    for label, key, default in (('FOV (m), M P S', 'fov', '0.3 0.22 0.008'),
                                ('Centre (m)', 'loc', '0 0 0'),
                                ('Rotation (deg)', 'rot', '0 0 0')):
      ttk.Label(self.controls, text=label).pack(anchor='w', pady=(6, 0))
      var = tk.StringVar(value=default)
      ttk.Entry(self.controls, textvariable=var).pack(fill='x')
      self._plan_vars[key] = var

    ttk.Button(self.controls, text='Apply plan',
               command=self.apply_plan).pack(fill='x', pady=(10, 0))
    self._submesh_label = ttk.Label(self.controls, text='', wraplength=270)
    self._submesh_label.pack(anchor='w', pady=(6, 0))

  # -- actions --------------------------------------------------------------

  def status(self, message: str) -> None:
    self._status.set(message)
    self.root.update_idletasks()

  def open_phantom(self) -> None:
    from tkinter import filedialog, messagebox

    path = filedialog.askopenfilename(
      title='Open phantom',
      filetypes=[('Meshes', '*.xdmf *.vtu *.msh *.vtk'), ('All files', '*')])
    if not path:
      return
    self.status(f'loading {path}...')
    try:
      self.session.load_mesh(path)
    except Exception as exc:
      self.status('load failed')
      messagebox.showerror('Could not open the phantom', str(exc))
      return
    self.status(f'loaded {path}')

  def open_sequence(self) -> None:
    """Load a Pulseq `.seq` into the sequence panel.

    Import is the adapter's, not a second reader: `import_pulseq` is what the
    rest of the library uses, so what the panel draws is what a run would
    play. It needs pypulseq, which is reported rather than raised.
    """
    from tkinter import filedialog, messagebox

    path = filedialog.askopenfilename(
      title='Open sequence',
      filetypes=[('Pulseq', '*.seq'), ('All files', '*')])
    if not path:
      return
    self.status(f'reading {path}...')
    try:
      self.session.load_sequence(path)
    except Exception as exc:
      self.status('sequence load failed')
      messagebox.showerror('Could not open the sequence', str(exc))
      return
    blocks = len(self.session.sequence.spans)
    self.status(f'loaded {path} -- {blocks} blocks')

  def _on_mesh(self) -> None:
    s = self.session
    summary = s.summary()
    self._mesh_label.config(
      text=f"{summary['nodes']} nodes, {summary['elements']} elements, "
           f"{summary['frames']} frame(s)")

    _, point_data, _ = s.read_frame(0)
    scalars = [''] + sorted(k for k, v in point_data.items()
                            if np.asarray(v).ndim == 1)
    vectors = [''] + sorted(k for k, v in point_data.items()
                            if np.asarray(v).ndim == 2
                            and np.asarray(v).shape[1] == 3)
    self._field_box.config(values=scalars)
    self._warp_box.config(values=vectors)
    self._frame_scale.config(to=max(0, s.n_frames - 1))

    lo, hi = s.points.min(axis=0), s.points.max(axis=0)
    self._plan_vars['loc'].set(' '.join(f'{v:.4g}' for v in 0.5 * (lo + hi)))
    self.viewport.rebuild()

  def _on_plan(self) -> None:
    """Show the current plan in the entries, whoever changed it.

    Guarded against its own echo: writing a `StringVar` does not fire
    `apply_plan`, but a future binding might, and a plan panel that rewrites
    the session on every repaint would loop.
    """
    box = self.session.box
    if box is None or self._updating_plan:
      return
    self._updating_plan = True
    try:
      self._plan_vars['fov'].set(' '.join(f'{v:.6g}' for v in box.fov))
      self._plan_vars['loc'].set(' '.join(f'{v:.6g}' for v in box.loc))
      self._plan_vars['rot'].set(' '.join(f'{v:.6g}'
                                          for v in np.degrees(box.angles)))
      if self.session.has_mesh:
        markers = self.session.submesh_markers()
        self._submesh_label.config(
          text=f'{int(markers.sum())} of {markers.size} elements in the slab')
    finally:
      self._updating_plan = False

  def _set_frame(self) -> None:
    try:
      self.session.frame = int(float(self._frame.get()))
    except IndexError:
      pass

  def apply_plan(self) -> None:
    from tkinter import messagebox

    try:
      fov = self._numbers('fov')
      loc = self._numbers('loc')
      rot = np.radians(self._numbers('rot'))
    except ValueError as exc:
      messagebox.showerror('Plan', str(exc))
      return

    self.session.box = FOVBox(fov=fov, loc=loc, angles=rot)
    if self.session.has_mesh:
      markers = self.session.submesh_markers()
      self._submesh_label.config(
        text=f'{int(markers.sum())} of {markers.size} elements in the slab')
    self.status('plan applied')

  def _numbers(self, key: str) -> np.ndarray:
    parts = self._plan_vars[key].get().replace(',', ' ').split()
    if len(parts) != 3:
      raise ValueError(f'{key}: expected three numbers, got {len(parts)}')
    try:
      return np.array([float(p) for p in parts])
    except ValueError as exc:
      raise ValueError(f'{key}: {exc}') from exc

  def import_plan(self) -> None:
    from tkinter import filedialog, messagebox

    path = filedialog.askopenfilename(title='Import plan',
                                      filetypes=[('ParaView state', '*.pvsm')])
    if not path:
      return
    try:
      from feelmri.Parameters import PVSMParser
      parser = PVSMParser(path)
      self._plan_vars['fov'].set(' '.join(f'{v:.6g}'
                                          for v in parser.FOV.m_as('m')))
      self._plan_vars['loc'].set(' '.join(f'{v:.6g}'
                                          for v in parser.LOC.m_as('m')))
      self._plan_vars['rot'].set(' '.join(f'{v:.6g}'
                                          for v in parser.Rotation.m_as('deg')))
    except Exception as exc:
      messagebox.showerror('Could not read the plan', str(exc))
      return
    self.apply_plan()
    self.status(f'imported {path}')

  def export_plan(self) -> None:
    from tkinter import filedialog, messagebox

    if self.session.box is None:
      messagebox.showinfo('Nothing to export', 'Apply a plan first.')
      return
    path = filedialog.asksaveasfilename(
      title='Export plan', defaultextension='.pvsm',
      filetypes=[('ParaView state', '*.pvsm')])
    if not path:
      return
    try:
      from ..model.pvsm import write_pvsm
      box = self.session.box
      write_pvsm(path, box.fov, box.loc, np.degrees(box.angles))
    except Exception as exc:
      messagebox.showerror('Export failed', str(exc))
      return
    self.status(f'wrote {path}')

  def close(self) -> None:
    pump = getattr(self, '_pump_id', None)
    if pump is not None:
      try:
        self.root.after_cancel(pump)
      except Exception:
        pass
    try:
      self.viewport.close()
    finally:
      self.root.destroy()

  def run(self) -> None:
    self.root.mainloop()
