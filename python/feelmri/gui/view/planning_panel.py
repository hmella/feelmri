"""The field of view: its numbers, and the two files a plan travels in.

The plan is edited in two places at once -- these entries and the draggable
box in the 3D window -- so **it has to travel both ways**. The panel writes to
the session on Apply and reads back from `plan_changed` whoever changed it;
without the second half, dragging the box moved the field of view while the
numbers beside it went stale, and the submesh count with them.

Export writes BOTH files. The `.pvsm` carries the geometry and is what
ParaView and `PVSMParser` read; the `.yaml` beside it is the parameter file an
example actually opens, and its `planning` key names the `.pvsm`. They are
written together because that key is resolved against the SCRIPT directory,
so the pair has to travel together to mean anything.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..model.planning import FOVBox, format_triplet, parse_triplet

#: The three editable rows: label, key, and the default shown before a plan.
FIELDS = (('FOV (m), M P S', 'fov', '0.3 0.22 0.008'),
          ('Centre (m)', 'loc', '0 0 0'),
          ('Rotation (deg)', 'rot', '0 0 0'))


class PlanningPanel:
  """Field-of-view entries, the submesh count, and `.pvsm` / `.yaml` export.

  Usable on its own: give it any Tk parent and a `Session`. The shell packs it
  into the Plan tab and points its File menu at `import_plan` / `export_plan`.
  """

  def __init__(self, parent, session,
               on_status: Optional[Callable[[str], None]] = None):
    import tkinter as tk
    from tkinter import ttk

    self.session = session
    self.on_status = on_status or (lambda _: None)
    self._updating = False

    self.widget = ttk.Frame(parent)
    ttk.Label(self.widget, text='Field of view',
              font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')

    ttk.Label(self.widget, text='Mesh scale (file units to m)').pack(
      anchor='w', pady=(6, 0))
    row = ttk.Frame(self.widget)
    row.pack(fill='x')
    self.scale = tk.StringVar(value='1')
    ttk.Entry(row, textvariable=self.scale, width=10).pack(side='left')
    ttk.Button(row, text='Rescale', width=8,
               command=self.rescale).pack(side='left', padx=(6, 0))
    self._scale_note = ttk.Label(self.widget, text='', wraplength=270,
                                 style='Muted.TLabel')
    self._scale_note.pack(anchor='w', pady=(2, 8))

    self.vars = {}
    for label, key, default in FIELDS:
      ttk.Label(self.widget, text=label).pack(anchor='w', pady=(6, 0))
      variable = tk.StringVar(value=default)
      ttk.Entry(self.widget, textvariable=variable).pack(fill='x')
      self.vars[key] = variable

    ttk.Button(self.widget, text='Apply plan',
               command=self.apply).pack(fill='x', pady=(10, 0))
    self._submesh = ttk.Label(self.widget, text='', wraplength=270)
    self._submesh.pack(anchor='w', pady=(6, 0))

    session.plan_changed.connect(lambda *_: self.refresh())
    session.mesh_changed.connect(lambda *_: self._show_scale())

  # -- the unit the file is in ----------------------------------------------

  def _show_scale(self) -> None:
    """Report the scale in force and what the mesh now measures."""
    if not self.session.has_mesh:
      return
    factor = getattr(self.session, 'scale_factor', 1.0)
    self.scale.set(f'{factor:g}')
    lo, hi = self.session.points.min(axis=0), self.session.points.max(axis=0)
    self._scale_note.config(text=f'mesh is {np.max(hi - lo):.3g} m across')
    self._show_submesh()

  def rescale(self) -> None:
    """Re-read the mesh at the scale in the entry.

    The plan is in metres and **the shipped phantoms are not all in metres**,
    so without this the field of view is drawn a hundred or a thousand times
    too small and the submesh reads zero. Re-reading rather than scaling in
    place keeps one code path for what the coordinates are.
    """
    from tkinter import messagebox

    if not self.session.has_mesh:
      messagebox.showinfo('No phantom', 'Open a phantom first.')
      return
    try:
      factor = float(self.scale.get())
      self.session.load_mesh(self.session.mesh_path, scale_factor=factor)
    except Exception as exc:
      messagebox.showerror('Could not rescale', str(exc))
      return
    self.on_status(f'mesh reloaded at scale {factor:g}')

  # -- the two directions ---------------------------------------------------

  def apply(self) -> None:
    """Entries to session."""
    from tkinter import messagebox

    try:
      box = FOVBox(fov=parse_triplet(self.vars['fov'].get(), 'fov'),
                   loc=parse_triplet(self.vars['loc'].get(), 'loc'),
                   angles=np.radians(parse_triplet(self.vars['rot'].get(),
                                                   'rotation')))
    except ValueError as exc:
      messagebox.showerror('Plan', str(exc))
      return
    self.session.box = box
    self._show_submesh()
    self.on_status('plan applied')

  def refresh(self) -> None:
    """Session to entries, whoever changed it.

    Guarded against its own echo: writing a `StringVar` does not fire `apply`
    today, but a panel that rewrote the session on every repaint would loop,
    and the guard costs one flag.
    """
    box = self.session.box
    if box is None or self._updating:
      return
    self._updating = True
    try:
      self.vars['fov'].set(format_triplet(box.fov))
      self.vars['loc'].set(format_triplet(box.loc))
      self.vars['rot'].set(format_triplet(np.degrees(box.angles)))
      self._show_submesh()
    finally:
      self._updating = False

  def _show_submesh(self) -> None:
    """How much of the mesh the slab keeps, which is the number that decides
    whether a plan is usable at all."""
    if not self.session.has_mesh or self.session.box is None:
      return
    markers = self.session.submesh_markers()
    self._submesh.config(
      text=f'{int(markers.sum())} of {markers.size} elements in the slab')

  # -- the files ------------------------------------------------------------

  def import_plan(self) -> None:
    """Read an existing ParaView state, through the library's own parser."""
    from tkinter import filedialog, messagebox

    path = filedialog.askopenfilename(
      title='Import plan', filetypes=[('ParaView state', '*.pvsm'),
                                      ('All files', '*')])
    if not path:
      return
    try:
      from ....Parameters import PVSMParser
      parser = PVSMParser(path)
      self.vars['fov'].set(format_triplet(parser.FOV.m_as('m')))
      self.vars['loc'].set(format_triplet(parser.LOC.m_as('m')))
      self.vars['rot'].set(format_triplet(parser.Rotation.m_as('deg')))
    except Exception as exc:
      messagebox.showerror('Could not read the plan', str(exc))
      return
    self.apply()
    self.on_status(f'imported {path}')

  def export_plan(self) -> None:
    """Write the `.pvsm` and, beside it, the `.yaml` that names it."""
    from pathlib import Path
    from tkinter import filedialog, messagebox

    if self.session.box is None:
      messagebox.showinfo('Nothing to export', 'Apply a plan first.')
      return
    path = filedialog.asksaveasfilename(
      title='Export plan', defaultextension='.pvsm',
      filetypes=[('ParaView state', '*.pvsm'), ('All files', '*')])
    if not path:
      return
    try:
      from ..model.pvsm import write_plan_yaml, write_pvsm
      box = self.session.box
      write_pvsm(path, box.fov, box.loc, np.degrees(box.angles))
      parameters = Path(path).with_suffix('.yaml')
      write_plan_yaml(parameters, Path(path).name)
    except Exception as exc:
      messagebox.showerror('Export failed', str(exc))
      return
    self.on_status(f'wrote {Path(path).name} and {parameters.name}')
