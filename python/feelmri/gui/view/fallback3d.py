"""What the 3D panel becomes when PyVista is not installed.

The 3D view is an optional extra, and VTK is 1.1 GB, so a machine without it
is an ordinary case rather than a broken one. This panel says what is missing,
how to get it, and offers the route that already exists: write the mesh and
the plan out and open them in ParaView.
"""
from __future__ import annotations

from typing import Callable, Optional


class Fallback3D:
  """A panel explaining the absent 3D view and offering the export route."""

  def __init__(self, parent, session, reason: str = '',
               on_status: Optional[Callable[[str], None]] = None):
    import tkinter as tk
    from tkinter import ttk

    self.session = session
    self.on_status = on_status or (lambda _: None)

    self.widget = ttk.Frame(parent, padding=24)
    ttk.Label(self.widget, text='3D view unavailable',
              font=('TkDefaultFont', 12, 'bold')).pack(anchor='w')
    ttk.Label(self.widget, wraplength=460, justify='left', text=(
      reason or 'PyVista and VTK are not installed.')).pack(anchor='w',
                                                            pady=(8, 16))
    ttk.Label(self.widget, wraplength=460, justify='left', text=(
      'Install the optional extra to enable it:\n\n'
      '    pip install "feelmri[gui]"\n\n'
      'Everything else in this window works without it. The plan can still be '
      'exported and opened in ParaView, which is the workflow this GUI sits '
      'alongside rather than replaces.')).pack(anchor='w')

    ttk.Button(self.widget, text='Export mesh and plan for ParaView',
               command=self._export).pack(anchor='w', pady=(16, 0))

  def _export(self) -> None:
    from tkinter import filedialog, messagebox

    if not self.session.has_mesh:
      messagebox.showinfo('Nothing to export', 'Load a phantom first.')
      return
    directory = filedialog.askdirectory(title='Export to')
    if not directory:
      return
    try:
      paths = export_for_paraview(self.session, directory)
    except Exception as exc:                       # surfaced, never swallowed
      messagebox.showerror('Export failed', str(exc))
      return
    self.on_status(f'wrote {len(paths)} file(s) to {directory}')
    messagebox.showinfo('Exported', '\n'.join(paths))

  # The no-VTK path still needs the camera controls to exist as no-ops, so the
  # shell can wire the same menu entries either way.
  def look(self, name: str) -> None:
    self.on_status(f'{name} view needs the 3D panel')

  def reset_view(self) -> None:
    pass

  def rebuild(self, keep_camera: bool = False) -> None:
    pass

  def refresh_overlay(self) -> None:
    pass

  def close(self) -> None:
    pass


def export_for_paraview(session, directory) -> list:
  """Write the surface as `.vtu` and the plan as `.pvsm`. Returns the paths.

  Kept a module-level function rather than a method so the same export is
  reachable from a script, which is the point of keeping logic out of panels.
  """
  import os

  import meshio
  import numpy as np

  written = []
  mesh_path = os.path.join(directory, 'phantom_surface.vtu')
  tris = session.surface
  meshio.write_points_cells(mesh_path, session.points, [('triangle', tris)])
  written.append(mesh_path)

  if session.box is not None:
    from ..model.pvsm import write_pvsm
    plan_path = os.path.join(directory, 'plan.pvsm')
    write_pvsm(plan_path, session.box.fov, session.box.loc,
               np.degrees(session.box.angles))
    written.append(plan_path)
  return written
