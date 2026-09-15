"""A finished run: its image, its k-space and the trajectory that sampled it.

The plan's acceptance was that a result displays **without writing a `.vti`
first**, so this is the panel that closes the loop: a run writes `kspace.npz`,
the session reads it back, and this draws it. ParaView stays available and is
no longer the only way to look at an answer.

As in `sequence_panel`, the drawing is module-level functions taking an axis,
so every one of them is testable against a bare Agg `Figure` with no Tk and no
display. `ResultsPanel` is wiring over them.

**Reconstruction is the library's `Recon.reconstruct_nufft`**, reached through
`gui.model.results`; nothing here grids k-space itself.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from ..model import results as results_model

#: The four panes, in the order they are laid out.
PANES = ('magnitude', 'phase', 'trajectory', 'k-space')


def make_axes(figure) -> List:
  """Two by two: the image twice, then how it was sampled."""
  axes = figure.subplots(2, 2)
  flat = list(np.asarray(axes).ravel())
  for ax, name in zip(flat, PANES):
    ax.set_title(name, fontsize=9)
    ax.tick_params(labelsize=7)
  return flat


def draw_image(ax, image, kind: str = 'magnitude') -> None:
  """One slice of a complex image, as magnitude or as phase.

  Phase is drawn on a cyclic colour map and a fixed `[-pi, pi]` range: an
  autoscaled phase image invents contrast from whatever range the data
  happens to span, which reads as structure that is not there.
  """
  if image is None:
    ax.set_axis_off()
    return
  data = np.asarray(image)
  while data.ndim > 2:
    data = data[..., 0] if data.shape[-1] == 1 else data[..., data.shape[-1] // 2]
  if kind == 'phase':
    ax.imshow(np.angle(data).T, origin='lower', cmap='twilight',
              vmin=-np.pi, vmax=np.pi)
  else:
    ax.imshow(np.abs(data).T, origin='lower', cmap='gray')
  ax.set_xticks([])
  ax.set_yticks([])


def draw_trajectory(ax, result) -> None:
  """Where k-space was sampled, in plane.

  Drawn as points rather than a line: the samples of an EPI train are not
  evenly spaced -- ramp sampling puts them closer together on the ramps --
  and joining them draws a path the gradients did follow but hides that.
  """
  if not result:
    ax.set_axis_off()
    return
  kx, ky, _kz = results_model.trajectory(result)
  if kx.size == 0:
    ax.set_axis_off()
    return
  ax.scatter(kx, ky, s=0.5, linewidths=0, color='tab:blue')
  ax.set_xlabel('kx [1/m]', fontsize=8)
  ax.set_ylabel('ky [1/m]', fontsize=8)
  ax.set_aspect('equal', adjustable='datalim')


def draw_kspace(ax, result) -> None:
  """`|k|` against sample index, on a log scale.

  Log because the dynamic range is the point: the centre of k-space is orders
  of magnitude above the edges, and on a linear axis everything but the first
  few samples is a flat line at zero.
  """
  if not result or 'kspace' not in result:
    ax.set_axis_off()
    return
  magnitude = np.abs(np.asarray(result['kspace']).ravel())
  if magnitude.size == 0:
    ax.set_axis_off()
    return
  ax.semilogy(np.arange(magnitude.size), np.maximum(magnitude, 1e-30),
              linewidth=0.6, color='tab:green')
  ax.set_xlabel('sample', fontsize=8)
  ax.set_ylabel('|k|', fontsize=8)


def draw_results(axes, result, image=None) -> None:
  """Fill all four panes. The one entry point a panel or a script needs."""
  draw_image(axes[0], image, 'magnitude')
  draw_image(axes[1], image, 'phase')
  draw_trajectory(axes[2], result)
  draw_kspace(axes[3], result)


class ResultsPanel:
  """Show the session's result, and reconstruct it on demand."""

  def __init__(self, parent, session,
               on_status: Optional[Callable[[str], None]] = None):
    import tkinter as tk
    from tkinter import ttk

    try:
      from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
      from matplotlib.figure import Figure
    except ImportError as exc:                       # pragma: no cover
      raise RuntimeError(
        f'The results panel needs matplotlib ({exc})') from exc

    self.session = session
    self.on_status = on_status or (lambda _: None)
    self.image = None

    self.widget = ttk.Frame(parent, padding=10)

    ttk.Label(self.widget, text='Image matrix').pack(anchor='w')
    self.matrix = tk.StringVar(value='')
    ttk.Entry(self.widget, textvariable=self.matrix).pack(fill='x')
    # Two rows: side by side these are clipped at this column width.
    buttons = ttk.Frame(self.widget)
    buttons.pack(fill='x', pady=(6, 0))
    ttk.Button(buttons, text='Reconstruct',
               command=self.reconstruct).pack(side='left', expand=True,
                                              fill='x')
    ttk.Button(buttons, text='Open result...',
               command=self.open_result).pack(side='left', expand=True,
                                              fill='x', padx=(6, 0))

    self._caption = ttk.Label(self.widget, text='no result yet',
                              wraplength=280)
    self._caption.pack(anchor='w', pady=(6, 0))

    self._figure = Figure(figsize=(5.0, 4.4), dpi=100, layout='constrained')
    self._axes = make_axes(self._figure)
    self._canvas = FigureCanvasTkAgg(self._figure, master=self.widget)

    self._details = tk.Text(self.widget, height=9, width=36,
                            font=('TkFixedFont', 8), relief='flat',
                            background=self.widget.winfo_toplevel().cget('bg'))
    self._details.pack(side='bottom', fill='x', pady=(4, 0))
    self._details.configure(state='disabled')
    self._canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    session.result_changed.connect(lambda *_: self.refresh())
    self.refresh()

  # -- display --------------------------------------------------------------

  @property
  def result(self):
    return getattr(self.session, 'result', None)

  def refresh(self) -> None:
    """Redraw from the session. A new result drops the old image with it."""
    self.image = None
    self._redraw()

  def _redraw(self) -> None:
    for ax in self._axes:
      ax.clear()
      ax.set_axis_on()
    for ax, name in zip(self._axes, PANES):
      ax.set_title(name, fontsize=9)
      ax.tick_params(labelsize=7)

    result = self.result
    draw_results(self._axes, result, self.image)
    if not result:
      self._caption.config(text='no result yet -- run a simulation, or open '
                                'a kspace.npz')
      self._set_details([])
      self.matrix.set('')
    else:
      path = getattr(self.session, 'result_path', None)
      name = path.rsplit('/', 1)[-1] if path else 'result'
      self._caption.config(text=name)
      self._set_details(results_model.describe(result))
      if not self.matrix.get().strip():
        # The matrix the trajectory actually supports, so the default
        # reconstruction is at the resolution that was acquired rather than
        # a round number the user has to correct.
        self.matrix.set(' '.join(
          str(v) for v in results_model.default_matrix(result)))
    self._canvas.draw_idle()

  def _set_details(self, rows) -> None:
    self._details.configure(state='normal')
    self._details.delete('1.0', 'end')
    for key, value in rows:
      self._details.insert('end', f'{key:<16s} {value}\n')
    self._details.configure(state='disabled')

  # -- actions --------------------------------------------------------------

  def reconstruct(self) -> None:
    from tkinter import messagebox

    result = self.result
    if not result:
      messagebox.showinfo('No result', 'Run a simulation first, or open a '
                                       'kspace.npz.')
      return
    try:
      parts = self.matrix.get().replace(',', ' ').split()
      if not parts:
        parts = [str(v) for v in results_model.default_matrix(result)]
        self.matrix.set(' '.join(parts))
      if len(parts) != 3:
        raise ValueError('the matrix is three integers, for example "64 64 1"')
      self.on_status('reconstructing...')
      self.image = results_model.reconstruct(result,
                                             matrix=[int(p) for p in parts])
    except Exception as exc:
      self.on_status('reconstruction failed')
      messagebox.showerror('Could not reconstruct', str(exc))
      return
    self._redraw()
    self.on_status(f'reconstructed {self.image.shape}')

  def open_result(self) -> None:
    """Open a `kspace.npz` from an earlier run.

    A result outlives the run that made it, so this does not need a launch.
    """
    from tkinter import filedialog, messagebox

    path = filedialog.askopenfilename(
      title='Open result', filetypes=[('Run result', '*.npz'),
                                      ('All files', '*')])
    if not path:
      return
    try:
      self.session.load_result(path)
    except Exception as exc:
      messagebox.showerror('Could not open the result', str(exc))
      return
    self.on_status(f'loaded {path}')
