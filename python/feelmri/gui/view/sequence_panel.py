"""The sequence drawn against time, with block picking.

`Sequence.plot` draws the same five rows, but it builds its own figure and ends
in `plt.show()`, so it cannot be embedded and cannot be interrogated. This panel
is the embedded version: it holds a matplotlib figure on a Tk canvas and asks
`gui.model.sequence.SequenceModel` for every array it draws, so the two cannot
disagree about what the sequence contains.

**No pyplot.** Embedding goes through `Figure` and `FigureCanvasTkAgg`
directly; importing pyplot here would install a second, competing event loop
and pull in whatever backend the environment happens to prefer.

**Why the RF row shows the REAL PART and not just the magnitude.** An offset
enters as a phase factor, so two pulses can share an envelope exactly and
differ entirely in what they do -- a magnitude-only row draws them
identically. Mis-signed frequency and phase offsets have twice been real
defects in this library, and this is the row on which one would be noticed.
The row therefore draws `Re(B1)` as the line, inside a faint `+-|B1|`
envelope that makes it readable as a pulse.

The envelope is not an invariant to lean on, though: the analytic generator
rescales a pulse to hold its requested flip angle, so adding a frequency
offset to a real `RF` moves the peak `|B1|` too -- measured at 5.4x of peak on
a 90 degree hard pulse with a 3 kHz offset. Only the drawing contract is
guaranteed here; the physics belongs to the library's own tests.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from ..model.sequence import AXIS_LABELS, SequenceModel
from .theme import PALETTE, TEXT_OPTIONS, style_figure

#: Rows, top to bottom, with their relative heights. The three gradient rows
#: take their names from the model's own `AXIS_LABELS`, so the row a trace is
#: drawn on and the axis it came from cannot drift apart.
ROWS = ((('RF', 2.0),) + tuple((name, 2.0) for name in AXIS_LABELS)
        + (('ADC', 0.7),))

#: Above this many block edges the guide lines are dropped. They stop being
#: readable long before they stop being slow, and `mprage_pypulseq.seq` has
#: 5940 blocks, which would put a line every 0.2 pixels.
MAX_BOUNDARY_LINES = 600


# -- drawing ----------------------------------------------------------------
#
# These are module level and take an axis, so every one of them is testable
# against a bare `matplotlib.figure.Figure` with no Tk, no canvas and no
# display -- the same reason `gui.model` has no view imports. `SequencePanel`
# below is then only wiring.


def make_axes(figure) -> List:
  """Five rows sharing one time axis, so a zoom moves all of them."""
  axes = figure.subplots(len(ROWS), 1, sharex=True,
                         gridspec_kw={'height_ratios': [h for _, h in ROWS]})
  units = {'RF': 'Re(B1)\n[uT]', 'ADC': 'ADC'}
  for ax, (name, _) in zip(axes, ROWS):
    ax.set_ylabel(units.get(name, f'G{name}\n[mT/m]'), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.18, linewidth=0.5, color=PALETTE['border'])
  axes[-1].set_xlabel('time [ms]', fontsize=8)
  axes[-1].set_yticks([])
  axes = list(axes)
  style_figure(figure, axes)
  return axes


def draw_rf(ax, model: SequenceModel) -> None:
  """`Re(B1)` in microtesla, inside a faint `+-|B1|` envelope.

  The real part is the informative one: an offset enters as a phase factor, so
  two pulses can share an envelope exactly and differ entirely in what they
  do, and a magnitude-only row would draw them identically.
  """
  for t, w in model.rf_traces():
    if np.size(t) == 0:
      continue
    w = np.asarray(w)
    magnitude = np.abs(w) * 1e3                   # mT to uT
    ax.plot(t, np.real(w) * 1e3, linewidth=1.0, color=PALETTE['rf'])
    ax.plot(t, magnitude, linewidth=0.7, color=PALETTE['rf'], alpha=0.35)
    ax.plot(t, -magnitude, linewidth=0.7, color=PALETTE['rf'], alpha=0.35)


def draw_gradients(ax, model: SequenceModel, axis: int) -> None:
  """One line per gradient, never one joined line.

  The gaps between gradients are real; joining them draws a ramp that is not
  played, which is exactly the picture a reader would act on.
  """
  for t, a in model.gradient_traces(axis):
    if np.size(t):
      ax.plot(t, a, linewidth=1.0, color=PALETTE['trace'])


def draw_adc(ax, model: SequenceModel) -> None:
  times = model.adc_times()
  if times.size:
    ax.vlines(times, 0.0, 1.0, linewidth=0.5, color=PALETTE['adc'])
  ax.set_ylim(0.0, 1.0)


def draw_boundaries(axes, model: SequenceModel) -> int:
  """Block edges as guide lines. Returns how many were drawn.

  Dropped above `MAX_BOUNDARY_LINES`: they stop being readable long before
  they stop being cheap, and at 5940 blocks a line lands every 0.2 pixels,
  which is a grey wash rather than information. Picking still works.
  """
  edges = model.boundaries()
  if edges.size == 0 or edges.size > MAX_BOUNDARY_LINES:
    return 0
  for ax in axes:
    ax.vlines(edges, 0, 1, transform=ax.get_xaxis_transform(),
              linewidth=0.4, color=PALETTE['border'], zorder=0)
  return int(edges.size)


def row_target(row: int):
  """`(kind, axis)` for a row index, or `(None, None)` for an unknown one.

  The row a user clicked already fixes what kind of object they meant, and
  for a gradient which axis. Passing that to `SequenceModel.object_at` is
  what makes a click unambiguous where an RF, a gradient and an ADC overlap.
  """
  if row == 0:
    return 'rf', None
  if 1 <= row <= 3:
    return 'gradient', row - 1
  if row == len(ROWS) - 1:
    return 'adc', None
  return None, None


def draw_sequence(axes, model: SequenceModel) -> None:
  """Fill all five rows. The one entry point a panel or a script needs."""
  draw_rf(axes[0], model)
  for axis in (0, 1, 2):
    draw_gradients(axes[1 + axis], model, axis)
  draw_adc(axes[-1], model)
  draw_boundaries(axes, model)


class SequencePanel:
  """RF, the three gradient axes and the ADC against absolute time.

  Usable on its own -- give it any Tk parent and a `Session` -- so a script
  that wants to look at a sequence does not have to start the whole shell.
  See `show_sequence` at the bottom of this module for that shortcut.
  """

  def __init__(self, parent, session,
               on_status: Optional[Callable[[str], None]] = None,
               on_pick: Optional[Callable[[int], None]] = None):
    from tkinter import ttk

    try:
      from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                     NavigationToolbar2Tk)
      from matplotlib.figure import Figure
    except ImportError as exc:                        # pragma: no cover
      raise RuntimeError(
        f'The sequence panel needs matplotlib: pip install matplotlib '
        f'({exc})') from exc

    self.session = session
    self.on_status = on_status or (lambda _: None)
    self.on_pick = on_pick or (lambda *_: None)
    self.selected: Optional[int] = None
    self.selected_object = None
    self._highlight = []

    self.widget = ttk.Frame(parent)

    header = ttk.Frame(self.widget)
    header.pack(fill='x')
    ttk.Label(header, text='Sequence',
              font=('TkDefaultFont', 11, 'bold')).pack(side='left', padx=(6, 12))
    self._caption = ttk.Label(header, text='none loaded')
    self._caption.pack(side='left')
    ttk.Button(header, text='Fit', width=5,
               command=self.fit).pack(side='right', padx=6)

    self._figure = Figure(figsize=(7.0, 3.6), dpi=100, layout='constrained')
    self._axes = make_axes(self._figure)
    self._canvas = FigureCanvasTkAgg(self._figure, master=self.widget)

    # **Pack order is the layout.** Tk allocates in the order pack() is
    # called, so the expanding canvas has to come LAST or it takes the whole
    # panel and the toolbar and readout are squeezed off the bottom -- which
    # is exactly what happened, and what hides the numbers a picked block
    # exists to show.
    # Packed bottom-up, so the readout goes first and the toolbar lands
    # directly under the plot where it belongs.
    self._details = self._make_details(ttk)
    self._toolbar = self._make_toolbar(ttk, NavigationToolbar2Tk)
    self._canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
    self._canvas.mpl_connect('button_press_event', self._on_click)

    session.sequence_changed.connect(lambda *_: self.rebuild())
    session.labels_changed.connect(lambda *_: self._refresh_details())
    self.rebuild()

  # -- construction ---------------------------------------------------------

  def _make_toolbar(self, ttk, NavigationToolbar2Tk):
    """The toolbar needs its own frame: it calls pack() on itself, which would
    otherwise fight the canvas for the same parent."""
    bar = ttk.Frame(self.widget)
    bar.pack(side='bottom', fill='x')
    toolbar = NavigationToolbar2Tk(self._canvas, bar, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side='left', fill='x')
    return toolbar

  def _make_details(self, ttk):
    import tkinter as tk

    frame = ttk.Frame(self.widget, padding=(6, 4))
    frame.pack(side='bottom', fill='x')
    self._details_title = ttk.Label(
      frame, text='Click a block to inspect it.',
      font=('TkDefaultFont', 9, 'bold'))
    self._details_title.pack(anchor='w')
    text = tk.Text(frame, height=6, width=60, font=('TkFixedFont', 9),
                   **TEXT_OPTIONS)
    text.pack(fill='x')
    text.configure(state='disabled')
    return text

  # -- drawing --------------------------------------------------------------

  @property
  def model(self) -> Optional[SequenceModel]:
    return getattr(self.session, 'sequence', None)

  def rebuild(self) -> None:
    """Redraw from scratch. Cheap enough: the traces come from a cache."""
    for ax in self._axes:
      for artist in list(ax.lines) + list(ax.collections):
        artist.remove()
    self._highlight = []
    self.selected = None
    self.selected_object = None

    model = self.model
    if model is None or not model.spans:
      self._caption.config(text='none loaded')
      self._set_details('Click a block to inspect it.', [])
      self._canvas.draw_idle()
      return

    draw_sequence(self._axes, model)

    path = getattr(self.session, 'sequence_path', None)
    name = path.rsplit('/', 1)[-1] if path else 'sequence'
    self._caption.config(
      text=f'{name} -- {len(model.spans)} blocks, {model.duration:.2f} ms')
    self.fit()





  def fit(self) -> None:
    """Show the whole sequence. Also the way back from a toolbar zoom."""
    model = self.model
    if model is None or not model.spans:
      self._canvas.draw_idle()
      return
    pad = max(model.duration * 0.01, 1e-6)
    self._axes[0].set_xlim(model.t_start - pad, model.t_end + pad)
    for ax in self._axes[:-1]:
      ax.relim()
      ax.autoscale_view(scalex=False, scaley=True)
    self._canvas.draw_idle()

  # -- picking --------------------------------------------------------------

  def _on_click(self, event) -> None:
    """Pick the block under the cursor.

    Ignored while a toolbar tool is armed, or the first click of a zoom
    rectangle would also select whatever it started on.
    """
    if event.inaxes is None or event.xdata is None:
      return
    if getattr(self._toolbar, 'mode', ''):
      return
    model = self.model
    if model is None:
      return
    t = float(event.xdata)
    index = model.block_at(t)
    if index is None:
      return
    try:
      row = list(self._axes).index(event.inaxes)
    except ValueError:
      row = -1
    kind, axis = row_target(row)
    found = None if kind is None else model.object_at(t, kind=kind, axis=axis)
    self.select_block(index, obj=found)

  def select_block(self, index: int, obj=None) -> None:
    """Highlight one block, and one object within it when there is one.

    The object is what a label can attach to beyond the block, so it is
    carried rather than derived again by whoever handles `on_pick`.
    """
    model = self.model
    if model is None or not model.spans:
      return
    index = int(index) % len(model.spans)
    self.selected = index
    self.selected_object = obj
    span = model.spans[index]

    for patch in self._highlight:
      patch.remove()
    self._highlight = [
      ax.axvspan(span.t0, span.t1, color=PALETTE['highlight'], alpha=0.22, zorder=0)
      for ax in self._axes]
    if obj is not None:
      row = {'rf': 0, 'adc': len(ROWS) - 1}.get(
        obj.kind, 1 + (obj.axis or 0))
      self._highlight.append(
        self._axes[row].axvspan(obj.t0, obj.t1,
                                color=PALETTE['highlight'],
                                alpha=0.5, zorder=0))

    self._refresh_details()
    self._canvas.draw_idle()
    where = obj.label if obj is not None else f'block {index}'
    self.on_status(f'{where}: {span.t0:.3f} to {span.t1:.3f} ms')
    self.on_pick(index, obj)

  def _refresh_details(self) -> None:
    model = self.model
    if model is None or self.selected is None:
      return
    rows = model.describe_block(self.selected)
    rows += self._label_rows(self.selected)
    obj = self.selected_object
    if obj is not None:
      rows = ([('selected', obj.label),
               ('span', f'{obj.t0:.4f} to {obj.t1:.4f} ms')]
              + rows + self._object_label_rows(obj))
    title = obj.label if obj is not None else f'Block {self.selected}'
    self._set_details(title, rows)

  def _label_rows(self, index: int) -> List:
    """Whatever the label store says about this block, if there is one."""
    store = getattr(self.session, 'labels', None)
    if store is None:
      return []
    try:
      labels = store.labels_on(index)
    except Exception:
      return []
    if not labels:
      return [('labels', 'none')]
    return [('labels', ', '.join(f'{k}={v}' for k, v in sorted(labels.items())))]

  def _object_label_rows(self, obj) -> List:
    store = getattr(self.session, 'labels', None)
    if store is None:
      return []
    try:
      labels = store.labels_on_object(obj.block, obj.kind, obj.ordinal)
    except Exception:
      return []
    if not labels:
      return [('object labels', 'none')]
    return [('object labels',
             ', '.join(f'{k}={v}' for k, v in sorted(labels.items())))]

  def _set_details(self, title: str, rows) -> None:
    self._details_title.config(text=title)
    self._details.configure(state='normal')
    self._details.delete('1.0', 'end')
    for key, value in rows:
      self._details.insert('end', f'{key:<12s} {value}\n')
    self._details.configure(state='disabled')

  def close(self) -> None:
    """Release the figure. Tk destroys the widgets on its own."""
    try:
      self._canvas.get_tk_widget().destroy()
    except Exception:
      pass


def show_sequence(sequence, path=None, title='feelmri  sequence'):
  """Open a standalone window showing one sequence, and block until closed.

  The shortcut for a script or a notebook that wants the viewer without the
  rest of the shell:

      from feelmri.PulseqAdapter import import_pulseq
      from feelmri.gui.view.sequence_panel import show_sequence
      show_sequence(import_pulseq('gre_v15.seq').feelmri_seq)
  """
  import tkinter as tk

  from ..model.session import Session

  session = Session()
  session.set_sequence(sequence, path=path)

  root = tk.Tk()
  root.title(title)
  root.geometry('1000x700')
  panel = SequencePanel(root, session)
  panel.widget.pack(fill='both', expand=True)
  root.protocol('WM_DELETE_WINDOW', root.destroy)
  root.mainloop()
  return session
