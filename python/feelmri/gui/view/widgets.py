"""Small Tk pieces the control column is built from.

Tk ships neither a collapsible section nor a labelled row, and the Plan tab
needs both: it had grown to fourteen controls in one unbroken scroll with no
headings, which is navigable only by remembering the order they were added in.

Nothing here knows about a `Session`. They take a parent and give back a
widget, so a panel composes them and this file stays testable by inspection.
"""
from __future__ import annotations

from typing import Callable, Optional

#: Shown on a section header. Plain text rather than an image, so there is no
#: asset to ship and no scaling to get wrong on a high-DPI display.
OPEN_MARK, SHUT_MARK = '▼', '▶'          # down and right triangles

#: Shown on a visibility toggle. Two characters wide either way, so the names
#: beside them line up whatever the state.
SHOWN_MARK, HIDDEN_MARK = '◉', '○'       # filled and hollow circles


class CollapsibleSection:
  """A titled group whose body folds away, as ParaView's property groups do.

  Build into `.body`; `.widget` is what the caller packs.

  Collapsing matters more here than it looks. The control column carries the
  plan, the display settings and the pipeline at once, and a user placing a
  field of view wants the plan expanded and everything else out of the way --
  which on a 760-pixel window is the difference between scrolling and not.
  """

  def __init__(self, parent, title: str, expanded: bool = True,
               on_toggle: Optional[Callable[[bool], None]] = None):
    from tkinter import ttk

    self.on_toggle = on_toggle or (lambda _shown: None)
    self.expanded = bool(expanded)

    self.widget = ttk.Frame(parent)
    self._header = ttk.Label(self.widget, style='Section.TLabel',
                             text=self._title(title))
    self._header.pack(fill='x')
    self._title_text = title
    self.body = ttk.Frame(self.widget, padding=(8, 4, 2, 6))
    if self.expanded:
      self.body.pack(fill='x')

    # Bound on the label rather than built as a Button: a full-width ttk
    # button draws a raised border that reads as a control rather than as a
    # heading, which is the opposite of what a section title should look like.
    self._header.bind('<Button-1>', lambda _e: self.toggle())

  def _title(self, title: str) -> str:
    return f'{OPEN_MARK if self.expanded else SHUT_MARK}  {title}'

  def toggle(self) -> None:
    self.expanded = not self.expanded
    self._header.config(text=self._title(self._title_text))
    if self.expanded:
      self.body.pack(fill='x')
    else:
      self.body.pack_forget()
    self.on_toggle(self.expanded)

  def set_expanded(self, expanded: bool) -> None:
    if bool(expanded) != self.expanded:
      self.toggle()


def labelled_row(parent, text: str, widget_factory, width: int = 13):
  """A label on the left and one control on the right, on a single line.

  Returns whatever `widget_factory(row)` returns. Stacking a label ABOVE each
  control -- which is what this column did -- costs a line per control and
  makes a dozen of them twice as tall as the window.
  """
  from tkinter import ttk

  row = ttk.Frame(parent)
  row.pack(fill='x', pady=1)
  ttk.Label(row, text=text, width=width).pack(side='left')
  widget = widget_factory(row)
  widget.pack(side='right', fill='x', expand=True)
  return widget


class ValueSlider:
  """A scale with its number beside it, committing ON RELEASE.

  Not on every motion event: each write redraws the scene, and dragging across
  a hundred thousand triangles would rebuild it per pixel. The readout follows
  the handle live, so the drag still reads as continuous.
  """

  def __init__(self, parent, text: str, low: float, high: float,
               start: float, on_commit: Callable[[float], None],
               fmt: str = '{:.2f}'):
    import tkinter as tk
    from tkinter import ttk

    self.fmt = fmt
    self.on_commit = on_commit

    row = ttk.Frame(parent)
    row.pack(fill='x', pady=(4, 0))
    ttk.Label(row, text=text, width=13).pack(side='left')
    self._readout = ttk.Label(row, text=fmt.format(start), width=6,
                              anchor='e', style='Muted.TLabel')
    self._readout.pack(side='right')

    self.variable = tk.DoubleVar(value=start)
    self.scale = ttk.Scale(parent, from_=low, to=high, variable=self.variable)
    self.scale.pack(fill='x')
    self.scale.configure(command=lambda _v: self._readout.config(
      text=self.fmt.format(self.variable.get())))
    self.scale.bind('<ButtonRelease-1>',
                    lambda _e: self.on_commit(float(self.variable.get())))

  def set(self, value: float) -> None:
    """Move the handle and its readout without firing the commit."""
    self.variable.set(float(value))
    self._readout.config(text=self.fmt.format(float(value)))
