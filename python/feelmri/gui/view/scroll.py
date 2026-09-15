"""A scrollable container for a tab whose controls are taller than the window.

The control columns outgrew the window: the Plan tab ends in `Apply plan` and
an element count, the Sequence tab in three file buttons, and at the default
760-pixel height both were **cut off with no way to reach them** -- not
scrolled, simply absent. A theme with any padding at all makes it worse, and a
laptop screen makes it worse again.

Tk has no scrollable frame, so this is the standard construction: a `Canvas`
that scrolls, with a frame inside it as a canvas item. Two bindings are what
make it behave rather than merely exist -- the inner frame's `<Configure>`
keeps the scroll region at the content's real height, and the canvas's own
`<Configure>` forces the item to the canvas width, without which the content
keeps its requested width and a narrow column is left with a horizontal gap.
"""
from __future__ import annotations

from .theme import PALETTE


class ScrollableFrame:
  """`outer` goes in the notebook; build into `inner`."""

  def __init__(self, parent, width: int = 320):
    import tkinter as tk
    from tkinter import ttk

    self.outer = ttk.Frame(parent)
    self._canvas = tk.Canvas(self.outer, highlightthickness=0, borderwidth=0,
                             background=PALETTE['surface'], width=width)
    bar = ttk.Scrollbar(self.outer, orient='vertical',
                        command=self._canvas.yview)
    self.inner = ttk.Frame(self._canvas, padding=10)

    self._item = self._canvas.create_window((0, 0), window=self.inner,
                                            anchor='nw')
    self._canvas.configure(yscrollcommand=bar.set)
    self._canvas.pack(side='left', fill='both', expand=True)
    bar.pack(side='right', fill='y')

    self.inner.bind('<Configure>', self._on_content)
    self._canvas.bind('<Configure>', self._on_canvas)
    # Wheel events go to the widget under the pointer, which is usually a
    # child rather than the canvas, so they are bound on the whole subtree
    # when the pointer is inside.
    self.outer.bind('<Enter>', lambda _e: self._bind_wheel(True))
    self.outer.bind('<Leave>', lambda _e: self._bind_wheel(False))

  def _on_content(self, _event=None) -> None:
    self._canvas.configure(scrollregion=self._canvas.bbox('all'))

  def _on_canvas(self, event) -> None:
    self._canvas.itemconfigure(self._item, width=event.width)

  def _bind_wheel(self, on: bool) -> None:
    if on:
      self._canvas.bind_all('<Button-4>', self._wheel)
      self._canvas.bind_all('<Button-5>', self._wheel)
      self._canvas.bind_all('<MouseWheel>', self._wheel)
    else:
      for sequence in ('<Button-4>', '<Button-5>', '<MouseWheel>'):
        self._canvas.unbind_all(sequence)

  def _wheel(self, event) -> None:
    """X11 sends buttons 4 and 5; everything else sends `<MouseWheel>`."""
    if getattr(event, 'num', None) == 4:
      step = -1
    elif getattr(event, 'num', None) == 5:
      step = 1
    else:
      step = -1 if getattr(event, 'delta', 0) > 0 else 1
    self._canvas.yview_scroll(step, 'units')
