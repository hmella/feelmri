"""The pipeline browser: what is in the scene, and what is shown.

ParaView's top-left panel, with the part that applies here. There is no
pipeline to build -- no sources to add and no filters to chain -- so what is
listed is fixed: the phantom, the planned field of view, its M/P/S triad and
the glyph arrows. What carries over is the useful half, **an eye beside each
row**.

**Visibility is not opacity**, and conflating them is why this is worth a
panel. Opacity 0 still costs the render and still depth-sorts against
everything behind it; more to the point the two answer different questions --
"I do not want to see the plan right now" against "I want to see through it".
The viewer had only the second.

The rows are `ttk.Checkbutton`s with the indicator off and a filled or hollow
circle as their text, rather than a `ttk.Treeview` with a toggle column.
A Treeview looks more like ParaView and would give selection, which is what
drives ParaView's Properties panel -- but this Properties panel shows
everything at once by design, so the selection would be decoration, and
hit-testing a toggle column against a click is fiddly enough to get wrong.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from ..model.session import LAYERS
from .widgets import HIDDEN_MARK, SHOWN_MARK

#: Layer key to the name shown, in the order the panel lists them. A dict
#: rather than a parallel tuple, so a layer cannot be listed without a name.
LAYER_NAMES: Dict[str, str] = {
  'phantom': 'Phantom',
  'plan': 'Field of view',
  'arrows': 'M / P / S axes',
  'glyphs': 'Arrows (glyphs)',
}

assert tuple(LAYER_NAMES) == LAYERS, (
  'the pipeline browser and the session disagree about the layers')


class PipelineBrowser:
  """One row per drawable layer, each with an eye toggle."""

  def __init__(self, parent, session,
               on_status: Optional[Callable[[str], None]] = None):
    import tkinter as tk
    from tkinter import ttk

    self.session = session
    self.on_status = on_status or (lambda _: None)
    self.widget = ttk.Frame(parent)
    self.variables: Dict[str, tk.BooleanVar] = {}
    self._buttons: Dict[str, ttk.Checkbutton] = {}

    for layer, name in LAYER_NAMES.items():
      row = ttk.Frame(self.widget)
      row.pack(fill='x')
      variable = tk.BooleanVar(value=session.is_visible(layer))
      # `indicatoron=False` drops the tick box; the glyph IS the state, which
      # is what makes the column read as ParaView's eye rather than as a form.
      button = ttk.Checkbutton(
        row, variable=variable, style='Tool.TButton', width=2,
        text=SHOWN_MARK if variable.get() else HIDDEN_MARK,
        command=lambda l=layer: self._toggle(l))
      button.pack(side='left')
      ttk.Label(row, text=name).pack(side='left', padx=(6, 0))
      self.variables[layer] = variable
      self._buttons[layer] = button

    # The session is the authority: a layer can be switched from elsewhere,
    # and a browser showing the opposite of what is drawn is worse than none.
    session.view_changed.connect(lambda *_: self.refresh())
    session.plan_changed.connect(lambda *_: self.refresh())

  def _toggle(self, layer: str) -> None:
    self.session.set_visible(layer, bool(self.variables[layer].get()))
    self.refresh()
    self.on_status(f'{LAYER_NAMES[layer]} '
                   f'{"shown" if self.variables[layer].get() else "hidden"}')

  def refresh(self) -> None:
    for layer, variable in self.variables.items():
      shown = self.session.is_visible(layer)
      variable.set(shown)
      self._buttons[layer].config(
        text=SHOWN_MARK if shown else HIDDEN_MARK)
