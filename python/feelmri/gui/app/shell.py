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

from ..model.session import Session
from ..view.theme import TEXT_OPTIONS


class Shell:
  """The application window."""

  def __init__(self, session: Optional[Session] = None):
    import tkinter as tk
    from tkinter import ttk

    self.session = session or Session()
    self.root = tk.Tk()
    self.root.title('feelmri')
    self.root.geometry('1180x760')

    # Before any widget is built: a ttk Style set afterwards still reaches
    # them, but plain Tk widgets read their options at construction.
    from ..view.theme import apply as apply_theme
    self.palette = apply_theme(self.root)

    self._status = tk.StringVar(value='ready')

    # The status bar is packed BEFORE the body: Tk allocates in pack order, so
    # an expanding body packed first takes the whole window and squeezes the
    # bar off the bottom entirely.
    ttk.Label(self.root, textvariable=self._status, anchor='w',
              style='Status.TLabel').pack(fill='x', side='bottom')

    body = ttk.Frame(self.root)
    body.pack(fill='both', expand=True)

    self._tabs = ttk.Notebook(body, width=340)
    self._tabs.pack(side='left', fill='y')
    self._tabs.pack_propagate(False)
    # Each tab scrolls: the control columns are taller than the window at any
    # sensible size, and without this the last few widgets are simply
    # unreachable rather than merely off screen.
    self.controls = self._scrollable('Plan')

    # One window, and the right-hand side shows the view that belongs to the
    # selected tab: the 3D scene while planning, the sequence while labelling,
    # the images once there is a result. Everything is built once and only the
    # visible one is packed, so switching costs nothing and no state is lost.
    right = ttk.Frame(body)
    right.pack(side='right', fill='both', expand=True)
    # The toolbar strip is packed BEFORE the deck, which expands: Tk allocates
    # in pack order, and a toolbar packed after an expanding sibling is
    # squeezed to nothing.
    self._toolbar_host = ttk.Frame(right)
    self._toolbar_host.pack(fill='x', side='top')
    self._deck = ttk.Frame(right)
    self._deck.pack(fill='both', expand=True)

    self.viewport = self._make_viewport(self._deck)
    self.toolbar = self._make_toolbar(self._toolbar_host)
    self.sequence_panel = self._make_sequence_panel(self._deck)

    self.label_panel = self._make_label_panel(self._scrollable('Sequence'))
    if self.label_panel is not None:
      self.label_panel.widget.pack(fill='both', expand=True)

    self.run_panel = self._make_run_panel(self._scrollable('Run'))
    if self.run_panel is not None:
      self.run_panel.widget.pack(fill='both', expand=True)

    # The CONTROLS are the tab; the figure is a deck view with its own parent,
    # so swapping the right-hand side cannot unpack the tab.
    self.results_panel = self._make_results_panel(self._scrollable('Results'))
    if self.results_panel is not None:
      self.results_panel.controls.pack(fill='both', expand=True)

    # Which view each tab shows on the right. Run keeps the sequence up: it is
    # what a run is about to play, and the log lives in the tab itself.
    self._views = {
      'Plan': self.viewport,
      'Sequence': self.sequence_panel,
      'Run': self.sequence_panel,
      'Results': self.results_panel,
    }
    self._shown = None
    self._tabs.bind('<<NotebookTabChanged>>', lambda _e: self._show_view())
    self._show_view()

    # After the viewport: the View menu binds straight to its methods.
    self._build_menu()
    self._build_controls()
    self.session.mesh_changed.connect(lambda *_: self._on_mesh())
    self.root.protocol('WM_DELETE_WINDOW', self.close)

  # -- construction ---------------------------------------------------------

  def _scrollable(self, title: str):
    """Add a scrolling tab and return the frame to build into."""
    from ..view.scroll import ScrollableFrame

    scroller = ScrollableFrame(self._tabs)
    self._tabs.add(scroller.outer, text=title)
    return scroller.inner

  def _make_viewport(self, parent):
    """Pick a 3D backend, via `FEELMRI_GUI_VIEWPORT`.

    | value | what it does |
    |---|---|
    | `embedded` (default) | native VTK reparented INTO this window |
    | `window` | native VTK as its own top-level window |
    | `blit` | offscreen render blitted into a Tk canvas |

    Embedding is the default because it is the only one that gives both: a
    single window AND a real `vtkRenderWindowInteractor`, which is what
    `add_box_widget` -- the draggable field of view -- needs. The blit has no
    interactor at all, so the box would have to be typed rather than dragged.

    Reparenting is X11-specific. `Window3D` falls back to its own top-level
    window if it fails, so `window` is the explicit form of that rather than a
    different code path.
    """
    import os

    from ..view.fallback3d import Fallback3D

    choice = os.environ.get('FEELMRI_GUI_VIEWPORT', 'embedded').lower()
    if choice not in ('embedded', 'window', 'blit'):
      raise ValueError(
        f'FEELMRI_GUI_VIEWPORT must be "embedded", "window" or "blit", '
        f'got {choice!r}')
    try:
      if choice == 'blit':
        from ..view.canvas3d import Canvas3D
        return Canvas3D(parent, self.session, on_status=self.status)
      from ..view.window3d import Window3D
      viewport = Window3D(parent, self.session, on_status=self.status,
                          embed=choice == 'embedded')
      self._start_pump(viewport)
      return viewport
    except Exception as exc:
      self.status('3D view unavailable, see the panel')
      return Fallback3D(parent, self.session, reason=str(exc),
                        on_status=self.status)

  def _show_view(self) -> None:
    """Pack the view belonging to the selected tab, and unpack the last one.

    Unpacking rather than rebuilding: the 3D scene, the sequence figure and a
    reconstructed image are all expensive, and a tab switch must not throw
    them away. An embedded render window simply unmaps with its host frame and
    comes back when it is packed again.
    """
    try:
      name = self._tabs.tab(self._tabs.select(), 'text')
    except Exception:
      return                          # no tab yet, during construction
    view = self._views.get(name)
    if view is self._shown:
      return
    if self._shown is not None:
      try:
        self._shown.widget.pack_forget()
      except Exception:
        pass
    self._shown = view
    if view is not None:
      view.widget.pack(fill='both', expand=True)
      self.root.update_idletasks()

  def _make_toolbar(self, parent):
    """The strip over the view, or nothing if it cannot be built.

    Optional the same way every other panel is: a missing one costs the
    toolbar, not the application.
    """
    try:
      from ..view.toolbar import Toolbar
      bar = Toolbar(parent, self.session, self.viewport, on_status=self.status)
      bar.widget.pack(fill='x')
      return bar
    except Exception as exc:
      self.status(f'toolbar unavailable: {exc}')
      return None

  def _make_label_panel(self, parent):
    """The labels tab, or nothing if it cannot be built."""
    try:
      from ..view.label_panel import LabelPanel
      return LabelPanel(parent, self.session, on_status=self.status)
    except Exception as exc:
      self.status(f'label panel unavailable: {exc}')
      return None

  def _make_run_panel(self, parent):
    """The run tab, or nothing if it cannot be built."""
    try:
      from ..view.run_panel import RunPanel
      return RunPanel(parent, self.session, on_status=self.status)
    except Exception as exc:
      self.status(f'run panel unavailable: {exc}')
      return None

  def _make_results_panel(self, parent):
    """The results tab, or nothing if it cannot be built."""
    try:
      from ..view.image_panel import ResultsPanel
      return ResultsPanel(parent, self.session, on_status=self.status,
                          view_parent=self._deck)
    except Exception as exc:
      self.status(f'results panel unavailable: {exc}')
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

    from ..view.theme import MENU_OPTIONS

    bar = tk.Menu(self.root, **MENU_OPTIONS)

    file_menu = tk.Menu(bar, tearoff=0, **MENU_OPTIONS)
    file_menu.add_command(label='Open phantom...', command=self.open_phantom)
    file_menu.add_command(label='Open sequence (.seq)...',
                          command=self.open_sequence)
    file_menu.add_separator()
    file_menu.add_command(label='Import plan (.pvsm)...', command=self.import_plan)
    file_menu.add_command(label='Export plan (.pvsm)...', command=self.export_plan)
    file_menu.add_separator()
    file_menu.add_command(label='Quit', command=self.close)
    bar.add_cascade(label='File', menu=file_menu)

    view_menu = tk.Menu(bar, tearoff=0, **MENU_OPTIONS)
    for name in ('axial', 'coronal', 'sagittal'):
      view_menu.add_command(label=name.capitalize(),
                            command=lambda n=name: self.viewport.look(n))
    view_menu.add_separator()
    view_menu.add_command(label='Reset', command=self.viewport.reset_view)
    bar.add_cascade(label='View', menu=view_menu)

    self.root.config(menu=bar)

  def _build_controls(self) -> None:
    """The Plan tab, laid out the way ParaView lays out its left-hand side.

    Top to bottom: a pipeline browser of what is in the scene, then the
    properties of the plan with `Apply` at the head of them, then the display
    settings, then an information panel. Each is a collapsible section, which
    the column badly needed -- it had grown to fourteen controls in one
    unbroken scroll with no headings, navigable only by remembering the order
    they had been added in.
    """
    import tkinter as tk
    from tkinter import ttk

    from ..view.pipeline import PipelineBrowser
    from ..view.planning_panel import PlanningPanel
    from ..view.widgets import CollapsibleSection, ValueSlider

    self.sections = {}

    def section(title: str, expanded: bool = True):
      part = CollapsibleSection(self.controls, title, expanded=expanded)
      part.widget.pack(fill='x', pady=(0, 4))
      self.sections[title] = part
      return part.body

    # -- Pipeline ----------------------------------------------------------
    self.pipeline = PipelineBrowser(section('Pipeline'), self.session,
                                    on_status=self.status)
    self.pipeline.widget.pack(fill='x')

    # -- Properties: the plan, with Apply at the head of it ----------------
    body = section('Properties')
    self.planning_panel = PlanningPanel(body, self.session,
                                        on_status=self.status)
    self.planning_panel.widget.pack(fill='x')
    # Kept as the panel's own, so a caller reaching for the entries finds one
    # set rather than two that can disagree.
    self._plan_vars = self.planning_panel.vars

    # -- Display -----------------------------------------------------------
    body = section('Display')
    self.sliders = {}
    self.sliders['opacity'] = ValueSlider(
      body, 'Opacity', 0.0, 1.0, 1.0,
      lambda v: setattr(self.session, 'opacity', v))

    ttk.Separator(body).pack(fill='x', pady=8)
    ttk.Label(body, text='Warp by deformation',
              style='Muted.TLabel').pack(anchor='w')
    self._warp = tk.StringVar(value='')
    self._warp_box = ttk.Combobox(body, textvariable=self._warp,
                                  state='readonly', values=[''])
    self._warp_box.pack(fill='x', pady=(2, 0))
    self._warp_box.bind('<<ComboboxSelected>>',
                        lambda _e: setattr(self.session, 'warp_field',
                                           self._warp.get() or None))
    self.sliders['warp_scale'] = ValueSlider(
      body, 'Warp scale', 0.0, 10.0, 1.0,
      lambda v: setattr(self.session, 'warp_scale', v))

    ttk.Separator(body).pack(fill='x', pady=8)
    ttk.Label(body, text='Glyph arrows', style='Muted.TLabel').pack(anchor='w')
    self._glyph = tk.StringVar(value='')
    self._glyph_box = ttk.Combobox(body, textvariable=self._glyph,
                                   state='readonly', values=[''])
    self._glyph_box.pack(fill='x', pady=(2, 0))
    self._glyph_box.bind('<<ComboboxSelected>>',
                         lambda _e: setattr(self.session, 'glyph_field',
                                            self._glyph.get() or None))
    # Unitless: the session works out the factor that puts the longest arrow
    # at a fixed fraction of the mesh, so 1.0 shows something whether the
    # field is a displacement in metres or a velocity in metres per second.
    self.sliders['glyph_scale'] = ValueSlider(
      body, 'Arrow scale', 0.0, 5.0, 1.0,
      lambda v: setattr(self.session, 'glyph_scale', v))

    row = ttk.Frame(body)
    row.pack(fill='x', pady=(6, 0))
    ttk.Label(row, text='Max arrows', width=13).pack(side='left')
    self._glyph_count = tk.StringVar(value=str(self.session.glyph_count))
    count = ttk.Entry(row, textvariable=self._glyph_count, width=9)
    count.pack(side='right')
    for event in ('<Return>', '<FocusOut>'):
      count.bind(event, lambda _e: self._set_glyph_count())

    # -- Information -------------------------------------------------------
    body = section('Information', expanded=False)
    self._information = tk.Text(body, height=11, width=34,
                                font=('TkFixedFont', 8), **TEXT_OPTIONS)
    self._information.pack(fill='x')
    self._information.configure(state='disabled')
    self.session.view_changed.connect(lambda *_: self._show_information())
    self.session.plan_changed.connect(lambda *_: self._show_information())
    self._show_information()

  def _show_information(self) -> None:
    """Fill the Information panel, ParaView's own name for this.

    It earns its place for one specific reason: **the shipped phantoms are in
    three different units**, and the extent in metres beside the scale that
    was applied is what makes a wrong one obvious before the submesh silently
    reads zero.
    """
    self._information.configure(state='normal')
    self._information.delete('1.0', 'end')
    for key, value in self.session.information():
      self._information.insert('end', f'{key:<16s} {value}\n')
    self._information.configure(state='disabled')

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
      # The shipped phantoms are in three different units, and the plan is in
      # metres, so the scale is read off the mesh and APPLIED -- then shown in
      # the Plan tab, where it can be corrected. Loading at 1.0 and leaving
      # the user to notice an empty submesh is the worse failure.
      from ..model.mesh import load_mesh as read_mesh, suggest_scale_factor
      factor, why = suggest_scale_factor(read_mesh(path)[0])
      self.session.load_mesh(path, scale_factor=factor)
    except Exception as exc:
      self.status('load failed')
      messagebox.showerror('Could not open the phantom', str(exc))
      return
    self.status(f'loaded {path} -- scale {factor:g}, {why}')

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

    # Colour by lives on the toolbar and refreshes itself from
    # `mesh_changed`; what is left here are the two vector choosers and the
    # plan's centre.
    _colours, vectors = s.field_choices(0)
    self._warp_box.config(values=[''] + vectors)
    self._glyph_box.config(values=[''] + vectors)
    # A selection the new phantom does not have would otherwise stay in the
    # box and on the session, showing a name for a mesh drawn plain.
    self._keep_choice(self._warp, 'warp_field', vectors)
    self._keep_choice(self._glyph, 'glyph_field', vectors)

    lo, hi = s.points.min(axis=0), s.points.max(axis=0)
    self._plan_vars['loc'].set(' '.join(f'{v:.4g}' for v in 0.5 * (lo + hi)))
    self._show_information()
    self.viewport.rebuild()


  def _keep_choice(self, variable, attribute: str, offered) -> None:
    """Drop a field selection the newly loaded phantom does not have."""
    if variable.get() and variable.get() not in offered:
      variable.set('')
      setattr(self.session, attribute, None)

  def _set_glyph_count(self) -> None:
    """Read the arrow budget back, or restore what the session still holds."""
    try:
      self.session.glyph_count = int(float(self._glyph_count.get()))
    except (TypeError, ValueError):
      self.status('max arrows must be a positive whole number')
    self._glyph_count.set(str(self.session.glyph_count))






  def apply_plan(self) -> None:
    """Kept so the menu, the tests and any caller still have one entry point."""
    self.planning_panel.apply()

  def import_plan(self) -> None:
    self.planning_panel.import_plan()

  def export_plan(self) -> None:
    self.planning_panel.export_plan()

  def close(self) -> None:
    pump = getattr(self, '_pump_id', None)
    if pump is not None:
      try:
        self.root.after_cancel(pump)
      except Exception:
        pass
    bar = getattr(self, 'toolbar', None)
    if bar is not None:
      # A playing cine is a chain of `after` callbacks; one left running calls
      # into a viewport that is about to stop existing.
      try:
        bar.stop()
      except Exception:
        pass
    runner = getattr(self, 'run_panel', None)
    if runner is not None:
      # Cancel first: the ranks are in their own process group and would
      # outlive the window otherwise.
      try:
        runner.close()
      except Exception:
        pass
    try:
      self.viewport.close()
    finally:
      self.root.destroy()

  def run(self) -> None:
    self.root.mainloop()
