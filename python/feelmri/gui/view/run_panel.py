"""Launch a simulation as a subprocess, and stream what it says.

The GUI never joins an MPI communicator -- `gui.model.runner` explains why at
length -- so a run is a generated script under `mpirun`, and this panel is the
part that starts it and shows the output.

**Tk is not thread-safe**, and reading a subprocess pipe blocks until the child
speaks, so the two cannot share a thread. A worker iterates
`runner.stream_lines` and pushes into a `queue.Queue`; a `root.after` timer
drains the queue and is the only thing that touches a widget. Nothing else is
shared, which is what keeps this a queue rather than a lock.

The script is written to the output directory before it runs and is left there
afterwards. It is plain Python that needs no GUI, so a run made here can be
repeated, edited or reported from a shell -- which is also what makes the
`Preview` button worth having rather than decoration.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

from ..model.runner import (RunConfig, cancel, command_line, default_output_dir,
                            launch, parse_progress, render_script,
                            stream_lines, write_script)

#: How often the UI drains the worker's queue, in ms.
DRAIN_MS = 100

#: Numeric fields: label, attribute, default, and the type to read it as.
FIELDS = (('MPI ranks', 'ranks', '1', int),
          ('OMP threads per rank', 'omp_threads', '1', int),
          ('Voxel size (m)', 'voxel_size', '0.005', float),
          ('Readout T2 (ms)', 'readout_t2_ms', '50', float),
          ('Off-resonance (rad/ms)', 'phi_dB0', '0', float))

#: Solver settings, forwarded to `BlochSolver` through `simulate_pulseq`.
#:
#: **The solver's T2 is NOT the readout T2 above.** They are separate objects
#: with no code path between them: this one governs the Bloch evolution
#: between blocks, the other `exp(-t/T2)` during a readout, measured from the
#: magnetization snapshot. Passing different values is supported and is the
#: right thing whenever a refocusing pulse recovers the reversible part.
SOLVER_FIELDS = (('Solver T1 (ms)', 'T1', '800', float),
                 ('Solver T2 (ms)', 'T2', '50', float))

#: Solver choices, as (label, key, values). `magnus2` is the library default
#: and is exact on a piecewise-linear gradient; see `bloch-solver.md`.
SOLVER_CHOICES = (('Method', 'method', ('magnus2', 'magnus4', 'cayley_klein')),
                  ('Precision', 'dtype', ('float32', 'float64')))

#: How the ASSEMBLER integrates the signal over each element -- the schemes the
#: accuracy work calls IS1 to IS4. Each name maps to the two `set_assembler`
#: flags underneath it; `lorder` and `horder` are separate because they apply
#: to the quadrature either way.
#:
#: **Quadrature is the default, and nodal is the one to be careful with.**
#: Measured on a uniform disc against a converged reference, `nodal +
#: lumped` reads **82.7%** error where quadrature at a voxel-matched size
#: reads 0.0%: a nodal approximation is a point-mass model, right for reading
#: nodal quantities and wrong for synthesizing k-space from a graded mesh.
#: `kspace-assembler.md` carries the numbers.
INTEGRATION = (
  ('quadrature (IS3)', dict(nodal_approximation=False, lumped=True)),
  ('nodal, lumped (IS2)', dict(nodal_approximation=True, lumped=True)),
  ('nodal, consistent mass (IS2)',
   dict(nodal_approximation=True, lumped=False)),
)

#: Quadrature degrees. **`voxel_size` decides which each element gets**, and
#: that makes the large-element order INERT more often than one expects: an
#: element is integrated at `horder` only if its size reaches `voxel_size`,
#: and the size is `cbrt(element volume)`, not an in-plane diameter.
#:
#: Measured on `water_fat_P1_prism`, changing `horder` from 1 to 6:
#:
#: | voxel_size | elements promoted | effect of horder |
#: |---|---|---|
#: | 0.005 | **0 of 11904** | **0.000e+00, inert** |
#: | 0.002 | 2600 of 11904 | 1.64 relative |
#: | 0.0005 | all 11904 | **50.1 relative** |
#:
#: So a voxel size at or above every element leaves the control doing
#: nothing, and one below every element makes it dominant. The run's own log
#: prints the split (`N/M elements with size < voxel_size`) -- read it.
#:
#: With a nodal strategy and a voxel size that actually splits the mesh, this
#: is the adaptive IS4 dispatch: `mri_signal` sends group 0 through the nodal
#: path and every other group through quadrature.
ORDER_FIELDS = (('Quadrature order, small elements', 'lorder', '1', int),
                ('Quadrature order, large elements', 'horder', '6', int))


class RunPanel:
  """Compose a `RunConfig` from the session, launch it, and stream the log."""

  def __init__(self, parent, session,
               on_status: Optional[Callable[[str], None]] = None):
    import tkinter as tk
    from tkinter import ttk

    self.session = session
    self.on_status = on_status or (lambda _: None)
    self._proc = None
    self._queue: 'queue.Queue[object]' = queue.Queue()
    self._drain_id = None

    self.widget = ttk.Frame(parent, padding=10)

    self.vars = {}
    for label, key, default, _kind in FIELDS:
      ttk.Label(self.widget, text=label).pack(anchor='w', pady=(6, 0))
      variable = tk.StringVar(value=default)
      ttk.Entry(self.widget, textvariable=variable).pack(fill='x')
      self.vars[key] = variable

    ttk.Separator(self.widget).pack(fill='x', pady=(10, 4))
    ttk.Label(self.widget, text='Integration',
              font=('TkDefaultFont', 9, 'bold')).pack(anchor='w')
    self.integration = tk.StringVar(value=INTEGRATION[0][0])
    ttk.Combobox(self.widget, textvariable=self.integration, state='readonly',
                 values=[name for name, _ in INTEGRATION]).pack(fill='x')
    self.order_vars = {}
    for label, key, default, _kind in ORDER_FIELDS:
      ttk.Label(self.widget, text=label).pack(anchor='w', pady=(4, 0))
      variable = tk.StringVar(value=default)
      ttk.Entry(self.widget, textvariable=variable).pack(fill='x')
      self.order_vars[key] = variable
    ttk.Label(self.widget, wraplength=280, foreground='#555', text=(
      'An element uses the large-element order only if its size reaches the '
      'voxel size above. The run log prints the split; if it reads 0 of N, '
      'that order is doing nothing.')).pack(anchor='w', pady=(4, 0))

    ttk.Separator(self.widget).pack(fill='x', pady=(10, 4))
    ttk.Label(self.widget, text='Solver',
              font=('TkDefaultFont', 9, 'bold')).pack(anchor='w')
    self.solver_vars = {}
    for label, key, default, _kind in SOLVER_FIELDS:
      ttk.Label(self.widget, text=label).pack(anchor='w', pady=(4, 0))
      variable = tk.StringVar(value=default)
      ttk.Entry(self.widget, textvariable=variable).pack(fill='x')
      self.solver_vars[key] = variable
    for label, key, values in SOLVER_CHOICES:
      ttk.Label(self.widget, text=label).pack(anchor='w', pady=(4, 0))
      variable = tk.StringVar(value=values[0])
      ttk.Combobox(self.widget, textvariable=variable, values=list(values),
                   state='readonly').pack(fill='x')
      self.solver_vars[key] = variable
    self.concomitant = tk.BooleanVar(value=False)
    ttk.Checkbutton(self.widget, text='Concomitant (Maxwell) fields',
                    variable=self.concomitant).pack(anchor='w', pady=(6, 0))

    ttk.Separator(self.widget).pack(fill='x', pady=(10, 4))
    ttk.Label(self.widget, text='Output directory').pack(anchor='w',
                                                         pady=(4, 0))
    row = ttk.Frame(self.widget)
    row.pack(fill='x')
    self.output_dir = tk.StringVar(value=str(default_output_dir()))
    ttk.Entry(row, textvariable=self.output_dir).pack(side='left', fill='x',
                                                      expand=True)
    ttk.Button(row, text='...', width=3,
               command=self._browse).pack(side='left', padx=(4, 0))

    # Two rows: three buttons side by side are clipped at this column width,
    # and a button reading "Prev" is worse than one on its own line.
    buttons = ttk.Frame(self.widget)
    buttons.pack(fill='x', pady=(12, 0))
    self._run_button = ttk.Button(buttons, text='Run', command=self.run)
    self._run_button.pack(side='left', expand=True, fill='x')
    self._cancel_button = ttk.Button(buttons, text='Cancel',
                                     command=self.cancel, state='disabled')
    self._cancel_button.pack(side='left', expand=True, fill='x', padx=(6, 0))
    ttk.Button(self.widget, text='Preview script',
               command=self.preview).pack(fill='x', pady=(4, 0))

    ttk.Separator(self.widget).pack(fill='x', pady=12)
    self._progress = ttk.Label(self.widget, text='idle', wraplength=280)
    self._progress.pack(anchor='w')
    self._log = tk.Text(self.widget, height=14, width=40,
                        font=('TkFixedFont', 8), wrap='none')
    self._log.pack(fill='both', expand=True, pady=(4, 0))
    self._log.configure(state='disabled')

  # -- the configuration ----------------------------------------------------

  def config(self) -> RunConfig:
    """A `RunConfig` from the session and the fields. Raises on bad input.

    The scale factor comes from the SESSION, not from a field of its own: the
    viewer already scaled the mesh to metres to draw the plan against it, and
    a run at a different scale would silently simulate a different phantom
    from the one on screen.
    """
    if not self.session.has_mesh:
      raise ValueError('open a phantom first')
    values = {}
    for label, key, _default, kind in FIELDS:
      text = self.vars[key].get().strip()
      try:
        values[key] = kind(text)
      except ValueError:
        raise ValueError(f'{label}: {text!r} is not a number') from None
    solver = {}
    for label, key, _default, kind in SOLVER_FIELDS:
      text = self.solver_vars[key].get().strip()
      try:
        solver[key] = kind(text)
      except ValueError:
        raise ValueError(f'{label}: {text!r} is not a number') from None
    for _label, key, _values in SOLVER_CHOICES:
      solver[key] = self.solver_vars[key].get()
    if self.concomitant.get():
      solver['concomitant_fields'] = True

    # A COPY: `dict(INTEGRATION)` hands back the module-level dicts
    # themselves, and the orders are written in below, which would modify the
    # constant for every later call.
    chosen = dict(dict(INTEGRATION)[self.integration.get()])
    for label, key, _default, kind in ORDER_FIELDS:
      text = self.order_vars[key].get().strip()
      try:
        chosen[key] = kind(text)
      except ValueError:
        raise ValueError(f'{label}: {text!r} is not a number') from None

    return RunConfig(
      solver=solver,
      **chosen,
      phantom=self.session.mesh_path,
      output_dir=self.output_dir.get().strip() or str(default_output_dir()),
      box=self.session.box,
      sequence=getattr(self.session, 'sequence_path', None),
      scale_factor=getattr(self.session, 'scale_factor', 1.0),
      **values)

  def preview(self) -> None:
    """Show the script that would run, and the command that would run it."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    try:
      config = self.config()
    except Exception as exc:
      messagebox.showerror('Cannot build a run', str(exc))
      return

    window = tk.Toplevel(self.widget)
    window.title('feelmri  run script')
    window.geometry('760x600')
    ttk.Label(window, text=command_line(config, '<script>'),
              font=('TkFixedFont', 8), wraplength=740,
              padding=(8, 6)).pack(anchor='w')
    text = tk.Text(window, font=('TkFixedFont', 9), wrap='none')
    text.pack(fill='both', expand=True)
    text.insert('end', render_script(config))
    text.configure(state='disabled')

  # -- running --------------------------------------------------------------

  def run(self) -> None:
    from tkinter import messagebox

    if self._proc is not None and self._proc.poll() is None:
      messagebox.showinfo('Already running', 'Cancel the current run first.')
      return
    try:
      config = self.config()
      script = write_script(config)
    except Exception as exc:
      messagebox.showerror('Cannot start the run', str(exc))
      return

    self._clear_log()
    self._append(f'$ {command_line(config, script)}')
    try:
      self._proc = launch(config, script)
    except FileNotFoundError as exc:
      # `mpirun` missing is the common one, and the script still runs at one
      # rank, so say which command was not found rather than "failed".
      messagebox.showerror('Could not launch', f'{exc}\n\n'
                           f'At 1 rank no launcher is used at all.')
      self._proc = None
      return

    self._run_button.config(state='disabled')
    self._cancel_button.config(state='normal')
    self._progress.config(text='running...')

    proc = self._proc
    worker = threading.Thread(target=self._read, args=(proc,), daemon=True)
    worker.start()
    self._drain()

  def _read(self, proc) -> None:
    """Worker thread. **Touches no widget** -- Tk is not thread-safe."""
    try:
      for line in stream_lines(proc):
        self._queue.put(line)
    finally:
      self._queue.put(('done', proc.returncode))

  def _drain(self) -> None:
    """The only place the log is written, and it runs on the Tk thread."""
    finished = None
    while True:
      try:
        item = self._queue.get_nowait()
      except queue.Empty:
        break
      if isinstance(item, tuple) and item and item[0] == 'done':
        finished = item[1]
        continue
      self._append(item)
      message = parse_progress(item)
      if message:
        self._progress.config(text=message)
        self.on_status(message)

    if finished is None:
      self._drain_id = self.widget.after(DRAIN_MS, self._drain)
      return

    self._drain_id = None
    self._run_button.config(state='normal')
    self._cancel_button.config(state='disabled')
    ok = finished == 0
    self._progress.config(text='finished' if ok else f'exit code {finished}')
    self.on_status('run finished' if ok else f'run failed ({finished})')
    if ok:
      self._load_result()

  def _load_result(self) -> None:
    """Read the finished run's k-space back into the session.

    The run is a subprocess writing a file, so this is the only way a result
    returns -- nothing is shared with it in memory. A run without a sequence
    writes nothing, which is not a failure, so a missing file is quiet.
    """
    from pathlib import Path

    target = Path(self.output_dir.get().strip() or '.') / 'kspace.npz'
    if not target.exists():
      return
    try:
      result = self.session.load_result(target)
    except Exception as exc:
      self._append(f'-- could not read {target.name}: {exc} --')
      return
    self._append(f'-- loaded {target.name}, '
                 f'{result["kspace"].shape} k-space samples --')
    self.on_status(f'loaded {target.name}')

  def cancel(self) -> None:
    """Stop the run and everything it launched."""
    if self._proc is None or self._proc.poll() is not None:
      return
    cancel(self._proc)
    self._append('-- cancelled --')
    self.on_status('run cancelled')

  def close(self) -> None:
    """Cancel any run and stop the timer, so quitting leaves nothing behind."""
    if self._drain_id is not None:
      try:
        self.widget.after_cancel(self._drain_id)
      except Exception:
        pass
      self._drain_id = None
    self.cancel()

  # -- the log --------------------------------------------------------------

  def _append(self, line: str) -> None:
    self._log.configure(state='normal')
    self._log.insert('end', line + '\n')
    self._log.see('end')
    self._log.configure(state='disabled')

  def _clear_log(self) -> None:
    self._log.configure(state='normal')
    self._log.delete('1.0', 'end')
    self._log.configure(state='disabled')

  def _browse(self) -> None:
    from tkinter import filedialog

    chosen = filedialog.askdirectory(title='Output directory',
                                     initialdir=self.output_dir.get() or '.')
    if chosen:
      self.output_dir.set(chosen)
