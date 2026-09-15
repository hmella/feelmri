"""The shared document and the run configuration.

`Session` is what keeps the panels in step without them knowing about each
other, so the tests are about notification: a change must emit exactly once, a
no-op must stay silent, and a batch must be collapsible.

`RunConfig` produces a script rather than running anything in process. The
check that matters is that the script is real Python that actually executes,
not that it contains the right substrings.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from feelmri.gui.model.planning import FOVBox
from feelmri.gui.model.runner import (PROGRESS_PREFIX, RunConfig, build_command,
                                      build_env, command_line,
                                      default_output_dir, parse_progress,
                                      render_script, write_script)
from feelmri.gui.model.session import Session, Signal

PHANTOMS = __import__('pathlib').Path(__file__).resolve().parent.parent / 'examples' / 'phantoms'


# --- the observer --------------------------------------------------------

def test_a_signal_fires_in_order_and_can_be_blocked():
  seen = []
  sig = Signal('t')
  sig.connect(lambda *_: seen.append('a'))
  sig.connect(lambda *_: seen.append('b'))
  sig.emit()
  assert seen == ['a', 'b']

  with sig.blocked():
    sig.emit()
  assert seen == ['a', 'b'], 'blocked() did not suppress the emission'

  sig.emit()
  assert seen == ['a', 'b', 'a', 'b'], 'blocking leaked past the context'


def test_connecting_twice_subscribes_once_and_disconnect_works():
  seen = []
  sig = Signal()
  fn = sig.connect(lambda *_: seen.append(1))
  sig.connect(fn)
  assert len(sig) == 1
  sig.emit()
  assert seen == [1]
  sig.disconnect(fn)
  sig.disconnect(fn)              # removing twice must not raise
  sig.emit()
  assert seen == [1]


# --- the document --------------------------------------------------------

def test_setting_the_plan_notifies_once_and_a_no_op_stays_quiet():
  """A redraw per edit, and no redraw when nothing changed.

  The no-op case matters: a panel that writes its own value back on every
  repaint would otherwise loop.
  """
  s = Session()
  fired = []
  s.plan_changed.connect(lambda _: fired.append(1))

  box = FOVBox(fov=[0.3, 0.2, 0.01], loc=[0, 0, 0], angles=[0, 0, 0])
  s.box = box
  assert len(fired) == 1

  s.box = FOVBox(fov=[0.3, 0.2, 0.01], loc=[0, 0, 0], angles=[0, 0, 0])
  assert len(fired) == 1, 'an identical plan emitted a change'

  s.box = FOVBox(fov=[0.3, 0.2, 0.02], loc=[0, 0, 0], angles=[0, 0, 0])
  assert len(fired) == 2


def test_the_plan_has_undo_and_redo():
  s = Session()
  a = FOVBox(fov=[1, 1, 1], loc=[0, 0, 0], angles=[0, 0, 0])
  b = FOVBox(fov=[2, 2, 2], loc=[0, 0, 0], angles=[0, 0, 0])
  assert not s.can_undo and not s.can_redo

  s.box = a
  s.box = b
  assert s.can_undo
  assert s.undo() and np.array_equal(s.box.fov, a.fov)
  assert s.undo() and s.box is None          # back before anything was set
  assert not s.undo()                        # and no further

  assert s.redo() and np.array_equal(s.box.fov, a.fov)
  assert s.redo() and np.array_equal(s.box.fov, b.fov)
  assert not s.redo()

  # A fresh edit clears the redo branch, as undo stacks do.
  s.undo()
  s.box = FOVBox(fov=[3, 3, 3], loc=[0, 0, 0], angles=[0, 0, 0])
  assert not s.can_redo


def test_the_plan_setter_rejects_the_wrong_type():
  s = Session()
  with pytest.raises(TypeError, match='must be a FOVBox'):
    s.box = 'not a box'


def test_view_state_is_range_checked_and_notifies():
  s = Session()
  fired = []
  s.view_changed.connect(lambda _: fired.append(1))

  s.field = 'fat'
  s.warp_scale = 2.5
  assert len(fired) == 2
  s.field = 'fat'                              # unchanged
  assert len(fired) == 2

  with pytest.raises(ValueError, match='must be finite'):
    s.warp_scale = float('nan')

  s.n_frames = 4
  s.frame = 3
  with pytest.raises(IndexError, match='out of range'):
    s.frame = 4


def test_derived_geometry_refuses_to_run_without_a_mesh():
  s = Session()
  assert not s.has_mesh
  for attr in ('surface', 'centroids'):
    with pytest.raises(RuntimeError, match='no mesh is loaded'):
      getattr(s, attr)
  with pytest.raises(RuntimeError, match='no mesh is loaded'):
    s.submesh_markers()


@pytest.mark.slow
def test_loading_a_mesh_populates_and_caches():
  path = PHANTOMS / 'water_fat_P1_prism.xdmf'
  if not path.exists():
    pytest.skip('phantom not present')

  s = Session()
  fired = []
  s.mesh_changed.connect(lambda _: fired.append(1))
  s.load_mesh(path)
  assert len(fired) == 1
  assert s.has_mesh and s.points.shape == (63357, 3)

  # Derived arrays are computed once and reused.
  assert s.surface is s.surface
  assert s.centroids is s.centroids
  assert len(s.centroids) == sum(len(c) for _, c in s.cells)

  # A plan then selects a subset of the elements.
  lo, hi = s.points.min(axis=0), s.points.max(axis=0)
  s.box = FOVBox(fov=(hi - lo) * [1, 1, 0.25], loc=(lo + hi) / 2,
                 angles=[0, 0, 0])
  markers = s.submesh_markers()
  assert markers.dtype == bool and markers.size == len(s.centroids)
  assert 0 < markers.sum() < markers.size, 'the slab selected all or nothing'

  summary = s.summary()
  assert summary['nodes'] == 63357 and summary['planned'] is True


# --- the run configuration ----------------------------------------------

def test_the_output_directory_defaults_outside_any_checkout():
  """`.gitignore` covers neither .vti nor .pvd, and this has bitten before."""
  out = default_output_dir()
  assert out.is_absolute()
  repo = __import__('pathlib').Path(__file__).resolve().parent.parent
  assert repo not in out.parents and out != repo


def test_run_config_refuses_impossible_settings():
  base = dict(phantom='p.xdmf', output_dir='/tmp/x')
  for bad, match in (({'ranks': 0}, 'ranks must be'),
                     ({'omp_threads': 0}, 'omp_threads must be'),
                     ({'voxel_size': 0.0}, 'voxel_size must be'),
                     ({'submesh_axis': 3}, 'submesh_axis must be')):
    with pytest.raises(ValueError, match=match):
      RunConfig(**base, **bad)


def test_one_rank_needs_no_launcher_but_more_ranks_do(tmp_path):
  """A preview must run on a machine with no MPI at all."""
  cfg = RunConfig(phantom='p.xdmf', output_dir=str(tmp_path), ranks=1)
  assert build_command(cfg, 's.py') == [sys.executable, 's.py']

  cfg = RunConfig(phantom='p.xdmf', output_dir=str(tmp_path), ranks=4)
  cmd = build_command(cfg, 's.py')
  assert cmd[0] == 'mpirun' and '-n' in cmd and '4' in cmd
  # The flags that let a rank count exceed the core count, as conftest does.
  assert '--oversubscribe' in cmd and '--allow-run-as-root' in cmd
  assert 'mpirun' in command_line(cfg, 's.py')


def test_the_environment_sets_the_thread_limits_but_yields_to_the_caller(tmp_path):
  cfg = RunConfig(phantom='p.xdmf', output_dir=str(tmp_path), omp_threads=3,
                  env={'OMP_NUM_THREADS': '8', 'MY_FLAG': '1'})
  env = build_env(cfg)
  assert env['OPENBLAS_NUM_THREADS'] == '1'
  assert env['MPLBACKEND'] == 'Agg'
  assert env['OMP_NUM_THREADS'] == '8', 'the explicit override was ignored'
  assert env['MY_FLAG'] == '1'


def test_progress_lines_are_distinguishable_from_ordinary_output():
  assert parse_progress(f'{PROGRESS_PREFIX} loading mesh') == 'loading mesh'
  assert parse_progress('  ' + PROGRESS_PREFIX + ' done  ') == 'done'
  assert parse_progress('some library chatter') is None
  assert parse_progress('') is None


def test_the_generated_script_is_valid_python_and_orders_orient_first(tmp_path):
  """`orient` before `set_assembler`, which the library enforces by refusing.

  Compiling the script is the weakest useful check; the ordering is the one
  that would otherwise fail only at runtime on a real phantom.
  """
  box = FOVBox(fov=[0.3, 0.2, 0.01], loc=[0.01, 0.0, -0.02],
               angles=[0.1, -0.2, 0.3])
  cfg = RunConfig(phantom='/tmp/p.xdmf', output_dir=str(tmp_path), box=box,
                  voxel_size=2e-3)
  src = render_script(cfg)
  compile(src, 'run.py', 'exec')

  assert src.index('phantom.orient(') < src.index('phantom.set_assembler('), (
    'set_assembler is emitted before orient, which the library refuses')
  # The angles travel as angles, recomposed in the documented order.
  assert 'Rz(tz) @ Rx(tx) @ Ry(ty)' in src
  assert repr(0.1) in src and repr(-0.2) in src

  path = write_script(cfg, tmp_path / 'run.py')
  assert __import__('pathlib').Path(path).read_text() == src


def test_a_script_with_no_plan_skips_the_geometry(tmp_path):
  cfg = RunConfig(phantom='/tmp/p.xdmf', output_dir=str(tmp_path))
  src = render_script(cfg)
  compile(src, 'run.py', 'exec')
  assert 'phantom.orient(' not in src
  assert 'create_submesh' not in src
  assert 'set_assembler' in src


@pytest.mark.slow
def test_the_generated_script_actually_runs(tmp_path):
  """Execute it, rather than trusting that it compiles.

  The phantom line is replaced with a stub so this stays a test of the
  generated control flow and not of the solver. Everything else, including the
  progress protocol the run panel parses, is exercised as written.
  """
  box = FOVBox(fov=[1.0, 1.0, 0.4], loc=[0.5, 0.5, 0.5], angles=[0.0, 0.0, 0.3])
  cfg = RunConfig(phantom='/tmp/does_not_exist.xdmf',
                  output_dir=str(tmp_path), box=box)
  src = render_script(cfg)

  stub = '''
import numpy as np
class _Stub:
  global_nodes = np.random.default_rng(0).uniform(0, 1, size=(64, 3))
  global_elements = np.arange(64).reshape(16, 4)
  def orient(self, *a, **k): print("STUB orient")
  def create_submesh(self, m): print("STUB submesh", int(m.sum()))
  def set_assembler(self, *a, **k): print("STUB assembler")
  local_nodes = np.zeros((64, 3))
  def set_static_fields(self, **k):
    print("STUB static fields", sorted(k))
'''
  src = src.replace('phantom = FEMPhantom(', stub + '\nphantom = _Stub()  # (')
  script = tmp_path / 'run.py'
  script.write_text(src)

  proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                        text=True, timeout=300, env=build_env(cfg))
  assert proc.returncode == 0, proc.stdout + proc.stderr

  messages = [parse_progress(l) for l in proc.stdout.splitlines()]
  messages = [m for m in messages if m is not None]
  assert 'done' in messages, f'the script did not finish: {proc.stdout}'
  assert any('orienting' in m for m in messages)
  assert any('keeping' in m for m in messages), 'no submesh count was reported'
  assert 'STUB orient' in proc.stdout and 'STUB assembler' in proc.stdout


# -- what the generated script must contain to actually run -----------------

def test_the_script_sets_the_static_fields_or_the_run_cannot_finish():
  """`simulate_pulseq` refuses a phantom without them, and both signal paths
  need them.

  They were a commented-out hint, so a run launched from the GUI failed at
  the first readout. Emitting real uniform maps is what makes the generated
  script runnable as written, which is the whole point of generating it.
  """
  from feelmri.gui.model.runner import RunConfig, render_script

  script = render_script(RunConfig(phantom='p.xdmf', output_dir='/tmp/o',
                                   sequence='s.seq', t2_ms=42.0,
                                   phi_dB0=0.25))
  assert 'phantom.set_static_fields(' in script
  assert '42.0' in script and '0.25' in script

  # Order matters, and it is compared on the STATEMENTS. A first version used
  # `script.index('simulate_pulseq')`, which found the word in a comment
  # emitted above the call and read the order backwards.
  def line_of(statement):
    for number, line in enumerate(script.splitlines()):
      if line.strip().startswith(statement):
        return number
    raise AssertionError(f'{statement!r} is not in the script')

  assert line_of('phantom.set_assembler(') < \
         line_of('phantom.set_static_fields(')
  assert line_of('phantom.set_static_fields(') < line_of('result = ')


def test_the_generated_script_is_valid_python():
  """It is written as text, so nothing else checks that it parses."""
  import ast

  from feelmri.gui.model.planning import FOVBox
  from feelmri.gui.model.runner import RunConfig, render_script

  box = FOVBox(fov=np.array([0.3, 0.22, 0.008]), loc=np.array([0.0, 0.0, 0.05]),
               angles=np.radians([0.0, 0.0, 25.0]))
  for config in (RunConfig(phantom='p.xdmf', output_dir='/tmp/o'),
                 RunConfig(phantom='p.xdmf', output_dir='/tmp/o', box=box),
                 RunConfig(phantom='p.xdmf', output_dir='/tmp/o', box=box,
                           sequence='s.seq', solver={'scanner': None})):
    ast.parse(render_script(config))


@pytest.mark.parametrize('bad', [0.0, -1.0, float('inf')])
def test_a_t2_that_would_poison_every_sample_is_refused(bad):
  """T2 = 0 inverts to Inf and `exp(-t*Inf)` is NaN even at t = 0, so ONE bad
  value poisons every k-space sample rather than its own contribution. A
  negative one is worse: finite, no NaN to notice, the signal simply grows."""
  from feelmri.gui.model.runner import RunConfig

  with pytest.raises(ValueError, match='t2_ms'):
    RunConfig(phantom='p.xdmf', output_dir='/tmp/o', t2_ms=bad)


def test_launch_does_not_read_the_pipe_itself():
  """It returns a `Popen` and nothing more.

  An earlier version took an `on_line` callback and drained stdout inline
  while its docstring promised to return immediately -- which would freeze a
  UI thread for the whole run. `stream_lines` is the reader, so the caller
  chooses which thread blocks.
  """
  import inspect

  from feelmri.gui.model.runner import launch, stream_lines

  assert list(inspect.signature(launch).parameters) == ['config', 'script']
  assert inspect.isgeneratorfunction(stream_lines)


def test_the_scale_factor_reaches_the_generated_script():
  """The viewer scales the mesh to metres to draw the plan against it; a run
  at a different scale would simulate a different phantom from the one on
  screen."""
  from feelmri.gui.model.runner import RunConfig, render_script

  script = render_script(RunConfig(phantom='p.xdmf', output_dir='/tmp/o',
                                   scale_factor=0.001))
  assert 'scale_factor=0.001' in script
