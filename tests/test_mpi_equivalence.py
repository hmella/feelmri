"""Serial vs ``mpirun -n N`` numerical-equivalence test.

Runs ``tests/helpers/mpi_runner.py`` twice — once directly (1 rank)
and once via ``mpirun -n 2`` — and asserts that the gathered Mxy/Mz
arrays match within single-precision tolerance. Catches per-rank
ordering bugs and any non-deterministic divergence between serial
and parallel execution paths."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _phantom_fixtures import make_1d_rod_mesh


_RUNNER = (Path(__file__).resolve().parent / 'helpers' / 'mpi_runner.py')


def _run(cmd, env):
  proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=180)
  return proc


@pytest.mark.slow
@pytest.mark.requires_mpi
@pytest.mark.timeout(240)
def test_serial_matches_mpi_n2(tmp_path):
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  if shutil.which('mpirun') is None:
    pytest.skip('mpirun not on PATH')

  mesh_path = tmp_path / 'rod.vtu'
  # Use a denser rod mesh so pymetis can partition across 2 ranks
  # without leaving a rank empty (a 2-tet mesh collapses to a single
  # rank under DUAL partitioning).
  make_1d_rod_mesh(mesh_path, length=8e-3, n_segments=32,
                   transverse_width=2e-4)

  out_serial = tmp_path / 'M_serial.npz'
  out_mpi    = tmp_path / 'M_mpi.npz'

  env = os.environ.copy()
  env.setdefault('OPENBLAS_NUM_THREADS', '1')
  env.setdefault('MPLBACKEND', 'Agg')

  proc_serial = _run(
    [sys.executable, str(_RUNNER), '--mesh', str(mesh_path),
     '--output', str(out_serial)],
    env=env,
  )
  assert proc_serial.returncode == 0, (
    f'serial run failed:\n{proc_serial.stdout.decode(errors="replace")}'
  )
  assert out_serial.exists()

  proc_mpi = _run(
    ['mpirun', '--allow-run-as-root', '--oversubscribe', '-n', '2',
     sys.executable, str(_RUNNER),
     '--mesh', str(mesh_path), '--output', str(out_mpi)],
    env=env,
  )
  assert proc_mpi.returncode == 0, (
    f'mpi run failed:\n{proc_mpi.stdout.decode(errors="replace")}'
  )
  assert out_mpi.exists()

  with np.load(out_serial) as a, np.load(out_mpi) as b:
    np.testing.assert_allclose(a['Mxy'], b['Mxy'], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(a['Mz'],  b['Mz'],  rtol=1e-4, atol=1e-6)


_REALISM_RUNNER = (Path(__file__).resolve().parent / 'helpers'
                   / 'realism_mpi_runner.py')


@pytest.mark.slow
@pytest.mark.requires_mpi
@pytest.mark.timeout(240)
def test_realism_features_match_between_serial_and_mpi(tmp_path):
  """Concomitant fields, a per-node B1+ map and a per-node T2' sub-ensemble,
  all on at once, must give the same answer at 1 and 2 ranks.

  None of the three had any MPI coverage. All are indexed by LOCAL node, and
  t2_prime additionally allocates per-rank K-fold state and carries it across
  blocks, so a rank-ordering bug would be invisible to every serial test while
  producing a plausible image.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  if shutil.which('mpirun') is None:
    pytest.skip('mpirun not on PATH')

  mesh_path = tmp_path / 'realism_rod.vtu'
  # Long enough that the concomitant term (quadratic in position) and the T2'
  # map both vary appreciably along it, and dense enough for pymetis to fill
  # two ranks.
  make_1d_rod_mesh(mesh_path, length=0.08, n_segments=48,
                   transverse_width=2e-4)

  out_serial = tmp_path / 'R_serial.npz'
  out_mpi = tmp_path / 'R_mpi.npz'

  env = os.environ.copy()
  env.setdefault('OPENBLAS_NUM_THREADS', '1')
  env.setdefault('MPLBACKEND', 'Agg')

  proc_serial = _run(
    [sys.executable, str(_REALISM_RUNNER), '--mesh', str(mesh_path),
     '--output', str(out_serial)], env=env)
  assert proc_serial.returncode == 0, (
    f'serial run failed:\n{proc_serial.stdout.decode(errors="replace")}')

  proc_mpi = _run(
    ['mpirun', '--allow-run-as-root', '--oversubscribe', '-n', '2',
     sys.executable, str(_REALISM_RUNNER),
     '--mesh', str(mesh_path), '--output', str(out_mpi)], env=env)
  assert proc_mpi.returncode == 0, (
    f'mpi run failed:\n{proc_mpi.stdout.decode(errors="replace")}')

  with np.load(out_serial) as a, np.load(out_mpi) as b:
    # float64 throughout, so the only difference should be summation order.
    np.testing.assert_allclose(a['Mxy'], b['Mxy'], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(a['Mz'], b['Mz'], rtol=1e-9, atol=1e-12)
    # Guard against a vacuous pass: the features must have done something.
    assert np.abs(a['Mxy']).max() > 1e-3
    assert np.ptp(np.abs(a['Mxy'][:, -1])) > 1e-2, (
      'the per-node maps produced a spatially flat result; the comparison '
      'would not detect a rank-ordering bug')


@pytest.mark.slow
@pytest.mark.requires_mpi
@pytest.mark.timeout(180)
def test_a_bad_per_node_array_on_one_rank_raises_everywhere(tmp_path):
  """A per-node array validated against the LOCAL node count can be wrong on
  one rank and right on the others, so the refusal has to be collective.

  It was not: `_node_column` raised on the spot for T1, T2, delta_B and both
  initial magnetizations, while every other rank walked on into the allgather
  that reports such problems and blocked there. Reproduced as a hang under
  `mpirun -n 2` -- rank 1 raised, rank 0 never returned and had to be killed.

  This is not a contrived input. Under dual partitioning the bloch and signal
  layouts carry different per-rank node counts, so an array built from
  `phantom.local_nodes` while the wrong layout is active matches on some ranks
  and not on others.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  if shutil.which('mpirun') is None:
    pytest.skip('mpirun not on PATH')

  mesh_path = tmp_path / 'poison_rod.vtu'
  make_1d_rod_mesh(mesh_path, length=0.08, n_segments=48,
                   transverse_width=2e-4)

  env = os.environ.copy()
  env.setdefault('OPENBLAS_NUM_THREADS', '1')
  env.setdefault('MPLBACKEND', 'Agg')

  proc = subprocess.run(
    ['mpirun', '--allow-run-as-root', '--oversubscribe', '-n', '2',
     sys.executable, str(_REALISM_RUNNER), '--mesh', str(mesh_path),
     '--output', str(tmp_path / 'unused.npz'), '--poison-rank', '1'],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)

  # The timeout above is the real assertion: before the fix this never
  # returned. What follows pins the diagnosis.
  out = proc.stdout.decode(errors='replace')
  assert proc.returncode != 0, 'a short delta_B on one rank was accepted'
  assert out.count('ValueError') >= 2, (
    f'only one rank reported the problem, so the others did not reach the '
    f'collective that raises it:\n{out[-3000:]}')
  assert 'delta_B' in out and 'reported by rank 1' in out, out[-3000:]


_PULSEQ_RUNNER = (Path(__file__).resolve().parent / 'helpers'
                  / 'pulseq_mpi_runner.py')


@pytest.mark.slow
@pytest.mark.requires_mpi
@pytest.mark.pulseq
@pytest.mark.timeout(300)
def test_simulate_pulseq_matches_serial_under_mpi(tmp_path):
  """The Pulseq path must be rank-count independent, and `gather=True` must
  leave non-root ranks empty.

  Nothing covered either before: the existing MPI test builds a native hard
  pulse, reads no `.seq`, and never calls `mri_signal`. So `import_pulseq`,
  `simulate_pulseq` and the k-space assembly were all untested at more than one
  rank, and the `Reduce(root=0)` contract that `simulate_pulseq`'s docstring
  warns about at length was asserted nowhere.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  pytest.importorskip('pypulseq')
  if shutil.which('mpirun') is None:
    pytest.skip('mpirun not on PATH')

  from conftest import skip_if_pypulseq_too_old
  from _phantom_fixtures import make_cube_mesh

  seq_path = Path(__file__).resolve().parent / 'data' / 'gre_an_v15.seq'
  if not seq_path.exists():
    pytest.skip('run tests/data/generate_seq_fixtures.py')
  skip_if_pypulseq_too_old(seq_path)

  mesh_path = tmp_path / 'cube.vtu'
  make_cube_mesh(mesh_path, 'tetra', n=4, scale=1e-3)

  out_serial = tmp_path / 'k_serial.npz'
  out_mpi = tmp_path / 'k_mpi.npz'

  env = os.environ.copy()
  env.setdefault('OPENBLAS_NUM_THREADS', '1')
  env.setdefault('OMP_NUM_THREADS', '1')
  env.setdefault('MPLBACKEND', 'Agg')

  serial = _run([sys.executable, str(_PULSEQ_RUNNER), '--mesh', str(mesh_path),
                 '--seq', str(seq_path), '--output', str(out_serial)], env)
  assert serial.returncode == 0, serial.stdout.decode(errors='replace')[-4000:]

  parallel = _run(['mpirun', '-n', '2', sys.executable, str(_PULSEQ_RUNNER),
                   '--mesh', str(mesh_path), '--seq', str(seq_path),
                   '--output', str(out_mpi)], env)
  assert parallel.returncode == 0, parallel.stdout.decode(errors='replace')[-4000:]

  k1 = np.load(out_serial)['kspace']
  k2 = np.load(out_mpi)['kspace']
  assert k1.shape == k2.shape
  scale = np.abs(k1).max()
  assert scale > 0, 'the serial run produced no signal'
  worst = float(np.abs(k1 - k2).max() / scale)
  # The assembler accumulates in float32, so a different partition sums the
  # same terms in a different order. Measured 1.3e-5 at 2 ranks and 1.5e-5 at
  # 4 -- FLAT in rank count, which is what distinguishes reassociation from a
  # real duplication bug (the signal_sum interface-node defect grew with rank
  # count: 6.6e-2 / 1.5e-1 / 1.2e-1 at 2 / 4 / 8). Same 1e-4 scale as the
  # native-sequence test above.
  assert worst < 1e-4, (
      f'simulate_pulseq is rank-count dependent: 1 rank vs 2 ranks differ by '
      f'{worst:.3e} of peak, well above float32 reassociation')

  # gather_data is a Reduce to root, so a non-root rank receives zeros. This is
  # documented and load-bearing -- a caller that reconstructs on every rank
  # would silently image nothing.
  rank1 = out_mpi.parent / (out_mpi.name + '.rank1.npz')
  assert rank1.exists(), 'the rank-1 process wrote no output'
  assert not np.any(np.load(rank1)['kspace']), (
      'gather=True must leave non-root ranks empty; rank 1 received a '
      'non-zero signal, so the reduction is not Reduce(root=0)')


@pytest.mark.slow
@pytest.mark.requires_mpi
@pytest.mark.pulseq
@pytest.mark.timeout(420)
def test_a_receive_map_survives_mpi_and_dual_partitioning(tmp_path):
  """A receive map is a per-LOCAL-node array, so under dual partitioning it is
  redistributed into the signal layout before it is folded in -- and nothing
  covered that path.

  The map is tied to NODE POSITION rather than being constant, which is the
  whole point: with a constant map, a redistribution that moved the wrong rows
  changes k-space by exactly 0.000e+00, and the audit that found this class of
  hole found it exactly that way. Each coil also gets its own spatial
  weighting and phase, so a fold that collapsed the coil axis or reused one
  column is visible.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  pytest.importorskip('pypulseq')
  if shutil.which('mpirun') is None:
    pytest.skip('mpirun not on PATH')

  from conftest import skip_if_pypulseq_too_old
  from _phantom_fixtures import make_cube_mesh

  seq_path = Path(__file__).resolve().parent / 'data' / 'gre_an_v15.seq'
  if not seq_path.exists():
    pytest.skip('run tests/data/generate_seq_fixtures.py')
  skip_if_pypulseq_too_old(seq_path)

  mesh_path = tmp_path / 'cube.vtu'
  make_cube_mesh(mesh_path, 'tetra', n=4, scale=1e-3)

  env = os.environ.copy()
  env.setdefault('OPENBLAS_NUM_THREADS', '1')
  env.setdefault('OMP_NUM_THREADS', '1')
  env.setdefault('MPLBACKEND', 'Agg')

  runs = {}
  for label, argv in (
      ('serial', [sys.executable]),
      ('mpi2', ['mpirun', '-n', '2', sys.executable]),
      ('mpi2_dual', ['mpirun', '-n', '2', sys.executable])):
    out = tmp_path / f'k_{label}.npz'
    cmd = argv + [str(_PULSEQ_RUNNER), '--mesh', str(mesh_path),
                  '--seq', str(seq_path), '--output', str(out), '--coils', '3']
    if label.endswith('dual'):
      cmd.append('--dual')
    proc = _run(cmd, env)
    assert proc.returncode == 0, proc.stdout.decode(errors='replace')[-4000:]
    runs[label] = np.load(out)['kspace']

  reference = runs['serial']
  assert reference.ndim == 2 and reference.shape[1] == 3, (
    f'expected one column per coil, got shape {reference.shape}')
  scale = float(np.abs(reference).max())
  assert scale > 0, 'the serial run produced no signal'

  # The coils must actually differ, or rank-count agreement below is agreement
  # about three copies of one number.
  for c in range(1, 3):
    apart = float(np.abs(reference[:, c] - reference[:, 0]).max() / scale)
    assert apart > 0.05, (
      f'coil {c} is within {apart:.2e} of coil 0; this map does not separate '
      f'the coil axis')

  for label in ('mpi2', 'mpi2_dual'):
    got = runs[label]
    assert got.shape == reference.shape
    worst = float(np.abs(got - reference).max() / scale)
    # Same float32 reassociation bound the other rank-count tests use: a
    # different partition sums the same terms in a different order.
    assert worst < 1e-4, (
      f'{label} differs from serial by {worst:.3e} of peak, above float32 '
      f'reassociation -- the map is following the partition, not the node')


@pytest.mark.slow
@pytest.mark.requires_mpi
@pytest.mark.pulseq
@pytest.mark.timeout(420)
def test_the_bin_readout_survives_mpi_and_dual_partitioning(tmp_path):
  """The per-sub-spin readout must give the same k-space at any rank count and
  under either partitioning scheme.

  It had no coverage of either. The bins are per-LOCAL-node arrays and the
  readout runs on the SIGNAL layout, so under dual partitioning every
  `set_static_fields` and `update_magnetization` inside the bin loop is an
  Alltoallv that has to move the right rows -- and `bin_offsets` is indexed by
  the bloch layout, which is the one the caller set the static fields in.
  Nothing checked that those two agree.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  pytest.importorskip('pypulseq')
  if shutil.which('mpirun') is None:
    pytest.skip('mpirun not on PATH')

  from conftest import skip_if_pypulseq_too_old
  from _phantom_fixtures import make_cube_mesh

  seq_path = Path(__file__).resolve().parent / 'data' / 'cpmg_v15.seq'
  if not seq_path.exists():
    pytest.skip('run tests/data/generate_seq_fixtures.py')
  skip_if_pypulseq_too_old(seq_path)

  mesh_path = tmp_path / 'cube.vtu'
  make_cube_mesh(mesh_path, 'tetra', n=4, scale=1e-3)

  env = os.environ.copy()
  env.setdefault('OPENBLAS_NUM_THREADS', '1')
  env.setdefault('OMP_NUM_THREADS', '1')
  env.setdefault('MPLBACKEND', 'Agg')

  def run(tag, ranks, dual):
    out = tmp_path / f'{tag}.npz'
    cmd = [sys.executable, str(_PULSEQ_RUNNER), '--mesh', str(mesh_path),
           '--seq', str(seq_path), '--output', str(out),
           '--t2-prime', '8.0', '--spectral-bins', '16']
    if dual:
      cmd.append('--dual')
    if ranks > 1:
      cmd = ['mpirun', '--allow-run-as-root', '--oversubscribe',
             '-n', str(ranks)] + cmd
    proc = _run(cmd, env)
    assert proc.returncode == 0, proc.stdout.decode(errors='replace')[-4000:]
    return np.load(out)['kspace']

  single_1 = run('single_1', 1, False)
  scale = float(np.abs(single_1).max())
  assert scale > 0, 'the reference run produced no signal'

  # One rank, two layouts: no communication happens, so this must be exact.
  dual_1 = run('dual_1', 1, True)
  assert np.array_equal(dual_1, single_1), (
    'dual partitioning changed the answer at ONE rank, where it redistributes '
    'nothing')

  # Two ranks, either scheme. The assembler accumulates in float32, so a
  # different partition sums the same terms in a different order; the
  # no-ensemble path on this mesh reads 2.3e-5 by the same measure, so a
  # bin-specific defect would have to hide under the ordinary noise floor.
  for tag, dual in (('single_2', False), ('dual_2', True)):
    got = run(tag, 2, dual)
    worst = float(np.abs(got - single_1).max() / scale)
    assert worst < 1e-4, (
      f'{tag} disagrees with the serial single-partition run by {worst:.2e}')


@pytest.mark.slow
@pytest.mark.requires_mpi
@pytest.mark.pulseq
@pytest.mark.timeout(240)
def test_a_rank_asymmetric_refusal_does_not_hang(tmp_path):
  """Two refusals that fired on SOME ranks only, both of which hung.

  simulate_pulseq's row-count guard compares the remembered static fields
  against the sub-spin offsets. Under dual partitioning the bloch and signal
  layouts have different per-rank node counts, so the comparison is
  rank-local: measured on a 4-cube at 6 ranks, it fired on 4 of them. Calling
  the collective inside that branch left the other 2 walking into the
  redistribution while 4 waited in the allgather -- all six timed out.

  solve()'s length check had the same shape: it raised on the offending rank
  while every other rank waited in solve()'s closing Barrier, and the barrier
  counts then desynchronised so rank 0's solve() RETURNED and the deadlock
  surfaced later somewhere unrelated.

  The timeouts are the real assertion here: before the fix neither command
  came back at all.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  pytest.importorskip('pypulseq')
  if shutil.which('mpirun') is None:
    pytest.skip('mpirun not on PATH')

  from conftest import skip_if_pypulseq_too_old
  from _phantom_fixtures import make_cube_mesh

  env = os.environ.copy()
  env.setdefault('OPENBLAS_NUM_THREADS', '1')
  env.setdefault('MPLBACKEND', 'Agg')

  seq_path = Path(__file__).resolve().parent / 'data' / 'cpmg_v15.seq'
  if not seq_path.exists():
    pytest.skip('run tests/data/generate_seq_fixtures.py')
  skip_if_pypulseq_too_old(seq_path)
  cube = tmp_path / 'cube.vtu'
  make_cube_mesh(cube, 'tetra', n=4, scale=1e-3)

  # 6 ranks: the count at which some ranks' two layouts happen to agree.
  guard = subprocess.run(
    ['mpirun', '--allow-run-as-root', '--oversubscribe', '-n', '6',
     sys.executable, str(_PULSEQ_RUNNER), '--mesh', str(cube),
     '--seq', str(seq_path), '--output', str(tmp_path / 'unused.npz'),
     '--dual', '--fields-under-signal'],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
  out = guard.stdout.decode(errors='replace')
  assert guard.returncode != 0, 'the mismatched static fields were accepted'
  assert out.count('ValueError') >= 6, (
    f'only some ranks reported the mismatch, so the rest never reached the '
    f'collective that raises it:\n{out[-3000:]}')

  rod = tmp_path / 'rod.vtu'
  make_1d_rod_mesh(rod, length=0.08, n_segments=48, transverse_width=2e-4)
  at_solve = subprocess.run(
    ['mpirun', '--allow-run-as-root', '--oversubscribe', '-n', '2',
     sys.executable, str(_REALISM_RUNNER), '--mesh', str(rod),
     '--output', str(tmp_path / 'unused2.npz'), '--poison-at-solve', '1'],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
  out = at_solve.stdout.decode(errors='replace')
  assert at_solve.returncode != 0, 'a short delta_B at solve time was accepted'
  assert out.count('ValueError') >= 2, (
    f'the solve-time length check raised on one rank only:\n{out[-3000:]}')
  assert 'delta_B' in out, out[-2000:]


@pytest.mark.slow
@pytest.mark.requires_mpi
@pytest.mark.timeout(240)
@pytest.mark.parametrize('case', ['static_fields', 'update_mag', 'b1_map',
                                  'coil_map', 'b0_gradient'])
def test_every_per_node_refusal_reaches_every_rank(tmp_path, case):
  """Three more refusals whose condition is true on a SUBSET of ranks, each
  sitting upstream of a collective. All three hung.

  - `set_static_fields`: a T2 map with one air node at zero has that node on
    one rank, and the guard ran before the layout redistribution's Alltoallv.
    This one was introduced by the guard added earlier in this same audit.
  - `update_magnetization`: SPMD code passes ONE length, so it matches the
    local node count on some ranks and not others, and the raise sat upstream
    of the same Alltoallv.
  - `b1_map`: the one public per-node array left out of the collective row
    check, so the kernel's own length check threw rank-locally and the other
    ranks waited in solve()'s closing Barrier.
  - `coil_map`: the receive sensitivity, added later and given the collected
    form from the start rather than after a hang -- one NaN at a single global
    node lives on one rank, and the setter redistributes.
  - `b0_gradient`: the scanner field's per-node Eulerian channel, the same
    shape as `coil_map` and upstream of the same Alltoallv.

  The timeout is the assertion: before the fix none of these came back.
  """
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  if shutil.which('mpirun') is None:
    pytest.skip('mpirun not on PATH')

  mesh_path = tmp_path / 'rod.vtu'
  make_1d_rod_mesh(mesh_path, length=0.08, n_segments=48,
                   transverse_width=2e-4)
  env = os.environ.copy()
  env.setdefault('OPENBLAS_NUM_THREADS', '1')
  env.setdefault('MPLBACKEND', 'Agg')

  proc = subprocess.run(
    ['mpirun', '--allow-run-as-root', '--oversubscribe', '-n', '2',
     sys.executable, str(_REALISM_RUNNER), '--mesh', str(mesh_path),
     '--output', str(tmp_path / 'unused.npz'), '--refusal-case', case],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=150)
  out = proc.stdout.decode(errors='replace')
  assert proc.returncode != 0, f'{case} was accepted'
  # Count the runner's own per-rank markers, not tracebacks: ONE traceback
  # contains the word "Error" twice, so the obvious `count('Error') >= 2`
  # passes on a single-rank raise -- exactly the bug these cases exist for.
  # Verified by mutation: with the collective replaced by a bare raise, this
  # reads 1 refusal where the old form read 2 "Error"s and passed.
  assert out.count('REFUSED') == 2, (
    f'{case} was refused on {out.count("REFUSED")} of 2 ranks; a rank that '
    f'neither refused nor accepted is blocked in the collective that reports '
    f'it:\n{out[-3000:]}')
  assert 'ACCEPTED' not in out, (
    f'{case} was accepted on some rank:\n{out[-3000:]}')

