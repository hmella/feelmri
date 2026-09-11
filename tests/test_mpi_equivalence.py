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
