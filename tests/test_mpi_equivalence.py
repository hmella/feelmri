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
