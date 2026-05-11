"""CLI used by ``tests/test_mpi_equivalence.py``.

Runs a fixed two-block Bloch simulation on the minimal 5-node mesh,
gathers per-rank Mxy / Mz to rank 0 in global-node-index order, and
writes a ``.npz`` file. Invoked as

    python3 tests/helpers/mpi_runner.py --mesh /tmp/m.vtu --output /tmp/M.npz

both directly (1 rank) and via ``mpirun -n 2``. The MPI equivalence
test then asserts that the two ``.npz`` outputs match within
single-precision tolerance."""
from __future__ import annotations

import argparse
import os
import sys

# When the test invokes this script with ``-m tests.helpers.mpi_runner``
# the test package is on sys.path; when invoked as a plain file path
# (the safer cross-version option) we need to push the repo root in.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)
_TESTS_DIR = os.path.dirname(_THIS_DIR)
if _TESTS_DIR not in sys.path:
  sys.path.insert(0, _TESTS_DIR)

import numpy as np
from mpi4py import MPI
from pint import Quantity

from feelmri.Bloch import BlochSolver
from feelmri.MPIUtilities import MPI_comm, MPI_rank
from feelmri.MRObjects import Scanner
from feelmri.Phantom import FEMPhantom

from _seq_fixtures import (
  make_empty_block,
  make_hard_pulse_block,
)


def _build_sequence():
  """Hard 90 deg pulse followed by a short free-precession interval.
  Both blocks store magnetization so ``BlochSolver.solve()`` returns
  Mxy/Mz at the end of each."""
  from feelmri import Sequence

  seq = Sequence()
  pulse = make_hard_pulse_block(np.pi / 2, dur_ms=0.1)
  pulse.store_magnetization = True
  seq.add_block(pulse)
  precession = make_empty_block(1.0, dt_ms=0.05)
  precession.store_magnetization = True
  seq.add_block(precession)
  return seq


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('--mesh', required=True)
  ap.add_argument('--output', required=True)
  args = ap.parse_args(argv)

  if not os.path.exists(args.mesh):
    raise FileNotFoundError(
      f'mesh fixture missing: {args.mesh} — create it from the test '
      f'before invoking this runner.'
    )

  phantom = FEMPhantom(path=args.mesh)
  seq = _build_sequence()
  solver = BlochSolver(
    seq, phantom,
    scanner=Scanner(),
    M0=1.0,
    T1=Quantity(500.0, 'ms'),
    T2=Quantity(100.0, 'ms'),
    initial_Mxy=0.0,
    initial_Mz=1.0,
    perfect_spoiling=False,
  )
  Mxy_local, Mz_local = solver.solve()

  # Gather to rank 0 in canonical global-node ordering.
  l2g = np.asarray(phantom.local_to_global_nodes, dtype=np.int64)
  n_global = int(MPI_comm.allreduce(int(l2g.max()) + 1, op=MPI.MAX))
  n_marked = Mxy_local.shape[1]

  gathered_l2g  = MPI_comm.gather(l2g, root=0)
  gathered_Mxy  = MPI_comm.gather(np.asarray(Mxy_local, dtype=np.complex64), root=0)
  gathered_Mz   = MPI_comm.gather(np.asarray(Mz_local,  dtype=np.float32),  root=0)

  if MPI_rank == 0:
    Mxy_global = np.zeros((n_global, n_marked), dtype=np.complex64)
    Mz_global  = np.zeros((n_global, n_marked), dtype=np.float32)
    for indices, mxy, mz in zip(gathered_l2g, gathered_Mxy, gathered_Mz):
      Mxy_global[indices, :] = mxy
      Mz_global[indices, :]  = mz
    np.savez(args.output, Mxy=Mxy_global, Mz=Mz_global)


if __name__ == '__main__':
  main()
