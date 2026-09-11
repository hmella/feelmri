"""CLI used by ``tests/test_mpi_equivalence.py``.

Runs one Bloch simulation with ALL THREE solver features added on the
``realism`` branch active at once -- concomitant fields, a per-node B1+ map and
a per-node T2' sub-ensemble -- then gathers Mxy / Mz to rank 0 in global node
order and writes a ``.npz``. Invoked both directly and under ``mpirun -n 2``;
the test asserts the two agree.

The two per-node maps are built from node POSITIONS rather than from local
indices, so they describe the same physical field whatever the partition. That
is the point of the test: ``b1_map`` and ``t2_prime`` are indexed by LOCAL node,
and ``t2_prime`` additionally allocates per-rank K-fold state, so a rank
ordering bug would be invisible to every serial test."""
from __future__ import annotations

import argparse
import os
import sys

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

from feelmri import Sequence
from feelmri.Bloch import BlochSolver, SequenceBlock
from feelmri.MPIUtilities import MPI_comm, MPI_rank
from feelmri.MRObjects import Gradient, Scanner
from feelmri.Phantom import FEMPhantom

from _seq_fixtures import make_empty_block, make_hard_pulse_block

TAU_MS = 8.0
GRADIENT_MS = 3.0


def _build_sequence(scanner):
  """90 -- gradient -- tau -- 180 -- tau. The gradient block is what makes the
  concomitant term do anything; the 180 is what makes the T2' ensemble do
  anything."""
  seq = Sequence()
  pulse = make_hard_pulse_block(np.pi / 2, dur_ms=0.05)
  pulse.store_magnetization = True
  seq.add_block(pulse)

  gradients = [
    Gradient(timings=Quantity(np.array([0.0, GRADIENT_MS]), 'ms'),
             amplitudes=Quantity(np.array([amp, amp]), 'mT/m'),
             scanner=scanner, ref=Quantity(0.0, 'ms'),
             time=Quantity(0.0, 'ms'), axis=axis)
    for axis, amp in enumerate((14.0, -9.0, 20.0))]
  seq.add_block(SequenceBlock(gradients=gradients,
                              dur=Quantity(GRADIENT_MS, 'ms'),
                              dt=Quantity(0.02, 'ms'), empty=False,
                              store_magnetization=True))

  seq.add_block(make_empty_block(TAU_MS, dt_ms=TAU_MS))
  refocus = make_hard_pulse_block(np.pi, dur_ms=0.05)
  refocus.store_magnetization = True
  seq.add_block(refocus)
  seq.add_block(make_empty_block(TAU_MS, dt_ms=TAU_MS))
  return seq


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('--mesh', required=True)
  ap.add_argument('--output', required=True)
  ap.add_argument('--poison-rank', type=int, default=-1,
                  help='hand this rank a delta_B one node short. Every rank '
                       'must then raise; if only the poisoned rank does, the '
                       'others block in the collective that reports it.')
  ap.add_argument('--poison-at-solve', type=int, default=-1,
                  help='give this rank a short delta_B AFTER construction, so '
                       'the refusal has to come from solve()')
  args = ap.parse_args(argv)

  if not os.path.exists(args.mesh):
    raise FileNotFoundError(f'mesh fixture missing: {args.mesh}')

  phantom = FEMPhantom(path=args.mesh)
  scanner = Scanner()

  # Smooth spatial functions, so each node's value follows the node and not
  # its position in some rank-local array.
  nodes = phantom.local_nodes.astype(np.float64)
  # GLOBAL, not per-rank. A rank-local maximum makes the maps a function of
  # how the mesh was cut, so the two runs would be comparing different physical
  # fields -- which on this symmetric rod happens to agree at exactly 2 ranks
  # and stops agreeing at 3.
  reach = float(MPI_comm.allreduce(float(np.abs(nodes[:, 0]).max()),
                                   op=MPI.MAX)) or 1.0
  u = nodes[:, 0] / reach
  b1_map = (0.7 + 0.3 * np.cos(np.pi * u)) * np.exp(0.4j * u)
  t2_prime = 6.0 + 4.0 * u**2

  # A per-rank field built against the wrong node count is the realistic way
  # this goes wrong -- under dual partitioning the two layouts have different
  # per-rank counts, so an array can match on one rank and not on another.
  delta_B = np.zeros((nodes.shape[0], 1))
  if args.poison_rank == MPI_rank:
    delta_B = delta_B[:-1]

  solver = BlochSolver(
    _build_sequence(scanner), phantom,
    scanner=scanner, M0=1.0, delta_B=delta_B,
    T1=Quantity(1e9, 'ms'), T2=Quantity(400.0, 'ms'),
    initial_Mxy=0.0, initial_Mz=1.0,
    perfect_spoiling=False, dtype='float64',
    concomitant_fields=True,
    b1_map=b1_map,
    t2_prime=Quantity(t2_prime, 'ms'), spectral_bins=32)
  if args.poison_at_solve >= 0:
    solver.solve()
    if args.poison_at_solve == MPI_rank:
      solver.delta_B = np.zeros(nodes.shape[0] - 1)
    solver.solve()

  Mxy_local, Mz_local = solver.solve()

  l2g = np.asarray(phantom.local_to_global_nodes, dtype=np.int64)
  n_global = int(MPI_comm.allreduce(int(l2g.max()) + 1, op=MPI.MAX))
  n_marked = Mxy_local.shape[1]

  gathered_l2g = MPI_comm.gather(l2g, root=0)
  gathered_Mxy = MPI_comm.gather(np.asarray(Mxy_local, dtype=np.complex128), root=0)
  gathered_Mz = MPI_comm.gather(np.asarray(Mz_local, dtype=np.float64), root=0)

  if MPI_rank == 0:
    Mxy_global = np.zeros((n_global, n_marked), dtype=np.complex128)
    Mz_global = np.zeros((n_global, n_marked), dtype=np.float64)
    for indices, mxy, mz in zip(gathered_l2g, gathered_Mxy, gathered_Mz):
      Mxy_global[indices, :] = mxy
      Mz_global[indices, :] = mz
    np.savez(args.output, Mxy=Mxy_global, Mz=Mz_global)


if __name__ == '__main__':
  main()
