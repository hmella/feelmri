"""CLI used by ``tests/test_mpi_equivalence.py::test_simulate_pulseq_matches_serial_under_mpi``.

Runs :func:`feelmri.PulseqAdapter.simulate_pulseq` on a cube mesh and writes
rank 0's k-space plus every rank's own partial sum, so the test can assert two
things the suite never checked: that the gathered signal is rank-count
independent, and that ``gather=True`` really does leave non-root ranks empty
(``gather_data`` is an ``MPI_comm.Reduce(root=0)``, not an Allreduce).

    python3 tests/helpers/pulseq_mpi_runner.py --mesh m.vtu --seq s.seq --output k.npz
"""
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
from pint import Quantity


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument('--mesh', required=True)
  ap.add_argument('--seq', required=True)
  ap.add_argument('--output', required=True)
  args = ap.parse_args()

  from mpi4py import MPI
  from feelmri.Phantom import FEMPhantom
  from feelmri.PulseqAdapter import simulate_pulseq

  comm = MPI.COMM_WORLD
  rank, size = comm.Get_rank(), comm.Get_size()

  phantom = FEMPhantom(path=args.mesh)
  phantom.set_assembler(voxel_size=0.0, lorder=2, horder=2,
                        nodal_approximation=False, lumped=False)
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 60.0, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))

  sim = simulate_pulseq(args.seq, phantom, M0=1.0,
                        T1=Quantity(1e9, 'ms'), T2=Quantity(60.0, 'ms'),
                        dtype='float64')
  gathered = np.asarray(sim.kspace[0]).reshape(-1)

  if rank == 0:
    np.savez(args.output, kspace=gathered, size=np.int64(size))
  else:
    # Every non-root rank records what IT was handed, so the test can pin the
    # Reduce-to-root contract rather than assume it.
    np.savez(f'{args.output}.rank{rank}', kspace=gathered, size=np.int64(size))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
