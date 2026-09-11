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
  ap.add_argument('--t2-prime', type=float, default=0.0,
                  help='ms; non-zero turns on the spectral sub-ensemble, so '
                       'the readout is evaluated once per sub-spin')
  ap.add_argument('--spectral-bins', type=int, default=16)
  ap.add_argument('--dual', action='store_true',
                  help='build two partitions instead of one, so every static '
                       'field and magnetization handoff is redistributed')
  args = ap.parse_args()

  from mpi4py import MPI
  from feelmri.Phantom import FEMPhantom
  from feelmri.PulseqAdapter import simulate_pulseq

  comm = MPI.COMM_WORLD
  rank, size = comm.Get_rank(), comm.Get_size()

  phantom = FEMPhantom(path=args.mesh)
  if args.dual:
    phantom.enable_dual_partition(voxel_size=0.0, lorder=2, horder=2,
                                  nodal_approximation=False, lumped=False)
  else:
    phantom.set_assembler(voxel_size=0.0, lorder=2, horder=2,
                          nodal_approximation=False, lumped=False)
  # Under dual partitioning this is the BLOCH layout, which is what is active
  # here and what the solver's bin offsets are indexed by.
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n, 60.0, dtype=np.float32),
                            phi_dB0=np.zeros(n, dtype=np.float32))

  extra = {}
  if args.t2_prime > 0.0:
    extra = dict(t2_prime=Quantity(args.t2_prime, 'ms'),
                 spectral_bins=args.spectral_bins)

  sim = simulate_pulseq(args.seq, phantom, M0=1.0,
                        T1=Quantity(1e9, 'ms'), T2=Quantity(60.0, 'ms'),
                        dtype='float64', **extra)
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
