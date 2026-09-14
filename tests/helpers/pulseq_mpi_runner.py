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
  ap.add_argument('--fields-under-signal', action='store_true',
                  help='set the static fields while the SIGNAL layout is live, '
                       'so their row count disagrees with the solver on some '
                       'ranks and not others')
  ap.add_argument('--coils', type=int, default=0,
                  help='build this many receive coils, each a smooth function '
                       'of NODE POSITION, and pass them as coil_sensitivities. '
                       'Position-tied on purpose: a redistribution that moved '
                       'the wrong rows is invisible against a constant map.')
  ap.add_argument('--b0-nodal', action='store_true',
                  help='carry a scanner-fixed field that no polynomial can '
                       'represent, so it rides the per-node channel: the '
                       'values on phi_dB0 and the gradient on '
                       'set_b0_gradient, both redistributed under --dual')
  ap.add_argument('--pod', action='store_true',
                  help='hand simulate_pulseq a rigid-translation trajectory. '
                       'Without it `moving` is False, `readout_terms` returns '
                       'no gradient at all, and --b0-nodal exercises only the '
                       'phi_dB0 half.')
  ap.add_argument('--b0-frozen', action='store_true',
                  help='with --b0-nodal, install the field VALUES but not the '
                       'gradient, i.e. the Lagrangian description. The two '
                       'must differ, or a rank-count comparison passes with '
                       'the gradient channel disabled on every rank.')
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
  #
  # phi_dB0 VARIES IN SPACE, and that is the point: with a constant map (and a
  # magnetization that comes out spatially uniform on these fixtures) a
  # redistribution that moved the wrong rows changed the k-space by exactly
  # 0.000e+00 -- reversing the destination order of every redistributed array
  # was undetectable. Tied to the node position so it follows the node, not its
  # index in some rank-local array.
  if args.fields_under_signal:
    # The mistake the row-count guard exists to diagnose. Under dual
    # partitioning the two layouts have different per-rank node counts, so
    # this is rank-asymmetric: the guard fires on some ranks and not others.
    with phantom._using('signal'):
      m = phantom.local_nodes.shape[0]
      phantom.set_static_fields(T2=np.full(m, 60.0, dtype=np.float32),
                                phi_dB0=np.zeros(m, dtype=np.float32))
    sim = simulate_pulseq(args.seq, phantom, M0=1.0, T1=Quantity(1e9, 'ms'),
                          T2=Quantity(60.0, 'ms'), dtype='float64',
                          t2_prime=Quantity(8.0, 'ms'), spectral_bins=8)
    return 0

  nodes = phantom.local_nodes.astype(np.float64)
  reach = float(np.abs(nodes).max()) or 1.0
  n = nodes.shape[0]
  phantom.set_static_fields(
      T2=np.full(n, 60.0, dtype=np.float32),
      phi_dB0=(2.0 * nodes[:, 0] / reach).astype(np.float32))

  extra = {}
  if args.b0_nodal:
    from feelmri import B0Field
    # Curved on the scale of the object, so `on_phantom` falls back to the
    # per-node rung. Sampled in SCANNER coordinates, so it follows the node
    # through the redistribution rather than its index in a rank-local array.
    rough = (lambda q: 1.0e-3 * np.sin(q[:, 0] / (0.45 * reach))
             * np.cos(q[:, 1] / (0.55 * reach)))
    field = B0Field.on_phantom(rough, phantom)
    if field.kind != 'nodal':
      raise SystemExit(f'the fixture must need the per-node rung, '
                       f'got {field.kind}')
    if args.b0_frozen:
      # The field frozen onto the node: its values on phi_dB0 and no gradient
      # at all. `simulate_pulseq` is given no b0_field, so nothing installs one.
      from feelmri.MRObjects import Scanner as _Scanner
      gamma = _Scanner().gamma.m_as('rad/ms/mT')
      nodal = field.nodal_mT(phantom)
      phantom.set_static_fields(
          T2=np.full(n, 60.0, dtype=np.float32),
          phi_dB0=((2.0 * nodes[:, 0] / reach) + gamma * nodal).astype(np.float32))
    else:
      extra['b0_field'] = field
  if args.pod:
    from feelmri.Motion import POD
    n_frames = 4
    # GLOBAL snapshots plus the node map, which is what every shipped example
    # does and what the contract requires: built from per-rank data instead,
    # each rank runs its own SVD and its modes carry its own normalisation, so
    # dual partitioning -- which redistributes modes BETWEEN ranks -- pairs a
    # node's mode with another rank's weights. Measured 1.76e-02 of peak that
    # way against 8.8e-07 this way. `FEMPhantom._signal_modes` refuses it now,
    # but the fixture should be right regardless.
    disp = np.zeros((phantom.global_shape[0], 3, n_frames), dtype=np.float32)
    for axis, amp in enumerate((0.30, -0.20, 0.25)):
      disp[:, axis, :] = amp * reach
    extra['pod'] = POD(data=disp, times=np.linspace(0.0, 400.0, n_frames),
                       n_modes=1, is_periodic=True,
                       global_to_local=phantom.local_to_global_nodes)
  if args.t2_prime > 0.0:
    extra = dict(t2_prime=Quantity(args.t2_prime, 'ms'),
                 spectral_bins=args.spectral_bins)
  if args.coils > 0:
    # Tied to the node's POSITION, so the map follows the node through the
    # redistribution rather than its index in some rank-local array. Each coil
    # gets a different spatial weighting AND a different phase, so a fold that
    # collapsed the coil axis or reused one column is visible.
    span = nodes / (reach or 1.0)
    extra['coil_sensitivities'] = np.stack(
        [(1.0 + 0.5 * np.cos((c + 1) * np.pi * span[:, c % 3]))
         * np.exp(1j * (0.6 * c + 0.3 * span[:, (c + 1) % 3]))
         for c in range(args.coils)], axis=1).astype(np.complex64)

  sim = simulate_pulseq(args.seq, phantom, M0=1.0,
                        T1=Quantity(1e9, 'ms'), T2=Quantity(60.0, 'ms'),
                        dtype='float64', **extra)
  gathered = np.asarray(sim.kspace[0])
  gathered = gathered.reshape(-1, gathered.shape[-1]) if args.coils > 0 \
      else gathered.reshape(-1)

  if rank == 0:
    np.savez(args.output, kspace=gathered, size=np.int64(size))
  else:
    # Every non-root rank records what IT was handed, so the test can pin the
    # Reduce-to-root contract rather than assume it.
    np.savez(f'{args.output}.rank{rank}', kspace=gathered, size=np.int64(size))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
