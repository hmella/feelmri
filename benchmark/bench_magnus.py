"""Bloch-portion benchmark for the cayley_klein / magnus2 / magnus4 solvers.

Builds a SPAMM-like workload (one SPAMM preparation block followed by a
configurable number of soft-RF imaging blocks) on a small synthetic
tetrahedral phantom and times only the C++ kernel via
``BlochSolver.bloch_elapsed``. Per-step cost lets you compare methods
independent of total sequence length.

Run-time isolation:
  export OPENBLAS_NUM_THREADS=1
  export OMP_NUM_THREADS=8
  taskset -c 0-7 python3 benchmark/bench_magnus.py

CSV output goes to ``benchmark/results/magnus_<hostname>.csv``.

For a small smoke run, pass ``--quick``: one imaging block and fewer
nodes so the whole thing finishes in seconds.
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import socket
import time
from pathlib import Path

# Match the spamm.py preamble so BLAS does not compete with OpenMP threads.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
from pint import Quantity

from feelmri import BlochSolver, FEMPhantom, Scanner
from feelmri.Bloch import Sequence, SequenceBlock
from feelmri.MRObjects import RF, Gradient


HERE = Path(__file__).resolve().parent
TESTS = HERE.parent / 'tests'

# Re-use the disk-mesh helper from the test fixtures so we get a phantom
# whose node count we can dial up or down.
import sys
sys.path.insert(0, str(TESTS))
from _phantom_fixtures import make_2d_disk_mesh  # noqa: E402


# ---------------------------------------------------------------------------
# Sequence builder (SPAMM-prep + soft-RF imaging, no ADC or k-space output)
# ---------------------------------------------------------------------------

def build_spamm_like_sequence(n_imaging_blocks: int = 20,
                              rf_dur_ms: float = 1.0,
                              dt_rf_ms: float = 0.01,
                              dt_gr_ms: float = 0.01,
                              dt_block_ms: float = 1.0) -> Sequence:
  """Build a sequence that exercises the inner kernel similarly to
  ``examples/spamm.py``: one SPAMM preparation block (two hard RF pulses
  bracketing a tag gradient) followed by ``n_imaging_blocks`` soft-RF
  imaging blocks. ADC, recon, and k-space are intentionally omitted —
  this benchmark times the Bloch evolution only."""
  scanner = Scanner()

  seq = Sequence()

  # SPAMM preparation block (small hard pulses + short tag gradient).
  rf_pre_a = RF(
    scanner=scanner, shape='hard',
    flip_angle=Quantity(np.pi / 2, 'rad'),
    dur=Quantity(0.2, 'ms'),
    time=Quantity(0.0, 'ms'),
  )
  rf_pre_b = RF(
    scanner=scanner, shape='hard',
    flip_angle=Quantity(np.pi / 2, 'rad'),
    dur=Quantity(0.2, 'ms'),
    time=Quantity(0.6, 'ms'),
  )
  g_tag = Gradient(
    scanner=scanner, axis=0,
    timings=Quantity(np.array([0.2, 0.6]), 'ms'),
    amplitudes=Quantity(np.array([5.0, 5.0]), 'mT/m'),
    time=Quantity(0.0, 'ms'),
  )
  prep = SequenceBlock(
    rf_pulses=[rf_pre_a, rf_pre_b],
    gradients=[g_tag],
    dur=Quantity(0.8, 'ms'),
    dt_rf=Quantity(dt_rf_ms, 'ms'),
    dt_gr=Quantity(dt_gr_ms, 'ms'),
    dt=Quantity(dt_block_ms, 'ms'),
    store_magnetization=False,
  )
  seq.add_block(prep)

  # Imaging blocks: apodized-sinc RF with a slice-select gradient. Many
  # short time steps inside each block, which is the regime where Magnus
  # orders matter.
  for _ in range(n_imaging_blocks):
    rf_img = RF(
      scanner=scanner,
      NbLobes=[2, 2],
      alpha=0.46,
      shape='apodized_sinc',
      flip_angle=Quantity(np.deg2rad(10.0), 'rad'),
      dur=Quantity(rf_dur_ms, 'ms'),
      nb_samples=512,
    )
    g_ss = Gradient(
      scanner=scanner, axis=2,
      timings=Quantity(np.array([0.0, rf_dur_ms]), 'ms'),
      amplitudes=Quantity(np.array([10.0, 10.0]), 'mT/m'),
      time=Quantity(0.0, 'ms'),
    )
    imaging = SequenceBlock(
      rf_pulses=[rf_img],
      gradients=[g_ss],
      dur=Quantity(rf_dur_ms + 1.0, 'ms'),
      dt_rf=Quantity(dt_rf_ms, 'ms'),
      dt_gr=Quantity(dt_gr_ms, 'ms'),
      dt=Quantity(dt_block_ms, 'ms'),
      store_magnetization=True,
    )
    seq.add_block(imaging)

  return seq


# ---------------------------------------------------------------------------
# Phantom factory
# ---------------------------------------------------------------------------

def build_phantom(tmp_dir: Path, n_radial: int, n_angular: int) -> FEMPhantom:
  mesh_path = tmp_dir / 'disk.vtu'
  make_2d_disk_mesh(mesh_path, radius=5e-3,
                    n_radial=n_radial, n_angular=n_angular,
                    thickness=1e-4)
  return FEMPhantom(path=str(mesh_path))


# ---------------------------------------------------------------------------
# One configuration run
# ---------------------------------------------------------------------------

def time_one(seq: Sequence, phantom: FEMPhantom, *,
             method: str, dtype: str, repeats: int = 3) -> dict:
  """Run ``solver.solve()`` ``repeats`` times and report the median wall-
  clock spent strictly inside the C++ kernel
  (``solver.bloch_elapsed``)."""
  bloch_times = []
  total_times = []
  n_steps_total = None

  for _ in range(repeats):
    solver = BlochSolver(
      seq, phantom,
      M0=1.0,
      T1=Quantity(1000.0, 'ms'),
      T2=Quantity(100.0, 'ms'),
      delta_B=5e-4,
      initial_Mxy=0.0 + 0.0j,
      initial_Mz=1.0,
      perfect_spoiling=False,
      method=method,
      dtype=dtype,
    )
    t0 = time.perf_counter()
    Mxy, Mz = solver.solve()
    total = time.perf_counter() - t0
    bloch_times.append(solver.bloch_elapsed)
    total_times.append(total)
    if n_steps_total is None:
      n_steps_total = sum(b.discrete_times.size for b in seq.blocks)

  median_bloch = float(np.median(bloch_times))
  median_total = float(np.median(total_times))
  n_nodes = phantom.local_nodes.shape[0]
  ns_per_step_per_node = (median_bloch * 1e9
                          / max(n_steps_total - len(seq.blocks), 1)
                          / max(n_nodes, 1))
  return dict(
    method=method,
    dtype=dtype,
    bloch_s=median_bloch,
    total_s=median_total,
    n_nodes=n_nodes,
    n_steps_total=n_steps_total,
    ns_per_step_per_node=ns_per_step_per_node,
  )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--quick', action='store_true',
                      help='Tiny phantom + few blocks for smoke testing.')
  parser.add_argument('--blocks', type=int, default=None,
                      help='Number of imaging blocks (overrides --quick).')
  parser.add_argument('--repeats', type=int, default=3,
                      help='Number of repeats per (method, dtype) for median.')
  parser.add_argument('--n-radial', type=int, default=None)
  parser.add_argument('--n-angular', type=int, default=None)
  parser.add_argument('--out', type=Path, default=None,
                      help='CSV output path (defaults to '
                           'benchmark/results/magnus_<hostname>.csv).')
  args = parser.parse_args()

  if args.quick:
    n_blocks = args.blocks if args.blocks is not None else 2
    n_radial = args.n_radial if args.n_radial is not None else 3
    n_angular = args.n_angular if args.n_angular is not None else 8
  else:
    n_blocks = args.blocks if args.blocks is not None else 20
    n_radial = args.n_radial if args.n_radial is not None else 6
    n_angular = args.n_angular if args.n_angular is not None else 32

  # Self-contained tmp dir for the mesh.
  import tempfile
  with tempfile.TemporaryDirectory(prefix='feelmri_bench_') as tmp_str:
    tmp_dir = Path(tmp_str)
    phantom = build_phantom(tmp_dir, n_radial=n_radial, n_angular=n_angular)
    seq = build_spamm_like_sequence(n_imaging_blocks=n_blocks)

    rows = []
    configs = [
      ('cayley_klein', 'float32'),
      ('cayley_klein', 'float64'),
      ('magnus2',      'float32'),
      ('magnus2',      'float64'),
      ('magnus4',      'float32'),
      ('magnus4',      'float64'),
    ]
    print(f"# Phantom local nodes: {phantom.local_nodes.shape[0]}")
    print(f"# Imaging blocks: {n_blocks}; repeats: {args.repeats}")
    print(f"# OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}, "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'unset')}")
    print()
    print(f"{'method':<14}{'dtype':<10}{'bloch_s':>12}"
          f"{'total_s':>12}{'ns/step/node':>16}")
    for method, dtype in configs:
      row = time_one(seq, phantom, method=method, dtype=dtype,
                     repeats=args.repeats)
      print(f"{row['method']:<14}{row['dtype']:<10}"
            f"{row['bloch_s']:>12.4f}{row['total_s']:>12.4f}"
            f"{row['ns_per_step_per_node']:>16.2f}")
      rows.append(row)

    # Write CSV.
    hostname = socket.gethostname()
    if args.out is None:
      out_dir = HERE / 'results'
      out_dir.mkdir(parents=True, exist_ok=True)
      out_path = out_dir / f'magnus_{hostname}.csv'
    else:
      out_path = args.out
      out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', newline='') as fh:
      writer = csv.writer(fh)
      writer.writerow(['hostname', 'platform', 'omp_threads',
                       'method', 'dtype',
                       'bloch_s', 'total_s',
                       'n_nodes', 'n_steps_total',
                       'ns_per_step_per_node'])
      for row in rows:
        writer.writerow([
          hostname,
          platform.platform(),
          os.environ.get('OMP_NUM_THREADS', 'unset'),
          row['method'], row['dtype'],
          f"{row['bloch_s']:.6f}",
          f"{row['total_s']:.6f}",
          row['n_nodes'],
          row['n_steps_total'],
          f"{row['ns_per_step_per_node']:.2f}",
        ])
    print()
    print(f"# Results written to {out_path}")


if __name__ == '__main__':
  main()
