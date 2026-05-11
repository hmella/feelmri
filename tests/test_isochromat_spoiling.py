"""Tests for the runtime-K isochromat spoiling refactor.

Covers:

* ``feelmri.PulseqAdapter.create_multi_isochromats`` accepting
  ``K``, ``distribution``, and ``seed`` as runtime parameters with
  determinism under a fixed seed.
* :func:`feelmri.spoiling_residual` decreasing monotonically with K
  under a Sobol sampler on a known phase ramp.
* Sobol outperforming uniform Monte-Carlo at the same K (the QMC
  promise: residual ~ (log K)^d / K vs Monte-Carlo's ~ K^{-1/2}).
* :func:`feelmri.plot_isochromat_voxel` producing a valid PNG without
  blowing up on a small synthetic batch.

All tests are sub-second; no MPI, no C++ extension, no phantom."""
from __future__ import annotations

import numpy as np
import pytest

from feelmri import (
  create_multi_isochromats,
  plot_isochromat_voxel,
  spoiling_residual,
)


# Use a kspace wavenumber chosen so that |k_sp| * voxel_size > 1
# (i.e. at least one phase winding across the voxel) — this is the
# regime where the spoiling sum's convergence rate becomes visible.
_K_SP = np.array([400.0, 0.0, 0.0], dtype=np.float64)   # 1/m
_R    = 2.5e-3                                          # m


def test_sobol_outperforms_uniform_at_fixed_K():
  """Same K, multi-trial uniform vs single-trial Sobol: Sobol's
  residual must be smaller than the EXPECTED uniform residual at
  the same K. Sobol is deterministic so n_trials=1 is reproducible;
  uniform is averaged across 8 seeds to suppress single-draw noise
  before the comparison."""
  K = 256
  rho_u, _ = spoiling_residual(K, _K_SP, _R,
                               distribution='uniform',
                               seed=0, n_trials=8)
  rho_s, _ = spoiling_residual(K, _K_SP, _R,
                               distribution='sobol', seed=0)
  assert rho_s < rho_u, (
    f'expected sobol < uniform; got sobol={rho_s:.4g}, '
    f'uniform={rho_u:.4g}'
  )


def test_residual_decays_with_K_under_sobol():
  """Sobol residual must shrink monotonically as K doubles, with a
  small absolute tolerance for the near-saturation regime."""
  rhos = [
    spoiling_residual(K, _K_SP, _R, distribution='sobol', seed=0)[0]
    for K in (32, 64, 128, 256, 512)
  ]
  for i in range(len(rhos) - 1):
    assert rhos[i + 1] <= rhos[i] + 1e-6, (
      f'sobol residual not monotone at K transition '
      f'{(32, 64, 128, 256, 512)[i]} -> {(32, 64, 128, 256, 512)[i+1]}: '
      f'{rhos[i]:.4g} -> {rhos[i+1]:.4g}'
    )
  assert rhos[-1] < rhos[0]


def test_create_multi_isochromats_respects_runtime_K_and_seed():
  """Direct API check: K controls the output row count, dtype is
  preserved, and a fixed (distribution, seed) pair produces
  bit-identical jitter on repeated calls."""
  N = 4
  K = 8
  x   = np.zeros((N, 3), dtype=np.float32)
  T1  = np.full((N, 1), 1000.0, dtype=np.float32)
  T2  = np.full((N, 1), 100.0,  dtype=np.float32)
  dB  = np.zeros((N, 1), dtype=np.float32)
  Mxy = np.zeros((N, 1), dtype=np.complex64)
  Mz  = np.ones((N, 1),  dtype=np.float32)

  x1, *_ = create_multi_isochromats(
    x, T1, T2, dB, Mxy, Mz,
    K=K, pos_jitter=1e-3, distribution='sobol', seed=42,
  )
  x2, *_ = create_multi_isochromats(
    x, T1, T2, dB, Mxy, Mz,
    K=K, pos_jitter=1e-3, distribution='sobol', seed=42,
  )

  assert x1.shape == (N * K, 3)
  assert x1.dtype == np.float32
  np.testing.assert_array_equal(x1, x2)

  # Different seed must produce different draws.
  x3, *_ = create_multi_isochromats(
    x, T1, T2, dB, Mxy, Mz,
    K=K, pos_jitter=1e-3, distribution='sobol', seed=43,
  )
  assert not np.array_equal(x1, x3)


def test_create_multi_isochromats_rejects_unknown_distribution():
  N = 2
  x   = np.zeros((N, 3), dtype=np.float32)
  T1  = np.full((N, 1), 1000.0, dtype=np.float32)
  T2  = np.full((N, 1), 100.0,  dtype=np.float32)
  dB  = np.zeros((N, 1), dtype=np.float32)
  Mxy = np.zeros((N, 1), dtype=np.complex64)
  Mz  = np.ones((N, 1),  dtype=np.float32)
  with pytest.raises(ValueError, match='unknown distribution'):
    create_multi_isochromats(
      x, T1, T2, dB, Mxy, Mz,
      K=4, pos_jitter=1e-3, distribution='gaussian', seed=0,
    )


def test_plot_isochromat_voxel_writes_png(tmp_path):
  """The visualisation helper must produce a non-empty PNG file when
  asked to export."""
  import matplotlib
  matplotlib.use('Agg')

  pts, _ = (lambda: (
    create_multi_isochromats(
      np.zeros((1, 3), dtype=np.float32),
      np.full((1, 1), 1000.0, dtype=np.float32),
      np.full((1, 1), 100.0,  dtype=np.float32),
      np.zeros((1, 1), dtype=np.float32),
      np.zeros((1, 1), dtype=np.complex64),
      np.ones((1, 1),  dtype=np.float32),
      K=128, pos_jitter=1e-3, distribution='sobol', seed=0,
    )[0],
    None,
  ))()

  png = tmp_path / 'iso.png'
  plot_isochromat_voxel(pts, R=1e-3, show=True, export_to=str(png))
  assert png.exists() and png.stat().st_size > 0
