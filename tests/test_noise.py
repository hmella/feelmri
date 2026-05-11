"""Unit tests for :func:`feelmri.add_cpx_noise`.

The function uses NumPy's legacy global random state, so determinism
is exercised by seeding it explicitly before the call."""
from __future__ import annotations

import numpy as np

from feelmri import add_cpx_noise


def test_add_cpx_noise_preserves_dtype_and_shape():
  K = (np.full((16, 16), 1 + 0.5j, dtype=np.complex64))[..., np.newaxis]
  np.random.seed(0)
  Kn = add_cpx_noise(K, relative_std=0.01)
  assert Kn.shape == K.shape
  assert np.iscomplexobj(Kn)


def test_add_cpx_noise_mean_close_to_input_at_small_relative_std():
  """At small std the empirical mean must stay close to the input."""
  K = np.ones((64, 64, 1), dtype=np.complex64) * (1.0 + 0.0j)
  np.random.seed(0)
  Kn = add_cpx_noise(K, relative_std=0.01)
  mean_offset = abs(complex(Kn.mean()) - 1.0)
  assert mean_offset < 0.05, f'mean offset {mean_offset} too large'


def test_add_cpx_noise_is_reproducible_under_explicit_seed():
  """Two calls preceded by the same np.random.seed must produce
  identical noisy outputs."""
  K = np.ones((32, 32, 1), dtype=np.complex64) * 0.5
  np.random.seed(42)
  Kn1 = add_cpx_noise(K.copy(), relative_std=0.1)
  np.random.seed(42)
  Kn2 = add_cpx_noise(K.copy(), relative_std=0.1)
  np.testing.assert_array_equal(Kn1, Kn2)


def test_add_cpx_noise_recover_returns_noise_field():
  K = np.zeros((8, 8, 1), dtype=np.complex64)
  np.random.seed(7)
  Kn, noise = add_cpx_noise(K, std=0.1, recover_noise=True)
  # Signal was zero, so the noisy output equals the noise field.
  np.testing.assert_array_equal(Kn, noise)
  assert np.iscomplexobj(noise)
