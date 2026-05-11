"""Unit tests for :mod:`feelmri.Filters` — Tukey and Riesz apodisation
windows used by Cartesian reconstruction. Pure-Python; sub-second."""
from __future__ import annotations

import numpy as np
import pytest

from feelmri import Riesz, Tukey


@pytest.mark.parametrize('size', [16, 32, 64, 65])
@pytest.mark.parametrize('width', [0.4, 0.6, 0.9])
@pytest.mark.parametrize('lift', [0.0, 0.3, 0.7])
@pytest.mark.parametrize('factory', [Tukey, Riesz])
def test_apodization_filter_is_bounded_and_symmetric(factory, size, width, lift):
  """Tukey and Riesz windows must be symmetric around the centre and
  bounded within [lift, 1.0] (up to a small FP slop)."""
  h = factory(size, width=width, lift=lift)
  assert h.shape == (size,)
  # Symmetry.
  np.testing.assert_allclose(h, h[::-1], atol=1e-12)
  # Bounded.
  assert h.max() <= 1.0 + 1e-12
  assert h.min() >= lift - 1e-12


@pytest.mark.parametrize('factory', [Tukey, Riesz])
def test_apodization_centre_attains_unity(factory):
  """The centre sample of either window must reach 1.0 (the
  pass-band peak)."""
  h = factory(64, width=0.6, lift=0.3)
  centre = h[len(h) // 2]
  assert centre == pytest.approx(1.0, abs=1e-6)


def test_riesz_decays_monotonically_from_centre():
  """Riesz is by construction a monotone-from-centre window for any
  positive lift, so the left half must be non-decreasing."""
  h = Riesz(128, width=0.8, lift=0.2)
  left = h[: len(h) // 2]
  diffs = np.diff(left)
  assert np.all(diffs >= -1e-9), f'left half not monotone: {diffs.min()}'
