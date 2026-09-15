"""Unit tests for :mod:`feelmri.Filters`: Tukey and Riesz apodisation
windows used by Cartesian reconstruction. Pure-Python; sub-second.

These were a grid over factory, size, width and lift, all asserting shape,
symmetry and ``lift <= h <= 1``. None of the three
can fail: ``Tukey`` is ``tukey(size, alpha=1-width) * (1-lift) + lift``, an
affine map of a scipy window, so the bounds are algebraic in ``lift`` whatever
scipy returns, and both windows are mirrored by construction. Some cells were
vacuous outright. ``Riesz(16, width=0.9)`` has ``s20 = 1`` and comes back all
ones, on which symmetry and bounds hold trivially.

What the grid never asserted is that ``width`` and ``lift`` do anything at all:
a Tukey that ignored ``width`` would have passed all 72. That is the third test
below.
"""
from __future__ import annotations

import numpy as np
import pytest

from feelmri import Riesz, Tukey

FACTORIES = [Tukey, Riesz]


def _ids(fs):
  return [f.__name__ for f in fs]


@pytest.mark.parametrize('factory', FACTORIES, ids=_ids(FACTORIES))
def test_window_is_symmetric_bounded_and_peaks_at_one(factory):
  """Length, mirror symmetry, the ``[lift, 1]`` envelope and a unit centre.

  Both parities are covered: an odd length has a true centre sample, an even
  one does not, and only the odd case pins the peak exactly.
  """
  for size in (64, 65):
    h = factory(size, width=0.6, lift=0.3)
    assert h.shape == (size,)
    np.testing.assert_allclose(h, h[::-1], atol=1e-12)
    assert h.max() <= 1.0 + 1e-12
    assert h.min() >= 0.3 - 1e-12
    assert h[size // 2] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('factory', FACTORIES, ids=_ids(FACTORIES))
def test_width_and_lift_actually_shape_the_window(factory):
  """The two parameters have to reach the output.

  ``lift`` is the floor exactly, not merely a lower bound, and a wider
  pass-band must taper strictly fewer samples. Measured at size 64,
  ``width = 0.3 / 0.6 / 0.9`` tapers 46 / 26 / 8 samples for Tukey and
  42 / 24 / 4 for Riesz.
  """
  for lift in (0.0, 0.3, 0.7):
    h = factory(64, width=0.6, lift=lift)
    assert h.min() == pytest.approx(lift, abs=1e-12), 'lift is not the floor'

  tapered = [int((factory(64, width=w, lift=0.0) < 1.0 - 1e-12).sum())
             for w in (0.3, 0.6, 0.9)]
  assert tapered[0] > tapered[1] > tapered[2] > 0, (
    f'width does not set the taper: {tapered} samples below unity '
    f'at width 0.3 / 0.6 / 0.9')


def test_riesz_decays_monotonically_from_centre():
  """Riesz is by construction a monotone-from-centre window for any
  positive lift, so the left half must be non-decreasing."""
  h = Riesz(128, width=0.8, lift=0.2)
  left = h[: len(h) // 2]
  diffs = np.diff(left)
  assert np.all(diffs >= -1e-9), f'left half not monotone: {diffs.min()}'
