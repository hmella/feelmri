"""Unit tests for :mod:`feelmri.Math`: Fourier helpers and rotations.

Pure-Python; no C++ extensions, no MPI.

Orthogonality and ``det = 1`` used to be swept over 3 axes x 5 angles. That is
one identity run fifteen times, and one that cannot discriminate the error
worth catching, since a transposed or sign-flipped rotation matrix is still
orthogonal with unit determinant. Only the handedness check can, and it used to
pin ``Rz`` alone, so ``Rx`` and ``Ry`` were untested. They are pinned below.
"""
from __future__ import annotations

import numpy as np
import pytest

from feelmri import Rx, Ry, Rz, itok, ktoi


# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('R, axis', [(Rx, 0), (Ry, 1), (Rz, 2)],
                         ids=['Rx', 'Ry', 'Rz'])
def test_rotation_is_a_rotation_about_its_axis(R, axis):
  """Orthogonal, unit determinant, and the named axis is invariant.

  ``theta = 0`` is excluded deliberately: it returns the identity, which
  satisfies all three for any implementation.
  """
  axis_vec = np.zeros(3)
  axis_vec[axis] = 1.0
  for theta in (0.3, 1.7, -0.9, np.pi):
    M = R(theta)
    assert M.shape == (3, 3)
    np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(M), 1.0, atol=1e-12)
    np.testing.assert_allclose(M @ axis_vec, axis_vec, atol=1e-12)


def test_rotations_are_right_handed():
  """The sign convention, which orthogonality cannot see.

  A quarter turn sends each axis to the next one cyclically:
  ``Rx: y -> z``, ``Ry: z -> x``, ``Rz: x -> y``. Transposing any of the
  three, or negating its angle, leaves it orthogonal with unit determinant
  and breaks only this.
  """
  x, y, z = np.eye(3)
  quarter = np.pi / 2
  np.testing.assert_allclose(Rx(quarter) @ y, z, atol=1e-12)
  np.testing.assert_allclose(Ry(quarter) @ z, x, atol=1e-12)
  np.testing.assert_allclose(Rz(quarter) @ x, y, atol=1e-12)


# ---------------------------------------------------------------------------
# Fourier roundtrip
# ---------------------------------------------------------------------------

def test_itok_ktoi_roundtrip_3d():
  rng = np.random.default_rng(0)
  data = (rng.standard_normal((8, 6, 4)) +
          1j * rng.standard_normal((8, 6, 4)))
  np.testing.assert_allclose(ktoi(itok(data)), data, atol=1e-10)
  np.testing.assert_allclose(itok(ktoi(data)), data, atol=1e-10)


def test_itok_axes_subset():
  """Specifying axes=[0, 1] must FFT only those axes; the third
  dimension passes through unchanged."""
  rng = np.random.default_rng(1)
  data = rng.standard_normal((4, 4, 3)) + 0j
  out = itok(data, axes=[0, 1])
  # Each slice along the third axis is its own 2-D FFT.
  for s in range(3):
    np.testing.assert_allclose(
      out[..., s],
      itok(data[..., s], axes=[0, 1]),
      atol=1e-12,
    )
