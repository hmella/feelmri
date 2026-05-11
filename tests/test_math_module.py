"""Unit tests for :mod:`feelmri.Math` — Fourier helpers and rotations.

Pure-Python; no C++ extensions, no MPI."""
from __future__ import annotations

import numpy as np
import pytest

from feelmri import Rx, Ry, Rz, itok, ktoi


# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('R, axis', [(Rx, 0), (Ry, 1), (Rz, 2)])
@pytest.mark.parametrize('theta', [0.0, 0.5, np.pi / 2, np.pi, 1.234])
def test_rotation_matrix_is_orthogonal_and_unit_det(R, axis, theta):
  M = R(theta)
  assert M.shape == (3, 3)
  np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)
  np.testing.assert_allclose(np.linalg.det(M), 1.0, atol=1e-12)


@pytest.mark.parametrize('R, axis', [(Rx, 0), (Ry, 1), (Rz, 2)])
def test_rotation_axis_is_fixed(R, axis):
  """R(theta) must leave the rotation axis invariant for any theta."""
  axis_vec = np.zeros(3)
  axis_vec[axis] = 1.0
  for theta in (0.3, 1.7, -0.9):
    np.testing.assert_allclose(R(theta) @ axis_vec, axis_vec, atol=1e-12)


def test_rotation_composes_in_expected_direction():
  """Rz(pi/2) maps x_hat -> y_hat."""
  x_hat = np.array([1.0, 0.0, 0.0])
  np.testing.assert_allclose(Rz(np.pi / 2) @ x_hat,
                             np.array([0.0, 1.0, 0.0]), atol=1e-12)


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
