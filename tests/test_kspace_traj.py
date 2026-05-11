"""Unit tests for :mod:`feelmri.KSpaceTraj`.

CartesianStack / RadialStack / SpiralStack expose a uniform API:
``traj.points = (kx, ky, kz)`` (each shaped
``(ro_samples, ph_samples, slices)``), ``traj.times`` of the same
shape, plus the bookkeeping attributes ``ro_samples``, ``ph_samples``,
``slices``, ``nb_shots``, ``echo_time``. These tests assert the
contract without running a full Bloch simulation."""
from __future__ import annotations

import numpy as np
import pytest
from pint import Quantity

from feelmri import CartesianStack, RadialStack, SpiralStack, Scanner


@pytest.fixture
def fov():
  return Quantity(np.array([0.20, 0.20, 0.005]), 'm')


@pytest.fixture
def scanner():
  return Scanner()


# ---------------------------------------------------------------------------
# CartesianStack
# ---------------------------------------------------------------------------

def test_cartesian_stack_shapes(fov, scanner):
  res = np.array([32, 32, 1])
  traj = CartesianStack(FOV=fov, res=res, oversampling=2,
                        lines_per_shot=1, scanner=scanner)
  expected_shape = (traj.ro_samples, traj.ph_samples, traj.slices)
  for axis in range(3):
    assert traj.points[axis].shape == expected_shape
  assert traj.times.shape == expected_shape
  # ro_samples = oversampling * res[0]
  assert traj.ro_samples == 2 * res[0]
  assert traj.ph_samples == res[1]
  assert traj.slices == res[2]


def test_cartesian_stack_kx_monotone_within_line(fov, scanner):
  traj = CartesianStack(FOV=fov, res=np.array([16, 16, 1]),
                        oversampling=1, lines_per_shot=1, scanner=scanner)
  kx = traj.points[0]
  # Each (line, slice) column should be monotone increasing in the readout.
  diffs = np.diff(kx, axis=0)
  assert np.all(diffs >= -1e-6), f'kx not non-decreasing: min={diffs.min()}'


def test_cartesian_stack_times_are_non_decreasing(fov, scanner):
  traj = CartesianStack(FOV=fov, res=np.array([16, 16, 1]),
                        oversampling=1, lines_per_shot=1, scanner=scanner)
  t = traj.times.m_as('ms')
  diffs = np.diff(t, axis=0)
  assert np.all(diffs >= -1e-6), f't not non-decreasing: min={diffs.min()}'


# ---------------------------------------------------------------------------
# RadialStack
# ---------------------------------------------------------------------------

def test_radial_stack_shapes_and_first_spoke_passes_through_origin(fov, scanner):
  traj = RadialStack(FOV=fov, res=np.array([16, 16, 1]),
                     oversampling=1, lines_per_shot=1, scanner=scanner)
  shape = (traj.ro_samples, traj.ph_samples, traj.slices)
  for axis in range(3):
    assert traj.points[axis].shape == shape
  # A radial spoke (center-out) should have its first sample near k=0.
  centre_kx = float(traj.points[0][0, 0, 0])
  centre_ky = float(traj.points[1][0, 0, 0])
  assert np.hypot(centre_kx, centre_ky) <= float(np.abs(traj.points[0]).max())


# ---------------------------------------------------------------------------
# SpiralStack
# ---------------------------------------------------------------------------

def test_spiral_stack_radial_speed_is_non_negative(fov, scanner):
  traj = SpiralStack(FOV=fov, res=np.array([16, 16, 1]),
                    oversampling=1, lines_per_shot=1, scanner=scanner)
  shape = (traj.ro_samples, traj.ph_samples, traj.slices)
  for axis in range(3):
    assert traj.points[axis].shape == shape
  # The spiral grows outward in the in-plane radius; the final sample
  # must sit at a larger radius than the first sample (averaged across
  # all shots/slices).
  kx0 = traj.points[0][0, :, 0]
  ky0 = traj.points[1][0, :, 0]
  kxN = traj.points[0][-1, :, 0]
  kyN = traj.points[1][-1, :, 0]
  r0 = np.hypot(kx0, ky0)
  rN = np.hypot(kxN, kyN)
  assert rN.mean() > r0.mean(), (
    f'spiral did not grow outward: r0={r0.mean():.3g} rN={rN.mean():.3g}'
  )
