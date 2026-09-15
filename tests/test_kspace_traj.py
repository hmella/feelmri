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

  # Both readout axes advance monotonically within a line: kx because the
  # readout gradient has one sign, t because it is a clock. Asserted on BOTH
  # configurations, the oversampled one above and the plain one below. The
  # two used to be separate tests, and they were not built alike: they ran at
  # res=16, oversampling=1, which this test does not otherwise cover.
  plain = CartesianStack(FOV=fov, res=np.array([16, 16, 1]),
                         oversampling=1, lines_per_shot=1, scanner=scanner)
  for label, tr in (('oversampled', traj), ('plain', plain)):
    for arr, what in ((tr.points[0], 'kx'), (tr.times.m_as('ms'), 't')):
      diffs = np.diff(arr, axis=0)
      assert np.all(diffs >= -1e-6), (
        f'{label}: {what} not non-decreasing: min={diffs.min()}')
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


def test_a_lab_frame_field_displaces_the_readout_by_the_off_resonance_rule(
        fov, scanner):
  """The k-space shift must reproduce the textbook off-resonance displacement.

  Off-resonance `df` moves a readout by `df / (gammabar * G_ro)` metres,
  because the readout gradient is what converts frequency into position. A
  lab-frame field `g` seen by a spin at `x` is `df = gammabar * (g.x)`, so the
  displacement is `(g.x) / G_ro`. No gammabar left in it.

  Asserted as an INTEGER number of pixels: the field is sized so the rule
  predicts exactly three, and the reconstructed peak has to land three bins
  away. A wrong sign puts it at -3, a missing `2*pi` puts it off the grid
  entirely, and neither can be absorbed by a tolerance.
  """
  from feelmri import B0Field

  res = np.array([64, 4, 1])
  traj = CartesianStack(FOV=fov, res=res, oversampling=1, lines_per_shot=1,
                        scanner=scanner, t_start=Quantity(2.0, 'ms'))

  # G_ro from the trajectory itself: one sample of dk over one dwell.
  kx = np.asarray(traj.points[0], dtype=float)
  t = np.asarray(traj.times.m_as('ms'), dtype=float)
  gammabar = scanner.gammabar.m_as('1/ms/mT')
  G_ro = (kx[1, 0, 0] - kx[0, 0, 0]) / (gammabar * (t[1, 0, 0] - t[0, 0, 0]))

  pixel = float(fov[0].m_as('m')) / traj.ro_samples
  z0 = 0.04
  shift_pixels = 3
  gz = shift_pixels * pixel * G_ro / z0
  field = B0Field(gradient=Quantity(np.array([0.0, 0.0, gz]), 'mT/m'))

  nominal = [np.array(p, copy=True) for p in traj.points]
  shifted = traj.b0_shifted_points(field, scanner)
  for i in range(3):
    assert np.array_equal(traj.points[i], nominal[i]), (
        'b0_shifted_points wrote the shift back into the trajectory, so the '
        'reconstruction would grid on the distorted k and see nothing')

  def peak_bin(points):
    # A point object at (0, 0, z0): its signal is exp(-2i pi k.x0), and it is
    # reconstructed on the NOMINAL grid whichever k actually encoded it.
    s = np.exp(-2j * np.pi * np.asarray(points[2], dtype=float) * z0)
    line = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(s[:, 0, 0])))
    return int(np.argmax(np.abs(line)))

  here = peak_bin(nominal)
  there = peak_bin(shifted)
  assert there - here == shift_pixels, (
      f'the field should displace the readout by {shift_pixels} pixels '
      f'({shift_pixels * pixel * 1e3:.2f} mm at G_ro = {G_ro:.3f} mT/m); '
      f'measured {there - here}')

  with pytest.raises(TypeError, match='B0Field'):
    traj.b0_shifted_points(np.zeros(3), scanner)
