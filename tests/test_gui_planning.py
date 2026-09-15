"""Scan geometry and the Z-X-Y Euler convention the .pvsm format requires.

`PVSMParser` composes `MPS = Rz(tz) @ Rx(tx) @ Ry(ty)`, so exporting a plan
means inverting that specific order. Any order produces a valid rotation, which
is what makes getting it wrong silent, so the tests below check the MATRIX
round trip rather than the angles: at gimbal lock the angles legitimately
differ from whatever produced them while the matrix must not move at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from feelmri.Math import Rx, Ry, Rz
from feelmri.gui.model.planning import (AXIS_NAMES, FOVBox,
                                         euler_to_mps, format_triplet,
                                         is_rotation, mps_to_euler,
                                         parse_triplet)

PLANNING = __import__('pathlib').Path(__file__).resolve().parent.parent / 'examples' / 'planning'
PVSM_FILES = ['4dflow', 'abdomen', 'beating_heart', 'phase_contrast', 'water_and_fat']


def test_euler_to_mps_is_the_composition_pvsmparser_uses():
  """Spelled out against Rz @ Rx @ Ry, so a reordering here fails loudly."""
  tx, ty, tz = 0.3, -0.7, 1.1
  np.testing.assert_allclose(euler_to_mps(tx, ty, tz),
                             Rz(tz) @ Rx(tx) @ Ry(ty), atol=1e-15)
  # A different order is a different matrix: the test is not vacuous.
  assert not np.allclose(euler_to_mps(tx, ty, tz), Rx(tx) @ Ry(ty) @ Rz(tz))


def test_the_matrix_round_trips_for_random_rotations():
  """`MPS -> angles -> MPS` to machine precision over 2000 draws.

  Angles are drawn to cover the whole range including both poles, and the
  worst case over all draws is reported so a rare bad branch cannot hide in an
  average.
  """
  rng = np.random.default_rng(20260914)
  worst = 0.0
  for _ in range(2000):
    tx = rng.uniform(-np.pi / 2, np.pi / 2)
    ty, tz = rng.uniform(-np.pi, np.pi, size=2)
    m = euler_to_mps(tx, ty, tz)
    back = euler_to_mps(*mps_to_euler(m))
    worst = max(worst, float(np.abs(back - m).max()))
  assert worst < 1e-12, f'worst matrix round-trip error {worst:.3e}'


def test_the_angles_round_trip_away_from_the_poles():
  """Away from lock the angles themselves are unique, so pin them too."""
  rng = np.random.default_rng(7)
  for _ in range(500):
    tx = rng.uniform(-1.3, 1.3)          # |tx| well below pi/2
    ty, tz = rng.uniform(-3.0, 3.0, size=2)
    got = mps_to_euler(euler_to_mps(tx, ty, tz))
    np.testing.assert_allclose(got, (tx, ty, tz), atol=1e-9)


@pytest.mark.parametrize('sign', [+1, -1], ids=['tx=+pi/2', 'tx=-pi/2'])
@pytest.mark.parametrize('tz', [0.0, 0.9, -2.4])
@pytest.mark.parametrize('ty', [0.0, 0.6, -1.7])
def test_gimbal_lock_still_reproduces_the_matrix(sign, tz, ty):
  """At `tx = +-pi/2` only `tz +- ty` survives, so the angles may differ.

  The contract is that the MATRIX comes back exactly, and that the convention
  taken is `ty = 0`. Both are asserted: without the second, a decomposition
  that returned arbitrary angles summing correctly would pass.
  """
  m = euler_to_mps(sign * np.pi / 2, ty, tz)
  ax, ay, az = mps_to_euler(m)
  np.testing.assert_allclose(euler_to_mps(ax, ay, az), m, atol=1e-12)
  assert ay == 0.0, 'the documented ty = 0 convention was not taken'
  assert abs(abs(ax) - np.pi / 2) < 1e-9


def test_the_near_pole_error_is_bounded_by_the_lock_tolerance():
  """Two regimes, and the boundary between them is the whole point.

  Outside the lock tolerance the ordinary branch is exact to machine epsilon.
  Inside it the branch pins `ty = 0` and so discards terms of size `cos(tx)`,
  which bounds the error by the tolerance rather than by epsilon. Measured:
  1e-16 down to `cos(tx) = 1e-7`, then 2.0e-08 at 1e-8 and 2.0e-09 at 1e-9.

  Asserting a flat 1e-9 everywhere, as a first version of this test did, is
  simply wrong about what the function can promise.
  """
  from feelmri.gui.model.planning import _LOCK_TOL

  for eps in (1e-2, 1e-4, 1e-6, 1e-7):
    for sign in (+1, -1):
      m = euler_to_mps(sign * (np.pi / 2 - eps), 0.4, -1.2)
      err = float(np.abs(euler_to_mps(*mps_to_euler(m)) - m).max())
      assert err < 1e-12, f'outside lock (eps={eps:g}) error {err:.3e}'

  for eps in (1e-8, 1e-9, 1e-12, 0.0):
    for sign in (+1, -1):
      m = euler_to_mps(sign * (np.pi / 2 - eps), 0.4, -1.2)
      err = float(np.abs(euler_to_mps(*mps_to_euler(m)) - m).max())
      assert err <= 2.0 * _LOCK_TOL, f'inside lock (eps={eps:g}) error {err:.3e}'


def test_mps_to_euler_refuses_a_bad_shape():
  with pytest.raises(ValueError, match='3x3'):
    mps_to_euler(np.eye(4))


def test_is_rotation_rejects_reflections_and_scalings():
  assert is_rotation(euler_to_mps(0.2, 0.3, 0.4))
  assert not is_rotation(np.diag([1.0, 1.0, -1.0])), 'a reflection passed'
  assert not is_rotation(2.0 * np.eye(3)), 'a scaling passed'
  assert not is_rotation(np.eye(2))


@pytest.mark.parametrize('name', PVSM_FILES)
def test_a_shipped_pvsm_survives_the_decomposition(name):
  """Every plan that ships must decompose and recompose to the same matrix.

  These are real ParaView states, so they carry whatever orientations the
  author actually used, which is a better sample than anything synthetic.
  """
  from feelmri.Parameters import PVSMParser
  path = PLANNING / f'{name}.pvsm'
  if not path.exists():
    pytest.skip(f'{path.name} not present')

  parser = PVSMParser(str(path))
  assert is_rotation(parser.MPS), f'{name}: PVSMParser produced a non-rotation'

  box = FOVBox.from_mps(parser.FOV.m_as('m'), parser.LOC.m_as('m'), parser.MPS)
  np.testing.assert_allclose(box.mps, parser.MPS, atol=1e-12)
  np.testing.assert_allclose(box.fov, parser.FOV.m_as('m'), atol=1e-12)
  np.testing.assert_allclose(box.loc, parser.LOC.m_as('m'), atol=1e-12)


def test_to_imaging_matches_the_arithmetic_orient_applies():
  """`(x - LOC) @ MPS`, which for row vectors is `R^T (x - LOC)`.

  Written out here rather than imported so that a change to either side shows
  up as a disagreement instead of both moving together.
  """
  box = FOVBox(fov=[0.3, 0.2, 0.01], loc=[0.01, -0.02, 0.03],
               angles=[0.2, -0.5, 1.0])
  rng = np.random.default_rng(3)
  x = rng.uniform(-0.5, 0.5, size=(200, 3))
  np.testing.assert_allclose(box.to_imaging(x), (x - box.loc) @ box.mps,
                             atol=1e-15)
  # And it must actually invert.
  np.testing.assert_allclose(box.to_lab(box.to_imaging(x)), x, atol=1e-12)


def test_the_box_contains_its_own_centre_and_corners_but_not_beyond():
  box = FOVBox(fov=[0.3, 0.2, 0.05], loc=[0.1, 0.0, -0.05],
               angles=[0.4, 0.1, -0.9])
  corners = box.corners()
  assert corners.shape == (8, 3)
  assert box.contains(box.loc[None, :])[0]
  # Corners sit exactly on the boundary, so they count as inside.
  assert box.contains(corners).all()
  # Push each corner 10% further out along its own diagonal: all must leave.
  outside = box.loc + 1.1 * (corners - box.loc)
  assert not box.contains(outside).any()


def test_the_axis_triad_is_the_columns_of_mps_in_m_p_s_order():
  """Column 0 is readout, 1 phase, 2 slice. A transpose here would be silent.

  Driven on a rotation with no symmetry so that the three columns are mutually
  distinguishable; with the identity a transpose changes nothing.
  """
  box = FOVBox(fov=[0.3, 0.2, 0.01], loc=[0.0, 0.0, 0.0],
               angles=[0.3, -0.8, 1.4])
  arrows = box.axis_arrows()
  assert [a[2] for a in arrows] == list(AXIS_NAMES)
  for i, (origin, direction, _) in enumerate(arrows):
    np.testing.assert_allclose(origin, box.loc, atol=1e-15)
    unit = direction / np.linalg.norm(direction)
    np.testing.assert_allclose(unit, box.mps[:, i], atol=1e-12)
    # Default length is 60% of that axis's own extent.
    np.testing.assert_allclose(np.linalg.norm(direction), 0.6 * box.fov[i],
                               rtol=1e-12)
  # A transpose would swap M and S here, so confirm they differ.
  assert not np.allclose(box.mps[:, 0], box.mps[:, 2])


def test_fov_box_refuses_impossible_input():
  with pytest.raises(ValueError, match='non-negative'):
    FOVBox(fov=[-1, 1, 1], loc=[0, 0, 0], angles=[0, 0, 0])
  with pytest.raises(ValueError, match='finite'):
    FOVBox(fov=[1, 1, np.nan], loc=[0, 0, 0], angles=[0, 0, 0])
  with pytest.raises(ValueError, match='finite'):
    FOVBox(fov=[1, 1, 1], loc=[0, np.inf, 0], angles=[0, 0, 0])
  with pytest.raises(ValueError, match='not a proper rotation'):
    FOVBox.from_mps([1, 1, 1], [0, 0, 0], np.diag([1.0, 1.0, -1.0]))


# -- the entry fields' text, which is where a typo becomes a message --------

def test_a_triplet_round_trips_through_the_text_it_is_shown_as():
  """Six significant digits must not move the plan.

  The entries are the only place a user edits the numbers, so a format that
  loses precision silently changes the field of view every time the panel
  repaints.
  """
  values = np.array([0.3, 0.22, 0.008])
  np.testing.assert_allclose(parse_triplet(format_triplet(values)), values,
                             rtol=1e-9)


def test_commas_and_extra_spaces_are_accepted():
  np.testing.assert_allclose(parse_triplet('0.3, 0.22 , 0.008'),
                             [0.3, 0.22, 0.008], rtol=1e-12)


@pytest.mark.parametrize('text', ['0.3 0.22', '0.3 0.22 0.008 0.1', ''])
def test_the_wrong_count_is_refused_rather_than_padded(text):
  """Two numbers and a default is a plausible plan and the wrong one."""
  with pytest.raises(ValueError, match='three numbers'):
    parse_triplet(text, name='fov')


def test_a_non_number_names_the_field_it_came_from():
  with pytest.raises(ValueError, match='fov'):
    parse_triplet('0.3 wide 0.008', name='fov')


def test_formatting_does_not_print_floating_point_noise():
  """`str(0.1 + 0.2)` in an entry field is what this exists to prevent."""
  assert format_triplet([0.1 + 0.2, 1 / 3, 2.0]) == '0.3 0.333333 2'
