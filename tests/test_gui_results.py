"""A finished run's k-space, and what it reconstructs to.

The risk here is not a crash. It is a **plausible wrong image**: a
reconstruction normalised against the wrong field of view produces smooth
interference texture that looks like an image and contains no object, and a
panel will display it without complaint. So these assert the derivation, and
one of them asserts the failure directly.

The drawing is module-level functions taking an axis, so it is checked against
a bare Agg `Figure` with no Tk and no display.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
from matplotlib.figure import Figure                              # noqa: E402

from feelmri.gui.model import results as R                        # noqa: E402
from feelmri.gui.view.image_panel import (draw_image, draw_kspace,  # noqa: E402
                                          draw_results,
                                          draw_trajectory, make_axes)


def cartesian_result(n=16, fov=(0.32, 0.24), phantom_radius=0.35):
  """A Cartesian k-space of a disc, built from its analytic transform.

  Synthetic on purpose: the FOV it encodes is known exactly, so a
  reconstruction can be checked against the object rather than against
  another reconstruction.
  """
  dkx, dky = 1.0 / fov[0], 1.0 / fov[1]
  kx = (np.arange(n) - n // 2) * dkx
  ky = (np.arange(n) - n // 2) * dky
  gx, gy = np.meshgrid(kx, ky, indexing='ij')
  kr = np.hypot(gx, gy)

  # The 2-D transform of a uniform disc is a jinc; the limit at k = 0 is its
  # area, which `j1(x)/x` cannot give directly.
  from scipy.special import j1
  radius = phantom_radius * fov[0]
  argument = 2 * np.pi * kr * radius
  signal = np.where(argument < 1e-12, np.pi * radius ** 2,
                    radius * j1(argument) / np.maximum(argument, 1e-12))

  points = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
  return {'kspace': signal.ravel().astype(complex).reshape(-1, 1, 1, 1),
          'times': np.linspace(0.0, 5.0, gx.size),
          'points': points,
          'plan_fov': np.array([fov[0], fov[1], 0.008])}


# -- the derivation ---------------------------------------------------------

def test_the_encoded_fov_comes_from_the_sample_spacing():
  """`1/dk`, which is a property of the trajectory and of nothing else."""
  result = cartesian_result(n=16, fov=(0.32, 0.24))
  fx, fy, fz = R.encoded_fov(result)
  assert fx == pytest.approx(0.32, rel=1e-6)
  assert fy == pytest.approx(0.24, rel=1e-6)
  assert fz == 0.0, 'a single-slice axis has no encoded extent'


def test_the_encoded_fov_is_not_the_plans_fov():
  """The two are different quantities that look alike, which is exactly how
  they get confused -- and confusing them is what produced a wrong image."""
  result = cartesian_result(n=16, fov=(0.32, 0.24))
  result['plan_fov'] = np.array([0.30, 0.22, 0.004])
  encoded = R.encoded_fov(result)
  assert encoded[0] != pytest.approx(result['plan_fov'][0])
  assert encoded[1] != pytest.approx(result['plan_fov'][1])


def test_the_default_matrix_is_what_the_trajectory_supports():
  """`2 * k_max / dk`, so the default reconstruction is at the resolution
  that was acquired rather than a round number."""
  result = cartesian_result(n=16, fov=(0.32, 0.24))
  matrix = R.default_matrix(result)
  assert matrix[0] == 16 and matrix[1] == 16
  assert matrix[2] == 1


def test_a_ramp_sampled_axis_reads_its_spacing_from_the_median():
  """The minimum gap would read the FOV off the ramp samples, which are
  closer together than the phase-encode step and describe nothing."""
  regular = (np.arange(20) - 10) * 4.0
  crowded = np.array([41.0, 41.5, 42.0])          # a few very close samples
  k = np.concatenate([regular, crowded])
  assert R._spacing(k) == pytest.approx(4.0)


# -- the reconstruction -----------------------------------------------------

def test_the_encoded_fov_reconstructs_the_object_and_the_plans_does_not():
  """The failure this module exists to prevent, asserted directly.

  A disc, reconstructed twice from the same k-space: once normalised with the
  FOV the trajectory encodes, once with the plan's. The first shows the disc;
  the second is interference texture with no object. Both look like images,
  which is why the check is on the CONTRAST between centre and corner rather
  than on whether anything was produced.
  """
  # The wrong FOV is passed explicitly, at twice the encoded one, which is
  # what the first version of `reconstruct` computed. On the real pair the two
  # were 0.32 vs 0.30 -- close enough to look like the same number.
  result = cartesian_result(n=32, fov=(0.32, 0.24))

  right = np.abs(np.asarray(R.reconstruct(result, matrix=(32, 32, 1))).squeeze())
  wrong = np.abs(np.asarray(R.reconstruct(result, matrix=(32, 32, 1),
                                          fov=(0.64, 0.48, 0.008))).squeeze())

  def contrast(image):
    centre = image[image.shape[0] // 2, image.shape[1] // 2]
    corner = max(float(image[1, 1]), 1e-30)
    return float(centre) / corner

  assert contrast(right) > 5.0, 'the disc is not at the centre'
  assert contrast(right) > 3 * contrast(wrong), (
    'the wrong field of view reconstructed just as well, so this proves '
    'nothing')


def test_reconstruction_needs_no_arguments():
  """Both the FOV and the matrix come from the data, so a panel can show
  something before the user has decided anything."""
  result = cartesian_result(n=16)
  image = np.asarray(R.reconstruct(result))
  assert image.shape[:2] == (16, 16)
  assert np.isfinite(image).all()


def test_a_result_without_a_trajectory_is_refused_by_name():
  """An older run saved no points. It can still be plotted, and saying so is
  better than reconstructing against a trajectory that is not there."""
  with pytest.raises(ValueError, match='points'):
    R.reconstruct({'kspace': np.zeros(4, dtype=complex)})


def test_a_trajectory_that_does_not_match_its_samples_is_refused():
  """A mismatched pair reconstructs to a plausible wrong image, so the count
  is checked rather than trusted."""
  result = cartesian_result(n=8)
  result['points'] = result['points'][:-3]
  with pytest.raises(ValueError, match='trajectory points'):
    R.reconstruct(result)


@pytest.mark.parametrize('matrix', [(0, 8, 1), (8, 8), (8, 8, 0)])
def test_a_nonsense_matrix_is_refused(matrix):
  with pytest.raises(ValueError, match='matrix'):
    R.reconstruct(cartesian_result(n=8), matrix=matrix)


# -- what the panel shows ---------------------------------------------------

def test_the_summary_shows_BOTH_fields_of_view():
  """They are different quantities and look alike, so showing one invites the
  confusion this module documents. The panel shows both, labelled."""
  rows = dict(R.describe(cartesian_result(n=16, fov=(0.32, 0.24))))
  assert 'FOV encoded (m)' in rows
  assert 'FOV of the plan' in rows
  assert rows['FOV encoded (m)'].startswith('0.32')
  assert rows['FOV of the plan'].startswith('0.32')


def test_every_pane_draws_without_a_display():
  result = cartesian_result(n=16)
  image = R.reconstruct(result)
  figure = Figure()
  axes = make_axes(figure)
  draw_results(axes, result, image)
  assert axes[0].images and axes[1].images       # magnitude and phase
  assert axes[2].collections                     # the trajectory scatter
  assert axes[3].lines                           # |k| against sample


def test_phase_is_drawn_on_a_fixed_range():
  """An autoscaled phase image invents contrast from whatever range the data
  happens to span, which reads as structure that is not there."""
  figure = Figure()
  axes = make_axes(figure)
  image = np.exp(1j * np.linspace(0.0, 0.2, 64)).reshape(8, 8)
  draw_image(axes[1], image, 'phase')
  assert axes[1].images[0].get_clim() == (-np.pi, np.pi)


def test_the_panes_are_blank_rather_than_broken_without_a_result():
  figure = Figure()
  axes = make_axes(figure)
  draw_results(axes, None, None)
  assert not any(ax.images or ax.lines or ax.collections for ax in axes)
