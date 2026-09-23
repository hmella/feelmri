"""The two directional encoders and the two analytic density compensations.

`MRImaging.VelocityEncoding` and `PositionEncoding` have 34 references across
the library and the examples and had none in `tests/`; `Recon`'s `dcf_*` trio
was named by no test either. Both are reached only through the `slow`
subprocess example runner, where a failure surfaces as "an example exited
non-zero" rather than as a named assertion.

Each is checked against its own closed form, which is what makes a sign or a
factor visible rather than merely different.
"""
from __future__ import annotations

import numpy as np
import pytest
from pint import Quantity as Q_

from feelmri.MRImaging import PositionEncoding, VelocityEncoding
from feelmri.Recon import dcf_local_speed_readout, dcf_radial_stack


def test_the_two_directional_encoders_match_their_closed_forms():
  """`phi_v = pi (v . d) / VENC + dphi` and `phi_x = ke (u . d)`.

  Driven on NON-orthogonal, non-unit directions and a velocity with a
  component along each: with axis-aligned unit directions the dot product
  collapses to one component, so a mixed-up direction index changes nothing.
  """
  dirs = np.array([[2.0, 0.0, 0.0],
                   [0.0, 3.0, 4.0],
                   [1.0, 1.0, 0.0]], dtype=np.float64)
  v = np.array([[0.30, -0.20, 0.10],
                [-0.05, 0.40, -0.25]], dtype=np.float64)

  # VENC is per direction, so a single wrong index rescales one column.
  venc = Q_(np.array([0.6, 0.9, 1.2]), 'm/s')
  enc = VelocityEncoding(venc, dirs.copy(), dtype=np.float64,
                         normalize_dirs=True)
  unit = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
  # Negative: encode() returns the phase the magnetization acquires under
  # exp(-i gamma B t), not the bipolar's first-moment sensitivity.
  want = -np.pi * (v @ unit.T) / np.array([0.6, 0.9, 1.2]) + 0.25
  assert np.abs(enc.encode(v, delta_phi=0.25) - want).max() < 1e-12

  # Without normalisation the raw direction length is part of the encoding.
  raw = VelocityEncoding(venc, dirs.copy(), dtype=np.float64)
  want_raw = -np.pi * (v @ dirs.T) / np.array([0.6, 0.9, 1.2])
  assert np.abs(raw.encode(v) - want_raw).max() < 1e-12

  # PositionEncoding always normalises, and multiplies rather than divides.
  ke = np.array([80.0, 120.0, 160.0])
  pos = PositionEncoding(ke, dirs.copy(), dtype=np.float64)
  u = v * 1e-3
  assert np.abs(pos.encode(u) - (u @ unit.T) * ke).max() < 1e-12

  # A scalar is broadcast over the directions; a wrong-length vector is not.
  assert VelocityEncoding(Q_(0.5, 'm/s'), dirs.copy()).VENC.shape == (3,)
  with pytest.raises(ValueError, match='same length'):
    VelocityEncoding(Q_(np.array([0.5, 0.6]), 'm/s'), dirs.copy())
  with pytest.raises(ValueError, match='same length'):
    PositionEncoding(np.array([80.0, 120.0]), dirs.copy())


def test_the_analytic_density_compensations_follow_their_own_rule():
  """The radial one is the polar Jacobian; the speed one is the arc length.

  Both normalise to a mean of one, so a constant factor is invisible, the
  SHAPE along the readout is the content, and each is compared against the
  rule its docstring states.
  """
  R, L, S = 9, 4, 2
  # A stack of radial spokes, each swept at a different angle and speed.
  r = np.linspace(-1.0, 1.0, R)[:, None, None]
  ang = np.linspace(0.0, np.pi, L, endpoint=False)[None, :, None]
  scale = np.array([1.0, 2.0])[None, None, :]
  kx = (r * np.cos(ang) * scale).astype(np.float64)
  ky = (r * np.sin(ang) * scale).astype(np.float64)

  w = dcf_radial_stack(kx, ky).reshape(R, L, S)
  radius = np.sqrt(kx**2 + ky**2)
  # The `eps` guarding the division is part of the rule, not slack: at this
  # scale it is a 2e-6 relative shift, which a 2e-6 tolerance would not
  # absorb on the outer samples.
  want = radius / (radius.mean(axis=(0, 1), keepdims=True) + 1e-6)
  assert np.abs(w - want).max() < 1e-6, 'the radial DCF is not the ramp'
  # Normalised PER SLICE, so the two slices agree despite the 2x scale.
  assert abs(w[..., 0].mean() - w[..., 1].mean()) < 1e-5

  # Without per-slice normalisation the outer slice must carry more weight.
  g = dcf_radial_stack(kx, ky, per_slice_normalize=False).reshape(R, L, S)
  assert g[..., 1].mean() > 1.9 * g[..., 0].mean()

  # The speed rule: a uniformly sampled spoke gives a flat weight, and the
  # end samples are a one-sided difference rather than a centred one.
  sp = dcf_local_speed_readout(kx, ky, None).reshape(R, L, S)
  assert np.abs(sp - 1.0).max() < 1e-5, 'a uniform sweep is not flat'

  # A spoke whose second half is sampled five times as densely must weight
  # that half five times less. That is the whole point of the compensation.
  # The step is 0.25 over the first five samples and 0.05 over the last four.
  t = np.concatenate([np.arange(5) * 0.25, 1.0 + np.arange(1, 5) * 0.05])
  kx2 = np.tile(t[:, None, None], (1, L, S))
  ky2 = np.zeros_like(kx2)
  sp2 = dcf_local_speed_readout(kx2, ky2, None).reshape(R, L, S)[:, 0, 0]
  assert sp2[1] == pytest.approx(5.0 * sp2[-2], rel=1e-5), (
    f'the 5x denser half should weigh 5x less; got {sp2[1]:.4f} against '
    f'{sp2[-2]:.4f}')

  # kz participates when it is given: adding a through-plane ramp lengthens
  # every step, and the normalisation then divides it back out.
  kz = np.tile(np.linspace(0.0, 1.0, R)[:, None, None], (1, L, S))
  assert np.abs(dcf_local_speed_readout(kx, ky, kz)
                - dcf_local_speed_readout(kx, ky, None)).max() < 1e-5
