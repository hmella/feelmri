"""Tests for ``feelmri.PulseqAdapter.as_signal_inputs``.

The helper centralises the rank-3 / float32 / C-contiguous contract
that ``Phantom.mri_signal`` (and the underlying C++
``SignalAssembler``) imposes on its ``kspace_points`` / ``kspace_times``
arguments. These tests cover the helper in isolation so
``mri_signal``-related regressions are caught without the C++
extension."""
from __future__ import annotations

import numpy as np
import pytest

from feelmri.PulseqAdapter import as_signal_inputs


def test_as_signal_inputs_default_shape_is_N_1_1():
  n = 64
  traj = {
    'kx':    np.linspace(0.0, 1.0, n),
    'ky':    np.linspace(-1.0, 1.0, n),
    'kz':    np.zeros(n),
    'times': np.arange(n, dtype=float),
  }

  pts, t = as_signal_inputs(traj)

  assert isinstance(pts, tuple) and len(pts) == 3
  for p, key in zip(pts, ('kx', 'ky', 'kz')):
    assert p.shape == (n, 1, 1)
    assert p.dtype == np.float32
    assert p.flags['C_CONTIGUOUS']
    np.testing.assert_array_equal(
      p.reshape(-1), traj[key].astype(np.float32)
    )
  assert t.shape == (n, 1, 1)
  assert t.dtype == np.float32
  assert t.flags['C_CONTIGUOUS']


def test_as_signal_inputs_explicit_shape_round_trips():
  shape = (4, 3, 2)
  n = int(np.prod(shape))
  traj = {k: np.arange(n, dtype=float) for k in ('kx', 'ky', 'kz', 'times')}

  pts, t = as_signal_inputs(traj, shape=shape)

  for p in pts:
    assert p.shape == shape
    assert p.dtype == np.float32
    assert p.flags['C_CONTIGUOUS']
  assert t.shape == shape
  np.testing.assert_array_equal(
    t.reshape(-1), np.arange(n, dtype=np.float32)
  )


def test_as_signal_inputs_mismatched_shape_raises():
  traj = {k: np.zeros(7) for k in ('kx', 'ky', 'kz', 'times')}
  with pytest.raises(ValueError, match='does not match'):
    as_signal_inputs(traj, shape=(3, 2, 1))


def test_as_signal_inputs_handles_non_contiguous_input():
  """Passing a strided / non-contiguous source ndarray must still
  yield C-contiguous float32 output (the helper copies on cast)."""
  buf = np.zeros((6, 2), dtype=np.float64)
  buf[:, 0] = np.arange(6.0)
  view = buf[:, 0]  # column-stride view, not C-contiguous for our purposes
  assert not view.flags['C_CONTIGUOUS']
  traj = {k: view for k in ('kx', 'ky', 'kz', 'times')}

  pts, t = as_signal_inputs(traj)

  for p in pts:
    assert p.flags['C_CONTIGUOUS']
    assert p.dtype == np.float32
  assert t.flags['C_CONTIGUOUS']
  np.testing.assert_array_equal(
    t.reshape(-1), np.arange(6, dtype=np.float32)
  )
