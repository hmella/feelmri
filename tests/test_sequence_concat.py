"""Targeted tests for the nested-Sequence path in ``Sequence.add_block``
and the delta-based behaviour of ``Gradient.change_time`` /
``RF.change_time``.

These cover the timing-mismatch regression introduced when
``add_block`` learned to accept a ``Sequence``: the per-block shift
used the wrong reference, parent-state was recomputed inside the loop
(so the shift drifted), and ``change_time`` left the interpolator
stale for user-supplied gradients (the Pulseq path)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from pint import Quantity

from feelmri.Bloch import Sequence, SequenceBlock
from feelmri.MRObjects import RF, Gradient


def _make_trap_block(duration_ms: float,
                     amp_mT_per_m: float = 5.0) -> SequenceBlock:
  """Return a SequenceBlock containing one user-supplied trapezoid of
  duration ``duration_ms`` anchored at block-local t = 0.

  ``Sequence.add_block`` shifts each block by parent_end - block_start
  when appended, so blocks that themselves start at zero land
  contiguously without surprises from
  ``SequenceBlock._get_extent``'s implicit (0, 0) RF placeholder
  (which pulls t_min down to 0 when the block carries no RF pulse)."""
  rise = min(0.05, duration_ms / 2.0)
  flat = max(duration_ms - 2.0 * rise, 0.0)
  timings = np.array([0.0, rise, rise + flat, duration_ms], dtype=float)
  amps = np.array([0.0, amp_mT_per_m, amp_mT_per_m, 0.0], dtype=float)

  g = Gradient(
    timings=Quantity(timings, 'ms'),
    amplitudes=Quantity(amps, 'mT/m'),
    axis=0,
  )
  return SequenceBlock(
    gradients=[g],
    dur=Quantity(duration_ms, 'ms'),
  )


def test_add_block_with_nested_sequence_preserves_absolute_offsets():
  """Appending a child Sequence after a non-empty parent must place
  the child's first block at parent_end and preserve internal
  offsets between child blocks."""
  child = Sequence()
  child.add_block(_make_trap_block(1.0))
  child.add_block(_make_trap_block(1.0))
  assert child.time_extent[1].m_as('ms') == pytest.approx(2.0)

  parent = Sequence()
  parent.add_block(_make_trap_block(3.0))
  assert parent.time_extent[1].m_as('ms') == pytest.approx(3.0)

  parent.add_block(child)

  assert parent.time_extent[0].m_as('ms') == pytest.approx(0.0)
  assert parent.time_extent[1].m_as('ms') == pytest.approx(5.0)
  assert len(parent.blocks) == 3
  assert parent.blocks[1].time_extent[0].m_as('ms') == pytest.approx(3.0)
  assert parent.blocks[1].time_extent[1].m_as('ms') == pytest.approx(4.0)
  assert parent.blocks[2].time_extent[0].m_as('ms') == pytest.approx(4.0)
  assert parent.blocks[2].time_extent[1].m_as('ms') == pytest.approx(5.0)

  # The original child must not be mutated by the merge.
  assert child.time_extent[1].m_as('ms') == pytest.approx(2.0)
  assert child.blocks[0].time_extent[1].m_as('ms') == pytest.approx(1.0)


def test_add_block_with_nested_sequence_repeated_append():
  """Appending the same child Sequence twice must produce four
  contiguous blocks; each insertion deep-copies its children, so
  the originals stay intact."""
  child = Sequence()
  child.add_block(_make_trap_block(1.0))
  child.add_block(_make_trap_block(1.0))

  parent = Sequence()
  parent.add_block(child)
  parent.add_block(child)

  assert len(parent.blocks) == 4
  expected = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
  for blk, (t0, t1) in zip(parent.blocks, expected):
    assert blk.time_extent[0].m_as('ms') == pytest.approx(t0)
    assert blk.time_extent[1].m_as('ms') == pytest.approx(t1)
  assert parent.time_extent[1].m_as('ms') == pytest.approx(4.0)


def test_gradient_interpolator_follows_change_time():
  """``change_time`` must shift a user-supplied gradient's timings AND
  rebuild the interpolator so it returns the original waveform shape
  at the new absolute timeline (and zero at the old position)."""
  g = Gradient(
    timings=Quantity(np.array([0.0, 0.1, 1.1, 1.2]), 'ms'),
    amplitudes=Quantity(np.array([0.0, 5.0, 5.0, 0.0]), 'mT/m'),
    axis=0,
  )
  baseline = float(g(0.5))
  assert math.isclose(baseline, 5.0, abs_tol=1e-6)

  g.change_time(Quantity(10.0, 'ms'))
  assert g.time.m_as('ms') == pytest.approx(10.0)
  np.testing.assert_allclose(
    g.timings.m_as('ms'),
    np.array([10.0, 10.1, 11.1, 11.2]),
    atol=1e-9,
  )

  assert math.isclose(float(g(10.5)), 5.0, abs_tol=1e-6)
  assert float(g(0.5)) == pytest.approx(0.0)


def test_gradient_change_time_idempotent_for_repeated_shifts():
  """Calling ``change_time`` repeatedly with monotonically increasing
  absolute times must apply each delta exactly once (no
  cumulative-offset drift)."""
  g = Gradient(
    timings=Quantity(np.array([0.0, 0.1, 1.1, 1.2]), 'ms'),
    amplitudes=Quantity(np.array([0.0, 5.0, 5.0, 0.0]), 'mT/m'),
    axis=0,
  )
  g.change_time(Quantity(4.0, 'ms'))
  g.change_time(Quantity(7.0, 'ms'))
  np.testing.assert_allclose(
    g.timings.m_as('ms'),
    np.array([7.0, 7.1, 8.1, 8.2]),
    atol=1e-9,
  )
  assert math.isclose(float(g(7.5)), 5.0, abs_tol=1e-6)


def test_rf_change_time_preserves_non_uniform_raster():
  """A custom RF with non-uniform timings must be shifted rigidly,
  preserving both the original sample times' relative spacing and
  the complex waveform values."""
  t0 = np.array([0.0, 0.05, 0.20, 0.40, 1.00])
  wf = np.array([0+0j, 1+0j, 1+0j, 0.5+0j, 0+0j])
  rf = RF(
    shape='hard',
    flip_angle=Quantity(np.pi / 2, 'rad'),
    dur=Quantity(1.0, 'ms'),
    timings=Quantity(t0, 'ms'),
    waveform=Quantity(wf, 'mT'),
  )

  rf.change_time(Quantity(5.0, 'ms'))

  np.testing.assert_allclose(rf.timings.m_as('ms'), t0 + 5.0, atol=1e-9)
  np.testing.assert_allclose(rf.waveform.m_as('mT'), wf, atol=1e-9)
  assert rf.time.m_as('ms') == pytest.approx(5.0)
