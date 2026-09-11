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


def _accumulated_phase(gradient, dur_ms, method, dt_gr, tmp_path):
  """Phase a spin at a known x accumulates under one gradient, per node.

  This goes through the SOLVER rather than re-implementing a quadrature, so it
  measures what the kernel does and not what a test thinks it does.
  """
  import meshio
  from feelmri.Bloch import BlochSolver
  from feelmri.MRObjects import Scanner
  from feelmri.Phantom import FEMPhantom

  x0 = 3e-3
  mesh = tmp_path / f'spin_{method}_{dt_gr}.vtu'
  meshio.write(str(mesh), meshio.Mesh(
      np.array([[x0, 0, 0], [x0 + 1e-5, 0, 0], [x0, 1e-5, 0], [x0, 0, 1e-5]]),
      [("tetra", np.array([[0, 1, 2, 3]]))]))

  phantom = FEMPhantom(path=str(mesh))
  phantom.set_assembler(voxel_size=0.0, lorder=1, horder=1,
                        nodal_approximation=False, lumped=False)
  block = SequenceBlock(gradients=[gradient], dur=Quantity(dur_ms, 'ms'),
                        dt_gr=Quantity(dt_gr, 'ms'))
  block.store_magnetization = True
  seq = Sequence()
  seq.add_block(block)
  Mxy, _Mz = BlochSolver(sequence=seq, phantom=phantom, M0=1.0,
                         T1=Quantity(1e9, 'ms'), T2=Quantity(1e9, 'ms'),
                         initial_Mxy=1.0 + 0j, dtype='float64', method=method,
                         perfect_spoiling=False).solve()
  gamma = Scanner().gamma.m_as('rad/ms/mT')
  return np.angle(Mxy[:, -1]), phantom.local_nodes[:, 0], gamma


@pytest.mark.parametrize('rise,fall', [
    (0.20, 0.05),      # both integer multiples of a 0.01 ms sub-raster
    (0.10, 0.10),      # symmetric: every quadrature gets this one right
    (0.0123, 0.0456),  # neither a multiple
    (0.002, 0.05),     # rise shorter than a gradient raster step
])
def test_trapezoid_phase_is_exact_under_the_default_solver(rise, fall, tmp_path):
  """A trapezoid must deliver its analytic moment whatever its ramps.

  The default `magnus2` integrates a piecewise-linear gradient EXACTLY -- the
  trapezoidal rule is exact on a straight segment, from the four corners alone,
  with no sub-sampling.

  `cayley_klein` charges each interval the field at its END, which over-charges
  the ramp up by `A*h_rise/2` and under-charges the ramp down by `A*h_fall/2`,
  leaving `A*(h_rise - h_fall)/2`. That is zero for a symmetric trapezoid, and
  it is also zero when both ramps are integer multiples of the sub-raster --
  which is why an earlier version of this test, parametrised only on
  `(0.20,0.05)`, `(0.05,0.20)` and `(0.10,0.10)`, passed for the wrong reason.
  The last two cases above are the ones that discriminate: measured 4.4e-4 and
  3.9e-3 relative under `cayley_klein`, against 4.3e-7 under `magnus2`.
  """
  A, flat = 10.0, 1.0
  timings = Quantity(np.array([0.0, rise, rise + flat, rise + flat + fall]), 'ms')
  amplitudes = Quantity(np.array([0.0, A, A, 0.0]), 'mT/m')
  grad = Gradient(timings=timings, amplitudes=amplitudes, axis=0)

  got, xn, gamma = _accumulated_phase(grad, rise + flat + fall, 'magnus2', -1, tmp_path)
  analytic = -gamma * xn * A * (flat + 0.5 * (rise + fall))
  err = float(np.abs(np.angle(np.exp(1j * (got - analytic)))).max())
  assert err < 1e-5, (
      f'rise={rise} fall={fall}: magnus2 is off by {err:.3e} rad; the '
      f'trapezoidal rule must be exact on a straight ramp')


def test_add_block_warns_instead_of_silently_dropping():
    """A dropped block shifts every later index by one, silently.

    `first_block`, `m_storage_block` and `block_labels` are all positional, so a
    zero-duration delay that vanishes without a word puts the readout
    bookkeeping one block out for the rest of the sequence.
    """
    seq = Sequence()
    seq.add_block(SequenceBlock(dur=Quantity(1.0, 'ms')))
    before = seq.Nb_blocks

    with pytest.warns(UserWarning, match='non-positive duration'):
        seq.add_block(Quantity(0.0, 'ms'))
    assert seq.Nb_blocks == before

    with pytest.warns(UserWarning, match='Nothing was appended'):
        seq.add_block(None)
    assert seq.Nb_blocks == before


def test_add_block_dt_is_rejected_for_an_existing_block():
    """`dt` builds a delay's raster; a SequenceBlock's is fixed at construction.

    Passing it there looked like it worked and did nothing at all.
    """
    seq = Sequence()
    with pytest.warns(UserWarning, match='applies only when'):
        seq.add_block(SequenceBlock(dur=Quantity(1.0, 'ms')), dt=Quantity(0.01, 'ms'))
    # The delay branch still honours it, and must not warn.
    seq2 = Sequence()
    seq2.add_block(Quantity(1.0, 'ms'), dt=Quantity(0.05, 'ms'))
    assert seq2.Nb_blocks == 1


def test_nested_sequence_carries_explicit_spoiling():
    """`explicit_spoiling` lives on the Sequence, so appending one to another
    used to drop it -- and `BlochSolver(perfect_spoiling=None)` then resolves
    back to True and zeroes Mxy at every block boundary, destroying exactly the
    coherence pathways a .seq spells its own spoilers out to preserve."""
    child = Sequence()
    child.add_block(SequenceBlock(dur=Quantity(1.0, 'ms')))
    child.explicit_spoiling = True

    parent = Sequence()
    parent.add_block(SequenceBlock(dur=Quantity(1.0, 'ms')))
    assert parent.explicit_spoiling is False
    parent.add_block(child)
    assert parent.explicit_spoiling is True


def test_adc_only_block_reports_a_real_duration():
    """_get_extent consults the ADC, so an ADC-only block is not zero-length.

    It used to report dur = 0 while `discrete_times` spanned the whole
    acquisition, so the next block chained 0 ms later and the two disagreed.
    """
    from feelmri.Bloch import ADC

    adc = ADC(times=Quantity(np.array([0.1, 0.2, 0.3]), 'ms'))
    block = SequenceBlock(adc=adc)
    lo, hi = (x.m_as('ms') for x in block.time_extent)
    assert hi > lo, f'ADC-only block still reports an empty extent [{lo}, {hi}]'
    assert hi == pytest.approx(0.3, abs=1e-9)
    t = block.discrete_times.m_as('ms')
    assert t[-1] <= hi + 1e-9, 'the raster runs past the block it belongs to'


def test_block_raster_never_leaves_its_block():
  """`discrete_times` must lie inside `time_extent`, always.

  Several sources feed the raster and they do not all share an origin:
  `adc.times` are block-local (per the ADC docstring and the timing gate), a
  user-defined `Gradient` keeps its own `timings` without applying `time`, and
  the `dt` arange runs over the extent. Concatenating the first of those raw
  agrees with the rest only while `time_extent[0] == 0` -- true for every
  IMPORTED block, because `_convert_*` puts every event at t = 0, and false for
  a natively built one.

  When it disagreed, the raster began before the block did, so the solver
  integrated time belonging to the previous block and `add_block` chained the
  next one from a duration that no longer matched.
  """
  from feelmri.Bloch import ADC

  grad = Gradient(timings=Quantity(np.array([0.0, 0.1, 1.1, 1.2]), 'ms'),
                  amplitudes=Quantity(np.array([0.0, 10.0, 10.0, 0.0]), 'mT/m'),
                  axis=0, time=Quantity(1.0, 'ms'))
  block = SequenceBlock(gradients=[grad],
                        adc=ADC(times=Quantity(np.linspace(0.5, 1.5, 5), 'ms')))

  lo = block.time_extent[0].m_as('ms')
  hi = block.time_extent[1].m_as('ms')
  t = block.discrete_times.m_as('ms')
  assert t.min() >= lo - 1e-9, (
      f'the raster starts at {t.min():.4f} ms, before the block at {lo:.4f}')
  assert t.max() <= hi + 1e-9, (
      f'the raster ends at {t.max():.4f} ms, after the block at {hi:.4f}')

  # And the ADC samples must land inside it, not at their bare block-local values.
  _rf, _g, mask = block(t)
  assert mask.sum() > 0, 'no ADC sample was placed on the raster'


def test_check_hardware_catches_an_over_spec_sequence():
  """An imported sequence is never checked against the scanner; this makes it
  checkable.

  `Gradient` copies the scanner's limits onto every instance as `Gr_max` /
  `Gr_sr`, but the user-defined branch of its constructor returns before
  comparing them -- and the Pulseq adapter only ever builds user-defined
  gradients. The only comparisons live in `calculate()` / `match_area()`, which
  the adapter never calls, so a `.seq` written for a stronger scanner imported
  and simulated silently. Peak B1 was not checkable at all.
  """
  from feelmri.MRObjects import Scanner

  grad = Gradient(timings=Quantity(np.array([0.0, 0.1, 1.1, 1.2]), 'ms'),
                  amplitudes=Quantity(np.array([0.0, 20.0, 20.0, 0.0]), 'mT/m'),
                  axis=0)
  seq = Sequence()
  seq.add_block(SequenceBlock(gradients=[grad], dur=Quantity(1.2, 'ms')))

  generous = Scanner(gradient_strength=Quantity(40, 'mT/m'),
                     gradient_slew_rate=Quantity(500, 'mT/m/ms'))
  assert seq.check_hardware(generous) == (), 'a legal sequence must report nothing'

  weak = Scanner(gradient_strength=Quantity(10, 'mT/m'),
                 gradient_slew_rate=Quantity(500, 'mT/m/ms'))
  amp_problems = seq.check_hardware(weak)
  assert any('gradient peaks' in p for p in amp_problems), amp_problems

  slow = Scanner(gradient_strength=Quantity(40, 'mT/m'),
                 gradient_slew_rate=Quantity(10, 'mT/m/ms'))
  slew_problems = seq.check_hardware(slow)
  assert any('slew reaches' in p for p in slew_problems), slew_problems


def test_adc_demodulation_is_reachable_without_simulate_pulseq():
  """A caller assembling signal by hand must be able to demodulate.

  `simulate_pulseq` applies the ADC frequency/phase offsets for its own readout
  windows, but the manual `update_magnetization` + `mri_signal` path -- which
  `examples/pulseq_run_epi_tagging.py` uses -- reached nothing, and the offsets
  carried on `feelmri.Bloch.ADC` were dead duplicates of the ReadoutWindow
  fields.
  """
  from feelmri.Bloch import ADC

  adc = ADC(times=Quantity(np.linspace(0.0, 1.0, 8), 'ms'),
            freq_offset=Quantity(500.0, 'Hz'),
            phase_offset=Quantity(0.3, 'rad'))
  phase = adc.demodulation_phase()
  assert phase.size == 8
  assert abs(phase[0] - 0.3) < 1e-12, 'the phase offset must apply at t = 0'
  # 500 Hz over 1 ms is half a cycle.
  assert abs((phase[-1] - phase[0]) - np.pi) < 1e-9

  signal = np.ones((8, 1, 1, 1), dtype=np.complex64)
  out = adc.demodulate(signal)
  assert out.dtype == np.complex64, 'demodulation must not promote the dtype'
  assert np.allclose(out.reshape(-1), np.exp(-1j * phase), atol=1e-6)

  quiet = ADC(times=Quantity(np.linspace(0.0, 1.0, 4), 'ms'))
  assert not np.any(quiet.demodulation_phase())
  probe = np.ones((4, 1, 1, 1), dtype=np.complex64)
  assert quiet.demodulate(probe) is probe
