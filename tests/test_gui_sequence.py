"""The sequence flattened into arrays, checked against the library's own view.

`SequenceModel` exists so a panel can draw a sequence without `Sequence.plot`
building its own figure. Since it reuses `block._discrete_objects()`, the risk
is not that the traces are wrong but that the TIME ORIGIN is: gradient timings
are absolute while `block.adc.times` are block-local, and mixing the two shifts
the ADC markers by one block start and still looks like a readout.
"""
from __future__ import annotations

import numpy as np
import pytest
from pint import Quantity

from feelmri.Bloch import ADC, Sequence, SequenceBlock
from feelmri.MRObjects import RF, Gradient, Scanner
from feelmri.gui.model.sequence import MRObjectRef, SequenceModel


def _block(t_start=0.0, dur=4.0, with_adc=True, n_grad=2):
  """A block with gradients on two axes, one RF and optionally an ADC."""
  scanner = Scanner()
  grads = []
  for axis in range(n_grad):
    g = Gradient(scanner=scanner, axis=axis,
                 timings=Quantity(np.array([0.0, 0.5, 2.5, 3.0]) + t_start, 'ms'),
                 amplitudes=Quantity([0.0, 10.0, 10.0, 0.0], 'mT/m'))
    grads.append(g)
  rf = RF(scanner=scanner, shape='hard', alpha=Quantity(90, 'deg'),
          dur=Quantity(0.4, 'ms'), time=Quantity(t_start + 0.2, 'ms'),
          ref=Quantity(0.2, 'ms'))
  adc = ADC(np.linspace(0.6, 2.4, 16)) if with_adc else None
  return SequenceBlock(gradients=grads, rf_pulses=[rf], adc=adc,
                       dur=Quantity(dur, 'ms'))


@pytest.fixture
def two_block_sequence():
  seq = Sequence()
  seq.add_block(_block())
  seq.add_block(_block(with_adc=False, n_grad=1))
  return seq


def test_spans_tile_the_timeline_without_gaps(two_block_sequence):
  """Blocks abut, so each span must start where the previous one ended."""
  model = SequenceModel(two_block_sequence)
  spans = model.spans
  assert len(spans) == len(two_block_sequence.blocks)
  for a, b in zip(spans, spans[1:]):
    assert b.t0 == pytest.approx(a.t1, abs=1e-9), 'a gap opened between blocks'
  assert model.duration == pytest.approx(spans[-1].t1 - spans[0].t0)
  assert [s.index for s in spans] == list(range(len(spans)))


def test_spans_report_what_each_block_carries(two_block_sequence):
  model = SequenceModel(two_block_sequence)
  first, second = model.spans
  assert first.has_adc and first.n_adc_samples == 16
  assert not second.has_adc and second.n_adc_samples == 0
  assert first.n_gradients == 2 and second.n_gradients == 1
  assert first.n_rf == second.n_rf == 1
  assert first.duration > 0


def test_adc_times_are_absolute_not_block_local(two_block_sequence):
  """The defect this class is most exposed to, pinned directly.

  `block.adc.times` is block-local. The model must add the block start; if it
  forgets, every marker lands inside block 0 whatever block it belongs to.
  """
  model = SequenceModel(two_block_sequence)
  block = two_block_sequence.blocks[0]
  local = np.asarray(block.adc.times.m_as('ms'), dtype=float)
  t0 = float(block.time_extent[0].m_as('ms'))

  got = model.adc_times()
  np.testing.assert_allclose(got, np.sort(local + t0), atol=1e-9)
  # Every sample must lie inside its own block's span.
  span = model.spans[0]
  assert got.min() >= span.t0 - 1e-9 and got.max() <= span.t1 + 1e-9

  # And the offset must be non-trivial, or the test cannot see the bug.
  seq2 = Sequence()
  seq2.add_block(_block(with_adc=False))
  seq2.add_block(_block())
  shifted = SequenceModel(seq2).adc_times()
  assert shifted.min() > SequenceModel(seq2).spans[1].t0 - 1e-9
  assert not np.allclose(shifted, np.sort(local)), 'block offset was not applied'


def test_gradient_traces_follow_the_axis(two_block_sequence):
  """Axis 0/1/2 select M/P/S, and the block-1 gradient exists only on axis 0."""
  model = SequenceModel(two_block_sequence)
  assert len(model.gradient_traces(0)) == 2      # one per block
  assert len(model.gradient_traces(1)) == 1      # only block 0 has axis 1
  assert len(model.gradient_traces(2)) == 0

  t, a = model.gradient_traces(0)[0]
  assert t.shape == a.shape and t.size >= 2
  assert a.max() == pytest.approx(10.0)
  # Times are already absolute for a gradient.
  assert t.min() >= model.spans[0].t0 - 1e-9

  with pytest.raises(ValueError, match='axis must be'):
    model.gradient_traces(3)


def test_the_played_pulse_and_the_bare_envelope_are_both_available():
  """The offsets live only in `rf.interp`, so the two traces differ.

  A pulse with a frequency offset is modulated in `rf(t)` and untouched in
  `rf.waveform`. Showing only one of them hides whether an offset exists at
  all, which this library has had two defects about.
  """
  scanner = Scanner()
  rf = RF(scanner=scanner, shape='hard', alpha=Quantity(90, 'deg'),
          dur=Quantity(1.0, 'ms'), time=Quantity(0.5, 'ms'),
          ref=Quantity(0.5, 'ms'), frequency_offset=Quantity(2000.0, 'Hz'))
  seq = Sequence()
  seq.add_block(SequenceBlock(rf_pulses=[rf], dur=Quantity(2.0, 'ms')))
  model = SequenceModel(seq)

  played = model.rf_traces()
  envelope = model.rf_envelopes()
  assert len(played) == len(envelope) == 1

  # The envelope is real-valued up to its own phase; the played pulse winds.
  phase_spread = np.ptp(np.angle(played[0][1][np.abs(played[0][1]) > 0]))
  assert phase_spread > 1.0, (
    f'the played pulse shows no modulation (phase spread {phase_spread:.3f} '
    f'rad), so the frequency offset is not reaching rf(t)')


def test_block_at_picks_the_block_under_a_time(two_block_sequence):
  model = SequenceModel(two_block_sequence)
  a, b = model.spans
  assert model.block_at(a.t0 + 0.25 * a.duration) == 0
  assert model.block_at(b.t0 + 0.25 * b.duration) == 1
  # Half-open intervals: an instant on a shared edge is the LATER block.
  assert model.block_at(a.t1) == 1
  # Both ends of the sequence resolve: the first start and the last end.
  assert model.block_at(a.t0) == 0
  assert model.block_at(b.t1) == 1
  # Outside the sequence, nothing.
  assert model.block_at(a.t0 - 1.0) is None
  assert model.block_at(b.t1 + 1.0) is None


def test_boundaries_are_the_block_edges(two_block_sequence):
  model = SequenceModel(two_block_sequence)
  edges = model.boundaries()
  assert edges[0] == pytest.approx(model.t_start)
  assert edges[-1] == pytest.approx(model.t_end)
  assert np.all(np.diff(edges) > 0), 'boundaries are not strictly increasing'


def test_objects_address_every_mr_object(two_block_sequence):
  """Each RF, gradient and ADC gets a reference that survives re-listing."""
  model = SequenceModel(two_block_sequence)
  objs = model.objects()
  kinds = [o.kind for o in objs]
  assert kinds.count('rf') == 2
  assert kinds.count('gradient') == 3          # 2 in block 0, 1 in block 1
  assert kinds.count('adc') == 1               # only block 0

  for o in objs:
    assert isinstance(o, MRObjectRef)
    assert o.t1 >= o.t0
    span = model.spans[o.block]
    assert o.t0 >= span.t0 - 1e-6 and o.t1 <= span.t1 + 1e-6, (
      f'{o.label} lies outside its own block')
    if o.kind == 'gradient':
      assert o.axis in (0, 1, 2)
    else:
      assert o.axis is None

  # Ordinals are unique within (block, kind), which is what makes a label stick.
  keys = [(o.block, o.kind, o.ordinal) for o in objs]
  assert len(set(keys)) == len(keys)
  assert 'G' in [o for o in objs if o.kind == 'gradient'][0].label


def test_a_block_selection_accepts_a_list_where_sequence_plot_does_not(
    two_block_sequence):
  """`Sequence.plot(blocks=[0])` raises, since it does `self.blocks[blocks]`.

  The model takes None, a slice, or any iterable of indices, so a panel can
  select scattered blocks. Negative indices wrap the usual way.
  """
  model = SequenceModel(two_block_sequence)
  assert len(model.gradient_traces(0, blocks=[0])) == 1
  assert len(model.gradient_traces(0, blocks=slice(0, 1))) == 1
  assert len(model.gradient_traces(0, blocks=[-1])) == 1
  assert len(model.gradient_traces(0, blocks=[])) == 0
  with pytest.raises(IndexError, match='out of range'):
    model.gradient_traces(0, blocks=[5])


def test_an_empty_sequence_does_not_crash():
  model = SequenceModel(Sequence())
  assert model.spans == []
  assert model.boundaries().size == 0
  assert model.adc_times().size == 0
  assert model.objects() == []
  assert model.block_at(0.0) is None
  assert model.duration == 0.0


@pytest.mark.pulseq
def test_an_imported_seq_flattens_consistently(pulseq_import):
  """On a real file, every derived quantity must agree with the source.

  `gre_v15.seq` is small and carries RF, gradients on all three axes and an
  ADC, so it exercises every branch without being slow.
  """
  from conftest import DATA_DIR, skip_if_pypulseq_too_old
  path = DATA_DIR / 'gre_v15.seq'
  if not path.exists():
    pytest.skip('gre_v15.seq not present')
  skip_if_pypulseq_too_old(path)

  imp = pulseq_import(path)
  seq = imp.feelmri_seq
  model = SequenceModel(seq)

  assert len(model.spans) == len(seq.blocks)
  np.testing.assert_allclose(model.t_start,
                             float(seq.time_extent[0].m_as('ms')), atol=1e-9)
  np.testing.assert_allclose(model.t_end,
                             float(seq.time_extent[1].m_as('ms')), atol=1e-9)

  # Every ADC sample the model reports must fall inside some block.
  for t in model.adc_times():
    assert model.block_at(t) is not None, f'ADC sample at {t} ms is in no block'

  # The object count must match the blocks' own contents.
  objs = model.objects()
  assert sum(1 for o in objs if o.kind == 'rf') == sum(s.n_rf for s in model.spans)
  assert (sum(1 for o in objs if o.kind == 'gradient')
          == sum(s.n_gradients for s in model.spans))
  assert (sum(1 for o in objs if o.kind == 'adc')
          == sum(1 for s in model.spans if s.has_adc))


def test_the_block_summary_reports_the_numbers_that_are_drawn(
    two_block_sequence):
  """The reading beside the picture must come from the picture's own arrays.

  `describe_block` goes through `gradient_traces` / `rf_traces` / `adc_times`
  rather than reaching into the block itself, so a summary cannot claim one
  amplitude while the trace shows another. Asserted here by recomputing each
  quoted number from those same accessors.
  """
  model = SequenceModel(two_block_sequence)
  rows = dict(model.describe_block(0))

  span = model.spans[0]
  assert rows['block'] == '0'
  assert rows['duration'].startswith(f'{span.duration:.4f}')

  peak_uT = float(np.abs(model.rf_traces(blocks=[0])[0][1]).max()) * 1e3
  assert f'{peak_uT:.4f} uT' in rows['RF[0]']

  t, a = model.gradient_traces(0, blocks=[0])[0]
  assert f'{float(np.abs(a).max()):.4f} mT/m' in rows['GM[0]']
  assert f'{float(np.trapezoid(a, t)):.5f}' in rows['GM[0]']

  adc = model.adc_times(blocks=[0])
  assert f'{adc.size} samples' in rows['ADC']
  assert f'{adc[0]:.4f} to {adc[-1]:.4f} ms' in rows['ADC']


def test_the_summary_quotes_ABSOLUTE_adc_times_like_the_axis_does():
  """A block-local ADC time would read plausibly and sit under the wrong block.

  The second block starts at 4 ms, so its samples cannot begin before that.
  This is the same origin trap the module docstring names, checked where a
  user actually reads the number.
  """
  seq = Sequence()
  seq.add_block(_block(t_start=0.0))
  seq.add_block(_block(t_start=4.0))
  model = SequenceModel(seq)

  rows = dict(model.describe_block(1))
  first = float(rows['ADC'].split('samples,')[1].split('to')[0])
  assert first >= model.spans[1].t0, 'the ADC was quoted block-local'
  assert np.isclose(first, model.adc_times(blocks=[1])[0], atol=1e-4)


def test_the_summary_names_the_flags_a_block_carries():
  """`spoiler` and `store_magnetization` change what a run does, so a user
  picking a block needs to see them without opening the file."""
  seq = Sequence()
  block = _block()
  block.store_magnetization = True
  block.spoiler = True
  seq.add_block(block)
  rows = dict(SequenceModel(seq).describe_block(0))
  assert 'spoiler' in rows['flags']
  assert 'stores magnetization' in rows['flags']


def test_a_block_with_no_adc_says_so_rather_than_omitting_the_row():
  """An absent row reads as 'I forgot to look', which is the wrong answer."""
  seq = Sequence()
  seq.add_block(_block(with_adc=False))
  assert dict(SequenceModel(seq).describe_block(0))['ADC'] == 'none'


def test_the_summary_accepts_a_negative_index_like_the_other_accessors():
  seq = Sequence()
  seq.add_block(_block(t_start=0.0))
  seq.add_block(_block(t_start=4.0))
  model = SequenceModel(seq)
  assert model.describe_block(-1) == model.describe_block(1)


def test_object_at_narrows_by_the_row_that_was_clicked(two_block_sequence):
  """A click fixes the kind, and for a gradient the axis, before the time.

  The fixture overlaps an RF, two gradients and an ADC in one block, so the
  time alone is ambiguous by construction -- which is the case a viewer hits
  constantly and the reason the filters are arguments rather than guesses.
  """
  model = SequenceModel(two_block_sequence)
  t = 1.5                                    # inside every object of block 0

  assert model.object_at(t, kind='rf') is None or \
         model.object_at(t, kind='rf').kind == 'rf'
  for axis in (0, 1):
    found = model.object_at(t, kind='gradient', axis=axis)
    assert found is not None and found.kind == 'gradient' and found.axis == axis
  adc = model.object_at(t, kind='adc')
  assert adc is not None and adc.kind == 'adc'


def test_object_at_returns_nothing_where_nothing_is_played(two_block_sequence):
  """Empty space must select no object, so the panel can fall back to the
  block rather than snapping to whatever is nearest."""
  model = SequenceModel(two_block_sequence)
  assert model.object_at(model.t_end + 1.0, kind='gradient', axis=0) is None
  assert model.object_at(3.8, kind='gradient', axis=0) is None


def test_object_at_prefers_the_shortest_span_when_several_match():
  """The short one is what a click can least easily hit by accident."""
  seq = Sequence()
  seq.add_block(_block())
  model = SequenceModel(seq)

  spans = [o for o in model.objects() if o.kind == 'gradient' and o.axis == 0]
  assert spans, 'the fixture must carry a gradient on axis 0'
  t = 0.5 * (spans[0].t0 + spans[0].t1)
  found = model.object_at(t, kind='gradient', axis=0)
  shortest = min((o for o in model.objects()
                  if o.kind == 'gradient' and o.axis == 0 and o.t0 <= t <= o.t1),
                 key=lambda o: o.t1 - o.t0)
  assert found == shortest


def test_an_object_owns_both_its_edges_unlike_a_block(two_block_sequence):
  """Blocks tile the timeline so their edges must belong to exactly one of
  them; an object is a thing being pointed at, and both ends are part of it."""
  model = SequenceModel(two_block_sequence)
  g = [o for o in model.objects() if o.kind == 'gradient' and o.axis == 0][0]
  assert model.object_at(g.t0, kind='gradient', axis=0) is not None
  assert model.object_at(g.t1, kind='gradient', axis=0) is not None
