"""The sequence panel's drawing, checked against `Sequence.plot` itself.

The panel exists so a sequence can be drawn inside a window instead of in the
figure `Sequence.plot` builds and shows itself. The risk is therefore not that
a trace is wrong -- both go through `block._discrete_objects()` -- but that the
panel puts the right array on the WRONG ROW, or scales it, or joins segments
that are separate. Every one of those produces a picture a reader would act on.

So these compare the panel's drawn line data against the line data
`Sequence.plot` produces for the same sequence, row by row.

**No display and no Tk is needed.** The drawing lives in module-level functions
taking an axis, so a bare Agg `Figure` is enough; `SequencePanel` itself is
only wiring over them.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from pint import Quantity

matplotlib.use('Agg')
from matplotlib.figure import Figure                            # noqa: E402

from feelmri.Bloch import ADC, Sequence, SequenceBlock          # noqa: E402
from feelmri.MRObjects import RF, Gradient, Scanner             # noqa: E402
from feelmri.gui.model.sequence import SequenceModel            # noqa: E402
from feelmri.gui.view.sequence_panel import (                   # noqa: E402
  MAX_BOUNDARY_LINES, ROWS, draw_adc, draw_boundaries, draw_gradients,
  draw_rf, draw_sequence, make_axes)


def _block(t_start=0.0, dur=4.0, with_adc=True, axes=(0, 1, 2), freq=0.0):
  scanner = Scanner()
  grads = [Gradient(scanner=scanner, axis=axis,
                    timings=Quantity(np.array([0.0, 0.5, 2.5, 3.0]) + t_start,
                                     'ms'),
                    amplitudes=Quantity([0.0, 5.0 * (axis + 1),
                                         5.0 * (axis + 1), 0.0], 'mT/m'))
           for axis in axes]
  rf = RF(scanner=scanner, shape='hard', alpha=Quantity(90, 'deg'),
          dur=Quantity(0.4, 'ms'), time=Quantity(t_start + 0.2, 'ms'),
          ref=Quantity(0.2, 'ms'), frequency_offset=Quantity(freq, 'Hz'))
  return SequenceBlock(gradients=grads, rf_pulses=[rf],
                       adc=ADC(np.linspace(0.6, 2.4, 16)) if with_adc else None,
                       dur=Quantity(dur, 'ms'))


@pytest.fixture
def sequence():
  seq = Sequence()
  seq.add_block(_block(t_start=0.0))
  seq.add_block(_block(t_start=4.0))
  return seq


@pytest.fixture
def drawn(sequence):
  """The panel's five rows, filled, on a figure that never reaches a screen."""
  figure = Figure(figsize=(6, 4))
  axes = make_axes(figure)
  draw_sequence(axes, SequenceModel(sequence))
  return axes


def _polylines(ax):
  """Every line on an axis as a set of rounded `(x, y)` tuples.

  A set, because neither the panel nor `Sequence.plot` promises an order, and
  ordering is not what is under test here.
  """
  out = set()
  for line in ax.lines:
    if line.get_linestyle() != '-':
      continue          # `Sequence.plot`'s axvline edges and axhline at zero
    x, y = line.get_xdata(), line.get_ydata()
    out.add((tuple(np.round(np.asarray(x, dtype=float), 9)),
             tuple(np.round(np.asarray(y, dtype=float), 9))))
  return out


def _reference_rows(sequence):
  """What `Sequence.plot` draws, without showing it.

  `plot` ends in `plt.show()`, which is a no-op under Agg, and is rank-0
  guarded, which is satisfied serially. Reading its figure is the only way to
  compare against the library's own picture rather than against a restatement
  of it.
  """
  import matplotlib.pyplot as plt

  plt.close('all')
  sequence.plot()
  figure = plt.gcf()
  rows = [_polylines(ax) for ax in figure.axes]
  plt.close('all')
  return rows


# -- the three gradient rows ------------------------------------------------

@pytest.mark.parametrize('axis,row', [(0, 1), (1, 2), (2, 3)])
def test_each_gradient_axis_lands_on_its_own_row(sequence, drawn, axis, row):
  """M, P and S must reach rows 1, 2 and 3, and no others.

  The fixture gives each axis a different amplitude precisely so a swap is
  visible: with equal amplitudes, transposing two rows changes nothing.
  """
  reference = _reference_rows(sequence)[row]
  assert _polylines(drawn[row]) == reference

  # And the control: the other two gradient rows must NOT carry it.
  for other in {1, 2, 3} - {row}:
    assert _polylines(drawn[other]) != reference, (
      f'G{ROWS[row][0]} also appears on the G{ROWS[other][0]} row')


def test_gradient_segments_are_separate_lines_not_one_joined_trace(sequence,
                                                                   drawn):
  """Joining the gaps draws a ramp the scanner never plays.

  Two blocks, each with a gradient on every axis, so each row must carry two
  polylines rather than one of twice the length.
  """
  for row in (1, 2, 3):
    lines = drawn[row].lines
    assert len(lines) == 2, f'row {ROWS[row][0]} joined its segments'
    assert all(len(line.get_xdata()) == 4 for line in lines)


# -- the RF row -------------------------------------------------------------

def test_the_rf_row_draws_the_real_part_in_microtesla(sequence, drawn):
  """`Sequence.plot` works in mT, the panel in uT, and the shape must agree.

  The factor is asserted explicitly rather than assumed: a missing 1e3 is a
  plausible-looking picture with the wrong number on the axis.
  """
  model = SequenceModel(sequence)
  played = model.rf_traces()
  assert played, 'the fixture must carry an RF pulse'

  drawn_lines = {tuple(np.round(line.get_ydata(), 9))
                 for line in drawn[0].lines}
  for t, w in played:
    expected = tuple(np.round(np.real(w) * 1e3, 9))
    assert expected in drawn_lines, 'Re(B1) in uT is not on the RF row'


def test_the_rf_row_brackets_the_real_part_with_its_envelope(sequence, drawn):
  """The envelope is what makes the real part readable as a pulse."""
  model = SequenceModel(sequence)
  drawn_lines = {tuple(np.round(line.get_ydata(), 9))
                 for line in drawn[0].lines}
  for _t, w in model.rf_traces():
    magnitude = np.abs(np.asarray(w)) * 1e3
    assert tuple(np.round(magnitude, 9)) in drawn_lines
    assert tuple(np.round(-magnitude, 9)) in drawn_lines


class _StubModel:
  """Just enough model for `draw_rf`, which asks only for `rf_traces`.

  Used so the drawing contract can be stated exactly, without the analytic RF
  generator in the way: it RESCALES a pulse to hold the requested flip angle,
  so adding a frequency offset to a real `RF` changes the amplitude as well as
  the phase (measured: peak |B1| moves by 5.4x) and nothing about the picture
  can be isolated.
  """

  def __init__(self, traces):
    self._traces = traces

  def rf_traces(self, blocks=None):
    return self._traces


def _rf_row(traces):
  figure = Figure()
  axes = make_axes(figure)
  draw_rf(axes[0], _StubModel(traces))
  return [np.asarray(line.get_ydata(), dtype=float) for line in axes[0].lines]


def test_the_rf_row_is_the_real_part_bracketed_by_the_envelope():
  """Three lines per pulse, and exactly which three."""
  t = np.linspace(0.0, 1.0, 25)
  w = 0.01 * np.exp(2j * np.pi * 3.0 * t)        # mT, rotating
  lines = _rf_row([(t, w)])

  assert len(lines) == 3
  np.testing.assert_allclose(lines[0], np.real(w) * 1e3, atol=1e-12)
  np.testing.assert_allclose(lines[1], np.abs(w) * 1e3, atol=1e-12)
  np.testing.assert_allclose(lines[2], -np.abs(w) * 1e3, atol=1e-12)


def test_a_magnitude_only_row_could_not_show_a_phase_offset():
  """The reason the row draws Re(B1) rather than |B1|.

  Two pulses with IDENTICAL magnitude and different phase: the envelope is
  bit-identical, so a magnitude-only row would draw the same picture for both,
  while the real part separates them completely. Mis-signed frequency and
  phase offsets have twice been real defects in this library, and this is the
  row on which a user would notice one.
  """
  t = np.linspace(0.0, 1.0, 64)
  envelope = 0.01 * np.sin(np.pi * t)
  plain = envelope.astype(complex)
  offset = envelope * np.exp(2j * np.pi * 5.0 * t)

  a, b = _rf_row([(t, plain)]), _rf_row([(t, offset)])

  np.testing.assert_allclose(np.abs(a[1]), np.abs(b[1]), atol=1e-15)
  assert not np.allclose(a[0], b[0], atol=1e-6), (
    'the drawn row is blind to a phase offset')


# -- the ADC row ------------------------------------------------------------

def test_the_adc_row_marks_absolute_sample_times(sequence, drawn):
  """Block-local times would put the second block's samples under the first.

  This is the origin trap the model docstring names, checked where it would
  actually mislead: on the picture.
  """
  model = SequenceModel(sequence)
  expected = model.adc_times()
  assert expected.size

  # Only the ADC markers: the boundary guides go on every row, this one
  # included, so reading every collection would mix the two.
  figure = Figure()
  axes = make_axes(figure)
  draw_adc(axes[-1], model)
  segments = np.asarray(axes[-1].collections[0].get_segments())
  marked = np.unique(np.round(segments[:, 0, 0], 9))
  np.testing.assert_allclose(marked, np.round(expected, 9), atol=1e-9)
  assert marked.min() >= model.t_start - 1e-9


def test_a_block_with_no_adc_draws_no_markers():
  seq = Sequence()
  seq.add_block(_block(with_adc=False))
  figure = Figure()
  axes = make_axes(figure)
  draw_adc(axes[-1], SequenceModel(seq))
  assert not axes[-1].collections


# -- block boundaries -------------------------------------------------------

def test_boundaries_are_drawn_on_every_row(sequence):
  """A guide line on one row only is worse than none: it reads as an event."""
  model = SequenceModel(sequence)
  figure = Figure()
  axes = make_axes(figure)
  drawn_count = draw_boundaries(axes, model)

  assert drawn_count == model.boundaries().size
  for ax in axes:
    segments = np.concatenate([c.get_segments() for c in ax.collections])
    np.testing.assert_allclose(np.unique(np.round(segments[:, 0, 0], 9)),
                               np.round(model.boundaries(), 9), atol=1e-9)


def test_boundaries_are_dropped_when_there_are_too_many_to_read():
  """At 5940 blocks a line lands every 0.2 pixels, which is a grey wash.

  Dropping them is a display decision, so it is asserted rather than left to
  be rediscovered: picking still works, only the guide lines go.
  """
  seq = Sequence()
  for k in range(MAX_BOUNDARY_LINES + 2):
    seq.add_block(_block(t_start=4.0 * k, with_adc=False, axes=(0,)))
  model = SequenceModel(seq)
  assert model.boundaries().size > MAX_BOUNDARY_LINES

  figure = Figure()
  axes = make_axes(figure)
  assert draw_boundaries(axes, model) == 0
  assert all(not ax.collections for ax in axes)

  # Picking is unaffected, which is the property that must survive. The time
  # comes from the model: `add_block` re-times a block onto the end of the
  # sequence, so the fixture's own t_start is not where it lands.
  span = model.spans[10]
  assert model.block_at(0.5 * (span.t0 + span.t1)) == 10


# -- the rows themselves ----------------------------------------------------

def test_the_gradient_rows_are_named_by_the_models_own_axis_labels():
  """The row a trace is drawn on and the axis it came from share one source."""
  from feelmri.gui.model.sequence import AXIS_LABELS
  assert tuple(name for name, _ in ROWS) == ('RF',) + AXIS_LABELS + ('ADC',)


def test_an_empty_sequence_draws_nothing_and_does_not_raise():
  figure = Figure()
  axes = make_axes(figure)
  draw_sequence(axes, SequenceModel(Sequence()))
  assert all(not ax.lines for ax in axes)


def test_drawing_does_not_mutate_the_sequence(sequence):
  """`tests/conftest.py` caches imported sequences for the whole session, so a
  panel that mutated one would leak into every other test."""
  model = SequenceModel(sequence)
  before = [(s.t0, s.t1, s.n_rf, s.n_gradients, s.n_adc_samples)
            for s in model.spans]
  figure = Figure()
  draw_sequence(make_axes(figure), model)
  after = [(s.t0, s.t1, s.n_rf, s.n_gradients, s.n_adc_samples)
           for s in SequenceModel(sequence).spans]
  assert before == after
