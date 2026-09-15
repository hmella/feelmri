"""A sequence flattened into plain arrays a panel can draw.

`Sequence.plot` already draws RF, M, P and S against time, but it builds its
own figure and calls `plt.show()`, so it cannot be embedded. This module is the
data half of that picture, separated out: it returns arrays and spans, holds no
figure, and imports no plotting library.

Everything is in ABSOLUTE sequence milliseconds. That matters because the two
sources disagree: a gradient's `timings` are already absolute, while
`block.adc.times` are block-local and have to be offset by the block start.
Getting that wrong shifts the ADC markers by one block and looks plausible.

`block._discrete_objects()` is reused rather than re-derived, so the panel and
`Sequence.plot` cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence as Seq, Tuple

import numpy as np

#: Gradient axis index to the label the library uses.
AXIS_LABELS = ('M', 'P', 'S')


@dataclass(frozen=True)
class BlockSpan:
  """Where one block sits on the time axis, and what it carries."""

  index: int
  t0: float                   # ms, absolute
  t1: float
  empty: bool
  store_magnetization: bool
  spoiler: bool
  n_rf: int
  n_gradients: int
  n_adc_samples: int

  @property
  def duration(self) -> float:
    return self.t1 - self.t0

  @property
  def has_adc(self) -> bool:
    return self.n_adc_samples > 0


@dataclass(frozen=True)
class MRObjectRef:
  """One RF pulse, gradient or ADC, addressed well enough to label it.

  `ordinal` counts within its own kind inside the block, so `('gradient', 1)`
  is the second gradient of that block whatever axis it is on. `axis` is 0, 1
  or 2 for a gradient and None otherwise.
  """

  block: int
  kind: str                   # 'rf' | 'gradient' | 'adc'
  ordinal: int
  axis: Optional[int]
  t0: float                   # ms, absolute
  t1: float

  @property
  def label(self) -> str:
    if self.kind == 'gradient' and self.axis is not None:
      return f'block {self.block} G{AXIS_LABELS[self.axis]}[{self.ordinal}]'
    return f'block {self.block} {self.kind}[{self.ordinal}]'


class SequenceModel:
  """Read-only view of a `Sequence`, shaped for drawing and picking.

  Holds a reference to the sequence and derives everything on demand. Nothing
  here mutates the sequence, which matters because `tests/conftest.py` caches
  imported sequences for the whole session and a mutation would leak.
  """

  def __init__(self, sequence):
    self.sequence = sequence
    self._spans: Optional[List[BlockSpan]] = None

  # -- geometry of the time axis -------------------------------------------

  @property
  def spans(self) -> List[BlockSpan]:
    """One `BlockSpan` per block, in sequence order. Computed once."""
    if self._spans is None:
      self._spans = [self._span(i, b)
                     for i, b in enumerate(self.sequence.blocks)]
    return self._spans

  @staticmethod
  def _span(index: int, block) -> BlockSpan:
    t0 = float(block.time_extent[0].m_as('ms'))
    t1 = float(block.time_extent[1].m_as('ms'))
    adc = getattr(block, 'adc', None)
    return BlockSpan(
      index=index, t0=t0, t1=t1,
      empty=bool(getattr(block, 'empty', False)),
      store_magnetization=bool(getattr(block, 'store_magnetization', False)),
      spoiler=bool(getattr(block, 'spoiler', False)),
      n_rf=len(getattr(block, 'rf_pulses', ())),
      n_gradients=len(getattr(block, 'gradients', ())),
      n_adc_samples=0 if adc is None else int(np.size(adc.times)))

  @property
  def t_start(self) -> float:
    return self.spans[0].t0 if self.spans else 0.0

  @property
  def t_end(self) -> float:
    return self.spans[-1].t1 if self.spans else 0.0

  @property
  def duration(self) -> float:
    """Total extent in ms, the span of the blocks rather than their sum."""
    return self.t_end - self.t_start

  def boundaries(self) -> np.ndarray:
    """Every block edge, sorted and de-duplicated, for drawing guide lines."""
    if not self.spans:
      return np.empty(0)
    edges = np.array([s.t0 for s in self.spans] + [self.spans[-1].t1])
    return np.unique(edges)

  def block_at(self, t: float) -> Optional[int]:
    """Index of the block containing absolute time `t`, or None.

    Blocks abut, so every shared edge belongs to two of them by arithmetic.
    The convention is half-open, `[t0, t1)`, so an instant exactly on an edge
    belongs to the LATER block, which is what `searchsorted(side='right')`
    gives with no special case. The one exception is the very end of the
    sequence, which has no later block and so stays with the last one.
    """
    if not self.spans:
      return None
    starts = np.array([s.t0 for s in self.spans])
    i = int(np.searchsorted(starts, t, side='right')) - 1
    if i < 0 or t > self.spans[i].t1:
      return None
    return i

  # -- traces ---------------------------------------------------------------

  def gradient_traces(self, axis: int,
                      blocks: Optional[Seq[int]] = None
                      ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """`(times_ms, amplitudes_mT_per_m)` for every gradient on one axis.

    One entry per gradient, not one concatenated line: the gaps between
    gradients are real and joining them would draw a ramp that is not played.
    """
    if axis not in (0, 1, 2):
      raise ValueError(f'gradient_traces: axis must be 0, 1 or 2, got {axis}')
    out = []
    for i in self._selected(blocks):
      discrete = self.sequence.blocks[i]._discrete_objects()
      for t, a in discrete[1 + axis]:
        out.append((np.asarray(t, dtype=float), np.asarray(a, dtype=float)))
    return out

  def rf_traces(self, blocks: Optional[Seq[int]] = None
                ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """`(times_ms, complex_mT)` for every RF pulse, AS PLAYED.

    This is the modulated pulse: `_discrete_objects` evaluates `rf(t)`, which
    goes through the interpolator, and the frequency and phase offsets are
    baked into that interpolator only. Use `rf_envelopes` for the bare shape.
    """
    out = []
    for i in self._selected(blocks):
      for t, w in self.sequence.blocks[i]._discrete_objects()[0]:
        out.append((np.asarray(t, dtype=float), np.asarray(w)))
    return out

  def rf_envelopes(self, blocks: Optional[Seq[int]] = None
                   ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """`(times_ms, complex_mT)` for every RF pulse, UNMODULATED.

    `rf.waveform` is what the file or the generator produced; the frequency and
    phase offsets never reach it. Plotting this beside `rf_traces` is how a
    user sees that a pulse carries an offset at all, which is otherwise
    invisible and has been the subject of two defects in this library.
    """
    out = []
    for i in self._selected(blocks):
      for rf in self.sequence.blocks[i].rf_pulses:
        t = np.asarray(_magnitude(rf.timings, 'ms'), dtype=float)
        out.append((t, np.asarray(_magnitude(rf.waveform, 'mT'))))
    return out

  def adc_times(self, blocks: Optional[Seq[int]] = None) -> np.ndarray:
    """Every ADC sample time in ABSOLUTE ms, concatenated and sorted.

    `block.adc.times` is block-local, so the block start is added. Forgetting
    that offset shifts every marker by one block start and still looks like a
    plausible readout.
    """
    chunks = []
    for i in self._selected(blocks):
      block = self.sequence.blocks[i]
      adc = getattr(block, 'adc', None)
      if adc is None or np.size(adc.times) == 0:
        continue
      local = np.asarray(adc.times.m_as('ms'), dtype=float)
      chunks.append(local + float(block.time_extent[0].m_as('ms')))
    if not chunks:
      return np.empty(0)
    return np.sort(np.concatenate(chunks))

  # -- what one block contains ----------------------------------------------

  def describe_block(self, index: int) -> List[Tuple[str, str]]:
    """`(label, value)` rows describing one block, ready to display.

    **Every number is derived from the SAME arrays the panel draws**, through
    `gradient_traces` / `rf_traces` / `adc_times`, so the reading beside the
    picture cannot disagree with the picture. Deriving it from the block
    separately is how a summary drifts from the trace it describes.

    Units follow what the panel puts on its axes: RF in microtesla, gradients
    in mT/m, their areas in mT/m*ms, times in ms.
    """
    i = self._selected([index])[0]
    span = self.spans[i]
    rows: List[Tuple[str, str]] = [
      ('block', str(i)),
      ('start', f'{span.t0:.4f} ms'),
      ('end', f'{span.t1:.4f} ms'),
      ('duration', f'{span.duration:.4f} ms'),
    ]
    flags = [n for n, on in (('empty', span.empty), ('spoiler', span.spoiler),
                             ('stores magnetization', span.store_magnetization))
             if on]
    rows.append(('flags', ', '.join(flags) if flags else 'none'))

    for k, (t, w) in enumerate(self.rf_traces(blocks=[i])):
      peak = float(np.abs(w).max()) * 1e3 if np.size(w) else 0.0
      rows.append((f'RF[{k}]',
                   f'{peak:.4f} uT peak over '
                   f'{_extent(t):.4f} ms'))

    for axis in (0, 1, 2):
      for k, (t, a) in enumerate(self.gradient_traces(axis, blocks=[i])):
        peak = float(np.abs(a).max()) if np.size(a) else 0.0
        area = float(np.trapezoid(a, t)) if np.size(a) > 1 else 0.0
        rows.append((f'G{AXIS_LABELS[axis]}[{k}]',
                     f'{peak:.4f} mT/m peak, area {area:.5f} mT/m*ms'))

    adc = self.adc_times(blocks=[i])
    if adc.size:
      dwell = float(np.diff(adc).mean()) if adc.size > 1 else 0.0
      rows.append(('ADC', f'{adc.size} samples, {adc[0]:.4f} to '
                          f'{adc[-1]:.4f} ms, dwell {dwell * 1e3:.3f} us'))
    else:
      rows.append(('ADC', 'none'))
    return rows

  # -- objects, for selection and labelling ---------------------------------

  def objects(self, blocks: Optional[Seq[int]] = None) -> List[MRObjectRef]:
    """Every MR object, addressed so it can be selected and labelled.

    The ordinal counts within a kind, and gradients keep their axis, so a
    reference survives anything except editing the block itself.
    """
    out: List[MRObjectRef] = []
    for i in self._selected(blocks):
      block = self.sequence.blocks[i]
      t0 = float(block.time_extent[0].m_as('ms'))
      t1 = float(block.time_extent[1].m_as('ms'))

      for k, rf in enumerate(getattr(block, 'rf_pulses', ())):
        start, stop = _support(rf, t0, t1)
        out.append(MRObjectRef(i, 'rf', k, None, start, stop))

      for k, g in enumerate(getattr(block, 'gradients', ())):
        times = np.asarray(_magnitude(g.timings, 'ms'), dtype=float)
        start = float(times.min()) if times.size else t0
        stop = float(times.max()) if times.size else t1
        out.append(MRObjectRef(i, 'gradient', k, int(g.axis), start, stop))

      adc = getattr(block, 'adc', None)
      if adc is not None and np.size(adc.times):
        local = np.asarray(adc.times.m_as('ms'), dtype=float)
        out.append(MRObjectRef(i, 'adc', 0, None,
                               t0 + float(local.min()), t0 + float(local.max())))
    return out

  # -- internals ------------------------------------------------------------

  def _selected(self, blocks: Optional[Seq[int]]) -> List[int]:
    """Normalise a block selection to a list of indices.

    Accepts None (all), a slice, or any iterable of indices. `Sequence.plot`
    takes only a slice, which is why passing it a list of indices raises; this
    does not inherit that.
    """
    n = len(self.sequence.blocks)
    if blocks is None:
      return list(range(n))
    if isinstance(blocks, slice):
      return list(range(*blocks.indices(n)))
    out = []
    for b in blocks:
      i = int(b)
      if not -n <= i < n:
        raise IndexError(
          f'SequenceModel: block {i} is out of range for {n} blocks')
      out.append(i % n)
    return out


def _magnitude(value, unit: str):
  """Magnitude of a pint Quantity, or the value unchanged if it is bare.

  `RF.timings` and `RF.waveform` are Quantities for an imported pulse and plain
  arrays for an analytic one, so both spellings reach here.
  """
  m_as = getattr(value, 'm_as', None)
  return m_as(unit) if m_as is not None else value


def _extent(t) -> float:
  """Span of a time array in ms, zero when it carries fewer than two points."""
  t = np.asarray(t, dtype=float)
  return float(t.max() - t.min()) if t.size > 1 else 0.0


def _support(rf, t0: float, t1: float) -> Tuple[float, float]:
  """Absolute time span of an RF pulse, falling back to the block extent."""
  times = np.asarray(_magnitude(getattr(rf, 'timings', None), 'ms'),
                     dtype=float) if getattr(rf, 'timings', None) is not None \
    else np.empty(0)
  if times.size:
    return float(times.min()), float(times.max())
  return t0, t1
