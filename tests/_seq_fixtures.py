"""Shared sequence builders for the closed-form physics tests.

These produce single-block Sequence objects with controlled content
(relaxation-only, hard RF pulse, gradient prephaser, …) so the
physics tests can assert closed-form Bloch results without having to
hand-author full Pulseq files."""
from __future__ import annotations

import numpy as np
from pint import Quantity

from feelmri.Bloch import Sequence, SequenceBlock
from feelmri.MRObjects import RF, Scanner


def make_empty_block(dur_ms: float, dt_ms: float = 0.1) -> SequenceBlock:
  """An empty (dead-time) block of the given duration. M evolves only
  under T1 / T2 / B0 — no RF, no gradient. The block is marked with
  ``empty=False`` so BlochSolver runs its full per-step kernel."""
  return SequenceBlock(
    dur=Quantity(float(dur_ms), 'ms'),
    dt=Quantity(float(dt_ms), 'ms'),
    empty=False,
    store_magnetization=True,
  )


def make_hard_pulse_block(flip_rad: float, dur_ms: float = 0.5,
                          scanner: Scanner | None = None,
                          dt_ms: float = 0.005) -> SequenceBlock:
  """A single hard (rectangular) RF pulse delivering the requested
  flip angle in ``dur_ms`` milliseconds, on resonance, with no
  gradient. The RF amplitude is chosen analytically via
  ``flip = gamma * B1 * dur`` so the rotation is exact."""
  if scanner is None:
    scanner = Scanner()
  rf = RF(
    scanner=scanner,
    shape='hard',
    flip_angle=Quantity(float(flip_rad), 'rad'),
    dur=Quantity(float(dur_ms), 'ms'),
    nb_samples=64,
  )
  return SequenceBlock(
    rf_pulses=[rf],
    dur=Quantity(float(dur_ms), 'ms'),
    dt_rf=Quantity(float(dt_ms), 'ms'),
    dt=Quantity(float(dur_ms), 'ms'),
    store_magnetization=True,
  )


def make_single_block_sequence(block: SequenceBlock) -> Sequence:
  """Wrap a single SequenceBlock in a fresh Sequence."""
  seq = Sequence()
  seq.add_block(block)
  return seq
