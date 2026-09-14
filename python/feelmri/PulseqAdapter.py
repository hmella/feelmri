"""
Reader and FEelMRI bridge for Pulseq `.seq` files.

The file format is a flat list of sections: a version, a table of definitions,
a block table indexing into per-event libraries, and a shape library holding
run-length-compressed waveforms. Reading it is therefore a two-stage job --
parse each section into its library, then resolve every block's event ids
against those libraries -- and this module keeps that separation.

Format versions 1.2 through 1.5 are supported. The RF, ADC and
arbitrary-gradient rows gained columns at 1.5, so those three reads dispatch on
the declared version and assert the row length afterwards; a file whose header
disagrees with its rows fails loudly rather than drifting silently by a column.

The section readers are `read_version`, `read_definitions`, `read_signature`,
`read_blocks`, `read_events`, `read_labels`, `read_extension_blocks` and
`read_shapes`; `compress_shape` / `decompress_shape` implement the format's
derivative-plus-repeat-count encoding. `read_Grad`, `read_RF`, `read_ADC` and
`read_extension` turn a library row into an event, and `read_seq` drives the
whole read and returns a `PulseqSequence`. `import_pulseq` is the entry point
that converts one into the `feelmri.Bloch` objects the solver takes.

The event types defined here (`Grad`, `RF`, `ADC`, `Trigger`, `LabelSet`,
`LabelInc`) carry only what the format stores, and stay separate from the
`feelmri.MRObjects` classes they are converted into.
"""

from __future__ import annotations

import cmath
import logging
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pint import Quantity

import numpy as np
import matplotlib.pyplot as plt

from feelmri.Bloch import _collective_raise
from feelmri.Bloch import ADC as feelmriADC
from feelmri.Bloch import Sequence as feelmriSequence
from feelmri.Bloch import SequenceBlock
from feelmri.MRObjects import RF as feelmriRF, Gradient, Scanner

# The `.seq` format reader. Re-exported so every existing
# `from feelmri.PulseqAdapter import read_seq` (and the parser internals the
# tests reach for) keeps resolving; this module is the feelmri-facing half.
from feelmri.PulseqFile import (  # noqa: F401
    ADC, GAMMA, Grad, LabelInc, LabelSet, PulseqSequence, RF, Rotation,
    Trigger, Version, _apply_rotation_to_grads, _grad_corners_seconds,
    _rotate_on_union_grid, _section_rows, _to_float_or_str,
    _warn_discontinuous_gradients, compress_shape, decompress_shape, dur_adc,
    dur_grad, dur_rf, fix_first_last_grads, read_ADC, read_Grad, read_RF,
    read_blocks, read_definitions, read_events, read_extension,
    read_extension_blocks, read_labels, read_seq, read_shapes, read_signature,
    read_version, skip_section, _shaped_waveform_seconds,
    _trap_waveform_seconds)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional 'pypulseq' dependency
# ---------------------------------------------------------------------------
#
# Only the k-space trajectory bridge functions require pypulseq; the rest of
# this adapter (parser, block conversion, label-driven partitioning) is
# pure-Python. Keep the import lazy and gated so 'import feelmri' works
# without pypulseq installed and so users hit a clear error message when
# they call a function that genuinely needs it.

def _require_pypulseq(feature: str):  # pragma: no cover (optional dep)
  """Import pypulseq on demand, raising a clear ImportError if absent.

  Parameters
  ----------
  feature : str
      Short name of the calling feature, used in the error message so the
      user knows which call site asked for pypulseq.

  Returns
  -------
  module
      The imported ``pypulseq`` module.
  """
  try:
    import pypulseq as pp
  except ImportError as exc:
    raise ImportError(
      f"{feature} requires the optional 'pypulseq' package. Install it with "
      f"'pip install pypulseq' (FEelMRI supports v1.2 through v1.5; the "
      f"'feelmri[pulseq]' extra installs a compatible version)."
    ) from exc
  return pp




# ---------------------------------------------------------------------------
# Convert PulseqSequence blocks to feelMRI objects
# ---------------------------------------------------------------------------



def _ppm_to_hz(scanner: Scanner) -> float:
  """Hz per ppm of the Larmor frequency, i.e. ``gammabar * B0``.

  Pulseq v1.5 stores RF and ADC offsets both in Hz (``freq``) and in ppm
  (``freq_ppm``); the effective offset is the sum, with the ppm term scaled
  by the Larmor frequency. This mirrors pypulseq's
  ``Sequence.waveforms_and_times`` (``freq_ppm * 1e-6 * gamma * B0``). B0 is
  NOT stored in the ``.seq`` file -- both libraries take it from the scanner
  definition -- so pass the right :class:`~feelmri.MRObjects.Scanner` to
  :func:`import_pulseq` when reading a sequence written for a field other
  than the 1.5 T default, or every ppm offset is scaled wrongly.
  """
  return float(scanner.gammabar.m_as('Hz/T') * scanner.field_strength.m_as('T')) * 1e-6


def _convert_gradient(g: "Grad", axis: int, scanner: Scanner) -> Optional[Gradient]:
  """Convert a parsed Pulseq Grad (trap or arbitrary) to a feelmri.Gradient.

  Returns None for zero-amplitude events. Output uses (ms, mT/m) units.
  """
  is_shaped = isinstance(g.A, np.ndarray)
  if is_shaped:
    if g.A.size == 0 or not np.any(np.abs(g.A) > 0.0):
      return None
    timings_s, amps_Tm = _shaped_waveform_seconds(g)
  else:
    if float(g.A) == 0.0:
      return None
    timings_s, amps_Tm = _trap_waveform_seconds(g)

  return Gradient(
    timings=Quantity(timings_s * 1e3, 'ms'),
    amplitudes=Quantity(amps_Tm, 'T/m').to('mT/m'),
    scanner=scanner,
    ref=Quantity(0.0, 'ms'),
    time=Quantity(0.0, 'ms'),
    axis=axis,
  )


def _convert_rf(rf: "RF", scanner: Scanner) -> Optional[feelmriRF]:
  """Convert a parsed Pulseq RF to a feelmri RF with custom waveform."""
  waveform = np.asarray(rf.waveform, dtype=np.complex128)
  if waveform.size == 0 or (waveform.size == 1 and np.abs(waveform[0]) == 0.0):
    return None
  if not np.any(np.abs(waveform) > 0.0):
    return None

  delay_s = float(rf.delay)
  if isinstance(rf.T, np.ndarray):
    dwells = np.asarray(rf.T, dtype=float)
    local = np.concatenate(([0.0], np.cumsum(dwells)))
    if local.size != waveform.size:
      n = min(local.size, waveform.size)
      local = local[:n]
      waveform = waveform[:n]
  else:
    total = float(rf.T)
    local = np.linspace(0.0, total, waveform.size)
  timings_s = delay_s + local

  duration_ms = float(timings_s[-1] - timings_s[0])
  if duration_ms <= 0.0:
    duration_ms = 1e-3
  duration_ms *= 1e3

  # v1.5 ppm offsets, scaled by the Larmor frequency. The .seq header gives
  # freq_ppm in ppm and phase_ppm in rad/MHz, so both take the same
  # gammabar*B0 factor: rad/MHz x MHz is rad. This matches pypulseq's
  # sequence.py, which is the path calculate_kspace and waveforms_and_times
  # take; its seq_plot.py drops the gamma factor and is the odd one out.
  ppm_to_hz = _ppm_to_hz(scanner)
  freq_hz = float(rf.df) + float(getattr(rf, 'freq_ppm', 0.0)) * ppm_to_hz
  phase_rad = float(getattr(rf, 'phase_ppm', 0.0)) * ppm_to_hz

  return feelmriRF(
    scanner=scanner,
    shape='custom',
    flip_angle=Quantity(np.pi / 2.0, 'rad'),
    dur=Quantity(duration_ms, 'ms'),
    ref=Quantity(0.0, 'ms'),
    time=Quantity(0.0, 'ms'),
    timings=Quantity(timings_s * 1e3, 'ms'),
    waveform=Quantity(waveform, 'T').to('mT'),
    frequency_offset=Quantity(freq_hz, 'Hz'),
    # The per-sample RF phase and the file's constant `phase` column are
    # already folded into the complex waveform by read_RF, so only the ppm
    # term is left to add here.
    phase_offset=Quantity(phase_rad, 'rad'),
    use=str(getattr(rf, 'use', 'undefined')),
  )


def _convert_adc(adc: "ADC", scanner: Scanner) -> Optional[feelmriADC]:
  """Convert a parsed ADC event to a feelmri.Bloch.ADC.

  Returns None when the ADC is inactive (num == 0). The v1.5 ppm offsets are
  folded into the Hz / rad offsets exactly as for RF, and the per-sample
  phase-modulation shape (the ``phase_id`` column) is carried through. All
  three are demodulation parameters: the solver does not sample the ADC, so
  it is the caller reconstructing from ``Phantom.mri_signal`` that applies
  them.
  """
  num = int(adc.num)
  if num <= 0:
    return None
  delay_s = float(adc.delay)
  T = float(adc.T)
  if num == 1:
    times_s = np.array([delay_s], dtype=float)
  else:
    dwell = T / (num - 1)
    times_s = delay_s + np.arange(num, dtype=float) * dwell
  ppm_to_hz = _ppm_to_hz(scanner)
  return feelmriADC(
    times_s * 1e3,
    freq_offset=Quantity(
      float(adc.df) + float(getattr(adc, 'freq_ppm', 0.0)) * ppm_to_hz, 'Hz'),
    phase_offset=Quantity(
      float(adc.phase) + float(getattr(adc, 'phase_ppm', 0.0)) * ppm_to_hz, 'rad'),
    phase_modulation=getattr(adc, 'phase_modulation', None),
  )


# ---------------------------------------------------------------------------
# K-space trajectory extraction
# ---------------------------------------------------------------------------

def pypulseq_version() -> Optional[Version]:
  """Version of the installed pypulseq, or None when it is not importable."""
  try:
    import pypulseq as pp
  except ImportError:
    return None
  raw = str(getattr(pp, '__version__', '') or '')
  parts = raw.split('.')[:3]
  try:
    nums = [int(''.join(c for c in p if c.isdigit()) or 0) for p in parts]
  except ValueError:  # pragma: no cover - defensive
    return None
  while len(nums) < 3:
    nums.append(0)
  return Version(*nums[:3])


def pypulseq_can_read(file_version: Version) -> bool:
  """Whether the installed pypulseq can read a file of this Pulseq version.

  pypulseq reads only formats up to its own, and the v1.5 layout is not
  backward readable: 1.4.x silently mis-parses a v1.5 file and then fails
  inside ``calculate_kspace``. FEelMRI's own reader handles both, but the
  trajectory comes from pypulseq, so a v1.5 file with an ADC needs
  pypulseq >= 1.5.
  """
  installed = pypulseq_version()
  if installed is None:
    return False
  return (installed.major, installed.minor) >= (file_version.major,
                                                file_version.minor)


def _read_with_pypulseq(filename, scanner: Optional[Scanner] = None):  # pragma: no cover (optional dep)
  """Read ``filename`` into a ``pp.Sequence``.

  One read serves three purposes -- the k-space trajectory, ``check_timing``
  and the excitation / refocusing anchor times -- and costs milliseconds even
  on a 231-block file, so it is not worth doing more than once.

  ``scanner`` supplies the transmit/receive dead times, which the file does not
  record. They must be set at CONSTRUCTION: pypulseq copies
  ``system.rf_dead_time`` / ``rf_ringdown_time`` onto every RF event as it is
  built and appends ``adc_dead_time`` into the ADC library during ``read``, so
  patching the ``Opts`` afterwards reaches nothing. With all three at their
  default zero this is identical to a bare ``pp.Sequence()``, which is what the
  adapter used before.
  """
  pp = _require_pypulseq('import_pulseq')
  if scanner is None:
    pp_seq = pp.Sequence()
  else:
    pp_seq = pp.Sequence(system=pp.Opts(
        rf_dead_time=float(scanner.rf_dead_time.m_as('s')),
        rf_ringdown_time=float(scanner.rf_ringdown_time.m_as('s')),
        adc_dead_time=float(scanner.adc_dead_time.m_as('s')),
    ))
  pp_seq.read(str(filename), detect_rf_use=False)
  return pp_seq


def _calculate_kspace_via_pypulseq(  # pragma: no cover (optional dep)
    filename,
) -> Tuple[np.ndarray, np.ndarray]:
  """Read ``filename`` with pypulseq and return the ADC-sample trajectory.

  Returns
  -------
  k_traj_adc : np.ndarray, shape ``(3, N_adc)``
      Gradient-moment-integrated k-space at every ADC sample, in 1/m.
  t_adc : np.ndarray, shape ``(N_adc,)``
      Absolute sequence times of those samples, in **seconds**.

  pypulseq's ``Sequence.calculate_kspace`` is authoritative for v1.5
  semantics: it resets k=0 at every RF pulse tagged
  ``use ∈ {'excitation', 'undefined'}`` and applies the spin-echo
  reflection ``k → −2 k_at_pulse − k`` at ``use == 'refocusing'``.
  Re-implementing this locally would force us to duplicate the
  use-label bookkeeping that pypulseq already encodes correctly.
  """
  k_traj_adc, _k_full, _t_exc, _t_ref, t_adc = (
      _read_with_pypulseq(filename).calculate_kspace())
  return (
      np.asarray(k_traj_adc, dtype=float),
      np.asarray(t_adc, dtype=float),
  )


def kspace_trajectory(pulseq_seq: "PulseqSequence") -> Dict[str, np.ndarray]:
  """Compute the flat k-space trajectory across all ADC samples.

  Returns a dict with arrays ``kx, ky, kz`` (1/m) and ``times`` (ms)
  of shape ``(N_total_adc_samples,)`` in sequence order. The function
  re-opens the underlying .seq file via pypulseq using the path stored
  on ``pulseq_seq.DEF['__pulseq_path__']`` (set by :func:`read_seq`).

  Sequences with zero ADC blocks short-circuit to empty arrays —
  pypulseq can fail on synthetic extensions in test fixtures, and there
  is no trajectory to compute anyway.
  """
  filename = pulseq_seq.DEF.get('__pulseq_path__')
  has_adc = any(int(a.num) > 0 for a in pulseq_seq.ADC)
  if filename is None or not has_adc:
    empty = np.array([], dtype=float)
    return {'kx': empty, 'ky': empty, 'kz': empty, 'times': empty}

  k_traj_adc, t_adc = _calculate_kspace_via_pypulseq(filename)
  return {
    'kx': k_traj_adc[0],
    'ky': k_traj_adc[1],
    'kz': k_traj_adc[2],
    'times': t_adc * 1e3,
  }


def _reshape_signal_inputs(kx, ky, kz, times_ms, shape):
  """Cast a flat trajectory to the rank-3 float32 tensors mri_signal takes.

  The C++ ``SignalAssembler`` kernels want ``kloc`` as
  ``std::vector<Eigen::Tensor<float, 3>>`` and ``t`` as one rank-3 float
  tensor of matching shape ``(nb_meas, nb_lines, nb_kz)``, so the flat
  arrays need a reshape and a dtype / contiguity cast. ``shape`` defaults to
  ``(N, 1, 1)``, the per-readout pattern used everywhere else in FEelMRI.
  """
  n = int(np.asarray(times_ms).size)
  if shape is None:
    shape = (n, 1, 1)
  shape = tuple(int(d) for d in shape)
  if int(np.prod(shape)) != n:
    raise ValueError(
      f'requested shape {shape} (prod={int(np.prod(shape))}) does not '
      f'match N={n} ADC samples'
    )
  points = tuple(
    np.ascontiguousarray(np.asarray(a).reshape(shape), dtype=np.float32)
    for a in (kx, ky, kz)
  )
  times = np.ascontiguousarray(
    np.asarray(times_ms).reshape(shape), dtype=np.float32)
  return points, times


def as_signal_inputs(traj: Dict[str, np.ndarray],
                     shape: Optional[Tuple[int, ...]] = None
                     ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                np.ndarray]:
  """Format a flat :func:`kspace_trajectory` dict for ``Phantom.mri_signal``.

  The C++ ``SignalAssembler`` kernels in ``cpp/feelmri/MRIAssemble.cpp``
  require their ``kloc`` argument as ``std::vector<Eigen::Tensor<float, 3>>``
  and the ``t`` argument as a single rank-3 float tensor of matching
  shape ``(nb_meas, nb_lines, nb_kz)``. The flat 1-D arrays returned by
  :func:`kspace_trajectory` therefore need a reshape + dtype/contiguity
  cast before reaching ``mri_signal``. This helper centralises that
  contract.

  Parameters
  ----------
  traj : dict
      Output of :func:`kspace_trajectory`. Keys ``kx``, ``ky``, ``kz``,
      ``times`` are each a 1-D ndarray of length ``N``.
  shape : tuple of int, optional
      Desired rank-3 shape ``(nb_meas, nb_lines, nb_kz)``. When
      ``None`` (default) the helper uses ``(N, 1, 1)``, matching the
      canonical per-readout pattern used elsewhere in FEelMRI. The
      product of ``shape`` must equal ``N``.

  Returns
  -------
  (kspace_points, kspace_times) : tuple
      ``kspace_points`` is a 3-tuple of float32 C-contiguous rank-3
      ndarrays (one per axis); ``kspace_times`` is a single float32
      C-contiguous rank-3 ndarray of the same shape.
  """
  return _reshape_signal_inputs(
      traj['kx'], traj['ky'], traj['kz'], traj['times'], shape)


def kspace_to_signal_inputs(  # pragma: no cover (optional dep)
    pp_seq,
    shape: Optional[Tuple[int, int, int]] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
  """Convert ``pp.Sequence.calculate_kspace`` output into the rank-3
  float32 tensors expected by ``feelmri.Phantom.FEMPhantom.mri_signal``.

  This is the explicit bridge between pypulseq's k-space helper and
  FEelMRI's signal assembler. It calls
  ``pp_seq.calculate_kspace()`` internally, takes the ADC samples
  (``k_traj_adc`` shape ``(3, N)`` in 1/m, ``t_adc`` shape ``(N,)`` in
  seconds), and returns them reshaped, cast to float32, made
  C-contiguous, with times converted to ms — the unit FEelMRI's
  ``mri_signal`` expects (see ``examples/phase_contrast.py:212`` and
  related call sites, all of which pass ``traj.times.m_as('ms')``).

  Parameters
  ----------
  pp_seq : pypulseq.Sequence
      A pypulseq Sequence on which ``calculate_kspace`` will be called.
      Pre-populated (built in Python) or freshly read from a .seq file.
  shape : (nb_meas, nb_lines, nb_kz), optional
      Rank-3 output shape. When ``None`` (default) the helper uses
      ``(N, 1, 1)``. The product must equal ``N`` (total ADC samples).

  Returns
  -------
  (kspace_points, kspace_times) : tuple
      ``kspace_points`` is a 3-tuple of C-contiguous rank-3 float32
      arrays for (kx, ky, kz) in 1/m. ``kspace_times`` is a single
      C-contiguous rank-3 float32 array in **ms**.
  """
  # pp_seq carries methods from the optional 'pypulseq' package; ensure it
  # is importable here so the user gets the same actionable message as
  # _calculate_kspace_via_pypulseq instead of an AttributeError further down.
  _require_pypulseq('kspace_to_signal_inputs')
  k_traj_adc, _k_full, _t_exc, _t_ref, t_adc = pp_seq.calculate_kspace()
  k_traj_adc = np.asarray(k_traj_adc, dtype=float)
  t_adc = np.asarray(t_adc, dtype=float)
  return _reshape_signal_inputs(
      k_traj_adc[0], k_traj_adc[1], k_traj_adc[2], t_adc * 1e3, shape)


# ---------------------------------------------------------------------------
# Partitioned import: dual-path workflow (Bloch prep + signal assembly)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadoutWindow:
  """One contiguous span of ADC blocks plus its trajectory and the
  index of the magnetization snapshot taken at the coherence-anchor
  RF pulse for the group.

  Attributes
  ----------
  first_block, last_block : int
      Inclusive range of ADC-bearing block indices grouped under a
      common coherence anchor. The range may include non-ADC blocks
      (phase-encode blips, spoilers) interleaved with the ADCs.
  m_storage_block : int
      Block index of the coherence anchor (most recent RF block with
      ``use ∈ {'excitation', 'refocusing', 'undefined'}``).
      ``store_magnetization`` is set to True on that block. ``-1`` when
      no use-labeled anchor exists; in that case the legacy
      "last non-ADC block strictly before the first ADC" rule is used.
  m_storage_idx : int
      Column index into ``BlochSolver.solve()``'s Mxy/Mz output
      (position of this readout's anchor among all blocks flagged
      ``store_magnetization``, in sequence order). ``-1`` mirrors
      ``m_storage_block == -1``.
  kspace : np.ndarray
      Shape (N, 3) float32 array of (kx, ky, kz) at every ADC sample
      inside the window, in 1/m.
  times : np.ndarray
      Shape (N,) float64 absolute sequence time of each ADC sample, in ms.
  adc_freq_offset : float
      Hz; constant within window (assumed identical across blocks).
  adc_phase_offset : float
      Rad; constant within window.
  adc_phase_modulation : np.ndarray or None
      Per-sample phase (rad) from the leading block's Pulseq v1.5 phase
      shape, or None. All three are demodulation parameters and are applied
      by the caller, not by the solver.
  kspace_file : np.ndarray
      The trajectory exactly as ``calculate_kspace`` reports it, measured
      from the excitation. ``kspace`` is this minus ``k_at_anchor``.
  k_at_anchor : np.ndarray, shape (3,)
      Gradient moment (1/m) already carried by the magnetization at
      ``m_storage_block``. It is subtracted from ``kspace`` because the
      assembler would otherwise wind it a second time.
  maxwell : np.ndarray or None
      Shape ``(N, 4)`` float64 of time-integrated gradient products in
      ``(mT/m)^2 ms``, integrated FORWARD from ``t_anchor`` -- the concomitant
      counterpart of ``kspace``. Unlike ``kspace`` there is no anchor value to
      subtract -- the origin is ours to choose, and the only correct one is the
      snapshot, since everything before it is already carried on the
      magnetization by the solver.

      ``maxwell[0]`` is therefore **not** generally zero: whatever plays
      between the anchor and the first ADC sample, typically a prephaser,
      belongs to the readout and is counted. Measured on the bundled fixtures,
      the first sample is non-zero on ``gre_v15`` and ``epi_v142`` and zero on
      ``cpmg_v15``, whose first sample sits at the anchor. That mirrors
      ``kspace``, which also starts away from the origin for the same reason.

      Pass it through :func:`maxwell_phase_coefficients` to get what
      ``mri_signal`` takes.
  t_anchor : float
      Absolute time (ms) of the snapshot, i.e. the end of the anchor block.
      ``times - t_anchor`` is the elapsed time the assembler's
      ``exp(-t/T2)`` and ``exp(i*phi*t)`` expect.
  """
  first_block: int
  last_block: int
  m_storage_block: int
  m_storage_idx: int
  kspace: np.ndarray
  times: np.ndarray
  adc_freq_offset: float
  adc_phase_offset: float
  adc_phase_modulation: Optional[np.ndarray] = None
  kspace_file: Optional[np.ndarray] = None
  k_at_anchor: Optional[np.ndarray] = None
  t_anchor: float = 0.0
  maxwell: Optional[np.ndarray] = None

  def demodulation_phase(self) -> np.ndarray:
    """Receiver phase Pulseq specifies for this window's samples, in rad.

    Delegates to :func:`feelmri.Bloch.demodulation_phase`, which is also what
    :meth:`feelmri.Bloch.ADC.demodulate` uses, so the window-level and
    block-level paths cannot drift apart.

    An EPI train collapses many ADC events into one window and the window
    carries the offsets of its HEAD event only, so this is exact just when
    every event shares them -- the usual case, since they come from one
    `make_adc` call. The origin is this window's first sample, while Pulseq
    references each ADC event's own start; for a train with a non-zero
    frequency offset that difference grows along the echo train.
    """
    from feelmri.Bloch import demodulation_phase as _phase
    return _phase(self.times, self.adc_freq_offset, self.adc_phase_offset,
                  self.adc_phase_modulation)

  def demodulate(self, signal: np.ndarray) -> np.ndarray:
    """Apply :meth:`demodulation_phase` to a signal sampled on this window.

    The solver never samples the ADC -- the readout is synthesized from the
    trajectory -- so the receiver's offsets are applied here or not at all.
    `simulate_pulseq` calls this for you; a caller driving
    `update_magnetization` + `mri_signal` by hand must call it explicitly, or
    use :meth:`feelmri.Bloch.ADC.demodulate` on the block's own ADC.
    """
    from feelmri.Bloch import apply_demodulation
    return apply_demodulation(signal, self.demodulation_phase())


@dataclass(frozen=True)
class PulseqImport:
  """Result of :func:`import_pulseq`. Exposes both the conventional
  feelmri Sequence (for direct ``BlochSolver`` consumption) and a
  partitioned view that splits prep blocks from ADC readout windows
  for use with the ``Phantom.update_magnetization`` /
  ``Phantom.mri_signal`` dual-path workflow.

  ``prep_storage_blocks`` lists the block indices flagged
  ``store_magnetization=True`` because they immediately follow a
  ``use='preparation'`` RF pulse. They give callers (notably
  myocardial-tagging readouts) a way to snapshot Mz after the prep even
  when no ADC window immediately follows. The corresponding
  ``prep_storage_indices`` mirror :attr:`ReadoutWindow.m_storage_idx`
  for indexing into ``BlochSolver.solve()``'s output Mxy/Mz columns.

  ``block_labels`` carries the per-block running state of every
  ``LABELSET`` / ``LABELINC`` extension encountered in the file (e.g.
  the user's ``make_label(label='SET', type='SET', value=A)`` tags).
  Use :meth:`filter_blocks` to query block indices that match a
  particular label state.

  ``timing_errors`` holds whatever ``pp.Sequence.check_timing`` reported
  for the file, as strings, and is empty both when the file is clean and
  when the check could not run (no pypulseq, or a file pypulseq refuses).
  The errors are raster and dead-time violations the scanner would reject;
  they are surfaced rather than raised because a file that fails them still
  simulates.

  ``hardware_problems`` holds whatever :meth:`Sequence.check_hardware`
  reported against the scanner -- gradient amplitude, slew rate and peak
  B1 -- and is empty when the sequence is within spec. Like
  ``timing_errors`` it is surfaced rather than raised: an over-spec file
  still simulates, and the numbers are what the file asks for, not what
  the simulation gets wrong. It stays a warning for a concrete reason --
  ``examples/pulseq/epi_pypulseq.seq`` sits at 98.4% of the default slew
  limit, so a raise would be one rounding change from rejecting a file
  that has always worked.

  ``feelmri_sim_seq`` is a parallel :class:`feelmriSequence` with the
  same block count and indices as ``feelmri_seq``, except that every
  block whose running ``SET`` label value matches the
  ``readout_set_values`` argument to :func:`import_pulseq` is replaced
  by an empty :class:`SequenceBlock` of identical duration. Storage
  flags (``store_magnetization=True``) on non-readout blocks are
  preserved verbatim, so :attr:`ReadoutWindow.m_storage_idx` columns
  remain valid against ``feelmri_sim_seq``. The list
  ``readout_sim_block_indices`` enumerates which block indices were
  collapsed (one entry per replaced block).
  """
  feelmri_seq: feelmriSequence
  pulseq_seq: PulseqSequence
  feelmri_sim_seq: feelmriSequence
  readouts: List[ReadoutWindow]
  prep_block_indices: List[int]
  adc_block_indices: List[int]
  prep_storage_blocks: List[int]
  prep_storage_indices: List[int]
  block_labels: List[Dict[str, int]]
  readout_sim_block_indices: List[int]
  timing_errors: Tuple[str, ...] = ()
  hardware_problems: Tuple[str, ...] = ()

  def filter_blocks(self, **labels: int) -> List[int]:
    """Return block indices whose running LABELSET/LABELINC state matches
    all given keyword constraints.

    Example
    -------
    Tag every readout-group boundary with
    ``pp.make_label(label='SET', type='SET', value=1)`` in the write
    script, then::

        imp = import_pulseq(path)
        readout_blocks = imp.filter_blocks(SET=1)

    AND semantics: multiple kwargs must all match
    (``imp.filter_blocks(SET=1, SLC=0)`` matches blocks where SET == 1
    *and* SLC == 0). Missing labels never match.
    """
    if not labels:
      return list(range(len(self.block_labels)))
    matches: List[int] = []
    for i, state in enumerate(self.block_labels):
      ok = True
      for key, want in labels.items():
        if state.get(key) != want:
          ok = False
          break
      if ok:
        matches.append(i)
    return matches

  @staticmethod
  def contiguous_groups(indices: List[int]) -> List[List[int]]:
    """Split block indices into runs of consecutive integers.

    A writer emits the blocks of one shot or one slice adjacently, so a
    label query returns several such runs concatenated:
    ``filter_blocks(SET=2)`` on a two-slice sequence gives
    ``[4, 5, 11, 12]`` and this returns ``[[4, 5], [11, 12]]``. Take
    ``[0]`` for the first shot without needing to know the slice count.
    """
    groups: List[List[int]] = []
    for i in sorted(indices):
      if groups and i == groups[-1][-1] + 1:
        groups[-1].append(i)
      else:
        groups.append([i])
    return groups

  def duration_of(self, indices: List[int], *, sim: bool = True) -> Quantity:
    """Total duration of the given blocks, in ms.

    Use it to size a placeholder delay that stands in for blocks whose
    physics is not being evolved -- a readout train during steady-state
    convergence, say -- so the timeline still adds up.
    """
    seq = self.feelmri_sim_seq if sim else self.feelmri_seq
    total = Quantity(0.0, 'ms')
    for i in indices:
      total = total + seq.blocks[i].dur.to('ms')
    return total

  def copy_block(self, index: int, *, sim: bool = True,
                 store_magnetization: bool = False,
                 spoiler: bool = False) -> SequenceBlock:
    """Return an independent copy of one block, ready to be assembled into
    a hand-built :class:`~feelmri.Bloch.Sequence`.

    ``store_magnetization`` is set explicitly rather than inherited: the
    import stamps it on prep and readout-anchor blocks for its own
    ``m_storage_idx`` bookkeeping, and a caller rebuilding the sequence
    wants its own snapshot points, not those. ``spoiler`` turns on the
    solver's multi-isochromat dephasing for the block.
    """
    seq = self.feelmri_sim_seq if sim else self.feelmri_seq
    block = seq.blocks[index].copy()
    block.store_magnetization = bool(store_magnetization)
    block.spoiler = bool(spoiler)
    return block



def _block_use_label(pulseq_seq: PulseqSequence, idx: int) -> Optional[str]:
  """Return the canonical 'use' label of the RF pulse on block ``idx``,
  or ``None`` when the block carries no active RF event."""
  rf = pulseq_seq.RF[idx]
  waveform = getattr(rf, 'waveform', None)
  if waveform is None:
    return None
  if isinstance(waveform, np.ndarray):
    if waveform.size == 0:
      return None
    if not np.any(np.abs(waveform) > 0.0):
      return None
  return str(getattr(rf, 'use', 'undefined'))


def _compute_block_labels(pulseq_seq: PulseqSequence) -> List[Dict[str, int]]:
  """Compute the running LABELSET / LABELINC state per block.

  Walks ``pulseq_seq.EXT`` in order. Each ``LabelSet(label, value)``
  overwrites the running entry; each ``LabelInc(label, value)`` adds
  to it (defaulting from 0). Blocks with no label extension inherit
  the previous block's dict unchanged.
  """
  n = len(pulseq_seq)
  out: List[Dict[str, int]] = []
  state: Dict[str, int] = {}
  for i in range(n):
    new_state = dict(state)
    for ext in pulseq_seq.EXT[i]:
      if isinstance(ext, LabelSet):
        new_state[str(ext.label)] = int(ext.value)
      elif isinstance(ext, LabelInc):
        new_state[str(ext.label)] = int(new_state.get(str(ext.label), 0)) + int(ext.value)
    out.append(new_state)
    state = new_state
  return out


def _gradient_moment_between(feelmri_seq, t0_ms: float, t1_ms: float,
                             gammabar_hz_per_t: float) -> np.ndarray:
  """Gradient moment accumulated over ``[t0_ms, t1_ms]``, in 1/m per axis.

  Integrates FEelMRI's own converted gradients rather than pypulseq's
  ``waveforms_and_times``: the latter concatenates each block's shape pieces
  with no padding between them, so interpolating across it silently bridges
  the gaps where no gradient is playing. Our blocks carry their own support
  and are zero outside it, so the sum is gap-free. The waveforms are
  piecewise linear, so evaluating at every corner inside the interval plus the
  two endpoints makes the trapezoid rule exact.
  """
  total = np.zeros(3, dtype=float)
  if t1_ms <= t0_ms:
    return total
  for block in feelmri_seq.blocks:
    if (block.time_extent[1].m_as('ms') <= t0_ms
        or block.time_extent[0].m_as('ms') >= t1_ms):
      continue
    for g in block.gradients:
      ts = g.timings.m_as('ms')
      amp = g.amplitudes.m_as('mT/m')
      lo = max(t0_ms, float(ts[0]))
      hi = min(t1_ms, float(ts[-1]))
      if hi <= lo:
        continue
      inner = ts[(ts > lo) & (ts < hi)]
      grid = np.concatenate(([lo], inner, [hi]))
      total[g.axis] += float(np.trapezoid(
          np.interp(grid, ts, amp, left=0.0, right=0.0), grid))
  # mT/m * ms -> T/m * s, then Hz/T -> 1/m
  return total * 1e-6 * gammabar_hz_per_t


def _flatten_gradients(source):
  """Every Gradient in a Sequence, or the iterable itself if it already is one.

  The moments need the SUMMED field per axis, so they are computed from one
  flat list rather than block by block: a gradient is zero outside its own
  support, so blocks that do not overlap contribute nothing and nothing has to
  be clipped.
  """
  blocks = getattr(source, 'blocks', None)
  if blocks is None:
    return list(source)
  return [g for block in blocks for g in block.gradients]


def _second_moment_increments(h, G):
  """Per-segment increments of the four concomitant moments.

  Exact on each piecewise-linear segment:
    integral A^2 dt  = h (A0^2 + A0 A1 + A1^2)/3
    integral A B dt  = h (2 A0 B0 + A0 B1 + A1 B0 + 2 A1 B1)/6

  ``h`` is the per-segment dwell and ``G`` the (N, 3) gradient at the segment
  corners, so ``h.size == G.shape[0] - 1``.
  """
  A0, A1 = G[:-1], G[1:]
  square = h[:, None] * (A0 * A0 + A0 * A1 + A1 * A1) / 3.0

  def _cross(i, j):
    return h * (2.0 * A0[:, i] * A0[:, j] + A0[:, i] * A1[:, j]
                + A1[:, i] * A0[:, j] + 2.0 * A1[:, i] * A1[:, j]) / 6.0

  increment = np.empty((h.size, 4), dtype=float)
  increment[:, 0] = square[:, 0] + square[:, 1]
  increment[:, 1] = square[:, 2]
  increment[:, 2] = _cross(0, 2)
  increment[:, 3] = _cross(1, 2)
  return increment


def maxwell_moments(source, t0_ms: float, sample_times_ms,
                    rotation=None) -> np.ndarray:
  """Time-integrated gradient products that drive the concomitant field.

  The Maxwell term is

      Bc = [(Gx^2+Gy^2) z^2 + (Gz^2/4)(x^2+y^2) - Gx Gz x z - Gy Gz y z] / (2 B0)

  a quadratic form in position whose coefficients depend only on time. Its time
  integral therefore FACTORISES exactly into four scalars times four fixed
  spatial monomials -- checked against :func:`feelmri.Bloch._concomitant_mT` on
  an oblique, time-varying waveform to 1.0e-14 relative:

      column 0: integral (Gx^2 + Gy^2) dt   multiplies z^2
      column 1: integral (Gz^2) dt          multiplies (x^2 + y^2)/4
      column 2: integral (Gx Gz) dt         multiplies -x z
      column 3: integral (Gy Gz) dt         multiplies -y z

  So this is four numbers per sample playing exactly the role the three
  components of ``k`` play for the linear term: a second trajectory, not a
  second field model. :func:`maxwell_phase_coefficients` turns them into the
  rad/m^2 the assembler wants.

  ``rotation`` maps the gradients into the PHYSICAL frame, in the sense
  ``G_physical = rotation @ G_logical``. The products below are B0-aligned, so
  a trajectory whose gradients are defined along logical (readout / phase /
  slice) axes must supply it or the squares are formed on the wrong axes. The
  same matrix then goes to :func:`maxwell_phase_coefficients`, which handles
  the matching rotation of the positions.

  ``source`` is a :class:`feelmri.Bloch.Sequence` or any iterable of
  :class:`feelmri.MRObjects.Gradient` -- the latter is what the native
  trajectory classes hand over, since they carry gradients without a sequence.

  Returns ``(N, 4)`` float64 in ``(mT/m)^2 ms``, integrated FORWARD from
  ``t0_ms``, so ``out[0]`` is zero whenever the first sample sits at ``t0_ms``.
  There is no anchor correction to make: unlike ``k``, which pypulseq measures
  from the excitation, this origin is ours to choose, and the only correct one
  is the magnetization snapshot -- everything before it is already carried on
  the magnetization by the solver.

  Three traps, each of which silently returns a plausible wrong answer:

  * **Sum each axis BEFORE squaring.** ``_gradient_moment_between`` may
    accumulate per gradient object because the integral is linear; here
    ``integral (Gx + Gx')^2 != integral Gx^2 + integral Gx'^2``. The cross
    terms cannot be decomposed per axis at all, which is why the grid is
    unioned across axes rather than built per gradient.
  * **Do not reach for pypulseq's ``waveforms_and_times``.** It concatenates
    each block's shape pieces with no padding, so interpolating across it
    bridges the gaps where nothing is playing -- measured at 1.14 1/m on the
    linear moment of ``arb_v15``, and one-signed here.
  * **The sample times must be ON the grid.** Between corners the cumulative
    moment is quadratic, so interpolating it afterwards is wrong at second
    order. They are unioned in rather than looked up.

  The segment rule is exact rather than sampled, which is what lets this avoid
  the raster error the solver has to sub-sample around (see
  ``Bloch.CONCOMITANT_DT_GR_MS``): over a ramp the trapezoid rule charges
  ``A^2 h/2`` where the exact second moment is ``A^2 h/3``, a 50% over-count.
  """
  times = np.asarray(sample_times_ms, dtype=float).reshape(-1)
  if times.size == 0:
    return np.zeros((0, 4), dtype=float)
  if times.min() < t0_ms:
    raise ValueError(
        f"maxwell_moments: sample time {times.min()} ms precedes the origin "
        f"{t0_ms} ms. The moments are integrated forward from the snapshot, "
        f"so a sample before it has no meaning.")
  t_last = float(times.max())
  if t_last <= t0_ms:
    return np.zeros((times.size, 4), dtype=float)

  gradients = _flatten_gradients(source)

  # The union grid: the origin, every sample, and every gradient corner that
  # falls inside. Adding corners to a piecewise-linear waveform is lossless,
  # and it is what makes the closed forms below exact.
  corners = [np.array([float(t0_ms), t_last]), times]
  for g in gradients:
    corners.append(np.asarray(g.timings.m_as('ms'), dtype=float))
  grid = np.unique(np.concatenate(corners))
  grid = grid[(grid >= t0_ms) & (grid <= t_last)]

  # The SUMMED amplitude per axis. Every gradient is zero outside its own
  # support (its interpolator is built with fill_value=0.0), so the sum is
  # gap-free and nothing needs clipping to a block.
  G = np.zeros((grid.size, 3), dtype=float)
  for g in gradients:
    ts = np.asarray(g.timings.m_as('ms'), dtype=float)
    amp = np.asarray(g.amplitudes.m_as('mT/m'), dtype=float)
    G[:, g.axis] += np.interp(grid, ts, amp, left=0.0, right=0.0)
  if rotation is not None:
    R = np.asarray(rotation, dtype=float)
    if R.shape != (3, 3):
      raise ValueError(f"maxwell_moments: rotation must be 3x3, got {R.shape}")
    G = G @ R.T

  h = np.diff(grid)
  increment = _second_moment_increments(h, G)

  cumulative = np.vstack((np.zeros((1, 4)), np.cumsum(increment, axis=0)))
  return cumulative[np.searchsorted(grid, times)]


def _maxwell_moments_array(moments, who: str) -> np.ndarray:
  """Validate and return the (N, 4) moment array."""
  m = np.asarray(moments, dtype=float)
  if m.ndim != 2 or m.shape[1] != 4:
    raise ValueError(f"{who}: expected (N, 4) moments, got {m.shape}")
  return m


def _maxwell_form(m: np.ndarray, scanner, who: str):
  """The concomitant quadratic form and its ``-gamma / (2 B0)`` scale.

  The sign convention, the 1/4 and the 1/B0 live here and nowhere else.
  ``who`` names the caller so each guard still reports itself: a refusal that
  cannot say which check fired is not coverage.

  Call this only where the caller has decided the term is live -- the B0
  refusal fires here, so hoisting it past an early return would refuse a
  configuration that previously did no work at all.
  """
  gamma = scanner.gamma.m_as('rad/ms/mT')
  B0 = scanner.field_strength.m_as('mT')
  if not B0 > 0:
    raise ValueError(
        f"{who}: the concomitant term scales as 1/B0, so a zero or negative "
        f"field strength ({scanner.field_strength}) is undefined, not merely "
        f"weak.")

  a, b, c, d = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
  form = np.zeros((m.shape[0], 3, 3), dtype=float)
  form[:, 0, 0] = form[:, 1, 1] = 0.25 * b
  form[:, 2, 2] = a
  form[:, 0, 2] = form[:, 2, 0] = -0.5 * c
  form[:, 1, 2] = form[:, 2, 1] = -0.5 * d
  return form, -gamma / (2.0 * B0)


def maxwell_phase_coefficients(moments, scanner, rotation=None) -> np.ndarray:
  """Turn :func:`maxwell_moments` output into the rad/m^2 the assembler adds.

  **The sign convention lives here and nowhere else.** The assembler ADDS

      p0 x^2 + p1 y^2 + p2 z^2 + p3 x y + p4 x z + p5 y z

  to its phase, so every sign, the factor of 4 on the transverse term and the
  1/B0 are folded in on this side. That keeps ``MRIAssemble.cpp`` free of any
  knowledge of B0, of the Maxwell expression, or of the imaging geometry --
  which matters, because three copies of that expression already exist (the
  kernel and the two Magnus seeds) and they must not drift apart.

  The solver precesses as ``exp(-i gamma Bz t)``, so the phase is
  ``-gamma * integral(Bc) dt``. ``Bc`` is a quadratic form in PHYSICAL
  position, ``x^T M x / (2 B0)``, with

      M = [[b/4,   0, -c/2],
           [  0, b/4, -d/2],
           [-c/2, -d/2,  a]]

  for the four moments ``(a, b, c, d)`` of :func:`maxwell_moments`.

  ``rotation`` is the 3x3 matrix relating the coordinates the assembler will
  use to the physical ones, in the sense ``x_physical = rotation @ x_assembler``
  -- which is what :meth:`FEMPhantom.orient` leaves behind, since it applies
  ``nodes @ MPS_ori``. Pass it whenever the phantom has been oriented, or the
  Maxwell expression is evaluated in the imaging frame and treats the slice
  normal as B0. With no rotation the form is B0-aligned: ``p0 == p1``, ``p3``
  is zero, and the six coefficients collapse to the four the field naturally
  has.
  """
  m = _maxwell_moments_array(moments, "maxwell_phase_coefficients")
  form, scale = _maxwell_form(m, scanner, "maxwell_phase_coefficients")
  if rotation is not None:
    R = np.asarray(rotation, dtype=float)
    if R.shape != (3, 3):
      raise ValueError(
          f"maxwell_phase_coefficients: rotation must be 3x3, got {R.shape}")
    # phi = x_phys^T M x_phys with x_phys = R x_assembler.
    form = np.einsum('ki,nkl,lj->nij', R, form, R)

  out = np.empty((m.shape[0], 6), dtype=float)
  out[:, 0] = scale * form[:, 0, 0]          # x^2
  out[:, 1] = scale * form[:, 1, 1]          # y^2
  out[:, 2] = scale * form[:, 2, 2]          # z^2
  out[:, 3] = scale * 2.0 * form[:, 0, 1]    # x y
  out[:, 4] = scale * 2.0 * form[:, 0, 2]    # x z
  out[:, 5] = scale * 2.0 * form[:, 1, 2]    # y z
  return out


def maxwell_recentre(moments, scanner, rotation=None, location=None):
  """The terms that move ``Bc`` from the slice centre back onto isocentre.

  :func:`maxwell_phase_coefficients` gives the QUADRATIC form only, and the
  assembler evaluates it at ``local_nodes`` -- which ``FEMPhantom.orient``
  measures from the slice centre. ``Bc`` is a quadratic form about ISOCENTRE, so
  an off-isocentre slab is otherwise imaged as though it sat in the middle of
  the bore. With ``x_physical = R u + L``,

      x^T F x = u^T (R^T F R) u  +  2 (R^T F L) . u  +  L^T F L

  the first term is what the six coefficients already carry, the second is
  linear in position -- so it is a k-space shift, exactly like a lab-frame field
  -- and the third a uniform phase.

  Returns ``(dk, phase)`` with the same convention as :func:`b0_kspace_shift`:
  ``dk`` is ``(N, 3)`` in 1/m and is ADDED to the samples, ``phase`` is ``(N,)``
  in rad and goes through :func:`feelmri.Bloch.apply_demodulation`.

  ``location`` is the slice offset in metres, in PHYSICAL coordinates -- the
  same vector :meth:`FEMPhantom.orient` was given. ``None`` or zero returns
  zeros, so an acquisition at isocentre pays nothing and stays bit-identical.
  """
  m = _maxwell_moments_array(moments, "maxwell_recentre")
  n = m.shape[0]
  if location is None:
    return np.zeros((n, 3)), np.zeros(n)
  L = np.asarray(location, dtype=float).reshape(3)
  if not np.any(L):
    return np.zeros((n, 3)), np.zeros(n)

  form, scale = _maxwell_form(m, scanner, "maxwell_recentre")

  # F L in physical coordinates, then into the frame the assembler's nodes are
  # in. Only the LINEAR term takes the rotation; the uniform one is a scalar.
  FL = np.einsum('nij,j->ni', form, L)
  if rotation is not None:
    R = np.asarray(rotation, dtype=float)
    if R.shape != (3, 3):
      raise ValueError(
          f"maxwell_recentre: rotation must be 3x3, got {R.shape}")
    FL = FL @ R
  lin = scale * 2.0 * FL                       # rad/m, added to the phase
  uniform = scale * np.einsum('i,nij,j->n', L, form, L)
  # The assembler carries -2 pi k . x in its phase, so a phase of `lin . x` is
  # a k-space offset of -lin / (2 pi); `apply_demodulation` applies exp(-i phi),
  # so the uniform term is handed over negated.
  return -lin / (2.0 * np.pi), -uniform


def b0_kspace_shift(field, times_ms, scanner, rotation=None, location=None):
  """The k-space shift a lab-frame B0 field puts on a readout.

  A field ``dB0 = b + g . x`` advances a spin at ``x`` by ``-gamma (b + g.x) t``.
  The position-dependent half is ``-2 pi (gammabar t g) . x``, which is exactly
  what moving the sample's k by ``gammabar t g`` does -- so it needs no
  assembler support, and the geometric distortion comes out of the
  reconstruction by itself. The uniform half is a per-sample phase and is
  returned alongside.

  Returns ``(dk, phase)``: ``dk`` has shape ``times_ms.shape + (3,)`` in 1/m and
  is ADDED to the k-space samples; ``phase`` has shape ``times_ms.shape`` in rad
  and goes through :func:`feelmri.Bloch.apply_demodulation`, which applies
  ``exp(-i phase)``.

  ``rotation`` and ``location`` describe the phantom the assembler holds. Its
  nodes are always in the imaging frame -- unlike the solver's, which are
  rotated into the physical frame when the concomitant term is on -- so the
  gradient is always taken as ``R^T g`` and the slice offset always lands in
  the constant.
  """
  if getattr(field, 'kind', None) == 'nodal':
    raise TypeError(
        "b0_kspace_shift: this field is a per-node expansion, and a k-space "
        "shift can only carry a field that is linear in position. The per-node "
        "part rides `phi_dB0` instead -- see `B0Field.readout_terms`.")
  b, g = field.in_frame(rotation=rotation, location=location, physical=False)
  t = np.asarray(times_ms, dtype=np.float64)
  gammabar = scanner.gammabar.m_as('1/ms/mT')
  gamma = scanner.gamma.m_as('rad/ms/mT')
  dk = (gammabar * t)[..., None] * g.reshape((1,) * t.ndim + (3,))
  return dk, gamma * b * t


def b0_readout_terms(field, times_ms, scanner, rotation=None, location=None):
  """Every channel a POLYNOMIAL lab-frame field needs on a readout.

  Returns ``(dk, phase, maxwell)``: the k-space offset to ADD to the samples,
  a per-sample phase in rad for :func:`feelmri.Bloch.apply_demodulation`, and
  ``(N, 6)`` quadratic coefficients to ADD to whatever the concomitant term
  already contributes -- ``None`` below degree 2.

  One call rather than :func:`b0_kspace_shift` plus something else, for the
  reason `Trajectory.b0_terms` exists: a field split across three channels is
  a field that can be half-applied, and `b0_kspace_shift` alone cannot carry a
  quadratic part at all -- it refuses one, which left `simulate_pulseq` unable
  to simulate a degree-2 field that `B0Field.on_phantom` builds happily.

  A per-node field rides the phantom instead and is refused here by name.
  """
  from feelmri.MRObjects import B0Field as _B0Field
  if not isinstance(field, _B0Field):
    raise TypeError(
        f"b0_readout_terms: expected a B0Field, got {type(field).__name__}.")
  # One implementation, on the field. This used to spell the expansion out a
  # second time, beside the copy in `Trajectory.b0_terms`, and the two had
  # already drifted in what the second return value MEANS -- a rate there, a
  # phase here -- with nothing comparing them.
  t = np.asarray(times_ms, dtype=np.float64)
  dk, phi_rate, maxwell = field.polynomial_readout_terms(
      t, scanner, rotation=rotation, location=location)
  return dk, phi_rate * t, maxwell


def maxwell_moments_from_kspace(kx, ky, kz, times_ms, scanner) -> np.ndarray:
  """:func:`maxwell_moments` for a trajectory given as sampled k(t).

  The native trajectory classes in :mod:`feelmri.KSpaceTraj` specify k
  directly and carry no gradient waveform, so the gradient is recovered as
  ``G = (dk/dt) / gammabar`` and the same closed forms are applied between
  samples.

  Two limitations, both structural:

  * It is **exact only where k is linear in t between samples** -- a Cartesian
    readout -- and approximate on a curved trajectory, where the recovered G is
    a finite-difference estimate. Refine by sampling k more densely.
  * The trajectory **begins at the first ADC sample**, so anything played
    before it (a prephaser, a slice rephaser) contributes no moment here. A
    caller holding the real gradient should use :func:`maxwell_moments`
    instead, which integrates from the snapshot.
  """
  t = np.asarray(times_ms, dtype=float).reshape(-1)
  k = np.stack([np.asarray(a, dtype=float).reshape(-1) for a in (kx, ky, kz)],
               axis=1)
  if k.shape[0] != t.size:
    raise ValueError(
        f"maxwell_moments_from_kspace: {k.shape[0]} k-points against {t.size} "
        f"times; they describe the same samples.")
  if t.size < 2:
    return np.zeros((t.size, 4), dtype=float)
  gammabar = scanner.gammabar.m_as('1/ms/mT')
  # (1/m)/ms divided by 1/(ms mT) is mT/m.
  G = np.gradient(k, t, axis=0) / gammabar

  h = np.diff(t)
  increment = _second_moment_increments(h, G)
  return np.vstack((np.zeros((1, 4)), np.cumsum(increment, axis=0)))


def _identify_readout_groups(pulseq_seq: PulseqSequence
                             ) -> List[Tuple[int, int, int]]:
  """Group ADC-bearing blocks by the last RF pulse that precedes them.

  Returns a list of ``(first_block, last_block, anchor_block)`` tuples.

  The anchor is the block holding the most recent ACTIVE RF pulse,
  whatever its ``use`` label. That is not a grouping convenience, it is
  what the dual-path factorisation requires. The readout is synthesized
  as

      M(t) = M(t_anchor) · exp(-i2π(k(t) − k(t_anchor))·x) · exp(-Δt/T2)

  which substitutes the snapshot for the magnetization at the ADC. It is
  valid only while nothing but gradients and relaxation act in between,
  so an RF pulse between the snapshot and the ADC breaks it. Anchoring a
  spin echo on its excitation leaves the refocusing pulse *after* the
  snapshot, and the inversion never reaches the signal at all: measured
  on tests/data/se_an_v15.seq, the off-resonance phase at the echo came
  out fully unrefocused (2.77 rad, i.e. ω·TE).

  An EPI echo train is unaffected — it carries no RF — so blip-separated
  ADCs sharing one excitation still collapse into a single window.

  When no RF precedes an ADC at all, that ADC becomes its own group
  anchored on the most recent non-ADC block.
  """
  n = len(pulseq_seq)
  anchor = -1
  groups: Dict[Any, List[int]] = {}
  anchor_order: List[Any] = []

  for i in range(n):
    has_adc = int(pulseq_seq.ADC[i].num) > 0
    # Any active RF anchors: what matters is that no pulse falls between the
    # snapshot and the ADC, not which coherence period the pulse opens.
    #
    # A block carrying BOTH an RF and an ADC must NOT become its own anchor:
    # the snapshot is taken at the block's END, which is after its own ADC
    # samples, so `times - t_anchor` goes negative and the assembler's
    # exp(-t/T2) becomes exponential GROWTH. Such a block (FID or
    # spectroscopy style) keeps the previous anchor and only updates it for
    # the blocks that follow.
    opens_anchor = _block_use_label(pulseq_seq, i) is not None
    if opens_anchor and not has_adc:
      anchor = i

    if not has_adc:
      continue

    if anchor < 0:
      # No use-labeled anchor yet -- fall back to legacy per-ADC rule.
      key = ('fallback', i)
      groups[key] = [i]
      anchor_order.append(key)
      if opens_anchor:
        anchor = i
      continue

    if anchor not in groups:
      groups[anchor] = []
      anchor_order.append(anchor)
    groups[anchor].append(i)

    if opens_anchor:
      # Its own readout belonged to the PREVIOUS anchor, but its pulse does
      # open a coherence period for everything after it.
      anchor = i

  out: List[Tuple[int, int, int]] = []
  for key in anchor_order:
    if isinstance(key, tuple) and key[0] == 'fallback':
      idx = key[1]
      m_block = -1
      for j in range(idx - 1, -1, -1):
        if int(pulseq_seq.ADC[j].num) == 0:
          m_block = j
          break
      out.append((idx, idx, m_block))
    else:
      adc_blocks = groups[key]
      out.append((min(adc_blocks), max(adc_blocks), key))
  out.sort(key=lambda tup: tup[0])
  return out


def import_pulseq(
    filename,
    *,
    scanner: Optional[Scanner] = None,
    validate: bool = True,
    readout_set_values: Tuple[int, ...] = (3,),
    placeholder_dt: Quantity = Quantity(1.0, 'ms'),
) -> PulseqImport:
  """Parse a ``.seq`` file and return a partitioned view ready for the
  dual-path workflow (``BlochSolver`` for prep + ``Phantom.mri_signal``
  for readouts).

  Parameters
  ----------
  filename : str or Path
      Path to a Pulseq ``.seq`` file.
  scanner : Scanner, optional
      Hardware definition used to convert the events. Its field strength
      sets the Hz-per-ppm scale of the v1.5 ``freq_ppm`` / ``phase_ppm``
      offsets, which the ``.seq`` file does not carry. Default is
      :class:`~feelmri.MRObjects.Scanner`'s 1.5 T; pass a matching scanner
      when reading a sequence written for another field.
  validate : bool, optional
      Run ``pp.Sequence.check_timing`` on the file and report what it finds
      on :attr:`PulseqImport.timing_errors`. Default True; the check costs
      milliseconds even on a few hundred blocks.
  readout_set_values : tuple of int, optional
      ``SET`` label values that mark readout-train blocks. Every block
      whose running ``LABELSET`` state has ``SET`` in this set is
      replaced by an empty delay of identical duration when building
      :attr:`PulseqImport.feelmri_sim_seq`. The default ``(3,)``
      matches the writer convention used by
      ``examples/pulseq_write_epi_tagging.py`` (prep=0/1, excitation=2,
      readout=3, spoiler=100). Pass ``()`` to disable substitution so
      ``feelmri_sim_seq`` is identical to ``feelmri_seq``.
  placeholder_dt : pint.Quantity, optional
      Time-step granularity for the replacement delay blocks. Controls
      the block's internal ``discrete_times`` grid. Default ``1 ms`` is
      fine enough for T1/T2 relaxation and off-resonance phase
      accumulation across a typical EPI readout train without paying
      for sub-millisecond sampling that has no event content.

  See :class:`PulseqImport` for the returned object's shape.
  """
  scanner_was_given = scanner is not None
  if scanner is None:
    scanner = Scanner()
  # Read with the scanner's own gyromagnetic ratio, so that gamma * B in the
  # solver reproduces the Hz/m and Hz the file specifies. GAMMA (42.576 MHz/T)
  # and Scanner.gammabar (42.58 MHz/T) differ by 9.4e-5 relative, which would
  # otherwise appear as that much error in every encoding phase.
  pulseq_seq = read_seq(str(filename), gamma=scanner.gammabar.m_as('Hz/T'))

  # One pypulseq read serves the trajectory below and the timing check here.
  # It is best-effort: a file using an extension pypulseq does not implement
  # (ROTATIONS) still imports, it just goes unvalidated -- and if it also has
  # an ADC the trajectory step below raises, since that one is not optional.
  pp_seq = None
  timing_errors: Tuple[str, ...] = ()
  try:
    pp_seq = _read_with_pypulseq(filename, scanner)
  except Exception as exc:
    logger.info("%s: pypulseq could not read the file (%s); timings are not "
                "validated", filename, exc)
  if pp_seq is not None and validate:
    ok, errors = pp_seq.check_timing()
    if not ok:
      timing_errors = tuple(str(e) for e in errors)
      logger.warning(
          "%s: check_timing reports %d violation(s), which a scanner would "
          "reject: %s%s", filename, len(timing_errors),
          '; '.join(timing_errors[:3]),
          '...' if len(timing_errors) > 3 else '')
  ppm_to_hz = _ppm_to_hz(scanner)
  gammabar = scanner.gammabar.m_as('Hz/T')

  # ppm offsets are a fraction of the Larmor frequency, and the .seq file does
  # not record B0 -- so they are scaled by the SCANNER's. Imported with the
  # default scanner, a sequence written for 3 T silently gets half the offset
  # it was designed with, and a fat-sat pulse lands on the wrong resonance.
  if not scanner_was_given:
    n_ppm = 0
    for i in range(len(pulseq_seq)):
      rf_i, adc_i = pulseq_seq.RF[i], pulseq_seq.ADC[i]
      for ev in (rf_i, adc_i):
        if ev is None:
          continue
        if (abs(float(getattr(ev, 'freq_ppm', 0.0) or 0.0)) > 0.0
            or abs(float(getattr(ev, 'phase_ppm', 0.0) or 0.0)) > 0.0):
          n_ppm += 1
    if n_ppm:
      logger.warning(
          "%s: %d event(s) carry a ppm frequency/phase offset, which is scaled "
          "by gammabar*B0*1e-6 = %.4g Hz/ppm from the DEFAULT scanner (B0 = %s). "
          "The .seq file does not record B0 -- pass import_pulseq(..., "
          "scanner=Scanner(field_strength=...)) if it was not written for "
          "this field.",
          filename, n_ppm, ppm_to_hz, scanner.field_strength)
  feelmri_seq = feelmriSequence()
  # A .seq file spells out its spoiler gradients and RF phase cycling, so the
  # solver must not zero Mxy between blocks on top of them -- that would
  # destroy the coherence pathways an EPI train, a FLASH or a bSSFP depends
  # on. BlochSolver reads this flag when perfect_spoiling is left at None.
  feelmri_seq.explicit_spoiling = True
  feelmri_seq.from_pulseq = True

  # Triggers are hardware handshakes with no simulated counterpart. A WAIT
  # trigger stalls the scanner for an unknown time, so the simulated timeline
  # and the executed one then differ by however long the scanner waited.
  trigger_blocks = [i for i, ext in enumerate(pulseq_seq.EXT)
                    if any(isinstance(e, Trigger) for e in ext)]
  if trigger_blocks:
    logger.warning(
        "%s: %d block(s) carry TRIGGERS extensions, which are ignored "
        "(blocks %s%s). A trigger that makes the scanner wait shifts every "
        "later event, and the simulated timing will not reflect that.",
        filename, len(trigger_blocks), trigger_blocks[:10],
        '...' if len(trigger_blocks) > 10 else '')

  # A gradient that ends away from zero with nothing continuing it leaves the
  # file ill-posed, and the two halves of the dual path then disagree: the
  # trajectory comes from pypulseq, whose calculate_kspace bridges the gap by
  # interpolating linearly from `last` to the next event on that axis, while
  # the solver integrates FEelMRI's own gradients, which are zero outside the
  # event. Neither is what a scanner does, so warn rather than guess.
  _warn_discontinuous_gradients(filename, pulseq_seq)

  for i in range(len(pulseq_seq)):
    gx, gy, gz = pulseq_seq.GR[i]
    rf_ev = pulseq_seq.RF[i]
    adc_ev = pulseq_seq.ADC[i]
    block_dur_s = float(pulseq_seq.DUR[i])

    Gx = _convert_gradient(gx, 0, scanner)
    Gy = _convert_gradient(gy, 1, scanner)
    Gz = _convert_gradient(gz, 2, scanner)
    Rf = _convert_rf(rf_ev, scanner)
    Adc = _convert_adc(adc_ev, scanner)

    gradients = [g for g in (Gx, Gy, Gz) if g is not None]
    rf_pulses = [Rf] if Rf is not None else []

    if not gradients and not rf_pulses and Adc is None:
      feelmri_seq.add_block(Quantity(block_dur_s * 1e3, 'ms'))
      continue

    block = SequenceBlock(
      gradients=gradients,
      rf_pulses=rf_pulses,
      adc=Adc,
      dur=Quantity(block_dur_s * 1e3, 'ms'),
    )
    feelmri_seq.add_block(block)

  n_blocks = len(pulseq_seq)
  adc_block_indices = [i for i in range(n_blocks)
                       if int(pulseq_seq.ADC[i].num) > 0]
  prep_block_indices = [i for i in range(n_blocks)
                        if int(pulseq_seq.ADC[i].num) == 0]

  # Per-block running LABELSET/LABELINC state (e.g. user-defined SET tags).
  block_labels = _compute_block_labels(pulseq_seq)

  # Pre-compute window time bounds from the in-house DUR list (seconds).
  dur_s = np.asarray(pulseq_seq.DUR, dtype=float)
  block_start_s = np.concatenate(([0.0], np.cumsum(dur_s)))
  block_end_s = block_start_s[1:]
  block_start_s = block_start_s[:-1]

  if adc_block_indices:
    try:
      if pp_seq is None:
        # Re-raise whatever the read failed with, rather than a bare None.
        pp_seq = _read_with_pypulseq(filename, scanner)
      k_traj_adc, _k_full, _t_exc, _t_ref, t_adc = pp_seq.calculate_kspace()
      k_traj_adc = np.asarray(k_traj_adc, dtype=float)
      t_adc = np.asarray(t_adc, dtype=float)
    except Exception as exc:  # pragma: no cover - surfaced as hard error
      file_version = pulseq_seq.DEF.get('PulseqVersion')
      hint = ''
      if file_version is not None and not pypulseq_can_read(file_version):
        installed = pypulseq_version()
        hint = (
            f" The file declares Pulseq v{file_version.major}."
            f"{file_version.minor}.{file_version.revision} and the installed "
            f"pypulseq is {installed and f'{installed.major}.{installed.minor}'}"
            f", which cannot read that layout. FEelMRI's own reader handles "
            f"it, but the k-space trajectory comes from pypulseq, so a file "
            f"with an ADC needs pypulseq >= "
            f"{file_version.major}.{file_version.minor}.")
      raise RuntimeError(
          f"Failed to compute k-space trajectory for {filename!s}: {exc}.{hint}"
      ) from exc
  else:
    k_traj_adc = np.zeros((3, 0), dtype=float)
    t_adc = np.zeros((0,), dtype=float)

  groups = _identify_readout_groups(pulseq_seq)

  # marked_in_order tracks every block flagged store_magnetization so
  # readout `m_storage_idx` and prep `prep_storage_idx` index the same
  # Mxy/Mz columns returned by BlochSolver.solve().
  marked_in_order: List[int] = []

  readouts: List[ReadoutWindow] = []
  for first, last, anchor_block in groups:
    m_block = anchor_block
    if m_block >= 0 and m_block not in marked_in_order:
      feelmri_seq.blocks[m_block].store_magnetization = True
      marked_in_order.append(m_block)
    m_idx = marked_in_order.index(m_block) if m_block >= 0 else -1

    t_start = float(block_start_s[first])
    t_end = float(block_end_s[last])
    mask = (t_adc >= t_start - 1e-12) & (t_adc < t_end + 1e-12)
    kspace_file = k_traj_adc[:, mask].T.astype(np.float32, copy=False)
    # float64: these are absolute sequence times, and float32 only resolves
    # about 6e-6 ms at 100 ms. The signal assembler takes float32, so callers
    # cast at that boundary, usually after subtracting the window start.
    times_arr = np.ascontiguousarray(t_adc[mask] * 1e3, dtype=np.float64)

    # The magnetization is snapshotted at the end of the anchor block, and
    # calculate_kspace measures k from the excitation, so the snapshot already
    # carries whatever moment had accumulated by then. Subtract it, so
    # rw.kspace is the encoding measured from the snapshot, which is what pairs
    # with rw's Mxy column.
    #
    # The anchor is the last RF before the readout, so nothing between it and
    # the first sample reflects k and a plain integral is valid there. Working
    # backwards from the first ADC sample also keeps any earlier refocusing
    # reflections, which pypulseq has folded into k_traj_adc.
    if m_block >= 0 and times_arr.size:
      t_anchor_ms = float(block_end_s[m_block]) * 1e3
      k_at_anchor = kspace_file[0].astype(float) - _gradient_moment_between(
          feelmri_seq, t_anchor_ms, float(times_arr[0]), gammabar)
    else:
      t_anchor_ms = float(t_start) * 1e3
      k_at_anchor = np.zeros(3, dtype=float)
    kspace = (kspace_file - k_at_anchor).astype(np.float32, copy=False)
    # The concomitant trajectory. A plain forward integral is valid over the
    # whole window: _identify_readout_groups anchors on ANY active RF, so no
    # pulse falls between the snapshot and the samples and no refocusing
    # reflection can occur inside the interval. That is also why this needs no
    # counterpart to k_at_anchor -- everything before the snapshot is already
    # carried on the magnetization by the solver.
    maxwell = (maxwell_moments(feelmri_seq, t_anchor_ms, times_arr)
               if times_arr.size else np.zeros((0, 4), dtype=float))

    head_adc = pulseq_seq.ADC[first]
    readouts.append(ReadoutWindow(
      first_block=first,
      last_block=last,
      m_storage_block=m_block,
      m_storage_idx=m_idx,
      kspace=kspace,
      kspace_file=kspace_file,
      k_at_anchor=k_at_anchor,
      t_anchor=t_anchor_ms,
      maxwell=maxwell,
      times=times_arr,
      adc_freq_offset=float(head_adc.df) + float(head_adc.freq_ppm) * ppm_to_hz,
      adc_phase_offset=float(head_adc.phase) + float(head_adc.phase_ppm) * ppm_to_hz,
      adc_phase_modulation=head_adc.phase_modulation,
    ))

  # Prep storage points: each block with use='preparation' flags the
  # immediately following block (skipping zero-duration ones) for an
  # Mz snapshot.
  prep_storage_blocks: List[int] = []
  prep_storage_indices: List[int] = []
  for i in range(n_blocks):
    if _block_use_label(pulseq_seq, i) != 'preparation':
      continue
    target = -1
    for j in range(i + 1, n_blocks):
      if float(pulseq_seq.DUR[j]) > 0.0:
        target = j
        break
    if target < 0:
      continue
    if target not in marked_in_order:
      feelmri_seq.blocks[target].store_magnetization = True
      marked_in_order.append(target)
    prep_storage_blocks.append(target)
    prep_storage_indices.append(marked_in_order.index(target))

  # ---------------------------------------------------------------------------
  # Simulation skeleton sequence
  # ---------------------------------------------------------------------------
  # Build a parallel Sequence with the same block count and indices as
  # feelmri_seq, but with every readout-tagged block replaced by an
  # empty delay of identical duration. Two integrity invariants:
  #
  #  1. Index parity with feelmri_seq (and pulseq_seq) so the
  #     ReadoutWindow / prep_storage_blocks indices stay valid.
  #  2. Storage flags carry through SequenceBlock.copy() inside
  #     Sequence.add_block, so BlochSolver.solve() returns Mxy/Mz
  #     columns at the same anchor points whether the runner uses
  #     feelmri_seq or feelmri_sim_seq.
  #
  # Readout-tagged blocks never carry storage flags by construction
  # (the anchor is the preceding RF, not the ADC), so dropping their
  # event content is safe with respect to the snapshot machinery.
  readout_set = set(int(v) for v in readout_set_values)
  feelmri_sim_seq = feelmriSequence()
  feelmri_sim_seq.explicit_spoiling = True
  feelmri_sim_seq.from_pulseq = True
  readout_sim_block_indices: List[int] = []
  for i, blk in enumerate(feelmri_seq.blocks):
    set_value = block_labels[i].get('SET') if i < len(block_labels) else None
    is_readout = set_value is not None and int(set_value) in readout_set
    if is_readout:
      dur_ms = float(blk.dur.m_as('ms'))
      if dur_ms > 0.0:
        feelmri_sim_seq.add_block(Quantity(dur_ms, 'ms'), dt=placeholder_dt)
      else:
        # Zero-duration readout block: preserve index alignment by
        # appending a fresh empty SequenceBlock directly (the
        # add_block(Quantity) path silently drops zero-duration input).
        feelmri_sim_seq.add_block(SequenceBlock(
          dur=Quantity(0.0, 'ms'),
          dt=placeholder_dt,
          empty=True,
          store_magnetization=False,
        ))
      readout_sim_block_indices.append(i)
    else:
      feelmri_sim_seq.add_block(blk)

  # Same contract as check_timing above: report, do not raise. check_timing
  # covers raster alignment and dead times; it says nothing about amplitude,
  # slew or peak B1, which is what check_hardware measures. Both together are
  # still only what the FILE asks of a scanner -- neither says the simulation
  # is wrong, which is why an over-spec sequence imports and runs.
  hardware_problems: Tuple[str, ...] = ()
  if validate:
    hardware_problems = tuple(feelmri_seq.check_hardware(scanner))
    if hardware_problems:
      logger.warning(
          "%s: %d hardware limit(s) exceeded for the given scanner: %s%s",
          filename, len(hardware_problems),
          '; '.join(hardware_problems[:3]),
          '...' if len(hardware_problems) > 3 else '')

  return PulseqImport(
    feelmri_seq=feelmri_seq,
    pulseq_seq=pulseq_seq,
    feelmri_sim_seq=feelmri_sim_seq,
    readouts=readouts,
    prep_block_indices=prep_block_indices,
    adc_block_indices=adc_block_indices,
    prep_storage_blocks=prep_storage_blocks,
    prep_storage_indices=prep_storage_indices,
    block_labels=block_labels,
    readout_sim_block_indices=readout_sim_block_indices,
    timing_errors=timing_errors,
    hardware_problems=hardware_problems,
  )


# ---------------------------------------------------------------------------
# One-call simulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PulseqSimulation:
  """Result of :func:`simulate_pulseq`.

  Attributes
  ----------
  kspace : list of np.ndarray
      One complex array per :class:`ReadoutWindow`, shaped
      ``(N_samples, 1, 1, n_coils)`` as ``Phantom.mri_signal`` returns it.
      Under MPI these are reduced onto rank 0 unless ``gather=False``, in
      which case each rank holds its own partial sum.
  times : list of np.ndarray
      Absolute sample times (ms) matching each ``kspace`` entry. The signal is
      assembled with ``times - ReadoutWindow.t_anchor``; these are kept
      absolute so a caller can place each window on the sequence timeline.
  Mxy, Mz : np.ndarray
      The solver's stored magnetization columns, one per block flagged
      ``store_magnetization``. ``ReadoutWindow.m_storage_idx`` indexes them.
  imp : PulseqImport
      The parsed sequence, so callers can reach the readout windows, the
      label state and the timing report without importing twice.
  """
  kspace: List[np.ndarray]
  times: List[np.ndarray]
  Mxy: np.ndarray
  Mz: np.ndarray
  imp: PulseqImport

  @property
  def kspace_flat(self) -> np.ndarray:
    """Every readout concatenated along the sample axis."""
    if not self.kspace:
      return np.zeros((0, 1, 1, 1), dtype=np.complex64)
    return np.concatenate(self.kspace, axis=0)

  @property
  def times_flat(self) -> np.ndarray:
    """Sample times matching :attr:`kspace_flat`, in ms."""
    if not self.times:
      return np.zeros((0,), dtype=np.float64)
    return np.concatenate(self.times)


@contextmanager
def _no_layout_change():
  """Stand-in for ``Phantom._using`` when there is only one partition."""
  yield


def readout_phase_terms(t_ms, *, scanner, b0_field=None, maxwell_moments=None,
                        rotation=None, location=None):
  """The k-space shift, the extra per-sample phase and the quadratic
  coefficients one readout window needs.

  Extracted from :func:`simulate_pulseq`'s readout loop, which is its only
  caller: the lab field's routing, the concomitant coefficients and the
  off-isocentre re-centring are one subject and are easier to check together
  than spread through the loop body. A caller assembling a readout by hand may
  use it, but nothing in the library does.

  It normalises both contributions onto the caller's own sample shape before
  adding them: :func:`maxwell_recentre` returns a flat sample list while
  :func:`b0_readout_terms` follows the shape of ``t_ms``, so added as they come
  they would broadcast rather than sum.

  Parameters
  ----------
  t_ms : np.ndarray
      Sample times measured FROM THE SNAPSHOT, not from the start of the file.
  b0_field : B0Field or None
      The scanner-fixed field, if a polynomial carries it. A field needing the
      per-node expansion rides the phantom instead and must not be passed here.
  maxwell_moments : np.ndarray or None
      ``(N, 4)`` concomitant moments for this window, or None when the solver
      did not model the term. Pass them exactly when the solver had
      ``concomitant_fields=True``: modelling it up to the snapshot and then
      dropping it for the readout is worse than not modelling it at all.

  Returns
  -------
  (dk, phase, maxwell)
      ``dk`` is ``(N, 3)`` to ADD to the sample's k, or None. ``phase`` is a
      per-sample phase in rad, or None. ``maxwell`` is the six-coefficient
      array for ``mri_signal(maxwell=)``, or None. A degree-2 lab field and
      the concomitant term share those six coefficients, so they add.
  """
  dk = None
  phase = None
  maxwell = None

  if b0_field is not None:
    dk, phase, maxwell = b0_readout_terms(
        b0_field, t_ms, scanner, rotation=rotation, location=location)

  if maxwell_moments is not None:
    conc = maxwell_phase_coefficients(maxwell_moments, scanner,
                                      rotation=rotation)
    maxwell = conc if maxwell is None else maxwell + conc
    # `Bc` is a quadratic form about ISOCENTRE while the assembler's nodes are
    # measured from the slice centre, so the rest of the expansion travels
    # with the six coefficients or an off-isocentre slab is imaged as though
    # it sat in the middle of the bore.
    conc_dk, conc_phase = maxwell_recentre(
        maxwell_moments, scanner, rotation=rotation, location=location)
    # `maxwell_recentre` works on a flat sample list while `b0_readout_terms`
    # follows the shape of `t_ms` -- a native trajectory hands in a
    # (ro, ph, slice) grid. Put both in the caller's own shape before adding,
    # or they broadcast against each other instead of summing.
    grid = np.shape(t_ms)
    if np.any(conc_dk):
      conc_dk = np.asarray(conc_dk).reshape(grid + (3,))
      dk = conc_dk if dk is None else dk + conc_dk
    if np.any(conc_phase):
      conc_phase = np.asarray(conc_phase).reshape(grid)
      phase = (conc_phase if phase is None
               else np.asarray(phase).reshape(grid) + conc_phase)

  return dk, phase, maxwell


def simulate_pulseq(seq_path,
                    phantom,
                    *,
                    scanner: Optional[Scanner] = None,
                    pod=None,
                    gather: bool = True,
                    coil_sensitivities=None,
                    import_kwargs: Optional[Dict[str, Any]] = None,
                    **solver_kwargs) -> PulseqSimulation:
  """Import a ``.seq``, evolve the magnetization and assemble its k-space.

  This is the dual-path workflow in one call: :func:`import_pulseq`, one
  :class:`~feelmri.Bloch.BlochSolver` pass over the whole sequence, then per
  readout window ``phantom.update_magnetization`` followed by
  ``phantom.mri_signal``. The readout is not evolved by the solver -- it is
  synthesized from the k-space trajectory -- which is why one solve serves
  every window.

  ``phantom`` must already have ``set_assembler`` and ``set_static_fields``
  called on it. Those carry modelling decisions (voxel size, quadrature
  order, the T2 and off-resonance maps) that no wrapper should guess.

  Parameters
  ----------
  seq_path : str or Path
      The Pulseq file.
  phantom : FEMPhantom
      Configured as above.
  scanner : Scanner, optional
      Passed to both the import and the solver, so the gamma the file is
      read with is the gamma the solver integrates. Default 1.5 T.
  pod : POD or PODSum or None
      Motion trajectory handed to ``mri_signal``. To move the spins during
      the Bloch evolution as well, pass it as ``pod_trajectory=`` too.
  gather : bool, optional
      Reduce each readout's signal onto rank 0 with
      :func:`~feelmri.MPIUtilities.gather_data`. Default True. ``mri_signal``
      has no collective of its own, so with ``gather=False`` every rank
      returns only its own nodes' contribution.

      ``gather_data`` is an ``MPI_comm.Reduce(root=0)``, **not** an Allreduce:
      with ``gather=True`` the complete signal exists on rank 0 alone and every
      other rank receives ZEROS. Guard reconstruction, plotting and file output
      with ``if MPI_rank == 0``, and do not test a non-root rank's ``kspace``
      for correctness -- it is expected to be empty, not wrong.
  coil_sensitivities : array, optional
      Per-node complex RECEIVE sensitivity, ``(n_local,)`` or
      ``(n_local, n_coils)``, handed to
      :meth:`~feelmri.Phantom.FEMPhantom.set_receive_sensitivity` and put back
      to whatever it was when the call RETURNS. Each window's signal then
      carries ``nv = n_enc * n_coils`` on its last axis with coils varying
      fastest, so ``signal.reshape(signal.shape[:-1] + (n_enc, n_coils))``
      recovers both.

      It has to be an explicit parameter rather than arriving through
      ``**solver_kwargs``: this is a signal-side quantity, so routing it to
      ``BlochSolver`` would raise. It is also NOT ``b1_map``, which is
      transmit and changes the magnetization itself.
  import_kwargs : dict, optional
      Forwarded to :func:`import_pulseq` (``readout_set_values``,
      ``placeholder_dt``, ``validate``).
  **solver_kwargs
      Forwarded to :class:`~feelmri.Bloch.BlochSolver` -- ``M0``, ``T1``,
      ``T2``, ``delta_B``, ``pod_trajectory``, ``method``, ``dtype`` and the
      isochromat controls. ``perfect_spoiling`` is left at its default,
      which resolves to False for an imported sequence.

  Returns
  -------
  PulseqSimulation
  """
  from feelmri.Bloch import BlochSolver, apply_demodulation
  from feelmri.MPIUtilities import gather_data, MPI_print

  if scanner is None:
    scanner = Scanner()
  imp = import_pulseq(seq_path, scanner=scanner, **(import_kwargs or {}))

  # Set while the CALLER's partition is still live, which is the layout
  # set_receive_sensitivity validates and redistributes from -- the same
  # contract set_static_fields follows. Restored at the end so the phantom
  # comes back as it was handed over.
  previous_sensitivity = getattr(phantom, '_receive_sensitivity', None)
  sensitivity_set = coil_sensitivities is not None
  if sensitivity_set:
    phantom.set_receive_sensitivity(coil_sensitivities)

  # Bound before the try so the finally can read them however early a readout
  # fails -- an unbound name there would mask the real exception.
  b0_phi_nodal = None
  b0_static_set = False
  remembered = None
  # The caller may already have installed a gradient of their own. Saved the
  # way `previous_sensitivity` is, so a temporary one put on here is removed
  # and theirs comes back rather than being cleared to None.
  previous_gradient = getattr(phantom, '_b0_gradient', None)
  previous_gradient_layout = getattr(phantom, '_b0_gradient_partition', None)

  # The temporary sensitivity map is removed in the finally below, so a
  # readout that raises does not leave it on the caller's phantom.
  try:
    solver = BlochSolver(sequence=imp.feelmri_seq, phantom=phantom,
                         scanner=scanner, **solver_kwargs)
    Mxy, Mz = solver.solve()

    # The scanner field belongs to the bore, so a spin sees it wherever it has
    # moved to rather than where it started. A field a polynomial can carry
    # reaches the readout as a shift of the sample's k, below; one that needs a
    # per-node expansion rides the PHANTOM instead -- the field itself on
    # `phi_dB0` and the gradient on its own channel, which the assembler
    # applies against the DISPLACEMENT `x(t) - x0`. (The solver spells the same
    # expansion against the absolute position and puts the bracket
    # `dB0(x0) - g.x0` on `delta_B`; the two differ deliberately, see
    # `FEMPhantom.set_b0_gradient`.)
    b0_field = solver_kwargs.get('b0_field', None)
    # Reduced, and reached from every rank: see `B0Field.is_live`. Spelled
    # here as a short-circuiting `and`, a rank with no field would skip the
    # allreduce the others are inside.
    from feelmri.MRObjects import B0Field as _B0FieldLive
    if not _B0FieldLive.is_live(b0_field):
      b0_field = None
    b0_nodal = b0_field is not None and b0_field.kind == 'nodal'
    b0_phi_nodal = None
    if b0_nodal:
      # No `location=`: the per-node path samples the expression at the
      # phantom's own nodes, which `_scanner_nodes` has already mapped back
      # through `_orientation` and `_location`, so an origin argument here
      # would be applied twice. It used to be accepted and discarded.
      b0_read = b0_field.readout_terms(
          phantom, scanner, moving=pod is not None,
          rotation=getattr(phantom, '_orientation', None))
      b0_phi_nodal = np.asarray(b0_read.phi_nodal,
                                dtype=np.float64).reshape(-1)
      # None on a static phantom, where the nodal value IS the Eulerian answer
      # and the channel would be pure cost. Cleared in the finally either way.
      phantom.set_b0_gradient(b0_read.node_gradient_rad_per_ms_per_m)

    # When the solver carried a spectral sub-ensemble, reproduce the readout from
    # it rather than from the collapsed magnetization. Needs the static fields the
    # caller set, which the phantom remembers for exactly this.
    bins = None
    if getattr(solver, 'bin_magnetization', None) is not None:
      # `phantom._static_fields` is rank-local, so the two arms below can be
      # taken by different ranks. The message is computed on both and the
      # collective is entered once, after the branch: with the raise inside
      # the `else`, a rank that only warned walked on while its peers blocked
      # in the allgather.
      remembered = getattr(phantom, '_static_fields', None)
      mismatch = ''
      if remembered is None:
        logger.warning(
            "t2_prime is set but set_static_fields was never called, so the "
            "readout cannot be reproduced per sub-spin and every echo will be "
            "attenuated by the dephasing standing at its anchor")
      else:
        offsets = solver.bin_offsets
        # The row counts are compared on local data, so under dual partitioning
        # this fires on some ranks and not others.
        if (remembered[0].shape[0] != offsets.shape[0]
                or remembered[1].shape[0] != offsets.shape[0]):
          mismatch = (
              f"simulate_pulseq: the remembered static fields have "
              f"{remembered[0].shape[0]} (T2) and {remembered[1].shape[0]} "
              f"(phi_dB0) rows against {offsets.shape[0]} sub-spin offsets. "
              f"set_static_fields must be called under the same partition the "
              f"solver was built on, or the two describe different nodes")
      _collective_raise(mismatch)
      if remembered is not None:
        bins = (solver.bin_magnetization, solver.bin_offsets,
                solver.bin_weights, remembered[0], remembered[1])

    # Under dual partitioning every set_static_fields and update_magnetization
    # is an Alltoallv into the signal layout, and the bin loop below would make
    # three per sub-spin. The window-independent arrays are moved once here and
    # the per-window ensemble once below, after which the loop runs with the
    # signal layout already active and communicates nothing.
    dual = (bins is not None and getattr(phantom, '_dual', False)
            and phantom._active_partition != 'signal')
    # `bins` stays in the BLOCH layout: it is what the finally below hands back
    # to set_static_fields, which redistributes it itself. `readout_bins` is the
    # same data already moved, for the loop that runs inside the signal layout.
    readout_bins = bins
    if dual:
      _ens, offsets, weights, T2_read, phi_read = bins
      move = lambda a: phantom.redistribute_nodal(
          np.ascontiguousarray(a), 'bloch', 'signal')
      readout_bins = (_ens, move(offsets), weights, move(T2_read),
                      move(phi_read))

    # The readout carries the concomitant term exactly when the SOLVER did. The
    # two halves describe one field, and modelling it up to the snapshot and then
    # dropping it for the readout would be worse than not modelling it at all --
    # it is the same coupling rule the off-resonance handoff follows.
    concomitant_readout = bool(solver_kwargs.get('concomitant_fields', False))

    # A per-node field is added to the off-resonance the caller set, once,
    # rather than per window: it is window independent, and under dual
    # partitioning every set_static_fields is an Alltoallv. Restored below.
    b0_static_set = False
    if b0_phi_nodal is not None:
      remembered = getattr(phantom, '_static_fields', None)
      _collective_raise(
          '' if remembered is not None else
          "simulate_pulseq: this b0_field needs a per-node expansion, which "
          "rides `phi_dB0`, but set_static_fields was never called -- there is "
          "nothing to add it to and the field would be silently dropped.")
      if bins is None:
        T2_prev, phi_prev = remembered
        phantom.set_static_fields(
            T2_prev,
            np.asarray(phi_prev, dtype=np.float64)
            + b0_phi_nodal.reshape(np.shape(phi_prev)))
        b0_static_set = True
      else:
        # The bin loop sets the fields itself, once per sub-spin, so the field
        # joins `phi_read` there instead. `bins` keeps the caller's own values,
        # because that is what the finally hands back.
        _e, _o, _w, _t2, _phi = readout_bins
        add = (phantom.redistribute_nodal(
                   np.ascontiguousarray(b0_phi_nodal), 'bloch', 'signal')
               if dual else b0_phi_nodal)
        readout_bins = (_e, _o, _w, _t2,
                        np.asarray(_phi, dtype=np.float64)
                        + add.reshape(np.shape(_phi)))

    kspace: List[np.ndarray] = []
    times: List[np.ndarray] = []
    for rw in imp.readouts:
      if rw.m_storage_idx < 0:
        logger.warning(
            "readout blocks %d-%d have no coherence anchor and are skipped",
            rw.first_block, rw.last_block)
        continue
      if bins is None:
        # On the bins path the loop below sets this once per sub-spin and the
        # finally puts the collapsed value back, so doing it here as well is a
        # wasted nodal store -- and, under dual partitioning, a wasted Alltoallv
        # per readout window (4 of 23 on cpmg_v15 at K = 16).
        phantom.update_magnetization(Mxy[:, rw.m_storage_idx])
      # Elapsed time since the snapshot, not absolute time from the start of the
      # file. mri_signal uses t for exp(-t/T2) and exp(i*phi*t), both of which
      # continue from the instant the magnetization was captured; feeding it
      # absolute times applies a spurious exp(-t_anchor/T2) to the whole window.
      points, t = _reshape_signal_inputs(
          rw.kspace[:, 0], rw.kspace[:, 1], rw.kspace[:, 2],
          rw.times - rw.t_anchor, None)
      # The shift stays LOCAL to this call: `rw.kspace` is the nominal
      # trajectory the reconstruction grids on, and the difference between the
      # two is the distortion the field produces.
      # The shift stays LOCAL to this call: `rw.kspace` is the nominal
      # trajectory the reconstruction grids on, and the difference between the
      # two IS the distortion the field produces. The readout carries the
      # concomitant term exactly when the SOLVER did.
      dk, b0_phase, maxwell = readout_phase_terms(
          t, scanner=scanner,
          b0_field=(b0_field if not b0_nodal else None),
          maxwell_moments=(rw.maxwell if concomitant_readout else None),
          rotation=getattr(phantom, '_orientation', None),
          location=getattr(phantom, '_location', None))
      if dk is not None:
        points = [np.ascontiguousarray(points[i] + dk[..., i],
                                       dtype=points[i].dtype)
                  for i in range(3)]
      # `t` is elapsed-since-snapshot, which the relaxation and off-resonance
      # factors need, but the POD weights need absolute sequence time, the frame
      # the motion is defined in. `get_weights` adds the trajectory's own
      # `timeshift`, so point it at this window's anchor and restore it after.
      # Composed with the caller's own shift, which `get_weights` folds in to
      # reach the cardiac phase, and restored in the finally below.
      shift = getattr(pod, 'timeshift', None) if pod is not None else None
      if shift is not None:
        pod.update_timeshift(float(shift) + float(rw.t_anchor))
      try:
        if bins is None:
          signal = phantom.mri_signal(list(points), t, pod, maxwell=maxwell)
        else:
          # Bin-by-bin readout. Collapsing the sub-ensemble at the snapshot and
          # letting the assembler replay a single exp(-t/T2) from there cannot
          # reproduce a readout: the snapshot sits at the coherence anchor, where the
          # ensemble is maximally dephased, and nothing downstream can bring it back.
          #
          # Each sub-spin is given its own off-resonance instead -- the bin offsets
          # are in the same rad/ms frame as phi_dB0, so they simply add -- and the
          # signals are weight-summed. Costs n_bins passes over the signal path.
          bin_Mxy, offsets, weights, T2_read, phi_read = readout_bins
          ensemble = bin_Mxy[:, :, rw.m_storage_idx]
          if dual:
            ensemble = phantom.redistribute_nodal(
                np.ascontiguousarray(ensemble), 'bloch', 'signal')
          signal = None
          with phantom._using('signal') if dual else _no_layout_change():
            for k, w in enumerate(weights):
              phantom.set_static_fields(T2_read, phi_read + offsets[:, k])
              phantom.update_magnetization(ensemble[:, k])
              contribution = w * phantom.mri_signal(list(points), t, pod,
                                                    maxwell=maxwell)
              signal = contribution if signal is None else signal + contribution
      finally:
        if shift is not None:
          pod.update_timeshift(shift)
        if bins is not None:
          phantom.set_static_fields(bins[3], bins[4])
          # The bin loop above leaves the phantom holding the last sub-spin, a tail
          # bin of the quadrature with weight ~1e-16, so restore the collapsed
          # magnetization for anything the caller evaluates afterwards.
          phantom.update_magnetization(Mxy[:, rw.m_storage_idx])
      if b0_phase is not None:
        # The uniform half of the field is position independent, so it cannot be
        # carried by a k-space offset; it is a per-sample phase instead.
        signal = apply_demodulation(signal, b0_phase.reshape(-1))
      # The receiver's frequency/phase offsets and any per-sample phase shape.
      signal = rw.demodulate(signal)
      kspace.append(gather_data(signal) if gather else signal)
      times.append(rw.times)
  finally:
    if b0_phi_nodal is not None:
      # Cleared unconditionally: a stale per-node gradient left on the phantom
      # is a wrong image with no symptom. The caller's own is then restored --
      # but only into the layout it was captured in. A `finally` that raises
      # replaces a good result, or the exception already propagating, with a
      # row-count complaint about the restore, so a gradient that no longer
      # describes the live partition is dropped with a warning instead.
      live = getattr(phantom, '_active_partition', None)
      if previous_gradient is not None and previous_gradient_layout != live:
        MPI_print("[simulate_pulseq] WARNING: the caller's per-node B0 "
                  "gradient was built under the '{}' layout and '{}' is live; "
                  "it has been cleared rather than paired with the wrong "
                  "nodes. Call set_b0_gradient again.".format(
                      previous_gradient_layout, live))
        previous_gradient = None
      phantom.set_b0_gradient(previous_gradient)
      if b0_static_set:
        phantom.set_static_fields(*remembered)
    if sensitivity_set:
      # Restored as the ALREADY-REDISTRIBUTED array, not by re-running the
      # setter: `previous_sensitivity` was read out of the signal layout, and
      # feeding it back through set_receive_sensitivity would redistribute it a
      # second time and pair it with the wrong nodes.
      phantom._receive_sensitivity = previous_sensitivity

  return PulseqSimulation(kspace=kspace, times=times, Mxy=Mxy, Mz=Mz, imp=imp)


# ---------------------------------------------------------------------------
# High-level sequence reader for FEelMRI
# ---------------------------------------------------------------------------

def read_seq_feelmri(filename, *, scanner: Optional[Scanner] = None,
                     validate: bool = True) -> Tuple[feelmriSequence, PulseqSequence]:
  """Backward-compatible wrapper around :func:`import_pulseq`.

  Returns ``(feelmri_seq, pulseq_seq)`` so existing callers keep working;
  new callers should prefer :func:`import_pulseq` for the partitioned
  view (with readout windows ready for the dual-path workflow).

  ``scanner`` and ``validate`` are forwarded. ``scanner`` is not optional in
  practice for anything but 1.5 T: the file carries no B0, so every ppm offset
  is scaled by the scanner's, and the gamma the file is read with must be the
  gamma the solver integrates.
  """
  imp = import_pulseq(filename, scanner=scanner, validate=validate)
  return imp.feelmri_seq, imp.pulseq_seq