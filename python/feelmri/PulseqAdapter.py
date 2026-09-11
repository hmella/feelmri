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
# Basic numerical / helper utilities
# ---------------------------------------------------------------------------

# Approximate gyromagnetic ratio for protons in Hz/T.
# You may override this constant from user code if needed.
GAMMA = 42.576e6  # [Hz/T]


def _to_float_or_str(token: str) -> Union[float, str]:
    """Try to parse a token as float, else keep as string."""
    try:
        return float(token)
    except ValueError:
        return token


# ---------------------------------------------------------------------------
# Version handling
# ---------------------------------------------------------------------------

@dataclass(order=True, frozen=True)
class Version:
    major: int
    minor: int
    revision: int

    @classmethod
    def from_file(cls, io) -> "Version":
        """
        Parse three lines:
            major X
            minor Y
            revision Z
        """
        def _read_int_line(expected_prefix: str) -> int:
            line = io.readline()
            if not line:
                raise EOFError("Unexpected end of file while reading version.")
            parts = line.strip().split()
            if len(parts) != 2 or parts[0] != expected_prefix:
                raise ValueError(f"Expected '{expected_prefix} <int>' line, got: {line!r}")
            return int(parts[1])

        major = _read_int_line("major")
        minor = _read_int_line("minor")
        revision = _read_int_line("revision")
        return cls(major, minor, revision)


def read_version(io) -> Version:
    """
    Read the [VERSION] section of a sequence file.
    """
    pulseq_version = Version.from_file(io)

    assert pulseq_version.major == 1, (
        f"Unsupported version_major {pulseq_version.major}"
    )
    if pulseq_version < Version(1, 2, 0):
        logger.error(
            "Unsupported Pulseq %s, only file format revision 1.2.0 and above are supported",
            pulseq_version,
        )
    elif pulseq_version < Version(1, 3, 1):
        logger.warning(
            "Loading older Pulseq %s; some code may not function as expected",
            pulseq_version,
        )
    elif pulseq_version >= Version(1, 6, 0):
        logger.warning(
            "Pulseq %s not yet supported by this Python translation "
            "(supported range is [1.2.0, 1.6.0)).",
            pulseq_version,
        )

    return pulseq_version


# ---------------------------------------------------------------------------
# Definitions & signature
# ---------------------------------------------------------------------------

def read_definitions(io) -> Dict[str, Any]:
    """
    Read the [DEFINITIONS] section as a dict of key->value(s).
    Numeric tokens become floats if parseable.
    """
    defs: Dict[str, Any] = {}
    while True:
        line = io.readline()
        if not line:
            break
        parts = line.split()
        if not parts:
            # break on whitespace / blank line
            break
        key = parts[0]
        value_tokens = parts[1:]
        parsed = [_to_float_or_str(tok) for tok in value_tokens]
        if len(parsed) == 1:
            defs[key] = parsed[0]
        else:
            defs[key] = parsed

    # Raster defaults for files that omit them, in seconds.
    defs.setdefault("BlockDurationRaster", 1e-5)
    defs.setdefault("GradientRasterTime", 1e-5)
    defs.setdefault("RadiofrequencyRasterTime", 1e-6)
    defs.setdefault("AdcRasterTime", 1e-7)

    return defs


def read_signature(io) -> str:
    """
    Read the [SIGNATURE] section and return the 'Hash' value, if present.
    """
    signature = ""
    while True:
        line = io.readline()
        if not line:
            break
        parts = line.split()
        if not parts:
            break
        key = parts[0]
        if key == "Hash":
            value_tokens = parts[1:]
            parsed = [_to_float_or_str(tok) for tok in value_tokens]
            signature = parsed[0] if len(parsed) == 1 else parsed
    return signature


# ---------------------------------------------------------------------------
# Blocks and events
# ---------------------------------------------------------------------------

def read_blocks(io, block_duration_raster: float, pulseq_version: Version):
    """
    Read the [BLOCKS] section.
    Returns:
        event_table: Dict[int, List[int]]
        block_durations: Dict[int, float]
        delay_ids_tmp: Dict[int, int]
    """
    event_table: Dict[int, List[int]] = {}
    block_durations: Dict[int, float] = {}
    delay_ids_tmp: Dict[int, int] = {}

    while True:
        number_block_events = 7 if pulseq_version <= Version(1, 2, 1) else 8

        line = io.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            break
        tokens = line.split()
        block_events = [int(t) for t in tokens]

        if block_events[0] != 0:
            bid = block_events[0]
            if pulseq_version <= Version(1, 2, 1):
                # Int64[0; blockEvents[3:end]...; 0]
                events = [0] + block_events[2:] + [0]
            else:
                # Int64[0; blockEvents[3:end]...]
                events = [0] + block_events[2:]
            event_table[bid] = events

            if pulseq_version >= Version(1, 4, 0):
                block_durations[bid] = block_events[1] * block_duration_raster
            else:
                # store delay index, to be resolved later
                delay_ids_tmp[bid] = block_events[1]

        if len(block_events) != number_block_events:
            # break on unexpected line length (e.g. whitespace / end of section)
            break

    return event_table, block_durations, delay_ids_tmp


def read_events(io, scale: List[float],
                type_: Any = -1,
                event_library: Optional[Dict[int, Dict[str, Any]]] = None
                ) -> Dict[int, Dict[str, Any]]:
    """
    General event reader. Reads lines of the form:
        id val1 val2 ... valN
    where the number of values matches len(scale).
    Values are multiplied element-wise by 'scale'.

    A ``np.nan`` entry in ``scale`` marks a string column (used in
    Pulseq v1.5 for the trailing ``use`` flag on RF rows). When the
    sentinel is present, ``data`` is returned as a mixed-type list of
    length ``len(scale)`` instead of a ``np.ndarray``; numeric columns
    are still scaled, the string column is kept as the raw token.
    """
    if event_library is None:
        event_library = {}

    n_vals = len(scale)
    scale_arr = np.array(scale, dtype=float)
    has_str_col = bool(np.any(np.isnan(scale_arr)))

    while True:
        line = io.readline()
        if not line:
            break
        parts = line.split()
        if not parts:
            break
        if len(parts) != n_vals + 1:
            # A short row ends the section: the table is over.
            break
        eid = int(float(parts[0]))

        if has_str_col:
            data: List[Any] = []
            for col, s in enumerate(scale_arr):
                token = parts[1 + col]
                if np.isnan(s):
                    data.append(token)
                else:
                    data.append(float(s) * float(token))
        else:
            raw_vals = np.array([float(p) for p in parts[1:]], dtype=float)
            data = scale_arr * raw_vals

        entry: Dict[str, Any] = {"data": data}
        if type_ != -1:
            entry["type"] = type_
        event_library[eid] = entry

    return event_library


def read_labels(io, event_library: Optional[Dict[int, Dict[str, Any]]] = None
                ) -> Dict[int, Dict[str, Any]]:
    """
    Read a label section:
        id int string
    """
    if event_library is None:
        event_library = {}

    while True:
        line = io.readline()
        if not line:
            break
        parts = line.split()
        if not parts:
            break
        if len(parts) < 3:
            break
        eid = int(float(parts[0]))
        val_int = int(float(parts[1]))
        val_str = parts[2]
        event_library[eid] = {"data": [val_int, val_str]}
        if len(parts) != 3:
            # A short row ends the section: the table is over.
            break

    return event_library


def skip_section(io) -> int:
    """Consume the body of a section we do not handle, up to the blank line
    that separates it from the next one. Returns the number of lines dropped.

    Every other reader consumes its own body, so a section that only warns
    leaves its data lines in the stream. The next iteration of read_seq's
    section loop then reads one of them as a section header and raises
    ``Unknown section code``.
    """
    dropped = 0
    while True:
        pos = io.tell()
        line = io.readline()
        if not line:
            break
        if not line.strip():
            break
        if line.lstrip().startswith('['):
            # A section header with no blank line before it: put it back.
            io.seek(pos)
            break
        dropped += 1
    return dropped


def read_extension_blocks(io, event_library: Optional[Dict[int, Dict[str, Any]]] = None
                          ) -> Dict[int, Dict[str, Any]]:
    """
    Read the extension blocks section:
        id type ref next_id
    """
    if event_library is None:
        event_library = {}

    while True:
        line = io.readline()
        if not line:
            break
        parts = line.split()
        if not parts:
            break
        if len(parts) < 4:
            break
        eid = int(float(parts[0]))
        vals = [int(float(p)) for p in parts[1:4]]
        event_library[eid] = {"data": vals}
        if len(parts) != 4:
            break

    return event_library


# ---------------------------------------------------------------------------
# Shape compression / decompression
# ---------------------------------------------------------------------------

def read_shapes(io, force_convert_uncompressed: bool):
    """
    Read the [SHAPES] section.
    Returns a dict: id -> (num_samples, data_array)
    """
    shape_library: Dict[int, Tuple[int, np.ndarray]] = {}

    # The line after [SHAPES] is a comment header and carries no data.
    _ = io.readline()

    while True:
        line = io.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            break

        if not line.startswith("shape_id"):
            break
        parts = line.split()
        if len(parts) != 2:
            break
        _, sid_str = parts
        shape_id = int(sid_str)

        line = io.readline()
        if not line:
            break
        parts = line.strip().split()
        if len(parts) != 2 or parts[0] != "num_samples":
            raise ValueError(f"Expected 'num_samples <int>' line, got: {line!r}")
        num_samples = int(parts[1])

        # read sample lines until we hit a non-float / blank / EOF
        samples: List[float] = []
        while True:
            pos = io.tell()
            line = io.readline()
            if not line:
                break
            line_stripped = line.strip()
            if not line_stripped:
                break
            try:
                val = float(line_stripped)
            except ValueError:
                # revert this line for outer logic and break
                io.seek(pos)
                break
            samples.append(val)

        data = np.asarray(samples, dtype=float)

        # For v1.4.x we use length(data)==num_samples as marker for uncompressed.
        # In older versions this condition could occur by chance; 'force_convert_uncompressed'
        # decides whether to attempt re-compression.
        if force_convert_uncompressed and len(data) == num_samples:
            # round-trip through decompress / compress
            w = decompress_shape(num_samples, data, force_decompression=True)
            num_samples2, data2 = compress_shape(w, force_compression=True)
            data = data2
            num_samples = num_samples2

        shape_library[shape_id] = (num_samples, data)

    return shape_library


def compress_shape(w: Union[np.ndarray, List[float]],
                   force_compression: bool = False) -> Tuple[int, np.ndarray]:
    """
    Compress a gradient/RF shape using the Pulseq scheme on the derivative.
    Returns (num_samples, compressed_data).
    """
    w = np.asarray(w, dtype=float)
    num_samples = w.size

    if not force_compression and num_samples <= 4:
        return num_samples, w.copy()

    quant_fac = 1e-7
    ws = w / quant_fac
    # first element + diffs
    datq = np.round(np.concatenate(([ws[0]], np.diff(ws))))
    qerr = ws - np.cumsum(datq)
    qcor = np.concatenate(([0.0], np.diff(np.round(qerr))))
    datd = datq + qcor

    mask_changes = np.concatenate(([True], np.diff(datd) != 0))
    vals = datd[mask_changes] * quant_fac

    k = np.nonzero(np.concatenate((mask_changes, [True])))[0]
    n = np.diff(k)  # number of repetitions

    n_extra = n.astype(float) - 2.0
    vals2 = vals.astype(float)

    # entries where n_extra < 0 are encoded as NaN, to be dropped
    mask_neg = n_extra < 0
    vals2[mask_neg] = np.nan
    n_extra[mask_neg] = np.nan

    v = np.concatenate((vals, vals2, n_extra))
    v = v[np.isfinite(v)]
    v[np.abs(v) <= 1e-10] = 0.0

    if force_compression or num_samples > v.size:
        data = v
    else:
        data = w.copy()

    return num_samples, data


def decompress_shape(num_samples: int,
                     data: Union[np.ndarray, List[float]],
                     force_decompression: bool = False) -> np.ndarray:
    """
    Decompress a Pulseq-compressed shape.
    Returns the uncompressed waveform of length 'num_samples'.
    """
    data_pack = np.asarray(data, dtype=float)
    data_pack_len = data_pack.size
    num_samples_int = int(num_samples)

    if not force_decompression and num_samples_int == data_pack_len:
        return data_pack.copy()

    w = np.zeros(num_samples_int, dtype=float)

    # Differences: when zero, subsequent samples are equal (marker for repeats).
    data_pack_diff = data_pack[1:] - data_pack[:-1]
    # A marker indexes the sample AFTER the repeated one, hence the +1.
    markers = np.where(data_pack_diff == 0)[0] + 1  # 1-based

    count_pack = 1       # 1-based index into compressed data
    count_unpack = 1     # 1-based index into uncompressed data

    for next_pack in markers:
        curr_unpack_samples = next_pack - count_pack
        if curr_unpack_samples < 0:
            # false positive, skip
            continue
        elif curr_unpack_samples > 0:
            # copy unpacked block
            w[count_unpack - 1: count_unpack - 1 + curr_unpack_samples] =                 data_pack[count_pack - 1: next_pack - 1]
            count_pack += curr_unpack_samples
            count_unpack += curr_unpack_samples

        # packed / repeated section
        if count_pack + 2 > data_pack_len:
            raise ValueError("Corrupted compressed shape (index out of range).")
        rep = int(math.floor(data_pack[count_pack - 1 + 2] + 2.0))
        w[count_unpack - 1: count_unpack - 1 + rep] = data_pack[count_pack - 1]
        count_pack += 3
        count_unpack += rep

    # any samples left?
    if count_pack <= data_pack_len:
        if data_pack_len - count_pack != num_samples_int - count_unpack:
            raise AssertionError("Unsuccessful unpacking of samples")
        w[count_unpack - 1:] = data_pack[count_pack - 1:]

    return np.cumsum(w)


# ---------------------------------------------------------------------------
# Domain objects: Grad, RF, ADC, Extensions, Sequence
# ---------------------------------------------------------------------------

@dataclass
class Grad:
    """Gradient event."""
    A: Union[float, np.ndarray]  # amplitude or shaped waveform
    T: Union[float, np.ndarray]  # total duration or per-sample durations
    rise: float = 0.0
    fall: float = 0.0
    delay: float = 0.0
    first: float = 0.0
    last: float = 0.0


@dataclass
class RF:
    """RF event."""
    waveform: np.ndarray  # complex RF samples
    T: Union[float, np.ndarray]  # total duration or per-sample durations
    df: float                # frequency offset (Hz)
    delay: float = 0.0
    # v1.5 additions. ``use`` is the canonical functional label
    # ('excitation' | 'refocusing' | 'inversion' | 'saturation' |
    #  'preparation' | 'other' | 'undefined'). ``freq_ppm`` and
    # ``freq_ppm`` / ``phase_ppm`` are ppm offsets, folded into the RF's
    # frequency_offset / phase_offset by _convert_rf.
    use: str = "undefined"
    freq_ppm: float = 0.0
    phase_ppm: float = 0.0


@dataclass
class ADC:
    """ADC event."""
    num: int
    T: float
    delay: float
    df: float = 0.0       # frequency offset (Hz)
    phase: float = 0.0    # phase offset (rad)
    # v1.5 additions. freq_ppm / phase_ppm are ppm offsets folded into df /
    # phase at conversion; phase_modulation is the per-sample phase shape
    # referenced by the event's phase_id column (rad, one entry per sample).
    freq_ppm: float = 0.0
    phase_ppm: float = 0.0
    phase_modulation: Optional[np.ndarray] = None


@dataclass
class Trigger:
    channel: int
    mode: int
    rise: float
    fall: float


@dataclass
class LabelSet:
    label: str   # e.g. 'SET', 'LIN', 'SLC', 'NAV', 'REF', ...
    value: int


@dataclass
class LabelInc:
    label: str
    value: int


@dataclass
class Rotation:
    """3x3 rotation matrix applied to (Gx, Gy, Gz) for a block."""
    matrix: np.ndarray  # shape (3, 3), row-major


Extension = Union[Trigger, LabelSet, LabelInc, Rotation]


@dataclass
class PulseqSequence:
    """Simplified sequence container."""
    GR: List[Tuple[Grad, Grad, Grad]] = field(default_factory=list)
    RF: List[RF] = field(default_factory=list)
    ADC: List[ADC] = field(default_factory=list)
    DUR: List[float] = field(default_factory=list)
    EXT: List[List[Extension]] = field(default_factory=list)
    DEF: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.DUR)

    def add_block(self,
                  gx: Grad,
                  gy: Grad,
                  gz: Grad,
                  rf: RF,
                  adc: ADC,
                  duration: float,
                  extensions: List[Extension]):
        self.GR.append((gx, gy, gz))
        self.RF.append(rf)
        self.ADC.append(adc)
        self.DUR.append(float(duration))
        self.EXT.append(list(extensions))


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------

def dur_grad(g: Grad) -> float:
    """Compute duration of a gradient event."""
    if isinstance(g.T, np.ndarray):
        t = float(np.sum(g.T))
    else:
        t = float(g.T)
    return float(g.delay + g.rise + t + g.fall)


def dur_rf(r: RF) -> float:
    """Compute duration of an RF event."""
    if isinstance(r.T, np.ndarray):
        t = float(np.sum(r.T))
    else:
        t = float(r.T)
    return float(r.delay + t)


def dur_adc(a: ADC) -> float:
    """Compute duration of an ADC event."""
    return float(a.delay + a.T)


def _grad_corners_seconds(g: "Grad") -> Tuple[np.ndarray, np.ndarray]:
  """(times_s, amplitudes_Tm) corner list for a Grad of either shape."""
  if isinstance(g.A, np.ndarray):
    return _shaped_waveform_seconds(g)
  return _trap_waveform_seconds(g)


def _rotate_on_union_grid(R: np.ndarray,
                          gx: "Grad", gy: "Grad", gz: "Grad"
                          ) -> Tuple["Grad", "Grad", "Grad"]:
  """Rotate three gradients that do NOT share a time grid.

  A rotation mixes the axes, so each output sample is a combination of all
  three inputs and is only defined where all three are. Sampling them on the
  union of their corners makes that true everywhere: a piecewise-linear
  waveform is exactly reproduced by adding corners to it, so this is lossless
  on each input and exact on the output.

  The result is emitted as an extended trapezoid -- ``A`` the per-sample
  amplitudes, ``T`` the per-step dwells -- which is the one Grad shape that
  can carry a non-uniform grid. Outside its own support a gradient is zero,
  which is what the ``left``/``right`` of the interpolation says.

  Inheriting one donor axis's geometry instead, as this used to, is wrong
  whenever the timings differ: on an ordinary block (in-plane prephasers
  0.5 ms flat, slice-select 2.0 ms) an IDENTITY matrix inflated the x and y
  moments by 250%.
  """
  corners = [_grad_corners_seconds(g) for g in (gx, gy, gz)]
  grid = np.unique(np.concatenate([t for t, _a in corners]))
  # Two axes naming one instant can differ in the last bits; a zero-length
  # step would put an infinite slew into the output.
  if grid.size > 1:
    grid = grid[np.concatenate(([True], np.diff(grid) > 1e-12))]
  if grid.size < 2:
    return gx, gy, gz

  stacked = np.vstack([np.interp(grid, t, a, left=0.0, right=0.0)
                       for t, a in corners])
  rotated = R @ stacked

  local = grid - grid[0]
  dwells = np.diff(local)
  out = []
  for axis in range(3):
    amps = rotated[axis]
    out.append(Grad(
      A=amps,
      T=dwells,
      rise=0.0,
      fall=0.0,
      delay=float(grid[0]),
      first=float(amps[0]),
      last=float(amps[-1]),
    ))
  return tuple(out)


def _warn_discontinuous_gradients(filename, pulseq_seq, rtol: float = 1e-3):
  """Warn about any gradient event that ends non-zero and is not continued.

  Pulseq expects a gradient to return to zero at a block boundary unless the
  next block carries it on. When one does not, the trajectory and the solver
  are reading two different waveforms -- see the call site for the measured
  cost. `check_timing` does not look at this.

  ``rtol`` is relative to the event's own peak, so a boundary sample at
  round-off is ignored and one at a percent of peak is not.
  """
  problems = []
  n_blocks = len(pulseq_seq)
  for i in range(n_blocks):
    for axis, g in enumerate(pulseq_seq.GR[i]):
      if g is None or not isinstance(g.A, np.ndarray) or g.A.size == 0:
        continue
      peak = float(np.abs(g.A).max())
      if peak <= 0.0:
        continue
      tail = float(abs(g.last))
      if tail <= rtol * peak:
        continue
      nxt = pulseq_seq.GR[i + 1][axis] if i + 1 < n_blocks else None
      carried = nxt is not None and (
          (isinstance(nxt.A, np.ndarray) and nxt.A.size
           and abs(float(nxt.first) - float(g.last)) <= rtol * peak)
          or (not isinstance(nxt.A, np.ndarray) and float(nxt.A) != 0.0))
      if not carried:
        problems.append((i, 'xyz'[axis], 100.0 * tail / peak))

  if problems:
    logger.warning(
        "%s: %d gradient event(s) end away from zero with nothing continuing "
        "them (%s). The k-space trajectory comes from pypulseq, which bridges "
        "the gap linearly to the next event on that axis, while the solver "
        "reads zero there -- so the readout carries phase the simulation never "
        "played. Make the waveform return to zero, or continue it in the next "
        "block.",
        filename, len(problems),
        ', '.join(f'block {b} G{a} ends at {pc:.2f}% of peak'
                  for b, a, pc in problems[:5]))


def _apply_rotation_to_grads(R: np.ndarray,
                             gx: Grad, gy: Grad, gz: Grad
                             ) -> Tuple[Grad, Grad, Grad]:
    """Apply a 3x3 rotation matrix to a (gx, gy, gz) triple.

    The rotation acts on gradient amplitudes pointwise. For trapezoidal
    gradients (scalar `A`), each output amplitude is a linear combination
    of the three scalars. For arbitrary shaped gradients (`A` is an
    ndarray of length N), the output is the row of R applied per sample,
    producing a new shaped gradient.

    Timing fields (`T`, `rise`, `fall`, `delay`) are inherited from one donor
    axis, which is exact ONLY when every active axis already shares them --
    the common case a Pulseq writer emits, and the only case accepted here.

    A rotation mixes the axes, so an output amplitude is a combination of
    three inputs and is only meaningful where all three are defined on the
    same grid. With a single donor and differing timings the result is
    silently wrong rather than approximate: measured on an ordinary block
    (x/y prephasers 0.5 ms flat, z slice-select 2 ms flat) an IDENTITY matrix
    inflated the x and y moments by **+250%**, and promoting a trapezoid to a
    constant array to match a shaped gradient discards its ramps (-9.5%).
    Both now raise instead. Doing it properly means resampling all three onto
    the union of their corners and emitting shaped gradients; that is a real
    feature, not a patch.
    """
    active = [g for g in (gx, gy, gz)
              if isinstance(g.A, np.ndarray) or float(g.A) != 0.0]
    if len(active) > 1:
        shapes = {('array', a.A.size) if isinstance(a.A, np.ndarray)
                  else ('trap', a.T, a.rise, a.fall, a.delay) for a in active}
        if len(shapes) > 1:
            return _rotate_on_union_grid(R, gx, gy, gz)

    # Coerce all amplitudes to a common shape: scalars stay scalar; arrays
    # broadcast to the longest. Mixed scalar/array becomes array.
    amps = []
    is_array = False
    for g in (gx, gy, gz):
        a = g.A
        if isinstance(a, np.ndarray):
            is_array = True
            amps.append(np.asarray(a, dtype=float))
        else:
            amps.append(float(a))

    if is_array:
        # Promote any scalars to constant arrays of matching length.
        target_len = max((a.size for a in amps if isinstance(a, np.ndarray)),
                         default=1)
        promoted = []
        for a in amps:
            if isinstance(a, np.ndarray):
                if a.size == target_len:
                    promoted.append(a)
                else:
                    # Linear-resample to target length to align rasters.
                    promoted.append(np.interp(
                        np.linspace(0.0, 1.0, target_len),
                        np.linspace(0.0, 1.0, a.size),
                        a,
                    ))
            else:
                promoted.append(np.full(target_len, a, dtype=float))
        stacked = np.vstack(promoted)             # (3, N)
        rotated = R @ stacked                     # (3, N)
        new_amps = [rotated[i] for i in range(3)]
    else:
        vec = np.asarray(amps, dtype=float)
        rotated = R @ vec
        new_amps = [float(rotated[i]) for i in range(3)]

    # Pick a timing donor: prefer the input with the longest amplitude
    # array, or the largest |A| if scalar.
    def _score(g: Grad) -> float:
        if isinstance(g.A, np.ndarray):
            return float(g.A.size)
        return abs(float(g.A))
    donor = max((gx, gy, gz), key=_score)

    # first/last are amplitudes on the same three axes, so they rotate with
    # the waveform. Dropping them would leave the boundary samples of a
    # rotated arbitrary gradient at zero.
    new_first = R @ np.array([gx.first, gy.first, gz.first], dtype=float)
    new_last = R @ np.array([gx.last, gy.last, gz.last], dtype=float)

    out = []
    for axis, new_A in enumerate(new_amps):
        out.append(Grad(
            A=new_A,
            T=donor.T,
            rise=donor.rise,
            fall=donor.fall,
            delay=donor.delay,
            first=float(new_first[axis]),
            last=float(new_last[axis]),
        ))
    return tuple(out)


# ---------------------------------------------------------------------------
# Fix first/last gradient samples (compatibility helper)
# ---------------------------------------------------------------------------

def fix_first_last_grads(seq: PulseqSequence) -> None:
    """
    Update Sequence with first/last points for gradients.

    Only needed below v1.5, where the boundary samples are absent from the file
    and have to be reconstructed from the shape and the preceding block.
    """
    grad_prev_last = [0.0, 0.0, 0.0]

    for bi in range(len(seq)):
        gx, gy, gz = seq.GR[bi]
        grads = [gx, gy, gz]
        if seq.DUR[bi] <= 0:
            continue

        for gi, gr in enumerate(grads):
            A = gr.A
            # treat scalar as length-1 array for the check
            if isinstance(A, np.ndarray):
                sum_abs = float(np.sum(np.abs(A)))
            else:
                sum_abs = abs(A)

            if sum_abs == 0.0:
                grad_prev_last[gi] = 0.0
                continue

            # only shaped gradients (A is an array) get first/last computed
            if isinstance(A, np.ndarray):
                if gr.delay > 0:
                    grad_prev_last[gi] = 0.0

                gr.first = grad_prev_last[gi]

                if isinstance(gr.T, np.ndarray):
                    # time-shaped case – last sample is last amplitude
                    gr.last = float(A[-1])
                else:
                    # Uniformly-shaped case (extended trapezoid): the
                    # boundary sample follows from an alternating-sign
                    # cumulative sum over the amplitudes.
                    odd_step1 = np.concatenate(([gr.first], 2.0 * A))
                    idx = np.arange(1, odd_step1.size + 1)
                    sign_vec = (idx % 2) * 2 - 1
                    odd_step2 = odd_step1 * sign_vec
                    waveform_odd_rest = np.cumsum(odd_step2) * sign_vec
                    gr.last = float(waveform_odd_rest[-1])

                grad_prev_last[gi] = gr.last
            else:
                # trapezoid case
                grad_prev_last[gi] = 0.0


# ---------------------------------------------------------------------------
# Reading Grad / RF / ADC for a block
# ---------------------------------------------------------------------------

# Map the single-character v1.5 RF 'use' code to its full name. Mirrors
# pypulseq/Sequence/sequence.py:rf_from_lib_data. Anything else (or
# missing) falls through to 'undefined'.
_USE_CODE_TO_STR: Dict[str, str] = {
    "e": "excitation",
    "r": "refocusing",
    "i": "inversion",
    "s": "saturation",
    "p": "preparation",
    "o": "other",
    "u": "undefined",
}


def read_Grad(grad_library: Dict[int, Dict[str, Any]],
              shape_library: Dict[int, Tuple[int, np.ndarray]],
              dt_gr: float,
              idx: int,
              pulseq_version: Version = Version(1, 4, 0)) -> Grad:
    """
    Construct a Grad object from gradient and shape libraries.

    Arbitrary-gradient rows have a version-dependent column layout:
    v1.4 has 4 columns ``(amp, amp_id, time_id, delay)``; v1.5 inserts
    the ``first`` and ``last`` boundary samples immediately after the
    amplitude, giving 6 columns ``(amp, first, last, amp_id, time_id,
    delay)``. Trapezoidal rows are unchanged.
    """
    if not grad_library or idx == 0:
        return Grad(0.0, 0.0)

    entry = grad_library[idx]
    gtype = entry.get("type")
    data = entry["data"]

    if gtype == ord("t") or gtype == "t":
        # trapezoidal gradient: (1)amplitude (2)rise (3)flat (4)fall (5)delay
        assert len(data) == 5, (
            f"[Grad id {idx}] expected 5 trapezoid columns, got {len(data)}"
        )
        g_A, g_rise, g_T, g_fall, g_delay = map(float, data)
        return Grad(g_A, g_T, g_rise, g_fall, g_delay)

    if gtype == ord("g") or gtype == "g":
        if pulseq_version >= Version(1, 5, 0):
            assert len(data) == 6, (
                f"[Grad id {idx}] v1.5 expects 6 arbitrary-grad columns, got {len(data)}"
            )
            amplitude = float(data[0])
            first_val = float(data[1])
            last_val = float(data[2])
            amp_shape_id = int(math.floor(float(data[3])))
            time_shape_id = int(math.floor(float(data[4])))
            delay = float(data[5])
        else:
            assert len(data) == 4, (
                f"[Grad id {idx}] v1.4 expects 4 arbitrary-grad columns, got {len(data)}"
            )
            amplitude = float(data[0])
            amp_shape_id = int(math.floor(float(data[1])))
            time_shape_id = int(math.floor(float(data[2])))
            first_val = 0.0
            last_val = 0.0
            delay = float(data[3])

        num_samp, amp_data = shape_library[amp_shape_id]
        gA = amplitude * decompress_shape(num_samp, amp_data)
        Nrf = gA.size - 1

        if time_shape_id <= 0:
            # no time waveform (uniform raster); v1.5.0 uses time_shape_id=-1 for half-raster.
            gT = Nrf * dt_gr
            g = Grad(gA, gT, dt_gr / 2.0, dt_gr / 2.0, delay)
        else:
            num_t, t_data = shape_library[time_shape_id]
            gt = decompress_shape(num_t, t_data)
            gT = np.diff(gt) * dt_gr
            g = Grad(gA, gT, 0.0, 0.0, delay)

        g.first = first_val
        g.last = last_val
        return g

    # fallback
    return Grad(0.0, 0.0)


def read_RF(rf_library: Dict[int, Dict[str, Any]],
            shape_library: Dict[int, Tuple[int, np.ndarray]],
            dt_rf: float,
            idx: int,
            pulseq_version: Version = Version(1, 4, 0)) -> RF:
    """
    Construct an RF object from libraries.

    Column layout is version-dependent:
      * v1.4: 7 columns ``(amp, mag_id, ph_id, time_id, delay, freq, phase)``.
      * v1.5: 11 columns ``(amp, mag_id, ph_id, time_id, center, delay,
        freq_ppm, phase_ppm, freq, phase, use)``. The final column is a
        single character functional label.
    """
    if not rf_library or idx == 0:
        return RF(np.zeros(1, dtype=complex), 0.0, 0.0, 0.0)

    data = rf_library[idx]["data"]
    amplitude = float(data[0])
    mag_id = int(math.floor(float(data[1])))
    phase_id = int(math.floor(float(data[2])))
    time_shape_id = int(math.floor(float(data[3])))

    if pulseq_version >= Version(1, 5, 0):
        assert len(data) == 11, (
            f"[RF id {idx}] v1.5 expects 11 columns, got {len(data)}"
        )
        # data[4] is 'center'; we keep the raw delay column (5).
        # v1.5 introduces time_shape_id=-1 for half-raster RF; treat it
        # like the uniform-raster case (time_shape_id == 0).
        delay = float(data[5]) + (dt_rf / 2.0 if time_shape_id <= 0 else 0.0)
        freq_ppm = float(data[6])
        phase_ppm = float(data[7])
        freq = float(data[8])
        phase = float(data[9])
        use_code = str(data[10]).strip().lower()[:1] or "u"
        use_label = _USE_CODE_TO_STR.get(use_code, "undefined")
    else:
        assert len(data) == 7, (
            f"[RF id {idx}] v1.4 expects 7 columns, got {len(data)}"
        )
        delay = float(data[4]) + (dt_rf / 2.0 if time_shape_id <= 0 else 0.0)
        freq = float(data[5])
        phase = float(data[6])
        freq_ppm = 0.0
        phase_ppm = 0.0
        use_label = "undefined"

    if amplitude != 0.0 and mag_id != 0:
        num_mag, mag_data = shape_library[mag_id]
        rfA = decompress_shape(num_mag, mag_data)
        num_phase, phase_data = shape_library[phase_id]
        rf_phi = decompress_shape(num_phase, phase_data)
        if not np.all(rf_phi >= 0.0):
            raise AssertionError(
                f"[RF id {idx}] Phase waveform rfϕ must have non-negative samples."
            )
        Nrf = num_mag - 1
        # amplitude * mag * exp(i*(2π*rfϕ + phase))
        rfAphi = amplitude * rfA * np.exp(1j * (2.0 * math.pi * rf_phi + phase))
    else:
        rfAphi = np.zeros(1, dtype=complex)
        Nrf = 1

    if time_shape_id <= 0:
        rfT = Nrf * dt_rf
    else:
        num_t, t_data = shape_library[time_shape_id]
        rft = decompress_shape(num_t, t_data)
        rfT = np.diff(rft) * dt_rf

    return RF(rfAphi, rfT, freq, delay,
              use=use_label, freq_ppm=freq_ppm, phase_ppm=phase_ppm)


def read_ADC(adc_library: Dict[int, Dict[str, Any]], idx: int,
             pulseq_version: Version = Version(1, 4, 0),
             shape_library: Optional[Dict[int, Tuple[int, np.ndarray]]] = None
             ) -> ADC:
    """
    Construct an ADC object from library.

    v1.4: 5 columns ``(num, dwell, delay, freq, phase)``.
    v1.5: 8 columns ``(num, dwell, delay, freq_ppm, phase_ppm, freq,
    phase, phase_id)``, where ``phase_id`` references a per-sample phase
    shape resolved against ``shape_library``.
    """
    if not adc_library or idx == 0:
        return ADC(0, 0.0, 0.0, 0.0, 0.0)

    data = adc_library[idx]["data"]
    if pulseq_version >= Version(1, 5, 0):
        assert len(data) == 8, (
            f"[ADC id {idx}] v1.5 expects 8 columns, got {len(data)}"
        )
        num = int(math.floor(float(data[0])))
        dwell = float(data[1])
        delay = float(data[2]) + dwell / 2.0
        freq_ppm = float(data[3])
        phase_ppm = float(data[4])
        freq = float(data[5])
        phase = float(data[6])
        # data[7] is a shape id into the shape library, holding one phase
        # value per ADC sample. 0 means no modulation.
        phase_shape_id = int(math.floor(float(data[7])))
        if phase_shape_id > 0:
            if shape_library is None or phase_shape_id not in shape_library:
                raise KeyError(
                    f"[ADC id {idx}] references phase shape {phase_shape_id}, "
                    f"which is not in the shape library"
                )
            num_ph, ph_data = shape_library[phase_shape_id]
            phase_modulation = decompress_shape(num_ph, ph_data)
        else:
            phase_modulation = None
    else:
        assert len(data) == 5, (
            f"[ADC id {idx}] v1.4 expects 5 columns, got {len(data)}"
        )
        num = int(math.floor(float(data[0])))
        dwell = float(data[1])
        delay = float(data[2]) + dwell / 2.0
        freq = float(data[3])
        phase = float(data[4])
        freq_ppm = 0.0
        phase_ppm = 0.0
        phase_modulation = None
    T = (num - 1) * dwell
    return ADC(num, T, delay, freq, phase,
               freq_ppm=freq_ppm, phase_ppm=phase_ppm,
               phase_modulation=phase_modulation)


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

def read_extension(extension_library: Dict[int, Dict[str, Any]],
                   extension_type: Dict[int, Dict[str, Any]],
                   trigger_library: Dict[int, Dict[str, Any]],
                   labelset_library: Dict[int, Dict[str, Any]],
                   labelinc_library: Dict[int, Dict[str, Any]],
                   idx: int,
                   rotation_library: Optional[Dict[int, Dict[str, Any]]] = None
                   ) -> List[Extension]:
    """
    Read extension(s) for a block.
    Returns a list of Extension objects.
    """
    if not extension_library or idx == 0:
        return []

    result: List[Extension] = []

    # Each entry in extension_library is: [type, ref, next_id]. The chain is
    # walked by next_id, which the file is free to make cyclic, so track what
    # has been visited rather than trusting it to terminate.
    entry = extension_library.get(idx)
    if entry is None:
        logger.warning("Extension list #%d does not exist", idx)
        return []
    type_id, ref, next_id = entry["data"]
    visited = {idx}

    while True:
        if type_id not in extension_type:
            logger.warning("Extension type #%d does not exist", type_id)
            break

        ext_type_name = extension_type[type_id]["data"]

        if ext_type_name == "LABELSET":
            # read_labels stored [val_int, val_str]; the canonical
            # Pulseq spec attaches the integer value to a named label,
            # so swap the order here when constructing the dataclass.
            val, lab = labelset_library[ref]["data"]
            result.append(LabelSet(str(lab), int(val)))
        elif ext_type_name == "LABELINC":
            val, lab = labelinc_library[ref]["data"]
            result.append(LabelInc(str(lab), int(val)))
        elif ext_type_name == "TRIGGERS":
            ch, mode, rise, fall = trigger_library[ref]["data"]
            result.append(Trigger(ch, mode, rise, fall))
        elif ext_type_name == "ROTATIONS" and rotation_library is not None:
            entry = rotation_library.get(ref)
            if entry is not None:
                matrix = np.asarray(entry["data"], dtype=float).reshape(3, 3)
                result.append(Rotation(matrix=matrix))
            else:
                logger.warning("Rotation extension ref #%d not found", ref)
        else:
            logger.warning("Extension type '%s' not implemented", ext_type_name)

        if next_id == 0:
            break
        if next_id in visited:
            logger.warning(
                "Extension list #%d is cyclic at entry #%d; stopping the walk",
                idx, next_id)
            break
        entry = extension_library.get(next_id)
        if entry is None:
            logger.warning("Extension list entry #%d does not exist", next_id)
            break
        visited.add(next_id)
        type_id, ref, next_id = entry["data"]

    return result


# ---------------------------------------------------------------------------
# High-level sequence reader
# ---------------------------------------------------------------------------

def read_seq(filename: str, gamma: float = GAMMA) -> PulseqSequence:
    """
    Read a Pulseq `.seq` file and return a PulseqSequence object.

    ``gamma`` (Hz/T) converts the file's Hz/m gradients and Hz RF amplitudes
    into T/m and T. It must be the SAME constant the solver later multiplies
    by, or the encoding is off by the ratio of the two: the file fixes the
    phase integral in Hz, and only a matching gamma reproduces it from
    ``gamma * B``. :func:`import_pulseq` therefore passes the scanner's
    ``gammabar`` rather than the module default.
    """
    logger.info("Loading sequence %s ...", os.path.basename(filename))

    pulseq_version = Version(0, 0, 0)
    grad_library: Dict[int, Dict[str, Any]] = {}
    defs: Dict[str, Any] = {}
    signature = ""
    block_events: Dict[int, List[int]] = {}
    block_durations: Dict[int, float] = {}
    delay_ind_tmp: Dict[int, int] = {}
    rf_library: Dict[int, Dict[str, Any]] = {}
    adc_library: Dict[int, Dict[str, Any]] = {}
    tmp_delay_library: Dict[int, Dict[str, Any]] = {}
    shape_library: Dict[int, Tuple[int, np.ndarray]] = {}
    extension_library: Dict[int, Dict[str, Any]] = {}
    trigger_library: Dict[int, Dict[str, Any]] = {}
    extension_type: Dict[int, Dict[str, Any]] = {}
    labelset_library: Dict[int, Dict[str, Any]] = {}
    labelinc_library: Dict[int, Dict[str, Any]] = {}
    rotation_library: Dict[int, Dict[str, Any]] = {}

    with open(filename, "r") as io:
        while True:
            section_line = io.readline()
            if not section_line:
                break
            section = section_line.strip()
            if not section or section.startswith("#"):
                continue

            if section == "[DEFINITIONS]":
                defs = read_definitions(io)
            elif section == "[VERSION]":
                pulseq_version = read_version(io)
            elif section == "[BLOCKS]":
                if pulseq_version == Version(0, 0, 0):
                    raise RuntimeError(
                        "Pulseq file MUST include [VERSION] section prior to [BLOCKS] section"
                    )
                block_events, block_durations, delay_ind_tmp = read_blocks(
                    io, defs["BlockDurationRaster"], pulseq_version
                )
            elif section == "[RF]":
                if pulseq_version >= Version(1, 5, 0):
                    # v1.5: amp, mag_id, ph_id, time_id, center, delay,
                    #       freq_ppm, phase_ppm, freq, phase, use
                    # The trailing 'use' column is a single character; the
                    # ``np.nan`` sentinel keeps it as a raw string token.
                    rf_library = read_events(
                        io,
                        [1.0 / gamma, 1.0, 1.0, 1.0, 1e-6, 1e-6,
                         1.0, 1.0, 1.0, 1.0, np.nan],
                    )
                elif pulseq_version >= Version(1, 4, 0):
                    rf_library = read_events(
                        io, [1.0 / gamma, 1.0, 1.0, 1.0, 1e-6, 1.0, 1.0]
                    )
                else:
                    rf_library = read_events(
                        io, [1.0 / gamma, 1.0, 1.0, 1e-6, 1.0, 1.0]
                    )
            elif section == "[GRADIENTS]":
                if pulseq_version >= Version(1, 5, 0):
                    # v1.5: amp, first, last, amp_id, time_id, delay. The two
                    # boundary samples are inserted directly after the
                    # amplitude, not after the shape ids, and carry the same
                    # Hz/m units as the amplitude.
                    grad_library = read_events(
                        io,
                        [1.0 / gamma, 1.0 / gamma, 1.0 / gamma,
                         1.0, 1.0, 1e-6],
                        type_=ord("g"), event_library=grad_library
                    )
                elif pulseq_version >= Version(1, 4, 0):
                    grad_library = read_events(
                        io, [1.0 / gamma, 1.0, 1.0, 1e-6],
                        type_=ord("g"), event_library=grad_library
                    )
                else:
                    grad_library = read_events(
                        io, [1.0 / gamma, 1.0, 1e-6],
                        type_=ord("g"), event_library=grad_library
                    )
            elif section == "[TRAP]":
                grad_library = read_events(
                    io, [1.0 / gamma, 1e-6, 1e-6, 1e-6, 1e-6],
                    type_=ord("t"), event_library=grad_library
                )
            elif section == "[ADC]":
                if pulseq_version >= Version(1, 5, 0):
                    # v1.5: num, dwell, delay, freq_ppm, phase_ppm,
                    #       freq, phase, phase_id
                    adc_library = read_events(
                        io, [1.0, 1e-9, 1e-6, 1.0, 1.0, 1.0, 1.0, 1.0]
                    )
                else:
                    adc_library = read_events(
                        io, [1.0, 1e-9, 1e-6, 1.0, 1.0]
                    )
            elif section == "[DELAYS]":
                if pulseq_version >= Version(1, 4, 0):
                    raise RuntimeError(
                        "Pulseq file revision 1.4.0 and above MUST NOT contain [DELAYS] section"
                    )
                tmp_delay_library = read_events(io, [1e-6])
            elif section == "[SHAPES]":
                force_convert = (pulseq_version.major == 1 and pulseq_version.minor < 4)
                shape_library = read_shapes(io, force_convert)
            elif section == "[EXTENSIONS]":
                extension_library = read_extension_blocks(io)
            elif section == "[SIGNATURE]":
                signature = read_signature(io)
            else:
                # extension sections like "extensionTRIGGERS..."
                if section.startswith("extension"):
                    extension_name = section[10:]  # after "extension_"
                    if extension_name.startswith("TRIGGERS"):
                        ext_id = int(extension_name[8:])
                        extension_type[ext_id] = {"data": "TRIGGERS"}
                        trigger_library = read_events(
                            io, [1.0, 1.0, 1e-6, 1e-6], event_library=trigger_library
                        )
                    elif extension_name.startswith("LABELSET"):
                        ext_id = int(extension_name[8:])
                        extension_type[ext_id] = {"data": "LABELSET"}
                        labelset_library = read_labels(io, event_library=labelset_library)
                    elif extension_name.startswith("LABELINC"):
                        ext_id = int(extension_name[8:])
                        extension_type[ext_id] = {"data": "LABELINC"}
                        labelinc_library = read_labels(io, event_library=labelinc_library)
                    elif extension_name.startswith("ROTATIONS"):
                        ext_id = int(extension_name[9:])
                        extension_type[ext_id] = {"data": "ROTATIONS"}
                        rotation_library = read_events(
                            io, [1.0] * 9, event_library=rotation_library
                        )
                    elif extension_name.startswith("DELAYS"):
                        n_dropped = skip_section(io)
                        logger.warning(
                            "DELAYS extension is not handled; %d row(s) ignored",
                            n_dropped)
                    else:
                        n_dropped = skip_section(io)
                        logger.warning(
                            "Ignoring unknown extension %s (%d row(s))",
                            extension_name, n_dropped)
                else:
                    raise RuntimeError(f"Unknown section code: {section}")

    # Fix blocks, gradients and RF objects imported from older versions
    if pulseq_version < Version(1, 4, 0):
        # RF: add a dummy time_shape field at position 4 (after first 3 entries)
        for i in range(len(rf_library)):
            if i in rf_library:
                data = rf_library[i]["data"]
                new_data = np.concatenate((data[:3], [0.0], data[3:]))
                rf_library[i]["data"] = new_data

        # Grad: update trapezoids ('t') and free-shape gradients ('g')
        grad_raster = defs.get("gradRasterTime", defs.get("GradientRasterTime", 1e-5))

        for i in range(len(grad_library)):
            if i not in grad_library:
                continue
            entry = grad_library[i]
            gtype = entry.get("type")
            data = entry["data"]

            if gtype == ord("t"):
                # (1)amplitude (2)rise (3)flat (4)fall (5)delay
                # fix missing rise/delay when amplitude is 0 and flat>0
                if data[1] == 0.0:  # rise
                    if abs(data[0]) == 0.0 and data[2] > 0.0:
                        data[2] -= grad_raster
                        data[1] = grad_raster
                if data[3] == 0.0:  # delay
                    if abs(data[0]) == 0.0 and data[2] > 0.0:
                        data[2] -= grad_raster
                        data[3] = grad_raster
                entry["data"] = data

            if gtype == ord("g"):
                # (1)amplitude (2)amp_shape_id (3)time_shape_id (4)delay
                # insert dummy 0 at position 3
                new_data = np.concatenate((data[:2], [0.0], data[2:]))
                entry["data"] = new_data

        # For versions prior to 1.4.0 blockDurations have not been initialized
        if not block_durations:
            # Durations are keyed by block id, not by position.
            for bid in block_events.keys():
                idelay = delay_ind_tmp.get(bid, 0)
                delay = 0.0
                if idelay > 0 and idelay in tmp_delay_library:
                    delay = float(tmp_delay_library[idelay]["data"][0])
                block_durations[bid] = delay

    # Transform to Sequence blocks
    seq = PulseqSequence()
    n_blocks = len(block_events)
    grad_raster_time = float(defs.get("GradientRasterTime", 1e-5))
    rf_raster_time = float(defs.get("RadiofrequencyRasterTime", 1e-6))

    for i in range(1, n_blocks + 1):
        if i not in block_events:
            continue
        idelay, irf, ix, iy, iz, iadc, iext = block_events[i]

        gx = read_Grad(grad_library, shape_library, grad_raster_time, ix,
                       pulseq_version=pulseq_version)
        gy = read_Grad(grad_library, shape_library, grad_raster_time, iy,
                       pulseq_version=pulseq_version)
        gz = read_Grad(grad_library, shape_library, grad_raster_time, iz,
                       pulseq_version=pulseq_version)

        rf = read_RF(rf_library, shape_library, rf_raster_time, irf,
                     pulseq_version=pulseq_version)
        adc = read_ADC(adc_library, iadc, pulseq_version=pulseq_version,
                       shape_library=shape_library)

        # block duration: max of blockDurations[i] and event durations
        d_list = [
            block_durations.get(i, 0.0),
            dur_grad(gx),
            dur_grad(gy),
            dur_grad(gz),
            dur_rf(rf),
            dur_adc(adc),
        ]
        D = float(max(d_list))

        ext_list = read_extension(
            extension_library,
            extension_type,
            trigger_library,
            labelset_library,
            labelinc_library,
            iext,
            rotation_library=rotation_library,
        )

        # Apply ROTATIONS extension(s) to the (gx, gy, gz) triple in place.
        # Per Pulseq spec §2.8.4 the rotation acts on gradient amplitudes,
        # whether a scalar trapezoid or a per-sample shape.
        for ext in ext_list:
            if isinstance(ext, Rotation):
                gx, gy, gz = _apply_rotation_to_grads(ext.matrix, gx, gy, gz)

        seq.add_block(gx, gy, gz, rf, adc, D, ext_list)

    # Add first and last points for gradients. v1.5 files already carry
    # these boundary samples on every arbitrary gradient row, so the
    # in-house recomputation is skipped — the file is authoritative.
    if pulseq_version < Version(1, 5, 0):
        fix_first_last_grads(seq)

    # Final details
    seq.DEF.update(defs)
    seq.DEF["FileName"] = os.path.basename(filename)
    seq.DEF["PulseqVersion"] = pulseq_version
    seq.DEF["signature"] = signature
    # Absolute path retained so downstream helpers (e.g. kspace_trajectory,
    # import_pulseq) can re-open the file via pypulseq when they need the
    # v1.5 k-space-calculation API.
    seq.DEF["__pulseq_path__"] = os.path.abspath(filename)

    # Recon dimensions are not in the file; infer them from the events.
    # Nx
    if "Nx" not in seq.DEF:
        nx = max((adc.num for adc in seq.ADC), default=0)
        seq.DEF["Nx"] = int(nx)

    # Nz
    if "Nz" not in seq.DEF:
        unique_df = {rf.df for rf in seq.RF}
        seq.DEF["Nz"] = int(len(unique_df)) if unique_df else 1

    # Ny
    if "Ny" not in seq.DEF:
        nz = seq.DEF.get("Nz", 1) or 1
        num_adc_on = sum(1 for adc in seq.ADC if adc.num > 0)
        seq.DEF["Ny"] = int(num_adc_on // nz)

    return seq


# ---------------------------------------------------------------------------
# Convert PulseqSequence blocks to feelMRI objects
# ---------------------------------------------------------------------------

def _trap_waveform_seconds(g: "Grad") -> Tuple[np.ndarray, np.ndarray]:
  """Build a 4-point trapezoid waveform from a parsed trapezoidal Grad.

  Returns (timings_seconds, amplitudes_Tm). The flat-top duration is
  ``g.T`` (a scalar in seconds) for trapezoidal events.
  """
  delay = float(g.delay)
  rise = float(g.rise)
  fall = float(g.fall)
  flat = float(g.T)
  amp = float(g.A)
  t = np.array([delay, delay + rise, delay + rise + flat, delay + rise + flat + fall], dtype=float)
  a = np.array([0.0, amp, amp, 0.0], dtype=float)
  return t, a


def _shaped_waveform_seconds(g: "Grad") -> Tuple[np.ndarray, np.ndarray]:
  """Build a (timings_seconds, amplitudes_Tm) pair for an arbitrary gradient.

  ``g.A`` is the per-sample amplitude array. ``g.T`` is either a per-step
  dwell array of length ``len(g.A) - 1`` (an extended trapezoid, whose shape
  already carries its own boundary samples) or a scalar total duration (an
  arbitrary gradient on the regular raster).

  In the regular-raster case the samples sit at raster CENTRES, not at the
  block boundaries: ``read_Grad`` records the two half-raster shoulders as
  ``rise`` and ``fall``, and the amplitudes at the boundaries themselves are
  the ``first`` / ``last`` columns of the file (v1.5) or the values
  ``fix_first_last_grads`` derives (v1.4). Both are prepended and appended
  here, which is what pypulseq's ``waveforms_and_times`` does. Without them
  the waveform is shifted half a raster and its two end ramps are missing.
  """
  delay = float(g.delay)
  amps = np.asarray(g.A, dtype=float)
  if isinstance(g.T, np.ndarray):
    dwells = np.asarray(g.T, dtype=float)
    times = np.concatenate(([0.0], np.cumsum(dwells)))
    if times.size != amps.size:
      n = min(times.size, amps.size)
      times = times[:n]
      amps = amps[:n]
    return delay + times, amps

  total = float(g.T)
  rise = float(g.rise)
  fall = float(g.fall)
  times = rise + np.linspace(0.0, total, amps.size)
  times = np.concatenate(([0.0], times, [rise + total + fall]))
  amps = np.concatenate(([float(g.first)], amps, [float(g.last)]))
  return delay + times, amps


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


# RF use labels that open a coherence period. Kept for reference and for
# callers that reason about grouping; the ANCHOR itself is not chosen from this
# set -- see _identify_readout_groups for why any RF must anchor.
_ANCHOR_USES = frozenset(('excitation', 'refocusing', 'undefined'))


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
  # event. Neither is what a scanner does -- one ramps absurdly slowly, the
  # other jumps infinitely fast -- so refusing to guess and saying so is the
  # only honest answer.
  #
  # Measured on the old tests/data/arb_v15.seq, whose shaped Gx stopped one
  # sample short of a full sine period and so ended at 1.57% of peak: the kx
  # handed to mri_signal ran to -8.43 1/m where the solver's own gradients
  # played ~0, i.e. **1.686 cycles of phase across the 0.2 m FOV**, invisible
  # to check_timing (which returns ok=True) and to every test.
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

    # The magnetization is snapshotted at the END of the anchor block, and
    # calculate_kspace measures k from the excitation, so the snapshot already
    # carries whatever moment had accumulated by then. Handing the assembler
    # the file's k as-is winds that moment a second time -- on the bundled
    # files that is 252 to 460 1/m of slice-select, one to two whole cycles
    # across the slice. Subtract it, so rw.kspace is the encoding measured
    # FROM THE SNAPSHOT, which is what pairs with rw's Mxy column.
    #
    # The anchor is the last RF before the readout, so nothing between it and
    # the first sample reflects k and a plain integral is valid there. Working
    # backwards from the first ADC sample also keeps any earlier refocusing
    # reflections, which pypulseq has already folded into k_traj_adc.
    if m_block >= 0 and times_arr.size:
      t_anchor_ms = float(block_end_s[m_block]) * 1e3
      k_at_anchor = kspace_file[0].astype(float) - _gradient_moment_between(
          feelmri_seq, t_anchor_ms, float(times_arr[0]), gammabar)
    else:
      t_anchor_ms = float(t_start) * 1e3
      k_at_anchor = np.zeros(3, dtype=float)
    kspace = (kspace_file - k_at_anchor).astype(np.float32, copy=False)

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


def simulate_pulseq(seq_path,
                    phantom,
                    *,
                    scanner: Optional[Scanner] = None,
                    pod=None,
                    gather: bool = True,
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
  from feelmri.Bloch import BlochSolver
  from feelmri.MPIUtilities import gather_data

  if scanner is None:
    scanner = Scanner()
  imp = import_pulseq(seq_path, scanner=scanner, **(import_kwargs or {}))

  solver = BlochSolver(sequence=imp.feelmri_seq, phantom=phantom,
                       scanner=scanner, **solver_kwargs)
  Mxy, Mz = solver.solve()

  # When the solver carried a spectral sub-ensemble, reproduce the readout from
  # it rather than from the collapsed magnetization. Needs the static fields the
  # caller set, which the phantom remembers for exactly this.
  bins = None
  if getattr(solver, 'bin_magnetization', None) is not None:
    remembered = getattr(phantom, '_static_fields', None)
    if remembered is None:
      logger.warning(
          "t2_prime is set but set_static_fields was never called, so the "
          "readout cannot be reproduced per sub-spin and every echo will be "
          "attenuated by the dephasing standing at its anchor")
    elif remembered[1].shape[0] != solver.bin_offsets.shape[0]:
      _collective_raise(
          f"simulate_pulseq: the remembered off-resonance map has "
          f"{remembered[1].shape[0]} rows against {solver.bin_offsets.shape[0]} "
          f"sub-spin offsets. set_static_fields must be called under the same "
          f"partition the solver was built on, or the two describe different "
          f"nodes")
    else:
      bins = (solver.bin_magnetization, solver.bin_offsets,
              solver.bin_weights, remembered[0], remembered[1])

  # Under dual partitioning every set_static_fields and update_magnetization is
  # an Alltoallv into the signal layout, and the bin loop below makes three of
  # them per sub-spin: measured on cpmg_v15 at K = 16, 4 redistributions for the
  # whole simulation became 208. The window-independent arrays are moved once
  # here and the per-window ensemble once below, after which the loop runs with
  # the signal layout already active and communicates nothing.
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

  kspace: List[np.ndarray] = []
  times: List[np.ndarray] = []
  for rw in imp.readouts:
    if rw.m_storage_idx < 0:
      logger.warning(
          "readout blocks %d-%d have no coherence anchor and are skipped",
          rw.first_block, rw.last_block)
      continue
    phantom.update_magnetization(Mxy[:, rw.m_storage_idx])
    # Elapsed time since the snapshot, not absolute time from the start of the
    # file. mri_signal uses t for exp(-t/T2) and exp(i*phi*t), both of which
    # continue from the instant the magnetization was captured; feeding it
    # absolute times applies a spurious exp(-t_anchor/T2) to the whole window.
    points, t = _reshape_signal_inputs(
        rw.kspace[:, 0], rw.kspace[:, 1], rw.kspace[:, 2],
        rw.times - rw.t_anchor, None)
    # `t` is elapsed-since-snapshot, which is what the relaxation and
    # off-resonance factors need -- but the POD weights need ABSOLUTE
    # sequence time, because that is the frame the motion is defined in.
    # `get_weights` reconciles the two by adding the trajectory's own
    # `timeshift`, so point it at this window's anchor. Without this every
    # window sampled the motion from cycle phase 0 and the readout
    # deformation disagreed with the one the solver used at the same
    # instant. Restored afterwards so the caller's object comes back
    # unchanged.
    shift = getattr(pod, 'timeshift', None) if pod is not None else None
    if shift is not None:
      pod.update_timeshift(float(rw.t_anchor))
    try:
      if bins is None:
        signal = phantom.mri_signal(list(points), t, pod)
      else:
        # Bin-by-bin readout. Collapsing the sub-ensemble at the snapshot and
        # letting the assembler replay a single exp(-t/T2) from there cannot
        # reproduce a readout: the snapshot sits at the coherence ANCHOR, where
        # the ensemble is maximally dephased, and nothing downstream can bring
        # it back. Measured on cpmg_v15 at T2' = 8 ms, every echo came out
        # scaled by exp(-0.5*(tau/T2')^2) = 0.82.
        #
        # Each sub-spin is instead given its own off-resonance -- the bin
        # offsets are in the same rad/ms frame as phi_dB0, so they simply add --
        # and the signals are weight-summed. Exact, and it costs n_bins passes
        # over the signal path per window.
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
            contribution = w * phantom.mri_signal(list(points), t, pod)
            signal = contribution if signal is None else signal + contribution
    finally:
      if shift is not None:
        pod.update_timeshift(shift)
      if bins is not None:
        phantom.set_static_fields(bins[3], bins[4])
        # The bin loop above left the phantom holding the LAST sub-spin --
        # a tail bin of the quadrature, weight ~1e-16 -- so anything the
        # caller evaluates afterwards reads that instead of the collapsed
        # magnetization. Measured on cpmg_v15 at T2' = 8 ms, K = 32: S(0)
        # came back 1.133x too large and with a spurious real part where
        # the correct value is purely imaginary.
        phantom.update_magnetization(Mxy[:, rw.m_storage_idx])
    # The receiver's frequency/phase offsets and any per-sample phase shape.
    signal = rw.demodulate(signal)
    kspace.append(gather_data(signal) if gather else signal)
    times.append(rw.times)

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