"""
The Pulseq ``.seq`` file format: reading it, and nothing else.

This half knows about sections, shapes, event libraries and the v1.2-v1.5
column layouts. It builds a :class:`PulseqSequence` of plain dataclasses and
stops there. Turning those into ``feelmri`` gradients, RF pulses and
sequence blocks is :mod:`feelmri.PulseqAdapter`, which is also where anything
needing ``pypulseq`` lives.

The in-house reader is kept rather than delegated to pypulseq deliberately: it
agrees with pypulseq on gradients, RF, ADC times, labels and anchors, and
handles both the v1.4 and v1.5 column layouts, where pypulseq 1.4 mis-parses a
v1.5 file rather than rejecting it.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


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
    """The ``[VERSION]`` section of a ``.seq`` file."""

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

def _section_rows(io):
    """Yield the whitespace-split rows of a section body.

    Stops at end of file or at the blank line that separates this section
    from the next, leaving anything after it for the caller. A reader that
    breaks out early leaves the remainder unconsumed, exactly as the
    hand-written loops did.
    """
    while True:
        line = io.readline()
        if not line:
            return
        parts = line.split()
        if not parts:
            return
        yield parts


def read_definitions(io) -> Dict[str, Any]:
    """
    Read the [DEFINITIONS] section as a dict of key->value(s).
    Numeric tokens become floats if parseable.
    """
    defs: Dict[str, Any] = {}
    for parts in _section_rows(io):
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
    for parts in _section_rows(io):
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

    for parts in _section_rows(io):
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

    for parts in _section_rows(io):
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

    for parts in _section_rows(io):
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

    # Pulseq spells a run of one as the value alone and a longer run as
    # value, value, (length - 2). The three pieces are INTERLEAVED per run:
    # the decoder finds a run by spotting two equal consecutive entries and
    # reads the count immediately after them, so emitting all the values and
    # then all the counts describes a different waveform. That layout happens
    # to be correct for a single run, which is why a constant shape survived
    # it and a ramp did not.
    pieces = []
    for value, run in zip(vals.astype(float), n):
        if run == 1:
            pieces.append(np.array([value]))
        else:
            pieces.append(np.array([value, value, float(run) - 2.0]))
    v = np.concatenate(pieces) if pieces else np.zeros(0, dtype=float)
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
    """One TRIGGERS extension entry.

    Parsed so the file round-trips, but the simulator has no counterpart: a
    WAIT trigger stalls a scanner for a time nothing here can know, so
    ``import_pulseq`` warns and ignores it.
    """

    channel: int
    mode: int
    rise: float
    fall: float


@dataclass
class LabelSet:
    """A LABELSET entry: assign ``value`` to counter ``label``."""

    label: str   # e.g. 'SET', 'LIN', 'SLC', 'NAV', 'REF', ...
    value: int


@dataclass
class LabelInc:
    """A LABELINC entry: add ``value`` to counter ``label``."""

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

  The result is emitted as an extended trapezoid, with ``A`` the per-sample
  amplitudes, ``T`` the per-step dwells, which is the one Grad shape that
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
  are reading two different waveforms. See the call site for the measured
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
        "reads zero there, so the readout carries phase the simulation never "
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
    # in-house recomputation is skipped: the file is authoritative.
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
