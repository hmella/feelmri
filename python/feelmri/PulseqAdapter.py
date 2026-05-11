
"""
Python translation of Pulseq.jl reader utilities (behavior-preserving, more Pythonic).

This module provides tools to read Pulseq `.seq` files in a way that mirrors the
logic of the original Julia implementation, but using idiomatic Python.

It implements:
- read_version
- read_definitions
- read_signature
- read_blocks
- read_events
- read_labels
- read_extension_blocks
- read_shapes
- compress_shape
- decompress_shape
- read_Grad
- read_RF
- read_ADC
- read_extension
- read_seq  (returns a Sequence object)
- fix_first_last_grads

NOTE: This is a self‑contained module. It does not depend on the rest of KomaMRI,
so types like Sequence, Grad, RF, ADC, Trigger, LabelSet, LabelInc are defined here
in a simplified but compatible fashion.
"""

from __future__ import annotations

import cmath
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pint import Quantity

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid

from feelmri.Bloch import ADC as feelmriADC
from feelmri.Bloch import Sequence as feelmriSequence
from feelmri.Bloch import SequenceBlock
from feelmri.MRObjects import RF as feelmriRF, Gradient, Scanner

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
    Mirrors the behavior of the Julia `read_version`.
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
    elif pulseq_version >= Version(1, 5, 0):
        logger.warning(
            "Pulseq %s not yet supported by this Python translation. "
            "This mirrors the Julia warning about versions >= 1.5.0.",
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

    # Default values (matching Julia code)
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
    """
    if event_library is None:
        event_library = {}

    n_vals = len(scale)

    while True:
        line = io.readline()
        if not line:
            break
        parts = line.split()
        if not parts:
            break
        if len(parts) != n_vals + 1:
            # Julia breaks when the scanf result count != EventLength
            break
        eid = int(float(parts[0]))
        raw_vals = np.array([float(p) for p in parts[1:]], dtype=float)
        data = np.array(scale, dtype=float) * raw_vals

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
            # Julia breaks when scanf result count != 3
            break

    return event_library


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

    # Skip the first line after [SHAPES] (Julia reads and discards it)
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
    # Julia uses 1-based indices for markers; we emulate that by +1.
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
    df: float                # frequency offset
    delay: float = 0.0


@dataclass
class ADC:
    """ADC event."""
    num: int
    T: float
    delay: float
    df: float = 0.0
    phase: float = 0.0


@dataclass
class Trigger:
    channel: int
    mode: int
    rise: float
    fall: float


@dataclass
class LabelSet:
    label: int
    value: int


@dataclass
class LabelInc:
    label: int
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


def _apply_rotation_to_grads(R: np.ndarray,
                             gx: Grad, gy: Grad, gz: Grad
                             ) -> Tuple[Grad, Grad, Grad]:
    """Apply a 3x3 rotation matrix to a (gx, gy, gz) triple.

    The rotation acts on gradient amplitudes pointwise. For trapezoidal
    gradients (scalar `A`), each output amplitude is a linear combination
    of the three scalars. For arbitrary shaped gradients (`A` is an
    ndarray of length N), the output is the row of R applied per sample,
    producing a new shaped gradient.

    Timing fields (`T`, `rise`, `fall`, `delay`) are inherited from the
    largest-amplitude input axis; if the three axes already share timing
    (the common case generated by Pulseq writers), this is exact.
    """
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

    out = []
    for new_A in new_amps:
        out.append(Grad(
            A=new_A,
            T=donor.T,
            rise=donor.rise,
            fall=donor.fall,
            delay=donor.delay,
            first=0.0,
            last=0.0,
        ))
    return tuple(out)


# ---------------------------------------------------------------------------
# Fix first/last gradient samples (compatibility helper)
# ---------------------------------------------------------------------------

def fix_first_last_grads(seq: PulseqSequence) -> None:
    """
    Update Sequence with first/last points for gradients.
    Mirrors the logic of Julia `fix_first_last_grads!`.
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
                    # uniformly-shaped case (extended trapezoid)
                    # replicate Julia odd-step construction
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

def read_Grad(grad_library: Dict[int, Dict[str, Any]],
              shape_library: Dict[int, Tuple[int, np.ndarray]],
              dt_gr: float,
              idx: int) -> Grad:
    """
    Construct a Grad object from gradient and shape libraries.
    """
    if not grad_library or idx == 0:
        return Grad(0.0, 0.0)

    entry = grad_library[idx]
    gtype = entry.get("type")
    data = entry["data"]

    if gtype == ord("t") or gtype == "t":
        # trapezoidal gradient: (1)amplitude (2)rise (3)flat (4)fall (5)delay
        g_A, g_rise, g_T, g_fall, g_delay = map(float, data)
        return Grad(g_A, g_T, g_rise, g_fall, g_delay)

    if gtype == ord("g") or gtype == "g":
        # arbitrary gradient waveform:
        # (1)amplitude (2)amp_shape_id (3)time_shape_id (4)delay
        amplitude = float(data[0])
        amp_shape_id = int(math.floor(data[1]))
        time_shape_id = int(math.floor(data[2]))
        delay = float(data[3])

        # amplitude waveform
        num_samp, amp_data = shape_library[amp_shape_id]
        gA = amplitude * decompress_shape(num_samp, amp_data)
        Nrf = gA.size - 1

        if time_shape_id <= 0:
            # no time waveform (uniform raster); v1.5.0 uses time_shape_id=-1 for half-raster.
            gT = Nrf * dt_gr
            return Grad(gA, gT, dt_gr / 2.0, dt_gr / 2.0, delay)
        else:
            num_t, t_data = shape_library[time_shape_id]
            gt = decompress_shape(num_t, t_data)
            gT = np.diff(gt) * dt_gr
            return Grad(gA, gT, 0.0, 0.0, delay)

    # fallback
    return Grad(0.0, 0.0)


def read_RF(rf_library: Dict[int, Dict[str, Any]],
            shape_library: Dict[int, Tuple[int, np.ndarray]],
            dt_rf: float,
            idx: int) -> RF:
    """
    Construct an RF object from libraries.
    """
    if not rf_library or idx == 0:
        return RF(np.zeros(1, dtype=complex), 0.0, 0.0, 0.0)

    data = rf_library[idx]["data"]
    # (1)amplitude (2)mag_id (3)phase_id (4)time_shape_id (5)delay (6)freq (7)phase
    amplitude = float(data[0])
    mag_id = int(math.floor(data[1]))
    phase_id = int(math.floor(data[2]))
    time_shape_id = int(math.floor(data[3]))
    # v1.5.0 introduces time_shape_id=-1 for half-raster RF; treat it like the
    # uniform-raster case (time_shape_id == 0).
    delay = float(data[4]) + (dt_rf / 2.0 if time_shape_id <= 0 else 0.0)
    freq = float(data[5])
    phase = float(data[6])

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

    return RF(rfAphi, rfT, freq, delay)


def read_ADC(adc_library: Dict[int, Dict[str, Any]], idx: int) -> ADC:
    """
    Construct an ADC object from library.
    """
    if not adc_library or idx == 0:
        # default ADC-off
        return ADC(0, 0.0, 0.0, 0.0, 0.0)

    data = adc_library[idx]["data"]
    # (1)num (2)dwell (3)delay (4)freq (5)phase
    num = int(math.floor(data[0]))
    dwell = float(data[1])
    delay = float(data[2]) + dwell / 2.0
    freq = float(data[3])
    phase = float(data[4])
    T = (num - 1) * dwell
    return ADC(num, T, delay, freq, phase)


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

    # Each entry in extension_library is: [type, ref, next_id]
    type_id, ref, next_id = extension_library[idx]["data"]

    while True:
        if type_id not in extension_type:
            logger.warning("Extension type #%d does not exist", type_id)
            break

        ext_type_name = extension_type[type_id]["data"]

        if ext_type_name == "LABELSET":
            lab, val = labelset_library[ref]["data"]
            result.append(LabelSet(lab, val))
        elif ext_type_name == "LABELINC":
            lab, val = labelinc_library[ref]["data"]
            result.append(LabelInc(lab, val))
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
        type_id, ref, next_id = extension_library[next_id]["data"]

    return result


# ---------------------------------------------------------------------------
# High-level sequence reader
# ---------------------------------------------------------------------------

def read_seq(filename: str) -> PulseqSequence:
    """
    Read a Pulseq `.seq` file and return a PulseqSequence object.
    This follows the control flow of the Julia `read_seq` function.
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
                if pulseq_version >= Version(1, 4, 0):
                    rf_library = read_events(
                        io, [1.0 / GAMMA, 1.0, 1.0, 1.0, 1e-6, 1.0, 1.0]
                    )
                else:
                    rf_library = read_events(
                        io, [1.0 / GAMMA, 1.0, 1.0, 1e-6, 1.0, 1.0]
                    )
            elif section == "[GRADIENTS]":
                if pulseq_version >= Version(1, 4, 0):
                    grad_library = read_events(
                        io, [1.0 / GAMMA, 1.0, 1.0, 1e-6],
                        type_=ord("g"), event_library=grad_library
                    )
                else:
                    grad_library = read_events(
                        io, [1.0 / GAMMA, 1.0, 1e-6],
                        type_=ord("g"), event_library=grad_library
                    )
            elif section == "[TRAP]":
                grad_library = read_events(
                    io, [1.0 / GAMMA, 1e-6, 1e-6, 1e-6, 1e-6],
                    type_=ord("t"), event_library=grad_library
                )
            elif section == "[ADC]":
                adc_library = read_events(io, [1.0, 1e-9, 1e-6, 1.0, 1.0])
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
                        logger.warning("DELAYS extension is not handled")
                    else:
                        logger.warning("Ignoring unknown extension: %s", extension_name)
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
            # blockDurations is treated as a list indexed by block index 1..N in Julia.
            # Here we construct a dict with same semantics.
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

        gx = read_Grad(grad_library, shape_library, grad_raster_time, ix)
        gy = read_Grad(grad_library, shape_library, grad_raster_time, iy)
        gz = read_Grad(grad_library, shape_library, grad_raster_time, iz)

        rf = read_RF(rf_library, shape_library, rf_raster_time, irf)
        adc = read_ADC(adc_library, iadc)

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
        # Per Pulseq spec §2.8.4 and KomaMRI ReadPulseq.jl `_apply_rotations_to_owned_sequence`,
        # the rotation acts on gradient amplitudes (scalar trapezoid or per-sample shaped).
        for ext in ext_list:
            if isinstance(ext, Rotation):
                gx, gy, gz = _apply_rotation_to_grads(ext.matrix, gx, gy, gz)

        seq.add_block(gx, gy, gz, rf, adc, D, ext_list)

    # Add first and last points for gradients
    fix_first_last_grads(seq)

    # Final details
    seq.DEF.update(defs)
    seq.DEF["FileName"] = os.path.basename(filename)
    seq.DEF["PulseqVersion"] = pulseq_version
    seq.DEF["signature"] = signature

    # Guessing recon dimensions (approximate mirrors of Julia logic)
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

  ``g.A`` is the per-sample amplitude array. ``g.T`` is either a scalar
  total duration (uniform raster) or a per-step dwell array of length
  ``len(g.A) - 1``.
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
  else:
    total = float(g.T)
    times = np.linspace(0.0, total, amps.size)
  return delay + times, amps


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

  return feelmriRF(
    scanner=scanner,
    shape='custom',
    flip_angle=Quantity(np.pi / 2.0, 'rad'),
    dur=Quantity(duration_ms, 'ms'),
    ref=Quantity(0.0, 'ms'),
    time=Quantity(0.0, 'ms'),
    timings=Quantity(timings_s * 1e3, 'ms'),
    waveform=Quantity(waveform, 'T').to('mT'),
    frequency_offset=Quantity(float(rf.df), 'Hz'),
    phase_offset=Quantity(0.0, 'rad'),
  )


def _convert_adc(adc: "ADC") -> Optional[feelmriADC]:
  """Convert a parsed ADC event to a feelmri.Bloch.ADC.

  Returns None when the ADC is inactive (num == 0). Preserves the
  per-event frequency and phase offsets so downstream signal demodulation
  has access to them.
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
  return feelmriADC(
    times_s * 1e3,
    freq_offset=Quantity(float(adc.df), 'Hz'),
    phase_offset=Quantity(float(adc.phase), 'rad'),
  )


# ---------------------------------------------------------------------------
# K-space trajectory extraction
# ---------------------------------------------------------------------------

def _sample_grad_seconds(g: "Grad", tt: np.ndarray) -> np.ndarray:
  """Sample a parsed Grad on a block-local time grid (seconds), in T/m."""
  is_shaped = isinstance(g.A, np.ndarray)
  if is_shaped:
    if g.A.size == 0 or not np.any(np.abs(g.A) > 0.0):
      return np.zeros_like(tt)
    timings_s, amps_Tm = _shaped_waveform_seconds(g)
  else:
    if float(g.A) == 0.0:
      return np.zeros_like(tt)
    timings_s, amps_Tm = _trap_waveform_seconds(g)
  return np.interp(tt, timings_s, amps_Tm, left=0.0, right=0.0)


def _walk_kspace(pulseq_seq: "PulseqSequence",
                 emit_indices: Optional[set] = None
                 ) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
  """Walk the sequence integrating k(t) = γ ∫₀ᵗ G(t') dt' continuously.

  Returns:
    per_block_samples — list of length len(pulseq_seq); element i is
      ``{'kx': ..., 'ky': ..., 'kz': ..., 'times': ...}`` with ADC
      samples for block i, or empty arrays if block i has no ADC or its
      index is not in ``emit_indices`` (when provided).
    block_start_times_s — ndarray of shape (n_blocks,) with absolute
      block start times in seconds.

  k accumulates across blocks (no reset on RF excitation); callers that
  need flat arrays simply concatenate the per-block samples.
  """
  n_blocks = len(pulseq_seq)
  per_block: List[Dict[str, np.ndarray]] = []
  block_start = np.zeros(n_blocks, dtype=float)

  t_block_start = 0.0
  k = np.zeros(3, dtype=float)
  dt_grad = float(pulseq_seq.DEF.get('GradientRasterTime', 1e-5))
  empty = np.array([], dtype=float)

  for i in range(n_blocks):
    block_start[i] = t_block_start
    block_dur = float(pulseq_seq.DUR[i])
    gx, gy, gz = pulseq_seq.GR[i]
    adc = pulseq_seq.ADC[i]

    if block_dur <= 0.0:
      per_block.append({'kx': empty, 'ky': empty, 'kz': empty, 'times': empty})
      continue

    n_steps = max(int(round(block_dur / dt_grad)), 1)
    tt = np.linspace(0.0, block_dur, n_steps + 1)
    amp_x = _sample_grad_seconds(gx, tt)
    amp_y = _sample_grad_seconds(gy, tt)
    amp_z = _sample_grad_seconds(gz, tt)
    kx_block = k[0] + GAMMA * cumulative_trapezoid(amp_x, tt, initial=0.0)
    ky_block = k[1] + GAMMA * cumulative_trapezoid(amp_y, tt, initial=0.0)
    kz_block = k[2] + GAMMA * cumulative_trapezoid(amp_z, tt, initial=0.0)

    samples = {'kx': empty, 'ky': empty, 'kz': empty, 'times': empty}
    has_adc = int(adc.num) > 0
    is_emitted = emit_indices is None or i in emit_indices
    if has_adc and is_emitted:
      n = int(adc.num)
      delay_s = float(adc.delay)
      T = float(adc.T)
      if n == 1:
        adc_local = np.array([delay_s], dtype=float)
      else:
        dwell = T / (n - 1)
        adc_local = delay_s + np.arange(n, dtype=float) * dwell
      samples = {
        'kx': np.interp(adc_local, tt, kx_block),
        'ky': np.interp(adc_local, tt, ky_block),
        'kz': np.interp(adc_local, tt, kz_block),
        'times': (t_block_start + adc_local) * 1e3,
      }
    per_block.append(samples)

    k = np.array([kx_block[-1], ky_block[-1], kz_block[-1]])
    t_block_start += block_dur

  return per_block, block_start


def kspace_trajectory(pulseq_seq: "PulseqSequence") -> Dict[str, np.ndarray]:
  """Compute the flat k-space trajectory across all ADC samples.

  Returns a dict with arrays ``kx, ky, kz`` (1/m) and ``times`` (ms)
  of shape ``(N_total_adc_samples,)`` in sequence order, ready for
  ``feelmri.Phantom.FEMPhantom.signal()``.
  """
  per_block, _ = _walk_kspace(pulseq_seq)
  empty = np.array([], dtype=float)
  out = {'kx': [], 'ky': [], 'kz': [], 'times': []}
  for s in per_block:
    if s['times'].size > 0:
      for key in out:
        out[key].append(s[key])
  if not out['times']:
    return {key: empty for key in out}
  return {key: np.concatenate(arrs) for key, arrs in out.items()}


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
  n = int(traj['times'].size)
  if shape is None:
    shape = (n, 1, 1)
  shape = tuple(int(d) for d in shape)
  if int(np.prod(shape)) != n:
    raise ValueError(
      f'requested shape {shape} (prod={int(np.prod(shape))}) does not '
      f'match N={n} ADC samples'
    )
  points = tuple(
    np.ascontiguousarray(traj[axis].reshape(shape), dtype=np.float32)
    for axis in ('kx', 'ky', 'kz')
  )
  times = np.ascontiguousarray(
    traj['times'].reshape(shape), dtype=np.float32
  )
  return points, times


# ---------------------------------------------------------------------------
# Partitioned import: dual-path workflow (Bloch prep + signal assembly)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadoutWindow:
  """One contiguous span of ADC blocks plus its trajectory and the
  index of the magnetization snapshot taken just before it.

  Attributes
  ----------
  first_block, last_block : int
      Inclusive range of block indices in ``feelmri_seq.blocks`` /
      ``pulseq_seq.GR/RF/ADC/DUR`` (0-based).
  m_storage_block : int
      Index of the last non-ADC block strictly before ``first_block``;
      that block has ``store_magnetization`` set to True. -1 when the
      readout starts before any prep block (no stored M available).
  m_storage_idx : int
      Column index into ``BlochSolver.solve()``'s output Mxy/Mz array
      (i.e. position of this readout's storage block among the marked
      blocks, in sequence order). -1 mirrors ``m_storage_block == -1``.
  kspace : np.ndarray
      Shape (N, 3) array of (kx, ky, kz) at every ADC sample inside the
      window, in 1/m.
  times : np.ndarray
      Shape (N,) absolute sequence time of each ADC sample, in ms.
  adc_freq_offset : float
      Hz; constant within window (assumed identical across blocks).
  adc_phase_offset : float
      Rad; constant within window.
  """
  first_block: int
  last_block: int
  m_storage_block: int
  m_storage_idx: int
  kspace: np.ndarray
  times: np.ndarray
  adc_freq_offset: float
  adc_phase_offset: float


@dataclass(frozen=True)
class PulseqImport:
  """Result of :func:`import_pulseq`. Exposes both the conventional
  feelmri Sequence (for direct ``BlochSolver`` consumption) and a
  partitioned view that splits prep blocks from ADC readout windows
  for use with the ``Phantom.update_magnetization`` /
  ``Phantom.mri_signal`` dual-path workflow.
  """
  feelmri_seq: feelmriSequence
  pulseq_seq: PulseqSequence
  readouts: List[ReadoutWindow]
  prep_block_indices: List[int]
  adc_block_indices: List[int]


def _identify_readout_groups(pulseq_seq: PulseqSequence
                             ) -> List[Tuple[int, int]]:
  """Walk pulseq_seq.ADC and yield (first, last) inclusive index ranges
  for each maximal run of consecutive blocks with adc.num > 0."""
  groups: List[Tuple[int, int]] = []
  start = -1
  n = len(pulseq_seq)
  for i in range(n):
    has_adc = int(pulseq_seq.ADC[i].num) > 0
    if has_adc and start < 0:
      start = i
    if start >= 0 and (not has_adc or i == n - 1):
      end = i if has_adc else i - 1
      groups.append((start, end))
      start = -1
  return groups


def import_pulseq(filename) -> PulseqImport:
  """Parse a ``.seq`` file and return a partitioned view ready for the
  dual-path workflow (``BlochSolver`` for prep + ``Phantom.mri_signal``
  for readouts).

  See :class:`PulseqImport` for the returned object's shape.
  """
  pulseq_seq = read_seq(str(filename))
  scanner = Scanner()
  feelmri_seq = feelmriSequence()

  # Convert blocks (same logic as the previous read_seq_feelmri body).
  for i in range(len(pulseq_seq)):
    gx, gy, gz = pulseq_seq.GR[i]
    rf_ev = pulseq_seq.RF[i]
    adc_ev = pulseq_seq.ADC[i]
    block_dur_s = float(pulseq_seq.DUR[i])

    Gx = _convert_gradient(gx, 0, scanner)
    Gy = _convert_gradient(gy, 1, scanner)
    Gz = _convert_gradient(gz, 2, scanner)
    Rf = _convert_rf(rf_ev, scanner)
    Adc = _convert_adc(adc_ev)

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

  # Block category indices.
  n_blocks = len(pulseq_seq)
  adc_block_indices = [i for i in range(n_blocks)
                       if int(pulseq_seq.ADC[i].num) > 0]
  adc_set = set(adc_block_indices)
  prep_block_indices = [i for i in range(n_blocks) if i not in adc_set]

  # Readout groups → ReadoutWindow list.
  groups = _identify_readout_groups(pulseq_seq)
  per_block_samples, _ = _walk_kspace(pulseq_seq, emit_indices=adc_set)

  readouts: List[ReadoutWindow] = []
  marked_in_order: List[int] = []
  for first, last in groups:
    # m_storage_block: largest non-ADC index strictly < first.
    m_block = -1
    for j in range(first - 1, -1, -1):
      if j not in adc_set:
        m_block = j
        break
    if m_block >= 0 and m_block not in marked_in_order:
      feelmri_seq.blocks[m_block].store_magnetization = True
      marked_in_order.append(m_block)
    m_idx = marked_in_order.index(m_block) if m_block >= 0 else -1

    kx_parts, ky_parts, kz_parts, t_parts = [], [], [], []
    for i in range(first, last + 1):
      s = per_block_samples[i]
      if s['times'].size > 0:
        kx_parts.append(s['kx'])
        ky_parts.append(s['ky'])
        kz_parts.append(s['kz'])
        t_parts.append(s['times'])
    if t_parts:
      kspace = np.stack([
        np.concatenate(kx_parts),
        np.concatenate(ky_parts),
        np.concatenate(kz_parts),
      ], axis=1)
      times_arr = np.concatenate(t_parts)
    else:
      kspace = np.zeros((0, 3), dtype=float)
      times_arr = np.zeros((0,), dtype=float)

    # ADC offsets are taken from the first ADC block in the group; if
    # they ever differ within a group the upstream sequence has
    # multi-segment ADC, which Pulseq does not model with a single
    # demodulation phase anyway.
    head_adc = pulseq_seq.ADC[first]
    readouts.append(ReadoutWindow(
      first_block=first,
      last_block=last,
      m_storage_block=m_block,
      m_storage_idx=m_idx,
      kspace=kspace,
      times=times_arr,
      adc_freq_offset=float(head_adc.df),
      adc_phase_offset=float(head_adc.phase),
    ))

  return PulseqImport(
    feelmri_seq=feelmri_seq,
    pulseq_seq=pulseq_seq,
    readouts=readouts,
    prep_block_indices=prep_block_indices,
    adc_block_indices=adc_block_indices,
  )


# ---------------------------------------------------------------------------
# High-level sequence reader for FEelMRI
# ---------------------------------------------------------------------------

def read_seq_feelmri(filename) -> Tuple[feelmriSequence, PulseqSequence]:
  """Backward-compatible wrapper around :func:`import_pulseq`.

  Returns ``(feelmri_seq, pulseq_seq)`` so existing callers keep working;
  new callers should prefer :func:`import_pulseq` for the partitioned
  view (with readout windows ready for the dual-path workflow).
  """
  imp = import_pulseq(filename)
  return imp.feelmri_seq, imp.pulseq_seq