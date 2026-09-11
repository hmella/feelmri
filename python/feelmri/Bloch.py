"""
Bloch-equation simulation of MRI pulse sequences on FEM phantom meshes.

Core classes:

* :class:`ADC` — analog-to-digital converter timing specification.
* :class:`SequenceBlock` — atomic unit containing gradients, RF pulses and an
  optional ADC window.
* :class:`Sequence` — ordered list of :class:`SequenceBlock` objects that
  defines a complete MRI pulse sequence.
* :class:`BlochSolver` — drives the C++ Bloch simulator
  (:mod:`feelmri.BlochSimulator`) over an :class:`~feelmri.Phantom.FEMPhantom`
  mesh and assembles the magnetization response.

Helper utilities:

* :func:`create_multi_isochromats` / :func:`collapse_isochromats` — build and
  reduce off-resonance isochromat ensembles for T2* simulation.
"""
import copy
import time
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


import numpy as np
from matplotlib.patches import Circle
from pint import Quantity as Quantity

from feelmri.BlochSimulator import solve_mri_f32, solve_mri_f64

# Timings reaching the raster from different sources can name the same instant
# and disagree in the last bits -- _get_extent pads t_max to honour a declared
# dur, and a shifted waveform corner is recomputed rather than moved. np.unique
# keeps both, and the kernel then takes a zero-length step there: harmless
# numerically, but it re-exponentiates every node and invalidates the dt cache
# twice (335 such steps on epi_v142.seq before this).
#
# 1e-6 ms is three orders below the 1 us finest raster Pulseq permits, but it is
# only a CEILING: _raster_tolerance lowers it further for a block that asks for
# finer steps, because a tolerance at or above the step size deletes the raster
# instead of de-duplicating it.
RASTER_TOL_MS = 1e-6

# How far outside an RF pulse's support to place the guard point. The interval
# arriving at the guard is charged full RF amplitude, so this length is a direct
# flip-angle error of eps/dur; it must stay comfortably above the collapse
# tolerance or the guard is removed again.
RF_EDGE_GUARD_MS = 1e-5


def demodulation_phase(times_ms, freq_offset_hz=0.0, phase_offset_rad=0.0,
                       phase_modulation=None):
    """Receiver phase Pulseq specifies for a set of ADC sample times, in rad.

    ``-2*pi*freq_offset*t + phase_offset + phase_modulation``, with ``t``
    measured from the FIRST sample given -- the ADC event's own origin, which
    is what the offsets are referenced to.

    **The frequency term is NEGATED, the phase terms are not.** The Pulseq
    specification calls ``adc.freq`` the "frequency offset of ADC receiver
    relative to the system frequency", so a receiver tuned ``+df`` must bring
    the spins precessing ``+df`` faster to DC. Applied as ``exp(-i*phase)``,
    that needs ``exp(+i*2*pi*df*t)``. Measured on a narrow rod at ``+4 mm``
    under ``Gx = 10 mT/m``, demodulating at ``gammabar*Gx*4mm``: the rod lands
    at ``-0.008 mm`` with the negation and at ``+7.927 mm`` without it -- the
    unnegated form pushes it FURTHER off centre, doubling the offset instead
    of removing it.

    ``phase_offset`` and ``phase_modulation`` keep their sign: ``MRObjects.RF``
    transmits ``exp(+i*phase_offset)``, so receive must conjugate it or RF
    spoiling stops cancelling. Same shape as the assembler fix -- only the one
    term is negated.

    The single implementation behind both :meth:`ADC.demodulate` and
    :meth:`feelmri.PulseqAdapter.ReadoutWindow.demodulate`, so the two cannot
    drift. The solver never samples the ADC -- readout is synthesized from the
    trajectory -- so applying this is the caller's job, and a caller driving
    ``mri_signal`` by hand must do it explicitly.
    """
    t = np.asarray(times_ms, dtype=float).reshape(-1)
    t = (t - t.min()) * 1e-3 if t.size else t                    # ms -> s
    phase = -2.0 * np.pi * float(freq_offset_hz) * t + float(phase_offset_rad)
    if phase_modulation is not None:
        pm = np.asarray(phase_modulation, dtype=float).reshape(-1)
        if pm.size == phase.size:
            phase = phase + pm
        else:
            warnings.warn(
                f"adc phase_modulation has {pm.size} entries for {phase.size} "
                f"samples; the per-sample phase is not applied.")
    return phase


def apply_demodulation(signal, phase):
    """Multiply ``signal`` by ``exp(-i*phase)`` along its first axis."""
    if not np.any(phase):
        return signal
    out = np.asarray(signal)
    factor = np.exp(-1j * np.asarray(phase).reshape(
        (np.size(phase),) + (1,) * (out.ndim - 1)))
    if np.iscomplexobj(out):
        # Keep the caller's dtype: a float64 phase would promote complex64 to
        # complex128, so the returned array silently doubles in size depending
        # on whether the sequence happened to set an offset.
        factor = factor.astype(out.dtype, copy=False)
    return out * factor


def _raster_tolerance(*steps):
    """Collapse tolerance for a block, below every step it means to take.

    A fixed tolerance is only safe while every real step is far above it. Ask
    for dt_rf = 1e-7 ms against a fixed 1e-6 and EVERY point is within tolerance
    of its neighbour, so the block collapses to a single time and is never
    integrated -- a finer raster silently producing no raster at all.
    """
    positive = [float(s) for s in steps if s is not None and float(s) > 0.0]
    return min([RASTER_TOL_MS] + [0.01 * min(positive)]) if positive \
        else RASTER_TOL_MS


def _sloped_segment_times(g, dt):
    """Uniform sub-raster over the segments of a gradient whose amplitude moves.

    Only needed under ``method='cayley_klein'``, whose end-of-interval rule
    mis-charges a ramp; the default ``magnus2`` integrates a straight segment
    exactly from its endpoints, which is why ``dt_gr`` defaults to disabled.
    """
    ts = g.timings.m_as('ms') if isinstance(g.timings, Quantity) else np.asarray(g.timings, dtype=np.float64)
    amp = g.amplitudes.m_as('mT/m') if isinstance(g.amplitudes, Quantity) else np.asarray(g.amplitudes, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    amp = np.asarray(amp, dtype=np.float64)
    if ts.size < 2 or not np.all(np.isfinite(ts)) or not np.all(np.isfinite(amp)):
        # A non-finite timing would make np.arange raise with a message naming
        # neither the block nor the axis; a non-finite amplitude would compare
        # False below and be silently treated as flat.
        if ts.size >= 2:
            warnings.warn(
                f"gradient on axis {getattr(g, 'axis', '?')} has non-finite "
                f"timings or amplitudes; its ramps are not sub-sampled.")
        return np.empty(0, dtype=np.float64)
    # Relative to the event's own peak: an absolute threshold in mT/m is a
    # different test for a 30 mT/m readout and a 1e-9 mT/m shim.
    floor = 1e-12 * max(float(np.abs(amp).max()), 1.0)
    out = []
    for i in range(ts.size - 1):
        seg = float(ts[i + 1]) - float(ts[i])
        if seg <= dt or abs(float(amp[i + 1]) - float(amp[i])) <= floor:
            # Already finer than the requested step, or flat: nothing to add.
            continue
        out.append(np.arange(float(ts[i]), float(ts[i + 1]), dt))
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)


def _rf_support_ms(rf):
    """(start, end) of an RF pulse's support in ms, read from its own timings.

    Authoritative on both construction paths: an analytic pulse builds timings
    as linspace(time - ref, time - ref + dur) so this reproduces that pair,
    while an imported pulse carries its delay only in timings -- _convert_rf
    sets time = ref = 0 and dur to the support LENGTH, so the (time, dur) pair
    describes [0, length] rather than [delay, delay + length].

    Used for the raster only. _get_extent must NOT use it: a block spans
    [0, dur] and its events sit inside, so taking the support as the extent
    would move the block start to the first pulse's delay.
    """
    t = rf.timings
    t = t.m_as('ms') if isinstance(t, Quantity) else np.asarray(t, dtype=np.float64)
    return float(t[0]), float(t[-1])


def _concomitant_mT(pos, G, B0_mT):
    """Maxwell term at ``pos`` (m) under gradient ``G`` (mT/m), in mT.

    ``Bc = [ (Gx^2+Gy^2) z^2 + (Gz^2/4)(x^2+y^2) - Gx Gz x z - Gy Gz y z ]
            / (2 B0)``

    The gradient coil cannot produce a purely linear Bz; Maxwell's equations
    force this second-order correction. It is the only channel in the field
    model that is non-linear in position.

    **This must stay identical to the kernel's own expression**
    (`BlochSimulator.cpp`, the `Bz_new` line). It exists as one function
    because the per-block Magnus seed recomputes the field in Python in TWO
    places -- the plain path and the spoiler path, the latter on jittered
    isochromat positions -- and a seed that disagrees with the kernel puts an
    O(dt) error at every block boundary, silently.

    ``B0_mT <= 0`` returns exactly zero, which is how the feature is disabled.
    """
    if B0_mT <= 0.0:
        return 0.0
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    gx, gy, gz = float(G[0]), float(G[1]), float(G[2])
    return ((gx * gx + gy * gy) * z * z
            + 0.25 * gz * gz * (x * x + y * y)
            - gx * gz * x * z - gy * gz * y * z) / (2.0 * B0_mT)


def _rf_waveform_mT(rf):
    """An RF pulse's B1 waveform in mT, on either construction path.

    ``_init_from_user_waveform`` stores a pint Quantity; the analytic
    generators store a bare ndarray already in mT (verified: gamma times the
    trapezoidal integral of a hard 90 reads 90.0000 deg either way). Callers
    that read ``.m_as`` unconditionally crash on every natively built pulse.
    """
    w = rf.waveform
    return np.asarray(w.m_as('mT') if isinstance(w, Quantity) else w)


def _collapse_near_duplicates(sorted_t, tol=RASTER_TOL_MS):
    """Drop entries within ``tol`` of their predecessor in a sorted array.

    Sound only while ``tol`` is below every real step, which is what
    :func:`_raster_tolerance` guarantees: then no two legitimate points are
    within tolerance of each other and comparing against the predecessor is the
    same as comparing against the last kept point.
    """
    if sorted_t.size < 2:
        return sorted_t
    keep = np.ones(sorted_t.size, dtype=bool)
    keep[1:] = np.diff(sorted_t) > tol
    return sorted_t[keep]


_METHOD_TO_ORDER = {
  'cayley_klein': 0,
  'magnus2': 2,
  'magnus4': 4,
}
from feelmri.Motion import POD
from feelmri.MPIUtilities import MPI_comm, MPI_print, MPI_rank
from feelmri.MRObjects import Scanner
from feelmri.Phantom import FEMPhantom


class ADC:
    """Analog-to-digital converter timing specification.

    Parameters
    ----------
    times : np.ndarray
        1-D array of ADC sampling times (ms), measured from the START of the
        parent :class:`SequenceBlock`, not from the start of the sequence.
        ``_convert_adc`` builds them that way and
        ``tests/test_pulseq_timing.py`` adds ``block.time_extent[0]`` to
        recover absolute times; ``change_time`` deliberately leaves them alone
        when the block is shifted.
    freq_offset : Quantity, optional
        Frequency offset applied to the ADC samples (Hz). Default 0 Hz.
    phase_offset : Quantity, optional
        Phase offset applied to the ADC samples (rad). Default 0 rad.
    phase_modulation : np.ndarray or None, optional
        Per-sample phase added on top of ``phase_offset`` (rad), one entry
        per ADC sample. Pulseq v1.5 carries it as a shape referenced by the
        ADC event; used for phase-cycled and CAIPI-style acquisitions.
        Default None.

    Notes
    -----
    The three offsets are demodulation parameters. ``BlochSolver`` does not
    sample the ADC -- readout is synthesized from the k-space trajectory by
    :meth:`~feelmri.Phantom.FEMPhantom.mri_signal` -- so applying them is the
    caller's job.
    """

    def __init__(self, times: np.ndarray,
                 freq_offset: Quantity = Quantity(0.0, 'Hz'),
                 phase_offset: Quantity = Quantity(0.0, 'rad'),
                 phase_modulation: np.ndarray | None = None):
        self.times = Quantity(times, 'ms')
        self.freq_offset = freq_offset.to('Hz')
        self.phase_offset = phase_offset.to('rad')
        self.phase_modulation = (
            None if phase_modulation is None
            else Quantity(np.asarray(phase_modulation, dtype=float), 'rad'))

    def demodulation_phase(self):
        """Receiver phase for this ADC's samples, in rad. See
        :func:`demodulation_phase`."""
        return demodulation_phase(self.times.m_as('ms'),
                                  self.freq_offset.m_as('Hz'),
                                  self.phase_offset.m_as('rad'),
                                  self.phase_modulation)

    def demodulate(self, signal):
        """Apply this ADC's frequency/phase offsets to a signal sampled on it.

        Needed by any caller assembling signal by hand -- `simulate_pulseq`
        does this for its own readout windows, but a manual
        `update_magnetization` + `mri_signal` pipeline gets nothing unless it
        calls this.
        """
        return apply_demodulation(signal, self.demodulation_phase())


class SequenceBlock:
    """Atomic building block of an MRI pulse sequence.

    Combines gradient waveforms, RF pulses, and an optional ADC sampling
    window into a single timed unit. The block automatically computes the
    time extent and a discretized timeline used by the Bloch solver.

    Parameters
    ----------
    gradients : list of Gradient, optional
        Gradient waveform objects, any axis. Default is empty.
    rf_pulses : list of RF, optional
        RF pulse objects. Default is empty.
    adc : ADC or None, optional
        ADC sampling specification. Default is None.
    dt_rf : Quantity, optional
        Time step for RF pulse discretization (ms). Default is 0.01 ms.
    dt_gr : Quantity, optional
        Sub-sample the SLOPED segments of each gradient at this step (ms).
        Negative disables; default -1 (disabled), which is correct under the
        default ``magnus2`` solver: its trapezoidal quadrature integrates a
        piecewise-linear ramp exactly from the corners alone, measured 4.3e-7
        rad against 3.6e-3 for ``cayley_klein`` on a 0.0123/0.0456 ms ramp
        pair. Set it only when solving with ``method='cayley_klein'``, whose
        end-of-interval rule leaves ``A*(rise - fall)/2`` per trapezoid; even
        then sub-sampling only shrinks that bias, it does not remove it.
    dt : Quantity, optional
        Coarse time step for the remaining sequence timeline (ms).
        Default is 10 ms.
    dur : Quantity, optional
        Explicit block duration (ms). Negative means inferred from waveforms.
        Default is -1 ms.
    empty : bool, optional
        If True, the block contains no waveforms (dead-time slot).
        Default is False.
    store_magnetization : bool, optional
        If True, the Bloch solver stores the magnetization at the end of
        this block. Default is False.
    spoiler : bool, optional
        If True, the Bloch solver runs its multi-isochromat dephasing path
        over this block, expanding each local node into ``isochromat_K``
        offset positions so the block's gradient actually dephases within a
        voxel. Without it a coarse mesh cannot resolve the intra-voxel phase
        spread a spoiler produces. Default is False.
    """

    def __init__(self, gradients: list = None,
                 rf_pulses: list = None,
                 adc: ADC | None = None,
                 dt_rf: Quantity = Quantity(0.01, 'ms'),
                 dt_gr: Quantity = Quantity(-1, 'ms'),
                 dt: Quantity = Quantity(10, 'ms'),
                 dur: Quantity = Quantity(-1, 'ms'),
                 empty: bool = False,
                 store_magnetization: bool = False,
                 spoiler: bool = False):
        # A list default in the signature is ONE shared list for every block
        # built without arguments, and change_time mutates the Gradient and
        # RF objects inside it.
        gradients = [] if gradients is None else list(gradients)
        rf_pulses = [] if rf_pulses is None else list(rf_pulses)
        self.gradients = gradients
        self.M_gradients = [g for g in self.gradients if g.axis == 0]
        self.P_gradients = [g for g in self.gradients if g.axis == 1]
        self.S_gradients = [g for g in self.gradients if g.axis == 2]
        self.rf_pulses = rf_pulses
        self.adc = adc
        self.dt_rf = dt_rf
        self.dt_gr = dt_gr
        self.dt = dt
        self.dur = dur
        self.time_extent = self._get_extent()
        self.discrete_times = self._discretization()
        self.Nb_times = len(self.discrete_times)
        self.empty = empty
        self.store_magnetization = store_magnetization
        self._spoiler = bool(spoiler)

    @property
    def spoiler(self) -> bool:
        """Whether the solver runs its multi-isochromat dephasing path here."""
        return self._spoiler

    @spoiler.setter
    def spoiler(self, value: bool):
        self._spoiler = bool(value)

    def copy(self):
        return copy.deepcopy(self)

    def __call__(self, t):
        rf = np.sum([rf(t) for rf in self.rf_pulses], axis=0)
        m_gr = np.sum([g(t) for g in self.M_gradients], axis=0)
        p_gr = np.sum([g(t) for g in self.P_gradients], axis=0)
        s_gr = np.sum([g(t) for g in self.S_gradients], axis=0)
        # Informational: which of the requested times are ADC samples. The
        # solver does not consume it -- readout is synthesized from the
        # k-space trajectory, not from the magnetization time course.
        rel = np.asarray(t, dtype=np.float64)
        adc_local = (np.sort(np.asarray(self.adc.times.m_as('ms'),
                                        dtype=np.float64))
                     if self.adc is not None else np.empty(0))
        if adc_local.size == 0:
            # No ADC, or one with no samples. np.clip(idx, 0, size-1) would give
            # -1 for an empty array and adc_local[-1] would raise instead of
            # returning an empty mask.
            adc_mask = np.zeros(rel.shape, dtype=bool)
        else:
            # adc.times are BLOCK-LOCAL (see the ADC docstring) while `t` is
            # absolute, so bring them onto one origin before comparing --
            # np.isin against raw absolute times never matched, and the mask
            # came back all-False on every block.
            #
            # searchsorted, not an (n, m) isclose broadcast: a single-shot
            # spiral ADC can carry tens of thousands of samples in one block,
            # and n x m float64 plus np.isclose's temporaries runs to GB.
            rel = rel - self.time_extent[0].m
            idx = np.clip(np.searchsorted(adc_local, rel), 0, adc_local.size - 1)
            prev = np.clip(idx - 1, 0, adc_local.size - 1)
            adc_mask = (np.abs(adc_local[idx] - rel) <= RASTER_TOL_MS) | \
                       (np.abs(adc_local[prev] - rel) <= RASTER_TOL_MS)

        return rf, (m_gr, p_gr, s_gr), adc_mask

    def __repr__(self):
        return f"Sequence(gradients={self.gradients}, rf_pulses={self.rf_pulses}, dt_rf={self.dt_rf}, dt_gr={self.dt_gr})"

    def __str__(self):
        return f"Sequence with {len(self.gradients)} gradients and {len(self.rf_pulses)} RF pulses."

    def __len__(self):
        return len(self.gradients) + len(self.rf_pulses)

    def _get_extent(self):
        # float64 throughout. The extents are absolute sequence times and every
        # block is chained off the end of the previous one, so a float32 rounding
        # here accumulates into the start time of every later block.
        extents = []

        # Get (t_min, t_max) for each gradient
        if self.gradients:
            extents.append(Quantity(
                np.array([(g.time.m, (g.time + g.dur).m) for g in self.gradients], dtype=np.float64),
                units=self.gradients[0].timings.u,
            ).m_as('ms'))

        # Get (t_min, t_max) for each RF pulse
        if self.rf_pulses:
            extents.append(Quantity(
                np.array([((rf.time - rf.ref).m, (rf.time - rf.ref + rf.dur).m) for rf in self.rf_pulses], dtype=np.float64),
                units=self.rf_pulses[0].ref.u,
            ).m_as('ms'))

        # An empty list contributes nothing. Adding a (0, 0) placeholder instead
        # would pull t_min down to zero for a block whose only event starts later.
        if not extents:
            extents.append(np.array([(0.0, 0.0)], dtype=np.float64))

        # Time extent
        t_min = float(np.min([e.min(axis=0) for e in extents]))
        t_max = float(np.max([e.max(axis=0) for e in extents]))

        # An ADC extends the block but must not move its START: a block spans
        # [0, dur] with its events inside, and an ADC's times begin at its own
        # delay. Without this an ADC-only block reports dur = 0 while its raster
        # spans the whole acquisition, so the next block chains 0 ms later.
        if self.adc is not None and np.size(self.adc.times):
            t_max = max(t_max, t_min + float(np.max(self.adc.times.m_as('ms'))))
        if (t_max - t_min) < self.dur.m_as('ms'):
            t_max += self.dur.m_as('ms') - (t_max - t_min)

        # Update duration if dur is negative
        if self.dur.m_as('ms') < 0:
            self.dur = Quantity(t_max - t_min, 'ms')

        return [Quantity(t_min, 'ms'), Quantity(t_max, 'ms')]

    def _discrete_objects(self):
        # TODO: make sure that both gradients and RF pulses keep the units. Do not use .m_as('ms') or .m here.
        # Get gradient timings and amplitudes
        M_d_gr = [(g.timings.m, g.amplitudes.m) for g in self.M_gradients]
        P_d_gr = [(g.timings.m, g.amplitudes.m) for g in self.P_gradients]
        S_d_gr = [(g.timings.m, g.amplitudes.m) for g in self.S_gradients]

        # Get (t_min, ref, t_max) for each rf pulse
        rf_d = []
        for rf in self.rf_pulses:
            eps   = self.dt_rf  # Small epsilon to avoid numerical issues
            start = (rf.time - rf.ref - eps).m
            end   = (rf.time - rf.ref + rf.dur + eps).m
            steps = int(np.ceil((end - start) / self.dt_rf.m))
            t  = np.linspace(start, end, steps)
            rf_d.append((t, rf(t)))

        return rf_d, M_d_gr, P_d_gr, S_d_gr

    def _discretization(self):
        # Get gradient timings while considering the dt_gr
        if self.gradients:
            gr_timings = np.concatenate([g.timings.m for g in self.gradients])
            if self.dt_gr > 0:
                # Sub-sample only the SLOPED segments. The kernel charges each
                # interval the field at its end, so a flat top is already exact
                # and a ramp is not: stored as bare corners, a trapezoid is
                # over-charged by A*rise/2 on the way up and under-charged by
                # A*fall/2 on the way down, leaving A*(rise - fall)/2 -- zero
                # only while rise == fall, which is why symmetric trapezoids
                # have never shown it. On a UNIFORM sub-raster the two errors
                # are +A*dt/2 and -A*dt/2 and cancel exactly, whatever the ramps.
                # Skipping flat tops costs x1.15-2.10 in raster size instead of
                # x1.59-5.64 for the same result.
                gr_timings = np.concatenate(
                    [_sloped_segment_times(g, self.dt_gr.m_as('ms')) for g in self.gradients]
                    + [gr_timings]
                )
        else:
            gr_timings = np.array([])

        # Get RF timings while considering the dt_rf
        if self.rf_pulses:
            rf_timings = np.concatenate([list(_rf_support_ms(rf)) for rf in self.rf_pulses])
            if self.dt_rf > 0:
                # One step OUTSIDE each edge of the support as well. The kernel
                # charges every interval the field at its END, so the interval
                # arriving at a pulse edge is charged full RF amplitude however
                # long it is: on ppm_v15 the raster jumped 0 -> 0.1 ms straight
                # onto the pulse start and billed the whole gap as pulse,
                # +1.25% of flip.
                #
                # The guard must sit a HAIR outside the support, not one dt_rf
                # outside: the guarded interval is still charged full amplitude,
                # so its length lands directly on the flip angle. At one dt_rf
                # that is dt_rf/dur -- +10.0% on a 0.1 ms imported hard pulse,
                # and magnus2 does not help because a pulse edge is a step, not
                # a ramp. At RF_EDGE_GUARD_MS it is 1e-4 of the pulse.
                guards = []
                lo_blk, hi_blk = self.time_extent[0].m, self.time_extent[1].m
                eps = min(RF_EDGE_GUARD_MS, float(self.dt_rf.m_as('ms')))
                for rf in self.rf_pulses:
                    lo, hi = _rf_support_ms(rf)
                    guards.append(np.array(
                        [max(lo - eps, lo_blk), min(hi + eps, hi_blk)],
                        dtype=np.float64))
                rf_timings = np.concatenate(
                    [np.arange(*_rf_support_ms(rf), self.dt_rf.m_as('ms')) for rf in self.rf_pulses]
                    + guards + [rf_timings]
                )
        else:
            rf_timings = np.array([])

        # Sequence timings. Every term is converted, not read raw: the raster
        # is in ms, and `dt` carries whatever unit the caller wrote. Reading
        # `.m` made dt=100 us give 2 points where dt=0.1 ms gave 21.
        seq_timings = np.arange(self.time_extent[0].m_as('ms'),
                                self.time_extent[1].m_as('ms'),
                                self.dt.m_as('ms'))

        # ADC timings
        if self.adc is not None:
            # adc.times are BLOCK-LOCAL (see the ADC docstring, and
            # test_pulseq_timing.py, which adds time_extent[0] to recover
            # absolute times), so they must be placed in the block's frame
            # before joining an otherwise absolute raster. They were
            # concatenated raw, which agrees only while time_extent[0] == 0 --
            # true for every imported block, since _convert_* puts every event
            # at t = 0, and false for a natively built one.
            adc_times = self.adc.times.m_as('ms') + self.time_extent[0].m
        else:
            adc_times = np.array([])

        # The block ends at time_extent[1], which np.arange never reaches. Without
        # it the last interval of every block is dropped, and a block whose only
        # timings come from that arange (an event-free delay shorter than dt) is
        # left with a single point and is not integrated at all.
        block_ends = np.array([self.time_extent[0].m, self.time_extent[1].m])

        # Concatenate all timings, sort them and remove duplicates
        all_timings = np.concatenate((gr_timings, rf_timings, seq_timings, adc_times, block_ends))
        # Keep the FIRST of each near-duplicate cluster, never the last, and do
        # not snap the ends onto time_extent. The block end can sit an ulp above
        # an event's last sample, and the interpolators are built with
        # right=0.0, so a raster point pushed past that sample evaluates the
        # event as zero: on a hard pulse defined by its two endpoints that costs
        # half the final interval, and 2.5% of the flip angle.
        tol = _raster_tolerance(self.dt.m_as('ms'), self.dt_rf.m_as('ms'),
                                self.dt_gr.m_as('ms'))
        all_timings = _collapse_near_duplicates(np.sort(all_timings), tol)

        # A block's raster must not leave the block. Several sources feed it and
        # they do not all share an origin -- a user-defined Gradient keeps its
        # own `timings` without applying `time`, so a block whose extent starts
        # late could otherwise be handed points before its own start. The solver
        # would then integrate time that belongs to the previous block, and
        # add_block would chain the next one from a duration that disagrees.
        lo, hi = self.time_extent[0].m, self.time_extent[1].m
        inside = (all_timings >= lo - tol) & (all_timings <= hi + tol)
        if not inside.all():
            all_timings = all_timings[inside]

        return Quantity(all_timings, units='ms')

    def change_time(self, time):
        # Update reference time for each gradient and RF pulse
        [g.change_time(g.time + time) for g in self.gradients]
        self.M_gradients = [g for g in self.gradients if g.axis == 0]
        self.P_gradients = [g for g in self.gradients if g.axis == 1]
        self.S_gradients = [g for g in self.gradients if g.axis == 2]
        [rf.change_time(rf.time + time) for rf in self.rf_pulses]
        self.time_extent[0] += time
        self.time_extent[1] += time
        self.discrete_times += time
        self.Nb_times = len(self.discrete_times)

    def plot(self, tight_layout=True, figsize=None, export_to=None):
        if MPI_rank == 0:
            # Plot RF pulses and MR gradients
            titles = ['RF', 'M', 'P', 'S']
            objects = self._discrete_objects()

            fig, ax = plt.subplots(4, 1, figsize=figsize)
            for i, obj in enumerate(objects):
                for t, amp in obj:
                    if titles[i] == 'RF':
                        for t, amp in obj:
                            ax[i].plot(t, np.real(amp), label='Real', color='b')
                            ax[i].plot(t, np.imag(amp), label='Imaginary', color='r')
                    else:
                        for t, amp in obj:
                            ax[i].plot(t, amp, color='b')
                ax[i].set_ylabel(titles[i])
                ax[i].set_xlim([self.time_extent[0].m, self.time_extent[1].m])

            # Add horizontal lines at zero
            [ax[k].axhline(0, color=mcolors.CSS4_COLORS['gray'], linestyle='--') for k in range(4)]

            ax[0].legend(['Real', 'Imaginary'], loc='upper right')
            ax[-1].set_xlabel('Time (ms)')
            if tight_layout:
                plt.tight_layout()
            if export_to is not None:
                plt.savefig(export_to, bbox_inches='tight')
            plt.show()

        # Synchronize all processes
        MPI_comm.Barrier()


class Sequence:
    """Ordered list of :class:`SequenceBlock` objects defining a pulse sequence.

    Parameters
    ----------
    blocks : list of SequenceBlock, optional
        Initial sequence blocks. Default is empty.
    """

    def __init__(self, blocks: list = None):
        blocks = [] if blocks is None else list(blocks)
        self.blocks = blocks
        self.Nb_blocks = len(self.blocks)
        # True when the blocks already carry the spoiler gradients and RF
        # phase cycling the sequence relies on, so the solver must not zero
        # Mxy between blocks on top of them. Set by import_pulseq; see
        # BlochSolver's perfect_spoiling argument.
        self.explicit_spoiling = False
        self.time_extent = self._get_extent()
        self.dur = self.time_extent[1] - self.time_extent[0]
        self.non_empty = [~block.empty for block in self.blocks if block is not None]

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.blocks)

    def __repr__(self):
        return f"Sequence(blocks={self.blocks})"

    def __str__(self):
        return f"Sequence with {len(self.blocks)} blocks."

    def add_block(self, block: SequenceBlock | Quantity, dt: Quantity = None):
        # `dt` builds the raster of a DELAY block and is meaningless for the
        # other two branches: a SequenceBlock's discrete_times are fixed at
        # construction and only ever shifted, and a nested Sequence carries its
        # blocks' own rasters.
        if dt is None:
            dt = Quantity(10, 'ms')
        elif not isinstance(block, Quantity):
            warnings.warn(
                "add_block(dt=...) applies only when `block` is a Quantity "
                "(a delay); the raster of an existing SequenceBlock is fixed "
                "at construction. Pass dt to SequenceBlock(...) instead.")
        # Add a block to the sequence
        if isinstance(block, SequenceBlock):
            block = block.copy()  # Ensure we work with a copy
            block.change_time(self.time_extent[-1].to('ms') - block.time_extent[0].to('ms'))
            self.blocks = [b for b in self.blocks + [block]]
            self.Nb_blocks = len(self.blocks)
            self.time_extent = self._get_extent()
            self.dur = self.time_extent[1] - self.time_extent[0]
            self.non_empty.append(not block.empty)
        elif isinstance(block, Quantity):
            # If a duration is provided, create a new block with that duration
            if not np.isfinite(block.m_as('ms')):
                raise ValueError(
                    f"add_block was given a non-finite duration ({block}). A "
                    f"NaN compares False against both 0 and itself, so it would "
                    f"be dropped silently and shift every later block index.")
            if block <= Quantity(0, 'ms'):
                # Dropping it silently shifts every later block index by one,
                # and the readout bookkeeping (first_block, m_storage_block,
                # block_labels) is built on those indices.
                warnings.warn(
                    f"add_block ignored a non-positive duration ({block}); no "
                    f"block was appended and every later index is unchanged. "
                    f"Guard zero-duration blocks at the call site if index "
                    f"alignment matters.")
            if block > Quantity(0, 'ms'):
                block = SequenceBlock(dur=block.to('ms'), dt=dt, empty=True, store_magnetization=False)
                block.change_time(self.time_extent[-1].to('ms'))
                self.blocks = [b for b in self.blocks + [block]]
                self.Nb_blocks = len(self.blocks)
                self.time_extent = self._get_extent()
                self.dur = self.time_extent[1] - self.time_extent[0]
                self.non_empty.append(not block.empty)
        elif isinstance(block, Sequence):
            # Append a nested Sequence. All child blocks receive the SAME
            # shift = parent_end - child_start, computed once so that the
            # child sequence's internal relative offsets are preserved.
            sequence = block
            seq_extent = sequence._get_extent()
            shift = self.time_extent[-1].to('ms') - seq_extent[0].to('ms')

            shifted = []
            for child in sequence.blocks:
                new_child = child.copy()
                new_child.change_time(shift)
                shifted.append(new_child)

            self.blocks = list(self.blocks) + shifted
            self.Nb_blocks = len(self.blocks)
            self.time_extent = self._get_extent()
            self.dur = self.time_extent[1] - self.time_extent[0]
            self.non_empty.extend(not c.empty for c in shifted)
            # The flag lives on the Sequence, so a child that spells out its own
            # spoilers would otherwise lose that on being appended and the
            # solver would resolve perfect_spoiling back to True, zeroing Mxy at
            # every block boundary.
            if getattr(sequence, 'explicit_spoiling', False):
                self.explicit_spoiling = True
        else:
            warnings.warn(
                f"add_block accepts a SequenceBlock, a Quantity (delay) or a "
                f"Sequence; got {type(block).__name__}. Nothing was appended, "
                f"so any duration already deducted for it is unaccounted for.")

    def check_hardware(self, scanner=None, rtol=1e-6):
        """Report where this sequence exceeds the scanner's limits.

        Returns a tuple of strings, empty when clean, so a caller can warn,
        raise or ignore. Nothing checks this implicitly.

        It matters most for an IMPORTED sequence. `Gradient` copies the
        scanner's limits onto every instance as `Gr_max`/`Gr_sr`, but the
        user-defined branch of its constructor returns before comparing them --
        the only comparisons live in `calculate()` and `match_area()`, which the
        Pulseq adapter never calls. A `.seq` written for a stronger scanner
        therefore imports and simulates silently. Peak B1 was not checkable at
        all until `Scanner.b1_max` existed.

        pypulseq's `check_timing` is not a substitute: it checks raster
        alignment and dead times, not amplitude or slew, and the adapter runs it
        with a default `Opts` where every dead time is zero.
        """
        from feelmri.MRObjects import Scanner as _Scanner
        sc = _Scanner() if scanner is None else scanner
        g_max = sc.gradient_strength.m_as('mT/m')
        s_max = sc.gradient_slew_rate.m_as('mT/m/ms')
        b1_max = getattr(sc, 'b1_max', None)
        b1_max = None if b1_max is None else b1_max.m_as('mT')

        problems = []
        for i, block in enumerate(self.blocks):
            for g in block.gradients:
                amp = np.abs(np.asarray(g.amplitudes.m_as('mT/m'), dtype=float))
                if amp.size and amp.max() > g_max * (1.0 + rtol):
                    problems.append(
                        f"block {i} axis {g.axis}: gradient peaks at "
                        f"{amp.max():.3f} mT/m, limit {g_max:.3f}")
                ts = np.asarray(g.timings.m_as('ms'), dtype=float)
                if ts.size > 1:
                    dt = np.diff(ts)
                    slew = np.abs(np.diff(np.asarray(
                        g.amplitudes.m_as('mT/m'), dtype=float)))
                    slew = np.divide(slew, dt, out=np.zeros_like(slew),
                                     where=dt > 0)
                    if slew.size and slew.max() > s_max * (1.0 + rtol):
                        problems.append(
                            f"block {i} axis {g.axis}: slew reaches "
                            f"{slew.max():.1f} mT/m/ms, limit {s_max:.1f}")
            if b1_max is not None:
                for rf in block.rf_pulses:
                    b1 = np.abs(_rf_waveform_mT(rf))
                    if b1.size and b1.max() > b1_max * (1.0 + rtol):
                        problems.append(
                            f"block {i}: RF peaks at {b1.max() * 1e3:.3f} uT, "
                            f"limit {b1_max * 1e3:.3f}")
        return tuple(problems)

    def flatten(self):
        # Flatten the sequence by creating a single block
        all_gradients = []
        all_rf_pulses = []
        for block in self.blocks:
            all_gradients.extend(block.gradients)
            all_rf_pulses.extend(block.rf_pulses)
        flattened_block = SequenceBlock(gradients=all_gradients, rf_pulses=all_rf_pulses)
        self.blocks = [flattened_block]
        self.Nb_blocks = 1
        self.time_extent = self._get_extent()
        self.dur = self.time_extent[1] - self.time_extent[0]
        self.non_empty = [not flattened_block.empty]

    def update_block_references(self):
        # Update reference time for each block
        for i, block in enumerate(self.blocks):
            shift = block.time_extent[-1].to('ms') + i * self.dt_blocks.to('ms') + self.dt_prep.to('ms')
            block.change_time(shift)

    def _get_extent(self):
        # Get (t_min, t_max) for each block
        time_extent_b = np.array([(b.time_extent[0].m, b.time_extent[1].m) for b in self.blocks if b is not None])

        # Time extent
        if time_extent_b.size == 0:
            # If no blocks, return zero extent
            t_min = 0.0
            t_max = 0.0
        else:
            t_min = np.min([time_extent_b.min(axis=0)])
            t_max = np.max([time_extent_b.max(axis=0)])

        return (Quantity(t_min, 'ms'), Quantity(t_max, 'ms'))

    def plot(self, blocks=None, tight_layout=True, figsize=None, export_to=None):
        if MPI_rank == 0:
            titles = ['RF', 'M', 'P', 'S']

            if blocks is None:  # Plot all
                discrete_blocks = [block._discrete_objects() for block in self.blocks]
                extents = [block.time_extent for block in self.blocks]
            else:               # Plot selected blocks
                discrete_blocks = [block._discrete_objects() for block in self.blocks[blocks]]
                extents = [block.time_extent for block in self.blocks[blocks]]

            # Create subplots (NO sharey, NO sharex → we sync manually)
            fig, ax = plt.subplots(4, 1, figsize=figsize)
            ax = np.asarray(ax)

            def on_xlims_change(event_ax):
                """Propagate x-limits from the modified axes."""
                if getattr(fig, "_syncing", False):
                    return
                fig._syncing = True
                new_xlim = event_ax.get_xlim()
                for other_ax in ax:
                    if other_ax is not event_ax:
                        other_ax.set_xlim(new_xlim)
                fig.canvas.draw_idle()
                fig._syncing = False

            # Attach callback only for x-axis
            for a in ax:
                a.callbacks.connect("xlim_changed", on_xlims_change)

            # -------- PLOTTING -------- #
            for i, objects in enumerate(discrete_blocks):
                for j, obj in enumerate(objects):
                    if titles[j] == 'RF':
                        for t, amp in obj:
                            ax[j].plot(t, np.real(amp), color='b')
                            ax[j].plot(t, np.imag(amp), color='r')
                    else:
                        for t, amp in obj:
                            ax[j].plot(t, amp, color='b')
                    ax[j].set_ylabel(titles[j])

                # Vertical block extent lines
                for k in range(4):
                    ax[k].axvline(extents[i][0].m, color=mcolors.CSS4_COLORS['pink'], linestyle=':')
                    ax[k].axvline(extents[i][1].m, color=mcolors.CSS4_COLORS['pink'], linestyle='--')

            # Horizontal zero lines
            for k in range(4):
                ax[k].axhline(0, color=mcolors.CSS4_COLORS['gray'], linestyle='--')

            # Initial x-limits
            for k in range(4):
                ax[k].set_xlim([extents[0][0].m, extents[-1][1].m])

            # Labels
            ax[0].legend(['Real', 'Imaginary'], loc='upper right')
            ax[-1].set_xlabel('Time (ms)')

            if tight_layout:
                plt.tight_layout()

            if export_to is not None:
                plt.savefig(export_to, bbox_inches='tight')

            plt.show()

        MPI_comm.Barrier()


class BlochSolver:
    """Bloch-equation solver for FEM-mesh MRI simulations.

    Drives the C++ :func:`~feelmri.BlochSimulator.solve_mri` kernel block by
    block over a :class:`~feelmri.Phantom.FEMPhantom`, tracking the full
    magnetization state (M0, T1, T2, B0 inhomogeneity) and optionally
    incorporating a POD motion trajectory.

    Parameters
    ----------
    sequence : Sequence
        Pulse sequence to simulate.
    phantom : FEMPhantom
        FEM mesh phantom providing the local signal assembler.
    scanner : Scanner, optional
        Scanner hardware definition. Default is a standard 1.5 T scanner.
    M0 : float, optional
        Scalar only -- the C++ kernel takes ``const T&``. A nodal array raises
        ``TypeError`` from the pybind signature, and would in any case broadcast
        ``M0 * ones((N, 1))`` to ``(N, N)``. For a spatially varying equilibrium
        use ``initial_Mz``, which is a nodal ``(N, 1)`` array.
        Default is 1.0.
    T1 : Quantity, optional
        Longitudinal relaxation time (ms). Default is 1000 ms.
    T2 : Quantity, optional
        Transverse relaxation time (ms). Default is 100 ms.

        Governs the EVOLUTION between blocks, and is independent of the T2
        handed to :meth:`Phantom.set_static_fields`, which governs the decay
        during a readout. The two are separate objects with no code path
        between them, so passing different values is supported and free.

        For a sequence with no refocusing pulse -- every gradient-echo example
        shipped here -- pass T2* to BOTH: the reversible dephasing is never
        recovered, so that is the correct model and splitting them is not.
        Split (T2 here, T2* on the signal side) only when a refocusing pulse
        recovers the reversible part. Neither choice reproduces a spin echo,
        whose reversible component rephases towards the echo; that needs
        sub-voxel isochromats rather than a scalar.
    delta_B : np.ndarray or float, optional
        Static B0 inhomogeneity field (nodal or scalar, in mT).
        Default is 0.0.
    b1_map : np.ndarray or complex or None, optional
        Transmit (B1+) sensitivity, one complex scale per local node: the RF
        each node actually sees is ``b1_map * rf``. ``1.0`` everywhere is the
        nominal field, ``0.8`` a node receiving 80% of the nominal flip, and a
        complex value adds a transmit phase. ``None`` (the default) is the off
        switch and hands the kernel an EMPTY map, which is not the same as a
        map of ones: a map of ones would still cost a complex load per node per
        time step.

        This is the TRANSMIT side only. Receive sensitivity is a signal-side
        quantity and belongs on the assembler's ``nv`` coil axis, which this
        does not touch.
    t2_prime : Quantity or None, optional
        Intra-voxel field-inhomogeneity time constant (ms), scalar or one
        entry per local node. ``None`` (the default) is the off switch. When set, every node carries
        ``spectral_bins`` sub-spins with static frequency offsets drawn from
        ``lineshape``, and the ensemble PERSISTS across blocks -- so the
        reversible dephasing it produces is genuinely REPHASED by a refocusing
        pulse, which no scalar T2* can do.

        This is the reversible part only. Pass the irreversible T2 to ``T2=``,
        and pass **T2, not T2\***, to ``Phantom.set_static_fields`` or the two
        double-count.
        Requires ``dtype='float64'`` for quantitative work: the bin offsets
        are ~1e-5 to 1e-3 mT and float32 loses them into the background field
        in proportion to it (2.4e-2 relative at ``|Bz|`` = 8 mT). The solver
        warns if you ask for both.
    spectral_bins : int, optional
        Number of sub-spins per node, as an UPPER BOUND -- :func:`lineshape_bins`
        prunes bins whose weight is below float64 epsilon, of which
        Gauss-Hermite produces many (4 of 32, 70 of 128). Default 16, which is
        machine-precision for the gaussian and uniform lineshapes. See
        :func:`lineshape_bins` for the measured accuracy of each rule.
    lineshape : {'gaussian', 'uniform', 'lorentzian'}, optional
        Shape of the intra-voxel field distribution. Default ``'gaussian'``.
        ``'lorentzian'`` is the only one that targets the conventional
        ``exp(-t/T2*)`` and the only inaccurate one -- see
        :func:`lineshape_bins`.
    pod_trajectory : POD or None, optional
        Motion trajectory for moving-phantom simulations. Default is None.
    initial_Mxy : np.ndarray or float, optional
        Initial transverse magnetization (complex, nodal or scalar).
        Default is 0.0.
    perfect_spoiling : bool or None, optional
        Zero the transverse magnetization between non-empty blocks. This
        stands in for gradient or RF spoiling, which a coarse mesh cannot
        dephase properly: the intra-voxel phase spread the spoiler is meant
        to produce is not resolved by the element size. It also destroys
        every coherence pathway that survives a block boundary, so it is
        wrong for FLASH, bSSFP, EPI echo trains and anything driven by a
        stimulated echo. ``None`` (the default) resolves to ``False`` when
        the sequence sets ``Sequence.explicit_spoiling`` -- which
        :func:`~feelmri.PulseqAdapter.import_pulseq` does, since a ``.seq``
        file spells its spoilers out -- and ``True`` otherwise. Pass a bool
        to override.
    """

    def __init__(self, sequence: Sequence,
                 phantom: FEMPhantom,
                 scanner: Scanner = Scanner(),
                 M0: np.ndarray | float = 1.0,
                 T1: Quantity = Quantity(1000.0, 'ms'),
                 T2: Quantity = Quantity(100.0, 'ms'),
                 delta_B: np.ndarray | float = 0.0,
                 b1_map: np.ndarray | complex | None = None,
                 t2_prime: Quantity | None = None,
                 spectral_bins: int = 16,
                 lineshape: str = 'gaussian',
                 pod_trajectory: POD | None = None,
                 initial_Mxy: np.ndarray | float = 0.0,
                 initial_Mz: np.ndarray | float = None,
                 perfect_spoiling: bool | None = None,
                 concomitant_fields: bool = False,
                 isochromat_K: int = 25,
                 isochromat_distribution: str = 'sobol',
                 isochromat_seed: int | None = 0,
                 method: str = 'magnus2',
                 dtype: str = 'float32'):
        method_key = str(method).lower()
        if method_key not in _METHOD_TO_ORDER:
          raise ValueError(
            f"BlochSolver: method must be one of {list(_METHOD_TO_ORDER)}; got {method!r}"
          )
        dtype_key = str(dtype).lower()
        if dtype_key not in ('float32', 'float64'):
          raise ValueError(
            f"BlochSolver: dtype must be 'float32' or 'float64'; got {dtype!r}"
          )

        self._method = method_key
        self._order = _METHOD_TO_ORDER[method_key]
        self._dtype = dtype_key
        self._np_real = np.float32 if dtype_key == 'float32' else np.float64
        self._np_cplx = np.complex64 if dtype_key == 'float32' else np.complex128
        self._py_cplx = complex  # pybind11 accepts either; cast at call site

        ones = np.ones((phantom.local_nodes.shape[0], 1), dtype=self._np_real)
        self.sequence = sequence
        self.scanner = scanner
        self.phantom = phantom
        # The solver allocates per-node state sized by this partition. Mark the
        # partition in use so a later repartition raises instead of leaving the
        # solver inconsistent.
        phantom._partition_bound = True
        self.M0 = M0

        def _node_column(value, name):
            """Broadcast a scalar or per-node value onto the (n, 1) node column.

            `ones` is (n, 1), so the bare `value * ones` idiom turns a plain
            (n,) array into an (n, n) OUTER PRODUCT rather than raising -- at
            63 357 nodes that is a silent 32 GB allocation, which is how this
            was found. A scalar or (n, 1) is the historical input and is
            unchanged; (n,) is now accepted and reshaped."""
            arr = np.asarray(value)
            if arr.ndim > 1 and arr.shape != ones.shape:
                raise ValueError(
                    f"BlochSolver: {name} must be a scalar, (n,) or (n, 1) with "
                    f"n = {ones.shape[0]} local nodes; got shape {arr.shape}")
            if arr.ndim == 1:
                if arr.size != ones.shape[0]:
                    raise ValueError(
                        f"BlochSolver: {name} has {arr.size} entries but there "
                        f"are {ones.shape[0]} local nodes")
                arr = arr.reshape(-1, 1)
            return arr * ones

        self.T1 = Quantity(_node_column(T1.m, 'T1'), T1.units)
        self.T2 = Quantity(_node_column(T2.m, 'T2'), T2.units)
        self.delta_B = _node_column(delta_B, 'delta_B')
        # Transmit sensitivity. Stored as a complex vector of local-node
        # length, or None. It is applied at USE TIME inside the kernel rather
        # than folded into rf_all, which keeps the carried Magnus state
        # (rf_old) a scalar and every unpack site unchanged -- b1 is
        # time-invariant, so scaling both trapezoid endpoints is identical to
        # scaling the pulse.
        # Accept a scalar, (n,) or (n, 1) and normalise to (n,). Deliberately
        # NOT the `value * ones` idiom used above: `ones` is (n, 1), so an (n,)
        # map would broadcast to (n, n) instead of raising.
        if b1_map is None:
            self.b1_map = None
        else:
            b1 = np.asarray(b1_map, dtype=self._np_cplx).reshape(-1)
            n_local = ones.shape[0]
            if b1.size == 1:
                b1 = np.full(n_local, b1[0], dtype=self._np_cplx)
            elif b1.size != n_local:
                raise ValueError(
                    f"BlochSolver: b1_map must be a scalar or have one entry "
                    f"per local node ({n_local}); got {b1.size}")
            self.b1_map = np.ascontiguousarray(b1)
        self.initial_Mxy = initial_Mxy * ones.astype(self._np_cplx)
        self.initial_Mz = initial_Mz * ones if initial_Mz is not None else M0 * ones
        self.pod_trajectory = pod_trajectory
        if perfect_spoiling is None:
            perfect_spoiling = not getattr(sequence, 'explicit_spoiling', False)
        self.perfect_spoiling = bool(perfect_spoiling)
        # A block flagged spoiler=True pays the full K-isochromat cost and then
        # has its Mxy zeroed anyway, so the two settings contradict each other.
        # Only import_pulseq sets explicit_spoiling, so a natively built
        # sequence using spoiler=True lands here by default.
        if self.perfect_spoiling and any(
                getattr(b, 'spoiler', False) for b in getattr(sequence, 'blocks', [])):
            warnings.warn(
                "the sequence has spoiler=True block(s) but perfect_spoiling is "
                "on, so their isochromat dephasing is computed and then "
                "discarded (Mxy is zeroed regardless). Pass "
                "perfect_spoiling=False to keep it, or drop the spoiler flag.")
        # Multi-isochromat dephasing controls for blocks with spoiler=True.
        # K          -- number of isochromats per local FE node.
        # distribution -- 'uniform' (Monte-Carlo, ~1/sqrt(K) residual) or
        #                 'sobol'/'halton' (QMC, ~(log K)^d / K residual).
        # seed       -- reproducibility for both samplers; default 0 makes
        #                 spoiler results deterministic across runs.
        self.isochromat_K = int(isochromat_K)
        self.isochromat_distribution = str(isochromat_distribution).lower()
        self.isochromat_seed = isochromat_seed
        # Concomitant (Maxwell) fields. OFF by default: switching it on moves
        # every existing result, so it must be a knowing choice. When off the
        # kernel is handed B0 = 0, which makes the term identically zero
        # rather than merely small -- the 24-case numerical A/B against the
        # feature-off build reads 0.000e+00.
        self.concomitant_fields = bool(concomitant_fields)
        self._B0_mT = (float(scanner.field_strength.m_as('mT'))
                       if self.concomitant_fields else 0.0)
        # Spectral sub-ensemble for reversible (T2') dephasing. OFF unless
        # t2_prime is given. Each node gets `spectral_bins` sub-spins whose
        # only difference is a static frequency offset, so the offsets ride the
        # existing per-node delta_B channel and the kernel is untouched.
        #
        # The ensemble PERSISTS across blocks -- that is the whole point. A
        # scalar T2* decays monotonically from the snapshot whatever constant
        # it is given, so it can never rephase at an echo; a real sub-ensemble
        # does, because each sub-spin simply runs backwards after a 180.
        self.lineshape = str(lineshape).lower()
        self.spectral_bins = int(spectral_bins)
        if t2_prime is None:
            self._t2_prime_ms = None
            self._bin_z = None
            self._bin_w = None
            self._n_bins = 1
        else:
            # Scalar, (n,) or (n, 1). Per-node is the useful case -- T2' is a
            # tissue property and is dominated by local susceptibility -- and it
            # costs nothing here, because the offsets ride delta_B rather than
            # the relaxation path that guard 2 protects.
            t2p = np.asarray(Quantity(t2_prime).m_as('ms'),
                             dtype=np.float64).reshape(-1)
            n_local = ones.shape[0]
            if t2p.size == 1:
                t2p = np.full(n_local, t2p[0])
            elif t2p.size != n_local:
                raise ValueError(
                    f"BlochSolver: t2_prime must be a scalar or have one entry "
                    f"per local node ({n_local}); got {t2p.size}")
            if not np.all(t2p > 0):
                raise ValueError(
                    "BlochSolver: every t2_prime entry must be positive; got a "
                    f"minimum of {t2p.min()}")
            self._t2_prime_ms = t2p
            if self.spectral_bins < 2:
                raise ValueError(
                    "BlochSolver: spectral_bins must be >= 2 when t2_prime is "
                    f"set; got {self.spectral_bins}. One bin is no ensemble.")
            self._bin_z, self._bin_w = lineshape_bins(self.spectral_bins,
                                                      self.lineshape)
            # Follow the rule, not the request: lineshape_bins prunes sub-spins
            # whose weight is below float64 epsilon.
            self._n_bins = int(self._bin_z.size)
            # The bin offsets are tiny -- z/(T2'*gamma) is 3e-5 to 5e-4 mT at
            # T2' = 50 ms -- and the kernel adds them to `curr.G + delta_B`,
            # which is O(1-10 mT) under any readout gradient. In float32 the
            # rounding quantum of that sum swamps the offset in proportion to
            # the background field. Measured on a gaussian FID at t = 2*T2',
            # against the float64 answer: 2.2e-4 with no gradient, 1.9e-3 at
            # |Bz| = 1 mT, and 2.4e-2 at |Bz| = 8 mT -- an ordinary readout.
            # It is a floor, not a step-size error, so a finer dt does not help.
            if self._dtype == 'float32':
                warnings.warn(
                    "BlochSolver: t2_prime with dtype='float32' loses the bin "
                    "offsets into the background field -- measured 1.9e-3 "
                    "relative at |Bz| = 1 mT and 2.4e-2 at 8 mT, and it does "
                    "not improve with dt. Use dtype='float64' for quantitative "
                    "T2' work.")
            if self.lineshape == 'lorentzian':
                warnings.warn(
                    f"lineshape='lorentzian' targets exp(-t/T2*) but cannot be "
                    f"represented by a finite spin ensemble: at "
                    f"spectral_bins={self._n_bins} the decay is wrong by "
                    f"~{_LORENTZIAN_ERR(self._n_bins):.1e}, improving only as "
                    f"K^-0.55. Use 'gaussian' or 'uniform' unless you need "
                    f"continuity with the exp(-t/T2*) convention.")
            # Guard 1: a K-fold node expansion also duplicates the POD mode
            # matrix, which is rebuilt and Fortran-transposed on every block
            # once the ensemble persists -- 3*N*K*M reals, against the 23 MB
            # the layout was tuned to avoid re-transposing. That is a kernel
            # redesign (a bin axis sharing positions and modes), not a tuning
            # knob, so refuse rather than quietly crawl.
            if pod_trajectory is not None:
                raise NotImplementedError(
                    "BlochSolver: t2_prime with a pod_trajectory is not "
                    "supported. The spectral ensemble duplicates every node, "
                    "which would duplicate the POD mode matrix and its "
                    "per-block transpose. It needs a kernel bin axis that "
                    "shares positions and modes across sub-spins.")
            # Guard 2: UniformRelax survives np.repeat of a constant, but a
            # per-node T1/T2 drops onto the kernel's per-node std::exp path,
            # which recomputes on every dt change -- 2*N*K libm calls on
            # roughly half of all steps, which would dominate the node loop.
            for name, arr in (('T1', self.T1.m), ('T2', self.T2.m)):
                a = np.asarray(arr).reshape(-1)
                if a.size and not np.all(a == a[0]):
                    raise NotImplementedError(
                        f"BlochSolver: t2_prime with a per-node {name} is not "
                        f"supported; the kernel would fall onto its per-node "
                        f"exp() path at K times the cost.")
            # Guard 3: perfect_spoiling zeroes Mxy at every non-empty block
            # boundary, which destroys the ensemble's coherence -- making the
            # whole feature a no-op that still costs K times.
            if self.perfect_spoiling:
                warnings.warn(
                    "BlochSolver: perfect_spoiling is on together with "
                    "t2_prime, so the sub-ensemble's coherence is zeroed at "
                    "every non-empty block boundary and can never rephase at "
                    "an echo. The spectral bins are then pure cost. Pass "
                    "perfect_spoiling=False.")
            # Guard 4: the spatial spoiler ensemble is a SECOND sub-voxel axis.
            # A tensor product is K_spatial * K_spectral (x400 at the default
            # isochromat_K=25), and merging them onto one index is wrong: the
            # quadrature weights span ~10 orders of magnitude, so the spatial
            # average would inherit them and its effective sample size would
            # collapse.
            if any(getattr(b, 'spoiler', False)
                   for b in getattr(sequence, 'blocks', [])):
                raise NotImplementedError(
                    "BlochSolver: t2_prime with a spoiler=True block is not "
                    "supported. They are two independent sub-voxel axes -- a "
                    "spatial spread is rewound by a gradient, a frequency "
                    "spread by a 180 -- so they would need a tensor product.")

        # Persistent Magnus state (per-node Bz, scalar rf) carried between
        # blocks so that order-2/4 maintain a continuous field history. For
        # order = 0 these arrays are written but never read by the kernel.
        self._Bz_old = np.zeros(phantom.local_nodes.shape[0], dtype=self._np_real)
        self._rf_old = self._np_cplx(0)
        # Carried sub-ensemble state, (n_nodes * n_bins, 1). Held separately so
        # the public initial_Mxy/initial_Mz keep their per-node shape and
        # meaning; a repeated solve(start=..., end=...) per shot must not lose
        # the intra-voxel coherence between calls.
        self._bin_Mxy = None
        self._bin_Mz = None
        # The collapse of (_bin_Mxy, _bin_Mz) as last published on the public
        # initial_Mxy / initial_Mz. Used to tell "the caller left the state
        # alone" from "the caller reset it".
        self._bin_collapsed = None
        # Wall-clock cumulative time spent inside the C++ kernel across all
        # solve() calls; populated by solve(). Useful for benchmarking.
        self.bloch_elapsed = 0.0
        # Cached contiguous per-component mode matrices, see _trajectory_modes.
        self._modes_cache = None

    def _trajectory_modes(self, nb_nodes):
        """Contiguous ``(3 * nb_nodes, n_modes)`` mode matrix for the kernel.

        The POD modes are static: ``POD.get_modes`` hands back the same array
        on every call and only the *weights* move with time. Rebuilding the
        kernel's view of them per block therefore repeats an identical copy for
        every block of the sequence -- and on a ``PODSum`` the
        ``np.concatenate`` of the two mode sets is repeated too. Both are
        hoisted here and cached until the trajectory object or the local node
        count changes.

        The ``(N, 3, M)`` tensor is flattened to ``(3N, M)`` rather than split
        into three ``(N, M)`` component matrices, so the kernel deforms the
        mesh with one GEMV over a single stream instead of three. It is
        returned in Fortran order because the kernel takes it column-major:
        the GEMV is then ``M`` long axpy passes over ``3N`` contiguous floats
        rather than an ``M``-long dot product per output element, which is 6%
        faster on the free-running block. Handing pybind11 the layout it
        declares also avoids a 23 MB transpose on every kernel call.
        """
        cache = self._modes_cache
        if cache is not None:
            pod_ref, n_ref, mat = cache
            if pod_ref is self.pod_trajectory and n_ref == nb_nodes:
                return mat

        modes = self.pod_trajectory.get_modes(nb_nodes)
        mat = np.asfortranarray(
            modes.reshape(3 * nb_nodes, -1), dtype=self._np_real
        )
        self._modes_cache = (self.pod_trajectory, nb_nodes, mat)
        return mat

    def _bin_state_is_current(self, initial_Mxy, initial_Mz):
        """Whether the carried sub-ensemble still matches the public state.

        The ensemble is resumed only when the caller has left ``initial_Mxy`` /
        ``initial_Mz`` at the values this solver published for them. If either
        has been reassigned or written in place -- an inversion-recovery or
        multi-TI loop resetting the magnetization between shots -- the caller's
        value wins and the ensemble is re-seeded from it, rather than silently
        continuing the previous shot's sub-voxel coherence.
        """
        if self._bin_collapsed is None:
            return False
        previous_Mxy, previous_Mz = self._bin_collapsed
        return (np.array_equal(previous_Mxy, initial_Mxy)
                and np.array_equal(previous_Mz, initial_Mz))

    def solve(self, start: int = 0, end: int = None):
        # Current machine time
        t0 = time.perf_counter()

        # Phantom position
        x = np.ascontiguousarray(self.phantom.local_nodes, dtype=self._np_real)

        # Blocks to be solved
        self._solve_calls = getattr(self, '_solve_calls', 0) + 1
        start_arg = start
        if start < 0:
            start += self.sequence.Nb_blocks
        if end is None:
            end = self.sequence.Nb_blocks
        blocks = self.sequence.blocks[start:end]
        MPI_print(
          f"[BlochSolver] Solving sequence blocks {start} to {end-1} "
          f"({len(blocks)} blocks) method={self._method} dtype={self._dtype}."
        )

        # Pick the right C++ entry point for this dtype.
        solve_kernel = solve_mri_f32 if self._dtype == 'float32' else solve_mri_f64

        # Dimensions
        nb_nodes  = x.shape[0]
        nb_blocks = len(blocks)

        # List of indices indicating which blocks need to be stored
        # These index into `blocks`, i.e. into the SLICE, while the readout
        # bookkeeping in PulseqAdapter counts store_magnetization blocks over
        # the WHOLE sequence. The two agree only for a whole-sequence solve.
        store_indices = [i for i, block in enumerate(blocks) if block.store_magnetization]
        # Once per solver, not once per call: the incremental steady-state
        # idiom (`solve(start=-2)` inside a per-shot loop) is legitimate and
        # would otherwise emit this on every shot. Testing the NORMALISED start
        # matters -- a negative start is the only non-zero start anywhere in
        # examples/, so guarding on the raw argument meant it never fired.
        if (start > 0 and not getattr(self, '_warned_start_storage', False)
                and any(b.store_magnetization
                        for b in self.sequence.blocks[:start])):
            self._warned_start_storage = True
            warnings.warn(
                f"solve(start={start_arg}) skips "  # report what was passed
                f"{sum(b.store_magnetization for b in self.sequence.blocks[:start])} "
                f"block(s) already flagged store_magnetization, so the returned "
                f"columns are numbered from `start` and no longer line up with "
                f"ReadoutWindow.m_storage_idx, which counts from block 0.")

        # Allocate magnetizations
        Mxy = np.zeros((nb_nodes, nb_blocks), dtype=self._np_cplx)
        Mz  = np.zeros((nb_nodes, nb_blocks), dtype=self._np_real)

        # Strip units of Bloch parameters just once
        T1 = np.ascontiguousarray(self.T1.m_as('ms'), dtype=self._np_real)
        T2 = np.ascontiguousarray(self.T2.m_as('ms'), dtype=self._np_real)
        delta_B = np.ascontiguousarray(self.delta_B, dtype=self._np_real)
        # Empty means "no map"; the kernel branches on size, not on content.
        b1_map = (np.empty(0, dtype=self._np_cplx) if self.b1_map is None
                  else np.ascontiguousarray(self.b1_map, dtype=self._np_cplx))
        initial_Mxy = np.ascontiguousarray(self.initial_Mxy, dtype=self._np_cplx)
        initial_Mz = np.ascontiguousarray(self.initial_Mz, dtype=self._np_real)
        Bz_old = np.ascontiguousarray(self._Bz_old, dtype=self._np_real).reshape(-1)
        rf_old = self._py_cplx(self._rf_old)

        # Gyromagnetic constant
        gamma = self.scanner.gamma.m_as('rad/ms/mT')

        # Spectral sub-ensemble. Every per-node array grows K-fold through
        # np.repeat, so node n occupies rows [n*K : (n+1)*K] -- the same
        # consecutive-duplicate ordering create_multi_isochromats uses. Only
        # delta_B actually differs between a node's bins: a sub-spin shares its
        # node's position, relaxation and transmit sensitivity.
        n_bins = self._n_bins
        bin_w = None
        if n_bins > 1:
            # z is dimensionless, z/T2' is rad/ms, and the kernel wants mT.
            # (n_nodes, n_bins) -> ravel puts node n's bins at rows
            # [n*K : (n+1)*K], matching np.repeat's ordering below.
            bin_dB = np.asarray(
                self._bin_z[None, :]
                / (self._t2_prime_ms[:, None] * gamma),
                dtype=self._np_real).reshape(-1, 1)
            bin_w = self._bin_w.astype(np.float64)
            x = np.repeat(x, n_bins, axis=0)
            T1 = np.repeat(T1, n_bins, axis=0)
            T2 = np.repeat(T2, n_bins, axis=0)
            delta_B = np.repeat(delta_B, n_bins, axis=0) + bin_dB
            delta_B = np.ascontiguousarray(delta_B, dtype=self._np_real)
            if b1_map.size:
                b1_map = np.ascontiguousarray(np.repeat(b1_map, n_bins, axis=0))
            Bz_old = np.ascontiguousarray(
                np.repeat(Bz_old.reshape(-1), n_bins, axis=0),
                dtype=self._np_real)
            # Resume the ensemble if a previous solve() left one of the right
            # shape; otherwise seed every bin of a node from its per-node value.
            want = (nb_nodes * n_bins, 1)
            # Resume only if the ensemble is still the one whose collapse the
            # public attributes currently hold. If the caller has reassigned
            # either, their value wins and the ensemble is re-seeded from it.
            resumable = (self._bin_Mxy is not None and self._bin_Mz is not None
                         and self._bin_Mxy.shape == want
                         and self._bin_Mz.shape == want
                         and self._bin_state_is_current(initial_Mxy, initial_Mz))
            if resumable:
                initial_Mxy = np.ascontiguousarray(self._bin_Mxy,
                                                   dtype=self._np_cplx)
                initial_Mz = np.ascontiguousarray(self._bin_Mz,
                                                  dtype=self._np_real)
            else:
                initial_Mxy = np.ascontiguousarray(
                    np.repeat(initial_Mxy, n_bins, axis=0), dtype=self._np_cplx)
                initial_Mz = np.ascontiguousarray(
                    np.repeat(initial_Mz, n_bins, axis=0), dtype=self._np_real)
            # The kernel sizes everything from r0.rows() and validates no other
            # length (only b1_map), so a forgotten expansion is an out-of-bounds
            # read under NDEBUG rather than an exception. Check here instead.
            n_rows = nb_nodes * n_bins
            for name, arr in (('x', x), ('T1', T1), ('T2', T2),
                              ('delta_B', delta_B), ('Bz_old', Bz_old),
                              ('initial_Mxy', initial_Mxy),
                              ('initial_Mz', initial_Mz)):
                if arr.shape[0] != n_rows:
                    raise RuntimeError(
                        f"BlochSolver: {name} has {arr.shape[0]} rows, expected "
                        f"{n_rows} = {nb_nodes} nodes x {n_bins} bins")

        def collapse_bins(arr):
            """Weighted sum over each node's bins, back to one row per node."""
            if n_bins == 1:
                return arr
            return (arr.reshape(nb_nodes, n_bins) * bin_w).sum(axis=1)

        # Solve the Bloch equations for each block
        for i, block in enumerate(blocks):

            # Discrete time points and time intervals
            discrete_times = block.discrete_times.m_as('ms')
            dt = np.diff(discrete_times, prepend=0).astype(self._np_real, copy=False)

            # Precompute RF and gradients
            n_steps = discrete_times.shape[0]
            rf_pulses = np.zeros((n_steps, 1), dtype=self._np_cplx)
            gradients = np.zeros((n_steps, 3), dtype=self._np_real)
            # The ADC mask is discarded: the solver evolves magnetization and
            # does not sample it. Readout is synthesized afterwards from the
            # k-space trajectory by Phantom.mri_signal.
            rf, G, _ = block(discrete_times)
            rf_pulses[:, 0] = rf
            gradients[:, 0] = G[0]
            gradients[:, 1] = G[1]
            gradients[:, 2] = G[2]

            # Indicator array
            regime_idx = np.abs(rf_pulses) != 0.0

            # Pre-compute the POD modes and weights for this block's timeframe
            has_traj = self.pod_trajectory is not None
            if has_traj:
                # `discrete_times` is already absolute sequence time, which is
                # the frame the motion is defined in, so hand it over as is.
                # `get_weights` adds the trajectory's own `timeshift` itself
                # (Motion.py: `_fold_time(t + self.timeshift)`), so a caller
                # who set one gets it honoured here.
                #
                # This used to call `update_timeshift(block_start)` and then
                # subtract the same value back off, which cancelled exactly --
                # the evolution was right, but a user-set `timeshift` was
                # silently discarded and the attribute was left mutated at the
                # LAST block's start time for every later consumer.
                weights = self.pod_trajectory.get_weights(discrete_times)

                # Get the static modes mapped to the original local nodes
                # (built once and cached -- they do not change between blocks)
                modes = self._trajectory_modes(nb_nodes)

                # Format weights securely for PyBind11
                total_modes = modes.shape[1]
                weights = np.ascontiguousarray(weights.reshape(-1, total_modes), dtype=self._np_real)
            else:
                # Dummies
                weights = np.empty((0, 0), dtype=self._np_real)
                modes = np.empty((0, 0), dtype=self._np_real, order='F')

            # Seed Magnus state (Bz_old per node, rf_old shared) from the
            # field at the start of this block. Without this seed, the very
            # first step of any Magnus order would average the block's
            # opening field with zero, producing an O(dt) boundary error
            # that propagates across block stitches. Block-local seeding is
            # the physically correct interpretation for sequences whose
            # blocks may have arbitrary deadtime between them.
            if self._order > 0:
                if has_traj and weights.size > 0:
                    c0 = x + (modes @ weights[0]).reshape(-1, 3)
                else:
                    c0 = x
                G0 = gradients[0, :]
                Bz_old = (c0 @ G0 + delta_B.reshape(-1)
                          + _concomitant_mT(c0, G0, self._B0_mT)).astype(
                    self._np_real, copy=False)
                rf_old = self._py_cplx(rf_pulses[0, 0])

            # Solve
            if block.spoiler is True:
                K = self.isochromat_K
                # pos_jitter is the sphere RADIUS and global_elem_size is
                # cbrt(element volume), a LENGTH -- which looks like it wants a
                # factor of 1/2. It does not, and halving it was measured worse:
                # the region a node must dephase over is its CONTROL VOLUME, not
                # one element. For P1 tets a node owns ~n_adj/4 elements, so the
                # equivalent sphere is R = (3*n_adj/(16*pi))**(1/3) * elem_size
                # = 0.89..1.13 * elem_size for 12..24 adjacent tets. The two
                # errors -- radius-vs-length and element-vs-control-volume --
                # very nearly cancel, so elem_size is right to ~6%.
                # Measured rho at K=200 for a spoiler winding one cycle per
                # element: 0.0775 at R = elem_size against 0.3044 at half that.
                #
                # The .min() over the whole mesh is still crude: on a graded
                # mesh the single smallest element sets the jitter everywhere,
                # so large elements are under-dephased by size_min/size_local.
                # Per-node radii need a vector pos_jitter through
                # create_multi_isochromats; not done here.
                elem_size = self.phantom.global_elem_size.min()
                # A fixed seed draws the IDENTICAL point set in every spoiler
                # block, so the residual is the same complex number each time
                # and accumulates coherently instead of averaging down as
                # 1/sqrt(n_blocks). Offsetting by the block index decorrelates
                # them while keeping the whole solve reproducible.
                # `i` indexes the SLICE, so solve(start=-4) called once per
                # shot would hand every shot the same seed and the residual
                # would again accumulate coherently. Offset by the absolute
                # block index AND by a per-call counter, so repeated solves of
                # the same blocks decorrelate too.
                block_seed = (None if self.isochromat_seed is None
                              else int(self.isochromat_seed)
                              + (start + i) + 7919 * self._solve_calls)
                (x_big, T1_big, T2_big,
                 deltaB_big, Mxy_big, Mz_big) = create_multi_isochromats(
                    x, T1, T2,
                    delta_B,
                    initial_Mxy,
                    initial_Mz,
                    K=K,
                    pos_jitter=elem_size,
                    distribution=self.isochromat_distribution,
                    seed=block_seed,
                )

                # CRITICAL FIX: Expand modes and Magnus state to match the
                # duplicated nodes in x_big!
                if has_traj:
                    # (3N, M) -> (N, 3, M), repeat per node, flatten back, so
                    # the expansion matches create_multi_isochromats' node
                    # ordering (each node duplicated K times consecutively).
                    modes_big = np.asfortranarray(
                        np.repeat(np.asarray(modes).reshape(nb_nodes, 3, -1),
                                  K, axis=0).reshape(3 * nb_nodes * K, -1))
                else:
                    modes_big = modes
                # Re-derive the Magnus seed from the JITTERED positions.
                # np.repeat(Bz_old, K) copies a field computed at the node
                # centres, which carries none of the intra-voxel dephasing this
                # block exists to produce, so the first trapezoidal step of
                # every spoiler block averaged the correct field with one that
                # had the spoiler's entire effect missing.
                if self._order > 0:
                    if has_traj and weights.size > 0:
                        c0b = x_big + (modes_big @ weights[0]).reshape(-1, 3)
                    else:
                        c0b = x_big
                    Bz_old_big = np.ascontiguousarray(
                        c0b @ gradients[0, :] + deltaB_big.reshape(-1)
                        + _concomitant_mT(c0b, gradients[0, :], self._B0_mT),
                        dtype=self._np_real)
                else:
                    Bz_old_big = np.ascontiguousarray(
                        np.repeat(Bz_old, K, axis=0), dtype=self._np_real)

                # b1 is a property of the NODE, so all K isochromats drawn
                # inside one node share it. Same consecutive-duplicate ordering
                # create_multi_isochromats uses for every other per-node array.
                b1_big = (b1_map if b1_map.size == 0
                          else np.ascontiguousarray(np.repeat(b1_map, K, axis=0)))

                # Solve for the expanded mesh
                t_call = time.perf_counter()
                Mxy_hist, Mz_hist, Bz_old_big_out, rf_old_out = solve_kernel(
                    x_big, T1_big, T2_big, deltaB_big, self.M0, gamma,
                    rf_pulses, gradients, dt, regime_idx, Mxy_big, Mz_big,
                    modes_big, weights, has_traj,
                    self._order, Bz_old_big, rf_old,
                    False, self._B0_mT, b1_big,
                )
                self.bloch_elapsed += time.perf_counter() - t_call

                Mxy_, Mz_ = collapse_isochromats(
                    Mxy_hist[:, -1],
                    Mz_hist[:, -1],
                    K=K,
                    mode="mean"
                )

                Mxy_ = Mxy_.reshape(-1, 1)
                Mz_ = Mz_.reshape(-1, 1)

                # Collapse the duplicated-node Magnus state back to per-node.
                # Within one voxel all K isochromats see (almost) the same
                # macroscopic field, so the mean is a faithful Bz_old to seed
                # the next block.
                Bz_old = Bz_old_big_out.reshape(-1, K).mean(axis=1).astype(
                    self._np_real, copy=False
                )
                rf_old = self._py_cplx(rf_old_out)

            else:
                # Solve normally
                t_call = time.perf_counter()
                Mxy_, Mz_, Bz_old_out, rf_old_out = solve_kernel(
                    x, T1, T2, delta_B, self.M0, gamma,
                    rf_pulses, gradients, dt, regime_idx,
                    initial_Mxy, initial_Mz,
                    modes, weights, has_traj,
                    self._order, Bz_old, rf_old,
                    False, self._B0_mT, b1_map,
                )
                self.bloch_elapsed += time.perf_counter() - t_call

                Bz_old = np.ascontiguousarray(Bz_old_out, dtype=self._np_real).reshape(-1)
                rf_old = self._py_cplx(rf_old_out)

            # Update magnetizations. This is the ONLY place the ensemble is
            # reduced: the carried state below stays per sub-spin, so coherence
            # survives the block boundary and a 180 can rephase it.
            Mxy[:, i] = collapse_bins(Mxy_[:, -1])
            Mz[:, i]  = collapse_bins(Mz_[:, -1])

            # Update the initial magnetization for the next block. Keep the
            # cached column-vector initial_Mxy/initial_Mz in step with the
            # public self.initial_* attributes.
            if block.empty is True:
                next_Mxy = Mxy_[:, -1]
            else:
                if self.perfect_spoiling is True:
                    # Stand-in for gradient or RF spoiling, which a coarse mesh
                    # cannot dephase. A sequence that carries its own spoilers
                    # sets Sequence.explicit_spoiling and lands here with the
                    # flag already False.
                    next_Mxy = np.zeros_like(Mxy_[:, -1])
                else:
                    next_Mxy = Mxy_[:, -1]

            initial_Mxy[:, 0] = next_Mxy
            initial_Mz[:, 0]  = Mz_[:, -1]

        # Keep the public attributes in step with the working copies. They are
        # normally the very same buffers -- np.ascontiguousarray is a no-op
        # when dtype and layout already match -- so rebinding only does
        # anything when a dtype conversion forced a copy above.
        if n_bins > 1:
            # Keep the sub-ensemble for the next solve(), and expose the
            # collapsed per-node state on the public attributes. Both are
            # stamped with the per-node state they correspond to, so that a
            # caller who RESETS initial_Mxy/initial_Mz between calls -- an
            # inversion-recovery or multi-TI loop -- is honoured instead of
            # silently resuming the previous shot's ensemble. Every other knob
            # on this class is read live at solve time; these must be too.
            self._bin_Mxy = initial_Mxy
            self._bin_Mz = initial_Mz
            self.initial_Mxy = collapse_bins(
                initial_Mxy[:, 0]).reshape(-1, 1).astype(self._np_cplx)
            self.initial_Mz = collapse_bins(
                initial_Mz[:, 0]).reshape(-1, 1).astype(self._np_real)
            self._bin_collapsed = (self.initial_Mxy.copy(),
                                   self.initial_Mz.copy())
        else:
            self.initial_Mxy = initial_Mxy
            self.initial_Mz = initial_Mz

        # Persist final Magnus state for the next solve() call. Bz_old is stored
        # per NODE: the next call re-expands it K-fold, so storing the expanded
        # array made the second solve() build n_nodes * n_bins^2 rows and trip
        # the length check. Collapsing is safe because the Magnus seed is
        # re-derived from the block's own opening field for order > 0, and the
        # kernel ignores it entirely for order 0 -- only the LENGTH is
        # load-bearing here.
        self._Bz_old = collapse_bins(Bz_old.reshape(-1)) if n_bins > 1 else Bz_old
        self._rf_old = rf_old

        # Print elapsed time
        MPI_print('[BlochSolver] Elapsed time for solving the sequence: {:.2f} s'.format(time.perf_counter() - t0))

        # Synchronize all processes
        MPI_comm.Barrier()

        return Mxy[:, store_indices], Mz[:, store_indices]


LINESHAPES = ('gaussian', 'uniform', 'lorentzian')

# Measured worst error of the lorentzian rule against exp(-t/T2'),
# over tau in [0, 3*T2']: 3.3e-1 at K=8 falling as roughly K^-0.55.
def _LORENTZIAN_ERR(K):
  return 3.3e-1 * (K / 8.0)**-0.55



def lineshape_bins(K, lineshape='gaussian'):
  """Quadrature nodes and weights for an intra-voxel field distribution.

  Returns ``(z, w)``: ``K`` dimensionless frequency offsets and ``K``
  non-negative weights summing to exactly 1. Scaled by ``1/T2'`` the offsets
  are rad/ms, and the ensemble average

      F(tau) = sum_k w_k * exp(-i * z_k / T2' * tau)

  is the decay the sub-ensemble produces. The weights are a genuine probability
  distribution: an unconstrained least-squares fit reaches 4e-11 on the
  exponential but needs ``sum|w| = 1534``, i.e. cancellation that a spin
  ensemble cannot represent and float32 cannot carry.

  Measured worst error against the lineshape each rule claims, over
  ``tau`` in ``[0, 3*T2']``:

  ============  ========  ========  ========  ========
  K                    8        16        64       256
  ============  ========  ========  ========  ========
  gaussian       9.7e-03   1.7e-08   3.3e-16   5.0e-16
  uniform        2.2e-07   6.1e-16   3.1e-16   8.3e-16
  lorentzian     3.3e-01   2.4e-01   1.0e-01   3.8e-02
  ============  ========  ========  ========  ========

  * ``'gaussian'`` -- Gauss-Hermite. Decay ``exp(-tau^2 / 2 T2'^2)``.
    Machine precision by K=24.
  * ``'uniform'`` -- Gauss-Legendre on a half-width of ``sqrt(3)/T2'``, chosen
    so the variance matches the Gaussian of the same ``T2'``. Decay is a sinc,
    so the signal has true zero crossings and partial recoveries -- the right
    model for a linear susceptibility gradient across the voxel.
  **A finite bin set is quasi-periodic, so the decay REVIVES at long tau.**
  With K discrete frequencies the ensemble cannot stay cancelled forever; it
  recurs once the accumulated phase spread wraps. Largest ``tau/T2'`` at which
  each rule still tracks its lineshape to 1e-3:

  ============  ======  ======  ======  ======
  K                  8      16      32      64
  ============  ======  ======  ======  ======
  gaussian        2.50    4.67    7.86   12.47
  uniform         5.37   13.40     inf     inf
  ============  ======  ======  ======  ======

  So for the gaussian rule size ``K >= 5 * tau_max / T2'``, where ``tau_max``
  is the longest time coherence survives WITHOUT a refocusing pulse -- a 180
  restarts the clock, so echo-based sequences are far less demanding than the
  bound suggests. The uniform rule barely needs sizing.

  * ``'lorentzian'`` -- equal-probability quantile midpoints. The only rule
    that targets the conventional ``exp(-t/T2*)``, and the only inaccurate one:
    it converges roughly as ``K^-0.55``, so 3.8e-2 at K=256. That is not an
    implementation limit. ``exp(-t/T2*)`` is the Fourier transform of a
    Lorentzian, which has infinite variance, so reproducing it needs
    arbitrarily far off-resonance spins and no finite ensemble gets there.
  """
  K = int(K)
  if K < 1:
    raise ValueError(f"lineshape_bins: K must be >= 1; got {K}")
  key = str(lineshape).lower()
  if key not in LINESHAPES:
    raise ValueError(
      f"lineshape_bins: lineshape must be one of {list(LINESHAPES)}; "
      f"got {lineshape!r}")
  if key == 'gaussian':
    z, w = np.polynomial.hermite_e.hermegauss(K)
  elif key == 'uniform':
    z, w = np.polynomial.legendre.leggauss(K)
    z = z * np.sqrt(3.0)
  else:
    u = (np.arange(K) + 0.5) / K
    z, w = np.tan(np.pi * (u - 0.5)), np.ones(K)
  z = np.asarray(z, dtype=np.float64)
  w = np.asarray(w, dtype=np.float64)
  # The rules break down at large K and numpy does not say so. hermegauss
  # returns NaN weights somewhere between K=320 and K=400, and `spectral_bins`
  # has no natural ceiling, so without this check a large K silently produced
  # NaN magnetization for the whole phantom.
  if not (np.all(np.isfinite(z)) and np.all(np.isfinite(w)) and w.sum() > 0.0):
    raise ValueError(
      f"lineshape_bins: the {key} rule loses all precision at K = {K} "
      f"(non-finite nodes or weights). Use a smaller K; the gaussian rule is "
      f"reliable to about K = 320, and its weights are already negligible far "
      f"below that.")
  # Normalise in float64. The T1 recovery term (1 - e1) * M0 is AFFINE, so the
  # collapsed equilibrium is M0 * sum(w): raw Gauss-Hermite weights sum to
  # 2.5066 and Gauss-Legendre to 2.0, either of which would put the whole
  # phantom at the wrong M0.
  w = w / w.sum()
  # Drop sub-spins that cannot contribute. Gauss-Hermite spends its extreme
  # abscissae on weights far below machine epsilon -- 4 of 32, 70 of 128 and
  # 172 of 256 sit under 1e-16 -- and each one is a full sub-spin carried
  # through every time step and then multiplied by nothing. Pruning at float64
  # epsilon changes no result that float64 can represent, and it keeps large K
  # from being mostly waste.
  keep = w >= 1e-16
  if keep.sum() >= 1:
    z, w = z[keep], w[keep]
    w = w / w.sum()
  return z, w


def _draw_in_sphere_offsets(M, R, distribution='uniform', seed=None):
  """Draw ``M`` offset vectors uniformly distributed inside a 3-sphere of radius ``R``.

  Three samplers are supported. All three use the same inverse-CDF
  mapping from the unit cube to the sphere — ``r = R * u^(1/3)``,
  ``cos(theta) = 1 - 2v``, ``phi = 2*pi*w`` — and differ only in how
  ``(u, v, w) in [0, 1)^3`` is drawn:

  * ``'uniform'`` — i.i.d. ``Uniform([0, 1])`` via
    ``numpy.random.default_rng(seed)``. Monte-Carlo residual rate
    :math:`\\rho \\sim K^{-1/2}`.
  * ``'sobol'`` — :class:`scipy.stats.qmc.Sobol`, a low-discrepancy
    sequence. Quasi-Monte-Carlo residual rate
    :math:`\\rho = \\mathcal O((\\log K)^d / K)`.
  * ``'halton'`` — :class:`scipy.stats.qmc.Halton`, same QMC class as
    Sobol; cheaper to seed but empirically slightly weaker in 3-D
    due to higher-prime axis correlations.

  Parameters
  ----------
  M : int
      Number of points to draw.
  R : float
      Sphere radius (m, but the function is unit-agnostic).
  distribution : {'uniform', 'sobol', 'halton'}
  seed : int or None
      Forwarded to the underlying RNG / QMC engine. ``None`` retains
      the pre-refactor non-deterministic behaviour.

  Returns
  -------
  np.ndarray
      Float32 C-contiguous array of shape ``(M, 3)``.
  """
  dist = str(distribution).lower()
  if dist == 'uniform':
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=M)
    v = rng.uniform(0.0, 1.0, size=M)
    w = rng.uniform(0.0, 1.0, size=M)
  elif dist in ('sobol', 'halton'):
    from scipy.stats.qmc import Halton, Sobol
    M_int = int(M)
    if dist == 'sobol':
      # Sobol's (t, m, s)-net balance properties hold exactly when n
      # is a power of 2. Generate the smallest 2**m >= M and slice
      # rather than calling random(M) — strictly higher-quality, and
      # avoids the scipy UserWarning about non-power-of-2 sample counts.
      qmc = Sobol(d=3, seed=seed)
      m_exp = int(np.ceil(np.log2(max(M_int, 1))))
      pts = qmc.random_base2(m_exp)[:M_int]
    else:
      qmc = Halton(d=3, seed=seed)
      pts = qmc.random(M_int)
    u, v, w = pts[:, 0], pts[:, 1], pts[:, 2]
  else:
    raise ValueError(
      f"unknown distribution {distribution!r}; expected one of "
      f"'uniform', 'sobol', 'halton'"
    )
  radius = R * np.cbrt(u)
  cos_theta = 1.0 - 2.0 * v
  sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
  phi = 2.0 * np.pi * w
  out = np.empty((int(M), 3), dtype=np.float32)
  out[:, 0] = (radius * sin_theta * np.cos(phi)).astype(np.float32)
  out[:, 1] = (radius * sin_theta * np.sin(phi)).astype(np.float32)
  out[:, 2] = (radius * cos_theta).astype(np.float32)
  return out


def create_multi_isochromats(x, T1, T2, delta_B, Mxy0, Mz0,
                             K=100, pos_jitter=0.2e-3,
                             distribution='uniform', seed=None):
  """Replicate every node K times and offset by an in-sphere jitter.

  Every input array is repeated K times along axis 0 with
  :func:`numpy.repeat` (so node ``n`` produces the contiguous range
  ``[n*K : (n+1)*K]``). The positions ``x_big`` are then perturbed by
  in-sphere offsets drawn from ``distribution`` with radius
  ``pos_jitter``.

  Parameters
  ----------
  x : np.ndarray
      Node positions of shape ``(N, 3)``.
  T1, T2, delta_B, Mxy0, Mz0 : np.ndarray
      Nodal arrays repeated K times along axis 0.
  K : int, optional
      Number of isochromats per node. Default 100.
  pos_jitter : float, optional
      Radius of the in-sphere offset (m). Default 0.2 mm.
  distribution : {'uniform', 'sobol', 'halton'}, optional
      Sampler for the offsets — see :func:`_draw_in_sphere_offsets`.
      Default ``'uniform'`` preserves the pre-refactor behaviour.
  seed : int or None, optional
      RNG / QMC seed forwarded to the sampler. ``None`` is
      non-deterministic; ``BlochSolver`` defaults to ``0`` so the
      spoiler is reproducible.
  """
  x_big      = np.repeat(x, K, axis=0)
  T1_big     = np.repeat(T1, K, axis=0)
  T2_big     = np.repeat(T2, K, axis=0)
  deltaB_big = np.repeat(delta_B, K, axis=0)
  Mxy_big    = np.repeat(Mxy0, K, axis=0)
  Mz_big     = np.repeat(Mz0, K, axis=0)

  N = x.shape[0]
  jitter = _draw_in_sphere_offsets(N * K, pos_jitter,
                                   distribution=distribution, seed=seed)
  if x.shape[1] == 2:
    jitter = jitter[:, :2]
  x_big = x_big + jitter.astype(x_big.dtype, copy=False)

  return x_big, T1_big, T2_big, deltaB_big, Mxy_big, Mz_big


def collapse_isochromats(Mxy_big, Mz_big, K, mode="mean"):
    Mxy_big = np.asarray(Mxy_big)
    Mz_big  = np.asarray(Mz_big)

    if Mxy_big.ndim == 1:
        Mxy_big = Mxy_big.reshape(-1, 1)
    if Mz_big.ndim == 1:
        Mz_big = Mz_big.reshape(-1, 1)

    N_big = Mxy_big.shape[0]
    N = N_big // K

    # Reshape arrays to isolate the K isochromats for each node
    # Shapes become (N, K, 1)
    Mxy_reshaped = Mxy_big.reshape(N, K, -1)
    Mz_reshaped  = Mz_big.reshape(N, K, -1)

    # Compute mean or sum across the K axis (axis=1)
    if mode == "mean":
        Mxy_out = np.mean(Mxy_reshaped, axis=1)
        Mz_out  = np.mean(Mz_reshaped, axis=1)
    else:
        Mxy_out = np.sum(Mxy_reshaped, axis=1)
        Mz_out  = np.sum(Mz_reshaped, axis=1)

    return Mxy_out, Mz_out


def plot_isochromat_voxel(positions, *, R=None, ax=None,
                          color='steelblue', alpha=0.7, s=8,
                          title=None, show=True, export_to=None):
  """3-D scatter of K isochromat positions inside a voxel.

  Parameters
  ----------
  positions : np.ndarray
      Array of shape ``(K, 3)`` with the isochromat coordinates (m).
  R : float, optional
      Voxel radius. When supplied, a translucent reference sphere of
      that radius is drawn at the origin for spatial context.
  ax : matplotlib 3-D axis, optional
      Pre-existing axes to draw into. When ``None``, a fresh figure
      is created.
  color : str, optional
      Scatter colour.
  alpha : float, optional
      Scatter alpha.
  s : int or float, optional
      Scatter marker size.
  title : str, optional
      Axes title.
  show : bool, optional
      Call ``plt.show()`` after rendering. Default True.
  export_to : str or path-like, optional
      When supplied, save the figure to this path before showing.

  Notes
  -----
  Rank-0 guarded: on non-zero MPI ranks the function is a no-op and
  returns ``None`` to mirror the convention used by
  :meth:`SequenceBlock.plot` and :meth:`Sequence.plot`.
  """
  if MPI_rank != 0:
    MPI_comm.Barrier()
    return None

  positions = np.asarray(positions)
  if positions.ndim != 2 or positions.shape[1] != 3:
    raise ValueError(f'positions must have shape (K, 3); got {positions.shape}')

  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D proj)
  if ax is None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
  else:
    fig = ax.figure

  ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
             c=color, s=s, alpha=alpha, depthshade=True,
             label=f'K = {positions.shape[0]}')

  if R is not None and R > 0:
    n = 24
    u, v = np.meshgrid(
      np.linspace(0.0, 2.0 * np.pi, n),
      np.linspace(0.0, np.pi, n // 2 + 1),
    )
    xs = R * np.sin(v) * np.cos(u)
    ys = R * np.sin(v) * np.sin(u)
    zs = R * np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color='gray', linewidth=0.3, alpha=0.4)

  ax.set_xlabel('x (m)')
  ax.set_ylabel('y (m)')
  ax.set_zlabel('z (m)')
  ax.set_title(title or f'Isochromat voxel scatter (K = {positions.shape[0]})')
  ax.legend(loc='upper right')
  try:
    ax.set_aspect('equal')
  except (NotImplementedError, ValueError):
    pass

  if export_to is not None:
    fig.savefig(export_to, bbox_inches='tight')
  if show:
    plt.show()

  MPI_comm.Barrier()
  return ax


def spoiling_residual(K, k_sp, voxel_size, *,
                      distribution='sobol', seed=0, n_trials=1):
  """Numerical residual of the K-isochromat spoiling sum.

  Computes

  .. math::

     \\rho(K) \\;=\\; \\left|
       \\frac{1}{K}\\sum_{k=1}^{K}
         \\exp\\big(\\,i\\, 2\\pi\\, \\vec k_{\\rm sp}\\cdot \\vec r_k\\big)
     \\right|

  where :math:`\\vec r_k` are K isochromat offsets drawn from
  ``distribution`` inside a sphere of radius ``voxel_size`` (m).
  Useful for sizing ``BlochSolver.isochromat_K`` before launching a
  real simulation.

  Parameters
  ----------
  K : int
      Number of isochromats.
  k_sp : array-like of shape (3,)
      Spoiler wavenumber :math:`\\vec k_{\\rm sp} = \\gamma/(2\\pi) \\cdot \\int_0^T G(t)\\, dt`
      in 1/m. The phase per isochromat is :math:`2\\pi \\vec k_{\\rm sp}\\cdot\\vec r_k`.
  voxel_size : float
      Sphere radius for the isochromat draw (m).
  distribution : {'uniform', 'sobol', 'halton'}
      Sampler — see :func:`_draw_in_sphere_offsets`.
  seed : int or None
      Base seed; trial ``t`` uses ``seed + t``.
  n_trials : int
      Independent repeats for mean/std estimation.

  Returns
  -------
  (mean, std) : tuple of float
      Mean and sample standard deviation of :math:`\\rho(K)` across
      ``n_trials`` independent draws.
  """
  k_sp = np.asarray(k_sp, dtype=np.float64).reshape(3)
  rhos = np.empty(int(n_trials), dtype=np.float64)
  for t in range(int(n_trials)):
    s = None if seed is None else int(seed) + t
    r = _draw_in_sphere_offsets(int(K), float(voxel_size),
                                distribution=distribution, seed=s)
    phase = 2.0 * np.pi * (r.astype(np.float64) @ k_sp)
    rhos[t] = np.abs(np.mean(np.exp(1j * phase)))
  if n_trials == 1:
    return float(rhos[0]), 0.0
  return float(rhos.mean()), float(rhos.std(ddof=0))


def plot_multi_isochromat_dephasing(
        idx,
        x_big,
        Mxy_big,
        Mxy_hist,
        K,
        x_original=None,
        elem_radius=None,
        t_index=None,
        show_positions=True,
        title_prefix="Isochromat Dephasing"):
    """
    Visualizes the K isochromats from original FE node idx in the complex plane,
    together with the original node and element radius.

    Parameters
    ----------
    idx : int
        FE node index to inspect.
    x_big : array (N_big, dim)
        Enlarged coordinates from create_multi_isochromats().
    Mxy_big : array (N_big, 1)
        Initial transverse magnetization.
    Mxy_hist : array (N_big, n_time)
        Time-history of Mxy for all isochromats (complex).
    K : int
        Number of sub-isochromats per original node.
    x_original : array (N, dim), optional
        Original node coordinates. Only used for plotting reference.
    elem_radius : float, optional
        Radius for element visualization around original node.
    """

    # Determine which rows in x_big / Mxy_big correspond to node idx
    start = idx * K
    end   = start + K
    iso_slice = slice(start, end)

    # Pick the magnetizations to plot
    if t_index is None:
        M = Mxy_big[iso_slice, 0]
        title_t = "(initial)"
    else:
        if t_index >= Mxy_hist.shape[1]:
            raise IndexError(
                f"t_index={t_index} exceeds number of time points {Mxy_hist.shape[1]}"
            )
        M = Mxy_hist[iso_slice, t_index]
        title_t = f"(t index = {t_index})"

    # Prepare complex-plane coordinates
    Re = np.real(M)
    Im = np.imag(M)

    # Plot
    fig = plt.figure(figsize=(11, 5))

    # --- complex plane ---
    ax1 = fig.add_subplot(1, 2 if show_positions else 1, 1)
    ax1.scatter(Re, Im, s=60, c='blue', label='Isochromats')

    # Draw mean magnetization vector (spoiled result)
    M_mean = np.mean(M)
    ax1.scatter(np.real(M_mean), np.imag(M_mean),
                s=120, c='red', marker='x', label='Mean Mxy')

    ax1.arrow(0, 0, np.real(M_mean), np.imag(M_mean),
              head_width=0.02 * np.max(np.abs(Re + 1j * Im)),
              color='red', linewidth=1.8)

    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.set_xlabel("Real(Mxy)")
    ax1.set_ylabel("Imag(Mxy)")
    ax1.set_aspect("equal", "box")
    ax1.set_title(f"{title_prefix} for node {idx} {title_t}\nComplex plane")
    ax1.legend()

    # Arrows for each isochromat
    rmax = np.max(np.abs(Re + 1j * Im))
    for r, im in zip(Re, Im):
        ax1.arrow(0, 0, r, im, head_width=0.02 * rmax,
                  length_includes_head=True, color="gray", alpha=0.4)

    # --- jittered positions (2nd subplot) ---
    if show_positions:
        x_node = x_big[iso_slice]  # (K, dim)
        ax2 = fig.add_subplot(1, 2, 2)

        # Plot jittered isochromats
        ax2.scatter(x_node[:, 0], x_node[:, 1], c='blue', s=50, label="Isochromats")

        # Plot original node
        if x_original is not None:
            x0 = x_original[idx]
            ax2.scatter([x0[0]], [x0[1]], c='black', s=80, marker='*', label="Original node")

            # Draw element radius as circle
            if elem_radius is not None:
                circle = Circle((x0[0], x0[1]), elem_radius,
                                fill=False, linestyle='--', edgecolor='red', linewidth=1.2)
                ax2.add_patch(circle)
                ax2.set_xlim(x0[0] - elem_radius * 1.5, x0[0] + elem_radius * 1.5)
                ax2.set_ylim(x0[1] - elem_radius * 1.5, x0[1] + elem_radius * 1.5)

        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Isochromat jittered positions\n(with original node + element radius)")
        ax2.set_aspect("equal", "box")
        ax2.legend()

    plt.tight_layout()
    plt.show()
