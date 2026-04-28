"""
MR hardware object definitions: scanner, gradient, and RF pulse.

:class:`Scanner` holds hardware limits (gradient strength, slew rate,
gyromagnetic ratio). :class:`Gradient` represents a trapezoidal or
user-defined gradient waveform. :class:`RF` generates analytic or
user-defined RF excitation pulses with flip-angle normalization.
"""
import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity
from scipy.interpolate import interp1d

from feelmri.MPIUtilities import MPI_print, MPI_rank


class Scanner:
    """MRI scanner hardware specification.

    Parameters
    ----------
    field_strength : Quantity, optional
        Static field strength (T). Default is 1.5 T.
    gradient_strength : Quantity, optional
        Maximum gradient amplitude (mT/m). Default is 33 mT/m.
    gradient_slew_rate : Quantity, optional
        Maximum gradient slew rate (mT/m/ms). Default is 180 mT/m/ms.

    Attributes
    ----------
    gammabar : Quantity
        Gyromagnetic ratio for protons (42.58 MHz/T).
    gamma : Quantity
        Angular gyromagnetic ratio (2π × gammabar, rad·Hz/T).
    """
    def __init__(self, 
                field_strength: Quantity = Quantity(1.5, 'T'), 
                gradient_strength: Quantity = Quantity(33,'mT/m'),
                gradient_slew_rate: Quantity = Quantity(180,'mT/m/ms')):
        self.field_strength = field_strength
        self.gradient_strength = gradient_strength
        self.gradient_slew_rate = gradient_slew_rate
        self.gammabar = Quantity(42.58e6, 'Hz/T')
        self.gamma = Quantity(42.58e6*2*np.pi, 'rad*Hz/T')


class Gradient:
    """Gradient waveform for trapezoidal, triangular, or user-supplied shapes.

    The gradient may be defined either analytically (via slope, strength, and
    plateau length) or explicitly (via supplied ``timings`` and ``amplitudes``).
    If both ``timings`` and ``amplitudes`` are provided, the constructor
    immediately builds the interpolator and **skips all analytic gradient
    construction**.

    Parameters
    ----------
    slope : Quantity or None, optional
        Duration of each ramp (ms). If None, derived from gradient amplitude
        and slew rate.
    lenc : Quantity, optional
        Duration of the flat-top portion of the gradient (ms). Default is 1 ms.
    strength : Quantity or None, optional
        Gradient amplitude (mT/m). If None, the maximum scanner gradient is used.
    scanner : Scanner, optional
        Scanner object containing hardware limits.
    ref : Quantity, optional
        Reference time (ms) relative to the sequence block. Default is 0 ms.
    time : Quantity, optional
        Absolute start time (ms) of the gradient within the sequence. Default is 0 ms.
    axis : int, optional
        Gradient axis (0 = M, 1 = P, 2 = S). Default is 0.
    timings : Quantity or None, optional
        Explicit waveform timing samples. If provided together with ``amplitudes``,
        analytic construction is bypassed.
    amplitudes : Quantity or None, optional
        Explicit waveform amplitude samples corresponding to ``timings``.

    Notes
    -----
    When ``timings`` and ``amplitudes`` are supplied, all analytic parameters
    (slope, lenc, strength) are stored but *not used* for waveform generation.
    The total duration is inferred from the last timing sample.
    """

    def __init__(
        self,
        slope=None,
        lenc=Quantity(1.0, "ms"),
        strength=None,
        scanner=Scanner(),
        ref=Quantity(0.0, "ms"),
        time=Quantity(0.0, "ms"),
        axis=0,
        timings=None,
        amplitudes=None,
    ):

        self.scanner = scanner
        self.Gr_max = scanner.gradient_strength      # [mT/m]
        self.Gr_sr = scanner.gradient_slew_rate      # [mT/m/ms]

        self.ref = ref.to("ms")
        self.time = time.to("ms")
        self.axis = axis
        self.user_defined = False

        # ------------------------------------------------------------------
        # USER-SUPPLIED TIMINGS (override analytic construction)
        # ------------------------------------------------------------------
        if timings is not None and amplitudes is not None:
            """
            When the user directly supplies timing and amplitude samples,
            the gradient shape is defined entirely by these arrays.

            The duration is inferred from the last timepoint.
            All analytic construction (slope/lenc/strength) is bypassed.
            """

            self.timings = timings
            self.amplitudes = amplitudes

            # Duration of the gradient relative to start time and reference
            self.dur = (self.timings[-1] - self.time + self.ref).to("ms")
            self.dur2 = (self.dur - self.ref).to("ms")

            # Build interpolator from supplied samples
            self.interpolator = interp1d(
                self.timings.m,
                self.amplitudes.m,
                kind="linear",
                fill_value=0.0,
                bounds_error=False,
            )

            # Flag: user-defined gradient parameters
            self.user_defined = True

            # Preserve user-provided parameters but do not use them
            self.slope = slope
            self.lenc = lenc
            self.strength = strength
            return

        # ------------------------------------------------------------------
        # ANALYTIC GRADIENT CONSTRUCTION
        # ------------------------------------------------------------------
        self.strength = self.Gr_max if strength is None else strength
        self.lenc = lenc

        # Determine slope from amplitude and slew rate if not provided
        self.slope = (
            np.abs(self.strength) / self.Gr_sr if slope is None else slope
        )

        # Duration of full trapezoid
        if self.lenc <= 0.0:
            self.dur = (2 * self.slope).to("ms")
        else:
            self.dur = (self.slope + self.lenc + self.slope).to("ms")

        self.dur2 = (self.dur - self.ref).to("ms")

        # Compute default timing/amplitude arrays
        self.timings, self.amplitudes, self.interpolator = self.group_timings()

    # ======================================================================
    # Representations and arithmetic
    # ======================================================================
    def __copy__(self):
        """Return a deep copy of the gradient object."""
        return copy.deepcopy(self)

    def __repr__(self):
        """String representation including major gradient parameters."""
        return (
            f"Gradient(slope={self.slope}, lenc={self.lenc}, strength={self.strength}, "
            f"Gr_max={self.Gr_max}, Gr_sr={self.Gr_sr}, ref={self.ref}, "
            f"time={self.time}, dur={self.dur}, axis={self.axis})"
        )

    def __call__(self, t):
        """Evaluate the gradient at time `t` using the internal interpolator."""
        return self.interpolator(t)

    def __mul__(self, other):
        """Multiply the gradient amplitude by a scalar.

        Parameters
        ----------
        other : float
            Scalar multiplier.

        Returns
        -------
        Gradient
            A new gradient object with scaled amplitude.

        Notes
        -----
        Only numerical scalars are permitted. This returns a *new* object.
        """
        if isinstance(other, (int, float, np.number)):
            return Gradient(
                slope=self.slope,
                lenc=self.lenc,
                strength=self.strength * other,
                scanner=self.scanner,
                ref=self.ref,
                time=self.time,
                axis=self.axis,
            )
        raise TypeError("Gradient can only be multiplied by a scalar (int or float).")

    # ======================================================================
    # Gradient Construction Helpers
    # ======================================================================
    def evaluate(self, t):
        """Evaluate the gradient interpolator at a given time point.

        Parameters
        ----------
        t : float
            Time at which to evaluate the gradient (ms).

        Returns
        -------
        float
            Gradient amplitude at time ``t`` (mT/m).
        """
        return self.interpolator(t)

    def group_timings(self):
        """Generate timing and amplitude arrays for a trapezoidal or triangular gradient.

        Returns
        -------
        tuple
            ``(timings, amplitudes, interpolator)`` where timings are offset
            by ``(time - ref)`` and the interpolator uses linear interpolation.
        """

        if self.lenc <= 0.0:
            # Triangular gradient
            timings = Quantity(
                np.array(
                    [0.0, self.slope.m, self.slope.m + self.slope.m],
                    dtype=np.float32,
                ),
                self.slope.u,
            )
            amplitudes = Quantity(
                np.array([0.0, self.strength.m, 0.0], dtype=np.float32),
                self.strength.u,
            )
        else:
            # Trapezoidal gradient
            timings = Quantity(
                np.array(
                    [
                        0.0,
                        self.slope.m,
                        self.slope.m + self.lenc.m,
                        self.slope.m + self.lenc.m + self.slope.m,
                    ],
                    dtype=np.float32,
                ),
                self.slope.u,
            )
            amplitudes = Quantity(
                np.array(
                    [0.0, self.strength.m, self.strength.m, 0.0],
                    dtype=np.float32,
                ),
                self.strength.u,
            )

        # Shift timing by sequence offsets
        timings += self.time - self.ref

        interpolator = interp1d(
            timings.m,
            amplitudes.m,
            kind="linear",
            fill_value=0.0,
            bounds_error=False,
        )

        return timings, amplitudes, interpolator

    # ======================================================================
    # Time & Reference Adjustment
    # ======================================================================
    def change_ref(self, ref):
        """Update the reference time of the gradient.

        Parameters
        ----------
        ref : Quantity
            New reference time (ms).
        """
        self.ref = ref.to("ms")
        self.dur2 = (self.dur - self.ref).to("ms")

    def change_time(self, time):
        """Update the absolute time of the gradient and rebuild timing arrays.

        Parameters
        ----------
        time : Quantity
            New absolute start time (ms).
        """
        self.time = time.to("ms")
        if not self.user_defined:
          self.timings, self.amplitudes, self.interpolator = self.group_timings()
        else:
          self.timings += self.time - self.ref

    # ======================================================================
    # Gradient Calculation Based on Bandwidth (original code preserved)
    # ======================================================================
    def calculate(self, k_bw, receiver_bw=None, ro_samples=None, ofac=None):
        """Calculate gradient shape from k-space bandwidth and scanner constraints.

        Parameters
        ----------
        k_bw : Quantity
            k-space bandwidth (1/m).
        receiver_bw : Quantity, optional
            Receiver bandwidth (Hz). If provided, the flat-top duration is
            fixed to accommodate the ADC window.
        ro_samples : int, optional
            Number of readout samples (required when ``receiver_bw`` is set).
        ofac : float, optional
            Oversampling factor (required when ``receiver_bw`` is set).

        Notes
        -----
        Updates ``slope``, ``lenc``, ``strength``, and the total ``dur`` in place.
        """

        if receiver_bw is not None:
            # Fixed flat-top duration from receiver bandwidth
            self.lenc = (ro_samples / ofac) / receiver_bw.to("1/ms")
            self.strength = (
                k_bw.to("1/m") /
                (self.scanner.gammabar.to("1/mT/ms") * self.lenc.to("ms"))
            )

            # Enforce hardware amplitude limit
            if self.strength > self.Gr_max:
                self.strength = self.Gr_max
                self.lenc = (
                    k_bw.to("1/m") /
                    (
                        self.scanner.gammabar.to("1/mT/s") *
                        self.strength.to("mT/m")
                    )
                )
                receiver_bw = ((ro_samples / ofac) / self.lenc).to("Hz")
                warnings.warn(
                    "Required gradient amplitude exceeds maximum. "
                    f"Adjusted receiver BW to {receiver_bw.m_as('Hz'):.0f} Hz."
                )

            self.slope = np.abs(self.strength) / self.Gr_sr

        else:
            # Compute triangular ramps only
            slope_req = np.sqrt(
                np.abs(k_bw.to("1/m")) /
                (
                    self.scanner.gammabar.to("1/mT/ms") *
                    self.Gr_sr.to("mT/m/ms")
                )
            )
            slope_max = self.Gr_max / self.Gr_sr

            if slope_req < slope_max:
                self.slope = slope_req
                self.strength = self.Gr_sr * slope_req
                self.lenc = self.slope - slope_req
            else:
                self.slope = slope_max
                self.strength = self.Gr_max

                k_slopes = (
                    self.scanner.gammabar.to("1/mT/ms") *
                    self.Gr_sr.to("mT/m/ms") *
                    slope_max.to("ms")**2
                )
                self.lenc = (
                    (np.abs(k_bw.to("1/m")) - k_slopes.to("1/m")) /
                    (self.strength.to("mT/m") *
                     self.scanner.gammabar.to("1/mT/ms"))
                )

            self.strength *= np.sign(k_bw)

        # Update total duration
        if self.lenc < 0:
            self.dur = self.slope + self.slope
        else:
            self.dur = self.slope + self.lenc + self.slope

        self.dur2 = (self.dur - self.ref).to("ms")
        self.timings, self.amplitudes, self.interpolator = self.group_timings()

    # ======================================================================
    # Bipolar Gradient Construction
    # ======================================================================
    def make_bipolar(self, VENC):
        """Construct a bipolar velocity-encoding gradient lobe pair.

        Modifies this gradient in place to become the first lobe and returns
        the second (inverted) lobe shifted in time.

        Parameters
        ----------
        VENC : Quantity
            Velocity encoding value (m/s).

        Returns
        -------
        Gradient
            Second gradient lobe (inverted, shifted by this lobe's duration).

        Notes
        -----
        The duration is chosen to produce the desired first-moment (velocity
        phase sensitivity π/VENC) using the scanner slew rate and amplitude
        limits.
        """

        VENC_sign = np.sign(VENC)
        VENC = np.abs(VENC)

        slope_max = (self.Gr_max / self.Gr_sr).to("ms")

        # Required slope for pure triangular VENC lobe
        slope_req = np.cbrt(
            Quantity(np.pi, "rad") /
            (
                2 *
                self.scanner.gamma.to("rad/ms/mT") *
                self.Gr_sr.to("mT/m/ms") *
                VENC.to("m/ms")
            )
        )

        if slope_req <= slope_max:
            self.slope = slope_req.to("ms")
            self.strength = -self.Gr_sr.to("mT/m/ms") * slope_req.to("ms")
            self.lenc = self.slope - slope_req
        else:
            a = (
                self.scanner.gamma.to("rad/ms/mT") *
                VENC.to("m/ms") *
                self.Gr_max.to("mT/m")
            )
            b = 3 * a * slope_max.to("ms")
            c = 2 * a * slope_max.to("ms")**2 - np.pi

            lenc_req = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

            self.slope = slope_max.to("ms")
            self.strength = -self.Gr_max.to("mT/m") * VENC_sign
            self.lenc = lenc_req.to("ms")

        # Update duration
        if self.lenc < 0:
            self.dur = (self.slope + self.slope).to("ms")
        else:
            self.dur = (self.slope + self.lenc + self.slope).to("ms")

        self.dur2 = (self.dur - self.ref).to("ms")
        self.timings, self.amplitudes, self.interpolator = self.group_timings()

        # Second lobe (inverted)
        g = self.__copy__()
        g *= -1.0
        g.change_time(self.time + self.dur)

        return g

    # ======================================================================
    # Area Computation
    # ======================================================================
    def area(self, t0=None, nb_samples=1000):
        """Compute the zeroth moment (area) of the gradient waveform.

        Parameters
        ----------
        t0 : Quantity or None, optional
            Integration start time (ms). Default is ``time - ref``.
        nb_samples : int, optional
            Number of samples for trapezoidal integration. Default is 1000.

        Returns
        -------
        Quantity
            Gradient area (mT·ms/m).
        """
        if t0 is None:
            t0 = self.time - self.ref

        t = np.linspace(
            t0.m_as("ms"),
            (self.time - self.ref + self.dur).m_as("ms"),
            nb_samples,
        )
        a = np.trapezoid(self.interpolator(t), t)
        return Quantity(a, "mT*ms/m")

    # ======================================================================
    # Area Matching (unchanged)
    # ======================================================================
    def match_area(self, area, dur=None):
        """Adjust slope, lenc, and strength to achieve a target gradient area.

        Parameters
        ----------
        area : Quantity
            Desired zeroth moment (mT·ms/m).
        dur : Quantity or None, optional
            Desired total duration (ms). If None, uses the minimal achievable
            duration given the scanner limits.

        Notes
        -----
        Sign of ``area`` is preserved; magnitude is used internally for
        calculations and restored at the end.
        """

        sign = np.sign(area)
        area = abs(area).to("mT*ms/m")

        slope_max = (self.Gr_max / self.Gr_sr).to("ms")

        if dur is not None:
            dur = dur.to("ms")

            # Case A: triangular only
            if dur < 2 * slope_max:
                self.slope = dur / 2
                self.lenc = Quantity(0.0, "ms")
                self.strength = (area / self.slope).to(self.Gr_max.u)

                if self.strength > self.Gr_max:
                    raise ValueError(
                        f"Cannot achieve area={area} in dur={dur}: "
                        f"G={self.strength} > Gmax={self.Gr_max}"
                    )

                self.dur = dur

            # Case B: plateau needed
            else:
                self.slope = slope_max
                self.lenc = dur - 2 * slope_max
                self.strength = (area / (self.slope + self.lenc)).to(
                    self.Gr_max.u
                )

                if self.strength > self.Gr_max:
                    raise ValueError(
                        f"Cannot achieve area={area} in dur={dur}: "
                        f"G={self.strength} > Gmax={self.Gr_max}"
                    )

                self.dur = dur

        else:
            # Minimal duration case
            slope_min = slope_max
            area_max = slope_min * self.Gr_max.to("mT/m")

            # Pure triangular
            if area <= area_max:
                ratio = (area / area_max).m
                self.slope = (slope_min * np.sqrt(ratio)).to("ms")
                self.strength = (
                    self.Gr_max * np.sqrt(ratio)
                ).to(self.Gr_max.u)
                self.lenc = (
                    self.slope - slope_min * np.sqrt(ratio)
                ).to("ms")

            # Plateau required
            else:
                area_needed = area - area_max
                self.slope = slope_min
                self.strength = self.Gr_max
                self.lenc = (area_needed / self.Gr_max).to("ms")

            # Duration update
            if self.lenc.m <= 0:
                self.dur = 2 * self.slope
            else:
                self.dur = 2 * self.slope + self.lenc

        # Restore sign
        self.strength *= sign

        # Recompute waveform
        self.dur2 = (self.dur - self.ref).to("ms")
        self.timings, self.amplitudes, self.interpolator = self.group_timings()

    # ======================================================================
    # Gradient Rotation
    # ======================================================================
    def rotate(self, directions, normalize_dirs=False):
        """Decompose the gradient into axis components along given direction(s).

        Parameters
        ----------
        directions : np.ndarray
            Direction vector(s), shape ``(N, 3)`` in MPS coordinates.
        normalize_dirs : bool, optional
            If True, each direction vector is normalized before decomposition.
            Default is False.

        Returns
        -------
        list of Gradient or list of list of Gradient
            For a single direction, a list of up to 3 axis-specific gradients.
            For multiple directions, a list of such lists.
        """
        directions = directions.reshape((-1, 3))
        if directions.shape[1] != 3:
            raise ValueError("Direction must be a 3-element vector [M,P,S].")

        nb_dirs = directions.shape[0]
        gradients = [[] for _ in range(nb_dirs)]

        for d in range(nb_dirs):
            direction = directions[d, :]
            norm = np.linalg.norm(direction)

            if norm != 0 and normalize_dirs:
                direction = direction / norm

            area_val = self.area()

            for i, fraction in enumerate(direction):
                if fraction != 0.0:
                    g = self.__copy__()
                    g.axis = i
                    g.match_area(fraction * area_val)
                    gradients[d].append(g)
                elif fraction == 0.0 and norm == 0.0:
                    g = self.__copy__()
                    g *= 0.0
                    gradients[d].append(g)
                    break

        max_dur = max(
            [
                g.dur if g.strength != 0.0 else Quantity(0.0, "ms")
                for d in range(nb_dirs)
                for g in gradients[d]
            ]
        )

        for d in range(nb_dirs):
            for g in gradients[d]:
                g.match_area(g.area(), max_dur)

        return gradients[0] if nb_dirs == 1 else gradients

    # ======================================================================
    # Plotting
    # ======================================================================
    def plot(self, linestyle="-"):
        """Plot the gradient waveform.

        Parameters
        ----------
        linestyle : str, optional
            Matplotlib line style string. Default is ``'-'``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure()
        plt.plot(self.timings, self.amplitudes, linestyle)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (mT/m)")
        plt.title("Gradient waveform")
        plt.grid(True)
        plt.show()
        return fig


class RF:
    """
    Radiofrequency (RF) excitation pulse class.

    This class generates RF waveforms commonly used in MRI imaging and
    simulation. It supports both *analytic pulse generation* (sinc, apodized
    sinc, hard pulses), and *user-provided custom waveforms*. All timing,
    phase, magnitude, and flip-angle normalization behavior is preserved
    exactly as in the original implementation.

    Key Features
    ------------
    - Supports analytical "hard", "sinc", and "apodized_sinc" RF shapes.
    - Maintains full Quantity support for physical units (ms, rad, Hz, etc.).
    - Preserves historical behavior: windowing, t-shifting, apodization,
      and flip-angle normalization.
    - Flip-angle specification is enforced by integrating the amplitude
      and applying a normalization factor.
    - Fully complex-valued interpolation, avoiding the real/imag duplication.
    - Optional custom waveform definition via `timings` + `waveform`.
    - Stores computed `timings` and complex `waveform` automatically.
    - Safe default for `scanner` (no mutable default instances).

    Parameters
    ----------
    scanner : Scanner or None
        MRI system definition containing at least `gamma`.
        If None, a default Scanner() is created.

    NbLobes : list or tuple of int, default=[2, 2]
        Number of lobes on the left and right sides of the sinc pulse.

    alpha : float, default=0.46
        Apodization factor for the apodized sinc. Ignored if shape='sinc'.

    shape : {'sinc', 'apodized_sinc', 'hard'}, default='apodized_sinc'
        RF pulse shape to generate.

    flip_angle : Quantity, default=pi/2 rad
        Desired flip angle of the RF pulse.

    dur : Quantity, default=2 ms
        Total duration of the RF pulse.

    ref : Quantity, default=0 ms
        Reference time for phase and time-shifting.

    time : Quantity, default=0 ms
        Time origin of the pulse.

    nb_samples : int, default=200
        Number of time samples used to generate the interpolated waveform.

    phase_offset : Quantity, default=0 rad
        Constant phase offset to apply to the entire RF waveform.

    frequency_offset : Quantity, default=0 Hz
        Frequency offset of the RF waveform (modulates B1 via exp(i 2π f t)).

    timings : array-like of floats or Quantities, optional
        Custom time vector in ms (or convertible to ms). If provided together
        with `waveform`, the analytic pulse generator is bypassed.

    waveform : array-like of complex, optional
        Custom RF waveform values corresponding to `timings`.

    Notes
    -----
    - Interpolation uses linear complex interpolation.
    - The internal pulse is always generated in milliseconds.
    """
    def __init__(self,
                 scanner=None,
                 NbLobes=[2, 2],
                 alpha=0.46,
                 shape='apodized_sinc',
                 flip_angle=Quantity(np.pi/2, 'rad'),
                 dur=Quantity(2.0, 'ms'),
                 ref=Quantity(0.0, 'ms'),
                 time=Quantity(0.0, 'ms'),
                 nb_samples=200,
                 phase_offset=Quantity(0.0, 'rad'),
                 frequency_offset=Quantity(0.0, 'Hz'),
                 timings=None,
                 waveform=None):

        # Safe default for scanner (prevents mutable default hazards)
        self.scanner = scanner if scanner is not None else Scanner()

        self.NbLobes = NbLobes
        self.alpha = alpha
        self.shape = shape

        # Shape selection (exact original behavior)
        if self.shape == 'sinc':
            self._pulse = self._unit_sinc
            if self.alpha != 0.0 and MPI_rank == 0:
                warnings.warn("For 'sinc' shape, the alpha parameter is automatically set to 0.0")
            self.alpha = 0.0
        elif self.shape == 'apodized_sinc':
            self._pulse = self._unit_sinc
        elif self.shape == 'hard':
            self._pulse = self._unit_hard

        # Physical parameters with units
        self.flip_angle = flip_angle.to('rad')
        self.ref = ref.to('ms')
        self.time = time.to('ms')
        self.dur = dur.to('ms')
        self.dur2 = (self.dur - self.ref).to('ms')
        self.nb_samples = nb_samples

        # Lobe durations for “sinc-like” pulse shapes
        self.half1 = (self.NbLobes[0] + 1)/(np.sum(self.NbLobes) + 2)*self.dur.to('ms')
        self.half2 = (self.NbLobes[1] + 1)/(np.sum(self.NbLobes) + 2)*self.dur.to('ms')

        self.phase_offset = phase_offset.to('rad')
        self.frequency_offset = frequency_offset.to('Hz')

        # Public waveform storage
        self.timings = None    # in ms
        self.waveform = None   # complex RF samples
        self._custom_waveform = False

        # Complex-valued interpolator
        self.interp = None

        # If user provides custom waveform → bypass analytic generator
        if timings is not None and waveform is not None:
            self._init_from_user_waveform(timings, waveform)
            self._custom_waveform = True
        else:
            self._build_interpolator()

    # ======================================================================
    # Internal time helpers
    # ======================================================================
    def _window(self, t):
        """Return a rectangular window selecting times between (time-ref) and (time-ref+dur)."""
        start = (self.time - self.ref).m_as('ms')
        end   = (self.time - self.ref + self.dur).m_as('ms')
        return (t >= start)*(t <= end)

    def _t_shift(self, t):
        """Return shifted local time used inside the RF excitation model."""
        return t - (self.time - self.ref).m_as('ms') - self.half1.m_as('ms')

    # ======================================================================
    # Analytic pulse definitions (bit-for-bit identical to original)
    # ======================================================================
    def _unit_sinc(self, t):
        """
        Generate an (apodized) sinc pulse.

        This method preserves the exact amplitude, windowing, phase modulation,
        and apodization behavior of the original implementation.
        """
        N = max(self.NbLobes)
        t_shift = self._t_shift(t)

        bw = (self.NbLobes[0] + self.NbLobes[1] + 2)/self.dur.to('ms')

        B1e = (1/bw.m)
        B1e *= (1 - self.alpha) + self.alpha*np.cos(np.pi*bw.m*t_shift/N)
        B1e *= np.sinc(bw.m*t_shift)
        B1e *= self._window(t)

        # Construct complex B1 with real magnitude (exact original)
        B1 = B1e + 1j*0

        # Apply phase + frequency offsets
        if self.phase_offset.m != 0.0 or self.frequency_offset.m != 0.0:
            B1 *= np.exp(1j*(self.phase_offset.m_as('rad')
                            + 2*np.pi*self.frequency_offset.m_as('kHz')*t_shift))

        return B1

    def _unit_hard(self, t):
        """Generate a hard (rectangular) RF pulse."""
        t_shift = self._t_shift(t)
        B1e = 1.0 * self._window(t)
        B1 = B1e + 1j*0

        if self.phase_offset.m != 0.0 or self.frequency_offset.m != 0.0:
            B1 *= np.exp(1j*(self.phase_offset.m_as('rad')
                            + 2*np.pi*self.frequency_offset.m_as('kHz')*t_shift))

        return B1

    # ======================================================================
    # Flip-angle normalization
    # ======================================================================
    def _flip_angle_factor(self, t):
        """
        Compute normalization factor so that the final RF pulse integrates to
        the desired flip angle.

        Flip angle = γ ∫ B1(t) dt
        """
        dt = t[1] - t[0]
        amp = self._pulse(t)
        unit_FA = np.sum((amp[1:] + amp[:-1])/2)*dt*self.scanner.gamma.m_as('rad/mT/ms')
        return self.flip_angle.m_as('rad')/unit_FA

    # ======================================================================
    # Interpolator constructors
    # ======================================================================
    def _init_from_user_waveform(self, timings, waveform):
        """
        Initialize the RF using a custom user-supplied waveform.
        """
        self.timings = timings
        self.waveform = waveform

        # Dimensionless arrays for interpolation
        tt = timings.m_as('ms') if isinstance(timings, Quantity) else np.array(timings, dtype=np.float32)
        ww = waveform.m_as('mT') if isinstance(waveform, Quantity) else np.array(waveform, dtype=np.complex64)

        # Duration of the gradient relative to start time and reference
        self.dur = (self.timings[-1] - self.timings[0] + self.ref).to("ms")
        self.dur2 = (self.dur - self.ref).to("ms")

        self.interp = interp1d(
            tt, ww,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )

    def _build_interpolator(self):
        """
        Build the analytic RF waveform, scale it to the correct flip angle,
        and construct the complex interpolating function.
        """
        # print("_build_interpolator: Generating analytic RF waveform.")
        start = (self.time - self.ref).m_as('ms')
        end   = (self.time - self.ref + self.dur).m_as('ms')

        t = np.linspace(start, end, self.nb_samples)

        # Flip-angle controlled scaling
        scaling = self._flip_angle_factor(t)
        wf = np.abs(scaling) * self._pulse(t)

        self.timings = t
        self.waveform = wf

        self.interp = interp1d(
            t, wf,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )

    # ======================================================================
    # Public Methods
    # ======================================================================
    def __call__(self, t):
        """
        Evaluate the complex RF pulse at time `t`.

        Parameters
        ----------
        t : float, ndarray, or Quantity
            Time at which to evaluate the RF (in ms or convertible to ms).

        Returns
        -------
        complex or ndarray of complex
            Complex RF field B1(t).
        """
        if isinstance(t, Quantity):
            t = t.to('ms').m_as('ms')
        return self.interp(t)


    def change_ref(self, ref):
        """Change the reference time of the RF pulse."""
        self.ref = ref.to('ms')
        self.dur2 = (self.dur - self.ref).to('ms')
        if self._custom_waveform:
            self._init_from_user_waveform(self.timings, self.waveform)
        else:
            self._build_interpolator()


    def change_time(self, time):
        """Change the absolute timing of the RF pulse."""
        self.time = time.to('ms')
        if self._custom_waveform:
            start = (self.time - self.ref).m_as('ms')
            end   = (self.time - self.ref + self.dur).m_as('ms')
            t = Quantity(np.linspace(start, end, self.waveform.size), 'ms')
            self._init_from_user_waveform(t, self.waveform)
        else:
            self._build_interpolator()


    def plot(self, linestyle='-'):
        """
        Plot the analytical (unscaled) RF pulse shape used internally.

        This is primarily for visual inspection of the pulse design itself.
        """
        start = (self.time - self.ref).m_as('ms')
        end   = (self.time - self.ref + self.dur).m_as('ms')
        t = np.linspace(start, end, self.nb_samples)

        plt.figure()
        plt.plot(t, self._pulse(t).real, linestyle)
        plt.plot(t, self._pulse(t).imag, linestyle)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (mT)")
        plt.legend(["Real", "Imag"])
        plt.show()