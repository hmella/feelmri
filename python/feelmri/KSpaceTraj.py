"""
k-space trajectory generation for Cartesian, radial, and spiral MRI acquisitions.

All trajectory classes inherit from :class:`Trajectory` and produce arrays of
k-space sample coordinates (``points``) and acquisition times (``times``) that
can be passed directly to the Bloch solver and image reconstruction routines.
"""
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from feelmri.MPIUtilities import MPI_comm, MPI_rank
from feelmri.MRObjects import Gradient, Scanner


class Trajectory:
    """Base class for MRI k-space trajectories.

    Computes gradient waveform parameters (readout, phase encoding) and stores
    k-space extents, timing information, and hardware limits used by all
    derived trajectory types.

    Parameters
    ----------
    FOV : Quantity, optional
        Field-of-view ``[Mx, Py, Sz]`` (m). Default is ``[0.3, 0.3, 0.08]`` m.
    res : np.ndarray, optional
        Image matrix size ``[Nx, Ny, Nz]``. Default is ``[100, 100, 1]``.
    oversampling : int, optional
        Readout oversampling factor applied along the frequency-encoding
        direction. Default is 2.
    lines_per_shot : int, optional
        Number of phase-encoding lines acquired per excitation (EPI factor).
        Default is 1.
    scanner : Scanner, optional
        Scanner hardware definition. Default is a standard 1.5 T scanner.
    t_start : Quantity, optional
        Echo time offset — the absolute start time of the readout window
        relative to the RF pulse (ms). Default is 0 ms.
    receiver_bw : Quantity, optional
        Receiver bandwidth (Hz). Default is 128 kHz.
    plot_seq : bool, optional
        If True, plot gradient waveforms when computing k-space points.
        Default is False.
    MPS_ori : np.ndarray, optional
        3×3 machine-to-patient-space rotation matrix. Default is identity.
    LOC : Quantity or np.ndarray, optional
        3-element slice location. A ``Quantity`` is converted; a bare array is
        taken to be METRES. Default is ``[0, 0, 0]``.

        **Recorded for provenance -- the trajectory does not apply it.** No
        code path in ``feelmri`` reads ``self.LOC`` back. The slice offset
        reaches the physics through the PHANTOM instead,
        ``FEMPhantom.orient(MPS_ori, LOC)``, which moves the object into the
        imaging frame; applying it here as well would double-count it.
    dtype : np.dtype, optional
        Floating-point precision. Default is ``np.float32``.
    """

    def __init__(
        self,
        FOV: Quantity = None,
        res: np.ndarray = None,
        oversampling: int = 2,
        lines_per_shot: int = 1,
        scanner: Scanner = None,
        t_start: Quantity = Quantity(0, 'ms'),
        receiver_bw: Quantity = Quantity(128.0e+3, 'Hz'),
        plot_seq: bool = False,
        MPS_ori: np.ndarray = None,
        LOC: np.ndarray = None,
        dtype: np.dtype = np.float32,
    ):
        # Evaluated-once defaults in the signature would be ONE shared array
        # and ONE shared Scanner for every trajectory built without explicit
        # arguments -- and this class passes that scanner straight into every
        # Gradient it constructs, so mutating one trajectory's hardware limits
        # would change them for all of them.
        self.scanner = Scanner() if scanner is None else scanner
        self.FOV = Quantity(np.array([0.3, 0.3, 0.08]), 'm') if FOV is None \
            else FOV
        self.res = np.array([100, 100, 1]) if res is None else np.asarray(res)
        # Later lines read the local names, not the attributes.
        scanner, FOV, res = self.scanner, self.FOV, self.res
        self.oversampling = oversampling
        self.oversampling_arr = np.array([oversampling, 1.0, 1.0])
        self.Gr_max = scanner.gradient_strength   # [mT/m]
        self.Gr_sr  = scanner.gradient_slew_rate  # [mT/(m*ms)]
        self.gammabar = scanner.gammabar           # [Hz/T]
        self.lines_per_shot = lines_per_shot
        self.ro_samples = self.oversampling * self.res[0]  # number of readout samples
        self.ph_samples = self.check_ph_enc_lines(self.res[1])
        self.slices = self.res[2]                  # number of slices
        self.nb_shots = self.ph_samples // self.lines_per_shot
        self.shots = [[None, ] * self.lines_per_shot for _ in range(self.nb_shots)]
        self.pxsz = FOV / res
        self.k_bw = 1.0 / self.pxsz
        self.k_spa = 1.0 / (self.oversampling_arr * FOV)
        self.kx_extent = (
            (np.array([0, self.ro_samples - 1]) - self.ro_samples // 2) * self.k_spa[0]
            # + float((self.res[0] % 2 != 0)) * self.k_spa[0]
        )
        self.ky_extent = (np.array([0, self.ph_samples - 1]) - self.ph_samples // 2) * self.k_spa[1]
        self.kz_extent = (np.array([0, self.slices - 1]) - self.slices // 2) * self.k_spa[2]
        self.t_start = t_start.astype(dtype)
        self.plot_seq = plot_seq
        self.receiver_bw = receiver_bw          # [Hz]
        MPS_ori = np.eye(3) if MPS_ori is None else np.asarray(MPS_ori)
        # Converted rather than coerced: `np.asarray` on a Quantity DISCARDS
        # the unit (with a warning nobody reads), so a location written in cm
        # -- which several PVSM files are -- would have been stored as though
        # it were metres. It is the only unit-stripping site the example suite
        # had, and it was invisible because nothing reads the value back.
        if isinstance(LOC, Quantity):
            LOC = LOC.m_as('m')
        LOC = np.zeros([3, ]) if LOC is None else np.asarray(LOC, dtype=float)
        self.MPS_ori = MPS_ori.astype(dtype)   # orientation
        self.LOC = LOC.astype(dtype)           # location, metres
        self.dtype = dtype

    def maxwell_coefficients(self, scanner, t0=None, carried=None,
                             t_snapshot=None):
        """Concomitant phase coefficients for this readout, ready for
        ``mri_signal(..., maxwell=...)``.

        Returns ``(N, 6)`` in rad/m^2 laid out like :attr:`times` flattened.
        Integrated forward from ``t0`` (default 0 ms, the instant this
        trajectory's own timings are measured from), so everything between the
        magnetization snapshot and each sample is counted -- the PREPHASERS
        included, which is the point. Measured on the geometry
        ``examples/phase_contrast.py`` uses (120 x 60 mm, 60 x 30, x2
        oversampling, oblique): the readout prephaser and the phase encode
        carry **52%** of the whole window's ``integral(Gx^2 + Gy^2) dt``, and
        59% has accumulated by the first ADC sample. A moment estimated from
        the sampled k-space cannot see any of it, because the sampling starts
        after it is over.

        Both rotations implied by :attr:`MPS_ori` are applied here. The
        gradients are along logical (readout / phase / slice) axes while the
        Maxwell products are B0-aligned, and the node coordinates the assembler
        works in are the imaging ones that ``FEMPhantom.orient`` leaves behind.
        Getting either wrong evaluates the expression as though the slice
        normal were B0 -- and since ``Bc`` singles out z, there is no rotation
        under which an oblique acquisition reduces to an axial one.

        **``t0`` must be the snapshot instant, and a trajectory built without
        ``t_start`` does not have one.** The prephasers run over
        ``[t_start - dur, t_start]``, so at the default ``t_start = 0`` they
        sit at NEGATIVE times and are outside any forward integration from 0.
        That is warned about rather than guessed at: the trajectory does not
        know where the excitation was.

        ``carried`` are gradients the solver has already integrated, given in
        this trajectory's own time frame. ``Bc`` is quadratic in G, so
        ``Bc(G_a + G_b) != Bc(G_a) + Bc(G_b)``: a carried set that overlaps
        this trajectory's gradients in time contributes a cross term that
        neither half computes alone. Passing them integrates the whole field
        once and subtracts what the solver already applied, the integral up to
        ``t_snapshot`` (default :attr:`t_start`) of the carried gradients
        alone.

        **Caveat specific to CartesianStack.** Both encoding axes are modelled
        by their largest prephaser rather than one waveform per line and per
        partition, so the y and z products are an upper bound. The readout
        axis, which dominates, is exact.
        """
        from feelmri.PulseqAdapter import (maxwell_moments as _moments,
                                           maxwell_phase_coefficients as _coef)

        gradients = getattr(self, 'gradients', None)
        if gradients is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not retain its gradient waveforms, "
                f"so the concomitant moments cannot be computed exactly. Only "
                f"CartesianStack does today; feelmri.maxwell_moments_from_kspace "
                f"is the approximate fallback, and it cannot see a prephaser.")
        t0 = 0.0 if t0 is None else float(t0)
        carried = [] if carried is None else list(carried)
        earliest = min(float(np.asarray(g.timings.m_as('ms')).min())
                       for g in gradients)
        if earliest < t0 - 1e-9:
            warnings.warn(
                f"{type(self).__name__}: gradient activity starts at "
                f"{earliest:.4f} ms, before the integration origin "
                f"{t0:.4f} ms, and everything before the origin is DROPPED. "
                f"On a Cartesian readout that is the prephaser, which can "
                f"carry more of the concomitant second moment than the "
                f"readout itself. Build the trajectory with `t_start` at the "
                f"magnetization snapshot, or pass `t0` explicitly.")
        R = np.asarray(self.MPS_ori, dtype=float)
        times = np.asarray(self.times.m_as('ms'), dtype=float).reshape(-1)
        if carried:
            t_snap = (float(self.t_start.m_as('ms')) if t_snapshot is None
                      else float(t_snapshot))
            # One integration of the summed field, so the cross terms are
            # present, minus what the solver already applied up to t_snap
            moments = (_moments(list(gradients) + carried, t0, times,
                                rotation=R)
                       - _moments(carried, t0, np.array([t_snap]),
                                  rotation=R)[0])
        else:
            moments = _moments(gradients, t0, times, rotation=R)
        return _coef(moments, scanner, rotation=R)

    def check_ph_enc_lines(self, ph_samples):
        """Verify that the number of phase-encoding lines is divisible by the multishot factor."""
        return np.int32(self.lines_per_shot * (ph_samples // self.lines_per_shot))

    def plot_trajectory(self, figsize=(12, 5), tight_layout=True, export_to=None):
        """Show k-space points and time map."""
        if MPI_rank == 0:
            # plt.rcParams['text.usetex'] = True
            # plt.rcParams.update({'font.size': 16})

            fig, ax = plt.subplots(1, 2, figsize=figsize)
            for shot in self.shots:
                kxx = np.concatenate((np.array([0]), self.points[0][:,shot,0].flatten('F')))
                kyy = np.concatenate((np.array([0]), self.points[1][:,shot,0].flatten('F')))
                ax[0].plot(kxx, kyy)
            ax[0].set_xlabel('$k_x ~(1/m)$')
            ax[0].set_ylabel('$k_y ~(1/m)$')

            im = ax[1].scatter(
                self.points[0][:,:,0], self.points[1][:,:,0],
                c=self.times[:,:,0].m_as('ms'), s=2.5, cmap='turbo',
            )
            ax[1].set_xlabel('$k_x ~(1/m)$')
            ax[1].set_yticklabels([])
            if tight_layout:
                plt.tight_layout()
            cbar = fig.colorbar(im, orientation='vertical', ax=ax)
            cbar.ax.set_title('Time [ms]')
            if export_to is not None:
                plt.savefig(export_to, bbox_inches='tight')
            plt.show()

        # Synchronize all processes
        MPI_comm.Barrier()


class CartesianStack(Trajectory):
    """Stack-of-Cartesian k-space trajectory (standard spin-warp or EPI).

    Acquires phase-encoding lines along :math:`k_y` and stacks slices along
    :math:`k_z`. Supports single- or multi-shot EPI-style acquisitions via
    ``lines_per_shot``.

    Parameters
    ----------
    shot_coverage : {'full', 'partial'}, optional
        Whether each shot covers the full or a partial set of k-space lines.
        Default is ``'full'``.
    *args, **kwargs
        Forwarded to :class:`Trajectory`.
    """

    def __init__(self, shot_coverage: Literal["full", "partial"] = "full", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shot_coverage = shot_coverage
        self.ph_samples = self.check_ph_enc_lines(self.res[1])
        self.nb_shots = self.ph_samples // self.lines_per_shot
        (self.points, self.times) = self.kspace_points()

    def kspace_points(self):
        """Compute Cartesian k-space points and acquisition times."""
        # k-space positioning gradients
        ph_grad = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
        ph_grad.calculate(0.5 * self.k_bw[1].to('1/m'))

        ro_grad0 = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
        ro_grad0.calculate(
            -0.5 * self.k_bw[0].to('1/m')
            - 0.5 * ro_grad0.scanner.gammabar.to('1/mT/ms')
            * ro_grad0.strength.to('mT/m')
            * ro_grad0.slope.to('ms')
        )

        blip_grad = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
        blip_grad.calculate(-self.k_bw[1].to('1/m') / self.ph_samples)

        # Gradient duration is not used because can be shorter than second half of the slice selection gradient
        enc_time = Quantity(
            self.t_start.m_as('ms') - np.max([ph_grad.dur.m_as('ms'), ro_grad0.dur.m_as('ms')]),
            'ms',
        )

        # Update timings
        ph_grad.change_time(enc_time)
        ro_grad0.change_time(enc_time)

        # Partition-encode gradient, at the largest |kz| the stack reaches. It
        # ends at `t_start` rather than starting with the in-plane prephasers,
        # which keeps it out of the first echo; `enc_time` is unchanged, so no
        # echo time and no sample time moves.
        enc_gradients = []
        if self.slices > 1:
            kz_max = np.max(np.abs(self.kz_extent.m_as('1/m')))
            kz_grad = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
            kz_grad.calculate(Quantity(kz_max, '1/m'))
            kz_grad.change_time(self.t_start - kz_grad.dur)
            kz_grad.axis = 2
            if kz_grad.time < Quantity(0.0, 'ms'):
                warnings.warn(
                    f"{type(self).__name__}: the partition-encode prephaser is "
                    f"{kz_grad.dur.m_as('ms'):.4f} ms and `t_start` is only "
                    f"{self.t_start.m_as('ms'):.4f} ms, so it starts before "
                    f"the trajectory's own origin and part of it is outside "
                    f"any forward integration from there.")
            enc_gradients.append(kz_grad)

        ro_gradients = [ro_grad0, ]
        ph_gradients = [ph_grad, ]
        for i in range(self.lines_per_shot):
            # Calculate readout gradient
            ro_grad = Gradient(time=ro_gradients[i].timings[-1], scanner=self.scanner)
            ro_grad.calculate(
                (-1)**i * self.k_bw[0].to('1/m'),
                receiver_bw=self.receiver_bw.to('Hz'),
                ro_samples=self.ro_samples,
                ofac=self.oversampling,
            )
            ro_gradients.append(ro_grad)

            # Calculate blip gradient
            if self.lines_per_shot > 1 and i < self.lines_per_shot - 1:
                ref = ro_gradients[-1].time + ro_gradients[-1].dur - 0.5 * blip_grad.dur
                blip_grad = Gradient(time=ref, scanner=self.scanner)
                blip_grad.calculate(-self.k_bw[1].to('1/m') / self.ph_samples)
                ph_gradients.append(blip_grad)

        if self.plot_seq:
            if MPI_rank == 0:
                # plt.rcParams['text.usetex'] = True
                plt.rcParams.update({'font.size': 16})

                fig, ax = plt.subplots(2, 1, figsize=(8, 4))

                # Phase encoding gradients
                for gr in ph_gradients:
                    ax1 = ax[1].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'r-', linewidth=2)

                # Readout gradients
                for gr in ro_gradients:
                    ax2 = ax[0].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'b-', linewidth=2)

                # Encoding gradients
                for gr in enc_gradients:
                    ax3 = ax[0].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'k--', linewidth=3, zorder=100)
                    ax4 = ax[1].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'k--', linewidth=3, zorder=100)

                # Add ADC readout
                for i in range(1, self.lines_per_shot + 1):
                    a = [
                        (ro_gradients[i].time + ro_gradients[i].slope).m_as('ms'),
                        (ro_gradients[i].time + ro_gradients[i].slope + ro_gradients[i].lenc).m_as('ms'),
                    ]
                    ax5 = ax[0].plot(a, [0, 0], 'm-', linewidth=4, zorder=101)

                # Set legend labels
                ax1[0].set_label('PH')
                ax2[0].set_label('RO')
                ax5[0].set_label('ADC')

                # Format plots
                for i in range(len(ax)):
                    ax[i].hlines(y=[0], xmin=0, xmax=[100], colors=['0.7'], linestyles='solid')
                    ax[i].tick_params('both', length=5, width=1, which='major', labelsize=16)
                    ax[i].tick_params('both', length=3, width=1, which='minor', labelsize=16)
                    ax[i].minorticks_on()
                    ax[i].set_ylabel('$G~\\mathrm{(mT/m)}$', fontsize=16)
                    ax[i].axis([
                        0,
                        ro_gradients[-1].time.m_as('ms') + ro_gradients[-1].dur.m_as('ms'),
                        -1.4 * self.Gr_max.m_as('mT/m'),
                        1.4 * self.Gr_max.m_as('mT/m'),
                    ])
                    ax[i].legend(fontsize=14, loc='upper right', ncol=4)
                ax[1].set_xlabel('$t~\\mathrm{(ms)}$', fontsize=16)
                plt.tight_layout()
                plt.show()

            # Synchronize all processes
            MPI_comm.Barrier()

        # Time needed to acquire one line
        # It depends on the k-space bandwidth, the gyromagnetic constant, and
        # the maximum gradient amplitude
        dt = np.linspace(0.0, ro_grad.lenc.m_as('ms'), self.ro_samples)

        # kspace locations
        kx = np.linspace(self.kx_extent[0].m_as('1/m'), self.kx_extent[1].m_as('1/m'), self.ro_samples)
        ky = self.ky_extent[0].m_as('1/m') * np.ones(kx.shape)
        kz = np.linspace(self.kz_extent[0].m_as('1/m'), self.kz_extent[1].m_as('1/m'), self.slices)

        kspace = (
            np.zeros([self.ro_samples, self.ph_samples, self.slices], dtype=self.dtype),
            np.zeros([self.ro_samples, self.ph_samples, self.slices], dtype=self.dtype),
            np.zeros([self.ro_samples, self.ph_samples, self.slices], dtype=self.dtype),
        )

        # Build shots locations
        for ph in range(self.ph_samples):
            if self.shot_coverage == "partial":
                self.shots[ph // self.lines_per_shot][ph % self.lines_per_shot] = ph
            elif self.shot_coverage == "full":
                self.shots[ph % self.nb_shots][ph // self.nb_shots] = ph

        # kspace times and locations
        t = np.zeros([self.ro_samples, self.ph_samples, self.slices], dtype=self.dtype)
        for shot in self.shots:
            for idx, ph in enumerate(shot):

                # Readout direction
                ro = (-1)**idx

                # Fill locations
                kspace[0][::ro, ph, :] = np.tile(kx[:, None], [1, self.slices])
                kspace[1][::ro, ph, :] = np.tile(
                    ky[:, None] + self.k_spa[1].m_as('1/m') * ph, [1, self.slices]
                )

                # Update timings
                if idx == 0:
                    t[::ro, ph, 0] = (
                        enc_time.m_as('ms') + ro_grad0.dur.m_as('ms')
                        + ro_grad.slope.m_as('ms') + dt
                    )
                else:
                    t[::ro, ph, 0] = (
                        t[:, shot[idx - 1]].max()
                        + ro_grad.slope.m_as('ms') + ro_grad.slope.m_as('ms')
                        + dt[::ro]
                    )

        # Fill kz coordinates
        for s in range(self.slices):
            kspace[2][:, :, s] = kz[s]
            t[:, :, s] = t[:, :, 0]

        # Calculate echo time
        self.echo_time = (enc_time + ro_grad0.dur + 0.5 * self.lines_per_shot * ro_grad.dur).to('ms')

        # Retained for callers that need the WAVEFORM rather than the sampled
        # k. The concomitant moments are the case in point: they depend on
        # integral(G^2), which cannot be recovered from k once the prephaser
        # has been played -- on a 300 mm / 128 geometry that prephaser carries
        # 1.49x the readout's own second moment, so a k-derived estimate would
        # miss the larger share. Setting `axis` here costs nothing: it is only
        # read by consumers, never by the gradient's own arithmetic.
        for g in ro_gradients:
            g.axis = 0
        for g in ph_gradients:
            g.axis = 1
        self.gradients = (list(ro_gradients) + list(ph_gradients)
                          + list(enc_gradients))

        return (kspace, Quantity(t, 'ms'))


class RadialStack(Trajectory):
    """Stack-of-radials k-space trajectory with golden-angle or uniform spoke ordering.

    Acquires radial spokes in the :math:`k_x`–:math:`k_y` plane and stacks
    them along :math:`k_z`. Spoke angles can follow a uniform distribution or
    the golden-angle increment for pseudo-random incoherent undersampling.

    Parameters
    ----------
    golden_angle : bool, optional
        If True, spoke angles follow the golden-angle increment
        (≈ 111.25°). Default is False (uniform angular spacing).
    full_spoke : bool, optional
        If True, each spoke covers the full diameter of k-space
        (center-out + back). Default is False (half-spoke, center-out).
    *args, **kwargs
        Forwarded to :class:`Trajectory`.
    """

    def __init__(self, *args,
                 golden_angle: bool = False,
                 full_spoke: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.golden_angle = golden_angle
        self.full_spoke = full_spoke
        self.ph_samples = self.check_ph_enc_lines(self.ph_samples)
        self.nb_shots = self.ph_samples // self.lines_per_shot
        (self.points, self.times) = self.kspace_points()

    def kspace_points(self):
        """Compute radial k-space points and acquisition times."""
        # k-space positioning gradients
        ph_grad = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
        ph_grad.calculate(0.5 * self.k_bw[1].to('1/m'))

        ro_grad0 = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
        blip_grad = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
        if self.golden_angle:
            blip_grad.calculate(-self.k_bw[0].to('1/m') / self.ro_samples)
            ro_grad0 = blip_grad.__copy__()
        else:
            ro_grad0.calculate(
                -0.5 * self.k_bw[0].to('1/m')
                - 0.5 * ro_grad0.scanner.gammabar.to('1/mT/ms')
                * ro_grad0.strength.to('mT/m')
                * ro_grad0.slope.to('ms')
            )
            blip_grad.calculate(-self.k_bw[1].to('1/m') / self.ph_samples)

        # Gradient duration is not used because can be shorter than second half of the slice selection gradient
        enc_time = Quantity(
            self.t_start.m_as('ms') - np.max([ph_grad.dur.m_as('ms'), ro_grad0.dur.m_as('ms')]),
            'ms',
        )

        # Update timings
        ph_grad.change_time(enc_time)
        ro_grad0.change_time(enc_time)

        enc_gradients = []
        ro_gradients = [ro_grad0, ]
        ph_gradients = [ph_grad, ]
        for i in range(self.lines_per_shot):
            # Calculate readout gradient
            ro_grad = Gradient(time=ro_gradients[i].timings[-1], scanner=self.scanner)
            ro_grad.calculate(
                (-1)**i * self.k_bw[0].to('1/m'),
                receiver_bw=self.receiver_bw.to('Hz'),
                ro_samples=self.ro_samples,
                ofac=self.oversampling,
            )
            ro_gradients.append(ro_grad)

            # Calculate blip gradient
            if self.lines_per_shot > 1 and i < self.lines_per_shot - 1:
                ref = ro_gradients[-1].time + ro_gradients[-1].dur - 0.5 * blip_grad.dur
                blip_grad = Gradient(time=ref, scanner=self.scanner)
                blip_grad.calculate(-self.k_bw[1].to('1/m') / self.ph_samples)
                ph_gradients.append(blip_grad)

        if self.plot_seq:
            if MPI_rank == 0:
                # plt.rcParams['text.usetex'] = True
                plt.rcParams.update({'font.size': 16})

                fig, ax = plt.subplots(2, 1, figsize=(8, 4))

                # Phase encoding gradients
                for gr in ph_gradients:
                    ax1 = ax[1].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'r-', linewidth=2)

                # Readout gradients
                for gr in ro_gradients:
                    ax2 = ax[0].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'b-', linewidth=2)

                # Encoding gradients
                for gr in enc_gradients:
                    ax3 = ax[0].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'k--', linewidth=3, zorder=100)
                    ax4 = ax[1].plot(gr.timings.m_as('ms'), gr.amplitudes.m_as('mT/m'), 'k--', linewidth=3, zorder=100)

                # Add ADC readout
                for i in range(1, self.lines_per_shot + 1):
                    a = [
                        (ro_gradients[i].time + ro_gradients[i].slope).m_as('ms'),
                        (ro_gradients[i].time + ro_gradients[i].slope + ro_gradients[i].lenc).m_as('ms'),
                    ]
                    ax5 = ax[0].plot(a, [0, 0], 'm-', linewidth=4, zorder=101)

                # Set legend labels
                ax1[0].set_label('PH')
                ax2[0].set_label('RO')
                ax5[0].set_label('ADC')

                # Format plots
                for i in range(len(ax)):
                    ax[i].hlines(y=[0], xmin=0, xmax=[100], colors=['0.7'], linestyles='solid')
                    ax[i].tick_params('both', length=5, width=1, which='major', labelsize=16)
                    ax[i].tick_params('both', length=3, width=1, which='minor', labelsize=16)
                    ax[i].minorticks_on()
                    ax[i].set_ylabel('$G~\\mathrm{(mT/m)}$', fontsize=16)
                    ax[i].axis([
                        0,
                        ro_gradients[-1].time.m_as('ms') + ro_gradients[-1].dur.m_as('ms'),
                        -1.4 * self.Gr_max.m_as('mT/m'),
                        1.4 * self.Gr_max.m_as('mT/m'),
                    ])
                    ax[i].legend(fontsize=14, loc='upper right', ncol=4)
                ax[1].set_xlabel('$t~\\mathrm{(ms)}$', fontsize=16)
                plt.tight_layout()
                plt.show()

            # Synchronize all processes
            MPI_comm.Barrier()

        # Time needed to acquire one line
        # It depends on the k-space bandwidth, the gyromagnetic constant, and
        # the maximum gradient amplitude
        dt = np.linspace(0.0, ro_grad.lenc.m_as('ms'), self.ro_samples)

        # kspace locations
        if self.full_spoke:
            kx = np.linspace(self.kx_extent[0], self.kx_extent[1], self.ro_samples)
        else:
            kx = np.linspace(0, self.kx_extent[1].m_as('1/m'), self.ro_samples)

        ky = np.zeros(kx.shape)
        kz = np.linspace(self.kz_extent[0].m_as('1/m'), self.kz_extent[1].m_as('1/m'), self.slices)

        kspace = (
            np.zeros([self.ro_samples, self.ph_samples, self.slices], dtype=self.dtype),
            np.zeros([self.ro_samples, self.ph_samples, self.slices], dtype=self.dtype),
            np.zeros([self.ro_samples, self.ph_samples, self.slices], dtype=self.dtype),
        )

        # Build shots locations
        for ph in range(self.ph_samples):
            self.shots[ph // self.lines_per_shot][ph % self.lines_per_shot] = ph
            # self.shots[ph % self.nb_shots][ph // self.nb_shots] = ph

        if self.full_spoke:
            # Full-spoke golden-angle radial sampling
            GR = np.deg2rad(111.25)
            theta = np.array([np.mod(np.pi / GR * n, 2 * np.pi) for n in range(self.ph_samples)])
        else:
            # Half-spoke golden-angle radial sampling
            GR = 1.61803398875
            theta = np.array([np.mod((2 * np.pi - 2 * np.pi / GR) * n, 2 * np.pi) for n in range(self.ph_samples)])

        # kspace times and locations
        t = np.zeros([self.ro_samples, self.ph_samples, self.slices], dtype=self.dtype)
        for shot in self.shots:
            for idx, ph in enumerate(shot):

                # Readout direction
                ro = (-1)**idx

                # Fill locations
                kspace[0][::ro, ph, :] = np.tile(
                    kx[:, None] * np.cos(theta[ph]) + ky[:, None] * np.sin(theta[ph]),
                    [1, self.slices],
                )
                kspace[1][::ro, ph, :] = np.tile(
                    -kx[:, None] * np.sin(theta[ph]) + ky[:, None] * np.cos(theta[ph]),
                    [1, self.slices],
                )

                # Update timings
                if idx == 0:
                    t[::ro, ph, 0] = (
                        enc_time.m_as('ms') + ro_grad0.dur.m_as('ms')
                        + ro_grad.slope.m_as('ms') + dt
                    )
                else:
                    t[::ro, ph, 0] = (
                        t[:, shot[idx - 1]].max()
                        + ro_grad.slope.m_as('ms') + ro_grad.slope.m_as('ms')
                        + dt[::ro]
                    )

        # Fill kz coordinates
        for s in range(self.slices):
            kspace[2][:, :, s] = kz[s]
            t[:, :, s] = t[:, :, 0]

        # Calculate echo time
        if self.full_spoke:
            self.echo_time = (enc_time + ro_grad0.dur + 0.5 * self.lines_per_shot * ro_grad.dur).to('ms')
        else:
            self.echo_time = (enc_time + ro_grad0.dur + 0.5 * self.lines_per_shot * ro_grad.dur - 0.5 * ro_grad.dur).to('ms')

        return (kspace, Quantity(t, 'ms'))


class SpiralStack(Trajectory):
    """Stack-of-spirals k-space trajectory with hardware-enforced gradient limits.

    Generates Archimedean spiral readout trajectories in the
    :math:`k_x`–:math:`k_y` plane and stacks them along :math:`k_z`.
    Spiral parameters (number of turns, readout duration) are derived
    analytically from the scanner's gradient amplitude and slew-rate limits
    and the receiver bandwidth, rather than being set manually.

    The total readout time is determined by the ADC (receiver bandwidth)
    and gradient hardware limits. The ``oversampling`` parameter only
    affects spatial density in k-space, not acquisition duration.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to :class:`Trajectory`.
    """

    def __init__(self, *args,
                 density_exponent: float = 1.0,
                 safety_margin: float = 0.95,
                 **kwargs):
        """
        Initialize the SpiralStack trajectory.

        Parameters
        ----------
        density_exponent : float, optional
            Power-law exponent controlling radial density.
            p > 1 increases density near the periphery (default = 1.0).
        safety_margin : float, optional
            Fractional margin applied to gradient and slew limits (default = 0.95).
        """
        super().__init__(*args, **kwargs)

        self.ph_samples = self.check_ph_enc_lines(self.ph_samples)
        self.nb_shots = self.ph_samples // self.lines_per_shot

        self.min_samples_per_turn = 32
        self.interleaves = self.ph_samples
        self.density_exponent = float(density_exponent)
        self.safety_margin = float(safety_margin)
        (self.points, self.times) = self.kspace_points()

    def _base_spiral(self, ro_samples: int, k_max: Quantity, turns: float, p: float):
        """
        Generate a variable-density 2D spiral trajectory.

        kr(u)  = k_max * u**p
        phi(u) = 2*pi*turns * u**(1/p)
        """
        u = np.linspace(0.0, 1.0, ro_samples, dtype=self.dtype)
        kr = k_max.m_as('1/m') * (u ** p)
        phi = 2.0 * np.pi * turns * (u ** (1.0 / p))
        K = kr * np.exp(1j * phi)
        return u, K

    def _enforce_hardware_limits(self, u: np.ndarray, K: np.ndarray):
        """
        Enforce gradient amplitude and slew-rate constraints to compute
        the continuous time law t(u).

        Returns
        -------
        t_final : ndarray (s)
            Monotonic time samples corresponding to u.
        T_ro : float
            Total readout duration in seconds.
        """
        # Scanner limits
        gamma = self.gammabar.to('Hz/T').m * 2 * np.pi       # [rad/s/T]
        Gmax = self.Gr_max.to('T/m').m * self.safety_margin
        Smax = self.Gr_sr.to('T/m/s').m * self.safety_margin

        # Derivatives of k(u)
        du = np.gradient(u)
        dK_du = np.gradient(K, u, edge_order=2)
        d2K_du2 = np.gradient(dK_du, u, edge_order=2)

        # Magnitudes
        abs_dK_du = np.abs(dK_du)
        abs_d2K_du2 = np.abs(d2K_du2)

        # Time per unit-u from amplitude and slew constraints
        dt_du_amp = abs_dK_du / (gamma * Gmax)
        dt_du_slew = np.sqrt(np.maximum(abs_d2K_du2, 0.0) / (gamma * Smax))
        dt_du = np.maximum(dt_du_amp, dt_du_slew)

        # Integrate over u to obtain t(u)
        t_final = np.cumsum(0.5 * (dt_du + np.roll(dt_du, 1)) * du)
        t_final[0] = 0.0
        T_ro = float(t_final[-1])
        return t_final.astype(self.dtype), T_ro

    def kspace_points(self):
        """
        Compute the full 3D stack-of-spirals k-space trajectory using
        ADC-based timing (independent of oversampling).

        Returns
        -------
        points : tuple of ndarray
            kx, ky, kz arrays of shape [ro_samples, interleaves, slices].
        times : Quantity
            Time array of shape [ro_samples, interleaves, slices], in ms.
        """
        # k-space positioning gradients
        ro_grad0 = Gradient(time=Quantity(0.0, 'ms'), scanner=self.scanner)
        ro_grad0.calculate(
            -0.5 * self.k_bw[0].to('1/m')
            - 0.5 * ro_grad0.scanner.gammabar.to('1/mT/ms')
            * ro_grad0.strength.to('mT/m')
            * ro_grad0.slope.to('ms')
        )

        # k-space extent
        k_max = 0.5 * self.k_bw[0]

        # Determine number of turns (independent of oversampling)
        k_spa_base = (1.0 / self.FOV)[0]                        # base grid spacing
        turns_nominal = max(1.0, float((k_max / k_spa_base).m_as('')))
        max_turns_from_sampling = max(1.0, self.res[0] / float(self.min_samples_per_turn))
        turns = min(turns_nominal, max_turns_from_sampling)

        # Generate base 2D spiral
        u, K = self._base_spiral(self.res[0], k_max, turns, self.density_exponent)

        # Enforce gradient limits to get continuous time law
        t_sec_cont, T_ro = self._enforce_hardware_limits(u, K)

        # ADC-based sampling grid (fixed by receiver bandwidth)
        dt_adc = 1.0 / self.receiver_bw.m_as('Hz')              # [s]
        N_adc = int(np.round(T_ro / dt_adc))
        t_adc = np.linspace(0.0, T_ro, N_adc * self.oversampling, endpoint=True)

        # Interpolate spiral onto uniform ADC time base
        K_adc_real = np.interp(t_adc, t_sec_cont, np.real(K))
        K_adc_imag = np.interp(t_adc, t_sec_cont, np.imag(K))
        K_adc = K_adc_real + 1j * K_adc_imag

        # 3D stack dimensions
        # ro_samples counts the *oversampled* ADC samples actually
        # written into the kspace arrays below (N_adc * oversampling
        # rows). Keep it consistent with Trajectory.__init__'s
        # convention (where ro_samples already folds in oversampling)
        # so that external callers allocating K = zeros([ro_samples,
        # ph, sl, ...]) see the same readout extent that
        # mri_signal returns.
        self.ro_samples = N_adc * self.oversampling
        dt_ms = t_adc * 1e3                          # [ms]
        kz = np.linspace(
            self.kz_extent[0].m_as('1/m'),
            self.kz_extent[1].m_as('1/m'),
            self.slices,
        )

        # Allocate arrays using the fixed ro_samples length
        kspace = (
            np.zeros([N_adc * self.oversampling, self.ph_samples, self.slices], dtype=self.dtype),
            np.zeros([N_adc * self.oversampling, self.ph_samples, self.slices], dtype=self.dtype),
            np.zeros([N_adc * self.oversampling, self.ph_samples, self.slices], dtype=self.dtype),
        )
        t = np.zeros([N_adc * self.oversampling, self.ph_samples, self.slices], dtype=self.dtype)

        # Interleaf rotation angles
        theta = np.linspace(0.0, 2.0 * np.pi, self.interleaves, endpoint=False, dtype=self.dtype)
        enc_time = Quantity(self.t_start.m_as('ms') - ro_grad0.dur.m_as('ms'), 'ms')

        # Build shots locations and time maps
        for ph in range(self.interleaves):
            # Map shot structure identically to Radial/Cartesian
            self.shots[ph // self.lines_per_shot][ph % self.lines_per_shot] = ph

            # Rotate base spiral
            R = np.exp(1j * theta[ph])
            K_rot = K_adc * R
            kx_ = np.real(K_rot)
            ky_ = np.imag(K_rot)

            # Fill k-space locations and time
            kspace[0][:, ph, :] = np.tile(kx_[:, None], [1, self.slices])
            kspace[1][:, ph, :] = np.tile(ky_[:, None], [1, self.slices])
            t[:, ph, :] = (enc_time.m_as('ms') + ro_grad0.dur.m_as('ms') + dt_ms)[:, None]

        # Fill kz coordinates
        for s in range(self.slices):
            kspace[2][:, :, s] = kz[s]

        # Echo time
        self.echo_time = enc_time + Quantity(0.5 * T_ro * 1e3, 'ms')

        return (kspace, Quantity(t, 'ms'))