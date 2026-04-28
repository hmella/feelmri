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

from feelmri.BlochSimulator import solve_mri
from feelmri.Motion import POD
from feelmri.MPIUtilities import MPI_comm, MPI_print, MPI_rank
from feelmri.MRObjects import Scanner
from feelmri.Phantom import FEMPhantom


class ADC:
    """Analog-to-digital converter timing specification.

    Parameters
    ----------
    times : np.ndarray
        1-D array of absolute ADC sampling times inside the parent
        :class:`SequenceBlock` (ms).
    """

    def __init__(self, times: np.ndarray):
        self.times = Quantity(times, 'ms')


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
        Time step for gradient discretization (ms). Negative disables.
        Default is -1 ms (disabled).
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
    """

    def __init__(self, gradients: list = [],
                 rf_pulses: list = [],
                 adc: ADC | None = None,
                 dt_rf: Quantity = Quantity(0.01, 'ms'),
                 dt_gr: Quantity = Quantity(-1, 'ms'),
                 dt: Quantity = Quantity(10, 'ms'),
                 dur: Quantity = Quantity(-1, 'ms'),
                 empty: bool = False,
                 store_magnetization: bool = False):
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
        self._spoiler = False

    def copy(self):
        return copy.deepcopy(self)

    def __call__(self, t):
        rf = np.sum([rf(t) for rf in self.rf_pulses], axis=0)
        m_gr = np.sum([g(t) for g in self.M_gradients], axis=0)
        p_gr = np.sum([g(t) for g in self.P_gradients], axis=0)
        s_gr = np.sum([g(t) for g in self.S_gradients], axis=0)
        if self.adc is not None:
            adc_mask = np.isclose(t, self.adc.times.m_as('ms'), rtol=1e-9, atol=1e-12)
        else:
            adc_mask = np.zeros_like(t, dtype=bool)

        return rf, (m_gr, p_gr, s_gr), adc_mask

    def __repr__(self):
        return f"Sequence(gradients={self.gradients}, rf_pulses={self.rf_pulses}, dt_rf={self.dt_rf}, dt_gr={self.dt_gr})"

    def __str__(self):
        return f"Sequence with {len(self.gradients)} gradients and {len(self.rf_pulses)} RF pulses."

    def __len__(self):
        return len(self.gradients) + len(self.rf_pulses)

    def _get_extent(self):
        # Get (t_min, t_max) for each gradient
        if self.gradients:
            time_extent_gr = Quantity(
                np.array([(g.time.m, (g.time + g.dur).m) for g in self.gradients], dtype=np.float32),
                units=self.gradients[0].timings.u,
            )
        else:
            time_extent_gr = Quantity(np.array([(0, 0)], dtype=np.float32), units='ms')

        # Get (t_min, t_max) for each RF pulse
        if self.rf_pulses:
            time_extent_rf = Quantity(
                np.array([((rf.time - rf.ref).m, (rf.time - rf.ref + rf.dur).m) for rf in self.rf_pulses], dtype=np.float32),
                units=self.rf_pulses[0].ref.u,
            )
        else:
            time_extent_rf = Quantity(np.array([(0, 0)], dtype=np.float32), units='ms')

        # Time extent
        t_min = np.min([time_extent_gr.m_as('ms').min(axis=0), time_extent_rf.m_as('ms').min(axis=0)])
        t_max = np.max([time_extent_gr.m_as('ms').max(axis=0), time_extent_rf.m_as('ms').max(axis=0)])
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
                gr_timings = np.concatenate(
                    [np.arange(g.timings[0].m, g.timings[-1].m, self.dt_gr.m) for g in self.gradients]
                    + [gr_timings]
                )
        else:
            gr_timings = np.array([])

        # Get RF timings while considering the dt_rf
        if self.rf_pulses:
            rf_timings = np.concatenate([[(rf.time - rf.ref).m, (rf.time - rf.ref + rf.dur).m] for rf in self.rf_pulses])
            if self.dt_rf > 0:
                rf_timings = np.concatenate(
                    [np.arange((rf.time - rf.ref).m, (rf.time - rf.ref + rf.dur).m, self.dt_rf.m) for rf in self.rf_pulses]
                    + [rf_timings]
                )
        else:
            rf_timings = np.array([])

        # Sequence timings
        seq_timings = np.arange(self.time_extent[0].m, self.time_extent[1].m, self.dt.m)

        # ADC timings
        if self.adc is not None:
            adc_times = self.adc.times.m_as('ms')
        else:
            adc_times = np.array([])

        # Concatenate all timings, sort them and remove duplicates
        all_timings = np.concatenate((gr_timings, rf_timings, seq_timings, adc_times))
        all_timings = np.unique(np.sort(all_timings))

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

    def __init__(self, blocks: list = []):
        self.blocks = blocks
        self.Nb_blocks = len(self.blocks)
        self.time_extent = self._get_extent()
        self.dur = self.time_extent[1] - self.time_extent[0]
        self.non_empty = [~block.empty for block in self.blocks if block is not None]

    def add_block(self, block: SequenceBlock | Quantity, dt: Quantity = Quantity(10, 'ms')):
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
            if block > Quantity(0, 'ms'):
                block = SequenceBlock(dur=block.to('ms'), dt=dt, empty=True, store_magnetization=False)
                block.change_time(self.time_extent[-1].to('ms'))
                self.blocks = [b for b in self.blocks + [block]]
                self.Nb_blocks = len(self.blocks)
                self.time_extent = self._get_extent()
                self.dur = self.time_extent[1] - self.time_extent[0]
                self.non_empty.append(not block.empty)
        else:
            warnings.warn("Only SequenceBlock or Quantity instances can be added to the sequence.")

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
    M0 : np.ndarray or float, optional
        Equilibrium magnetization (nodal array or scalar). Default is 1.0.
    T1 : Quantity, optional
        Longitudinal relaxation time (ms). Default is 1000 ms.
    T2 : Quantity, optional
        Transverse relaxation time (ms). Default is 100 ms.
    delta_B : np.ndarray or float, optional
        Static B0 inhomogeneity field (nodal or scalar, in mT).
        Default is 0.0.
    pod_trajectory : POD or None, optional
        Motion trajectory for moving-phantom simulations. Default is None.
    initial_Mxy : np.ndarray or float, optional
        Initial transverse magnetization (complex, nodal or scalar).
        Default is 0.0.
    """

    def __init__(self, sequence: Sequence,
                 phantom: FEMPhantom,
                 scanner: Scanner = Scanner(),
                 M0: np.ndarray | float = 1.0,
                 T1: Quantity = Quantity(1000.0, 'ms'),
                 T2: Quantity = Quantity(100.0, 'ms'),
                 delta_B: np.ndarray | float = 0.0,
                 pod_trajectory: POD | None = None,
                 initial_Mxy: np.ndarray | float = 0.0,
                 initial_Mz: np.ndarray | float = None,
                 perfect_spoiling: bool = True):
        ones = np.ones((phantom.local_nodes.shape[0], 1), dtype=np.float32)
        self.sequence = sequence
        self.scanner = scanner
        self.phantom = phantom
        self.M0 = M0
        self.T1 = Quantity(T1.m * ones, T1.units)
        self.T2 = Quantity(T2.m * ones, T2.units)
        self.delta_B = delta_B * ones
        self.initial_Mxy = initial_Mxy * ones.astype(np.complex64)
        self.initial_Mz = initial_Mz * ones if initial_Mz is not None else M0 * ones
        self.pod_trajectory = pod_trajectory
        self.perfect_spoiling = perfect_spoiling

    def solve(self, start: int = 0, end: int = None):
        # Current machine time
        t0 = time.perf_counter()

        # Phantom position
        x = self.phantom.local_nodes

        # Blocks to be solved
        if start < 0:
            start += self.sequence.Nb_blocks
        if end is None:
            end = self.sequence.Nb_blocks
        blocks = self.sequence.blocks[start:end]
        MPI_print(f"[BlochSolver] Solving sequence blocks {start} to {end-1} ({len(blocks)} blocks).")

        # Dimensions
        nb_nodes  = x.shape[0]
        nb_blocks = len(blocks)

        # List of indices indicating which blocks need to be stored
        store_indices = [i for i, block in enumerate(blocks) if block.store_magnetization]

        # Allocate magnetizations
        Mxy = np.zeros((nb_nodes, nb_blocks), dtype=np.complex64)
        Mz  = np.zeros((nb_nodes, nb_blocks), dtype=np.float32)

        # Strip units of Bloch parameters just once
        T1 = self.T1.m_as('ms')
        T2 = self.T2.m_as('ms')

        # Gyromagnetic constant
        gamma = self.scanner.gamma.m_as('rad/ms/mT')

        # Solve the Bloch equations for each block
        for i, block in enumerate(blocks):

            # Update POD trajectory time shift
            if self.pod_trajectory is not None:
                self.pod_trajectory.update_timeshift(block.time_extent[0].m_as('ms'))

            # Discrete time points and time intervals
            discrete_times = block.discrete_times.m_as('ms')
            dt = np.diff(discrete_times, prepend=0)

            # Precompute RF and gradients
            rf_pulses = np.zeros((discrete_times.shape[0], 1), dtype=np.complex64)
            gradients = np.zeros((discrete_times.shape[0], 3), dtype=np.float32)
            rf, G, adc_mask = block(discrete_times)
            rf_pulses[:, 0] = rf
            gradients[:, 0] = G[0]
            gradients[:, 1] = G[1]
            gradients[:, 2] = G[2]

            # Indicator array
            regime_idx = np.abs(rf_pulses) != 0.0

            # Solve
            if block._spoiler is True:
                # Build multi-isochromats quickly via vectorization
                K = 25
                elem_size = self.phantom.global_elem_size.min()
                (x_big, T1_big, T2_big,
                 deltaB_big, Mxy_big, Mz_big) = create_multi_isochromats(
                    x, T1, T2,
                    self.delta_B,
                    self.initial_Mxy,
                    self.initial_Mz,
                    K=K,
                    pos_jitter=elem_size
                )

                # Solve for the expanded mesh
                Mxy_hist, Mz_hist = solve_mri(
                    x_big, T1_big, T2_big, deltaB_big, self.M0, gamma,
                    rf_pulses, gradients, dt, regime_idx, Mxy_big, Mz_big,
                    self.pod_trajectory
                )

                # Collapse back to the FE resolution using only the final time step
                Mxy_, Mz_ = collapse_isochromats(
                    Mxy_hist[:, -1],
                    Mz_hist[:, -1],
                    K=K,
                    mode="mean"
                )

                # Reshape to keep consistency with solve_mri's usual 2D return format
                # so the downstream assignment Mxy[:, i] = Mxy_[:, -1] still works.
                Mxy_ = Mxy_.reshape(-1, 1)
                Mz_ = Mz_.reshape(-1, 1)

            else:
                Mxy_, Mz_ = solve_mri(
                    x, T1, T2, self.delta_B, self.M0, gamma,
                    rf_pulses, gradients, dt, regime_idx,
                    self.initial_Mxy, self.initial_Mz,
                    self.pod_trajectory
                )

            # Update magnetizations
            Mxy[:, i] = Mxy_[:, -1]
            Mz[:, i]  = Mz_[:, -1]

            # Update the initial magnetization for the next block
            if block.empty is True:
                self.initial_Mxy[:, 0] = Mxy_[:, -1]
            else:
                # TODO: verify if there is a better way to know beforehand if the sequence will contain spoilers
                if self.perfect_spoiling is True:
                    # This is done because gradient or RF spoiling cannot be applied on coarse meshes.
                    # Therefore, we need to artificially spoil the magnetization.
                    self.initial_Mxy[:, 0] = 0.0
                else:
                    self.initial_Mxy[:, 0] = Mxy_[:, -1]

            self.initial_Mz[:, 0] = Mz_[:, -1]

        # Print elapsed time
        MPI_print('[BlochSolver] Elapsed time for solving the sequence: {:.2f} s'.format(time.perf_counter() - t0))

        # Synchronize all processes
        MPI_comm.Barrier()

        return Mxy[:, store_indices], Mz[:, store_indices]


def create_multi_isochromats(x, T1, T2, delta_B, Mxy0, Mz0, K=100, pos_jitter=0.2e-3):
    # np.repeat duplicates each row K times sequentially
    x_big      = np.repeat(x, K, axis=0)
    T1_big     = np.repeat(T1, K, axis=0)
    T2_big     = np.repeat(T2, K, axis=0)
    deltaB_big = np.repeat(delta_B, K, axis=0)
    Mxy_big    = np.repeat(Mxy0, K, axis=0)
    Mz_big     = np.repeat(Mz0, K, axis=0)

    # Generate jitter for all sub-isochromats at once and add to positions
    N, dim = x.shape
    radius = pos_jitter * 0.5

    # Ensure the jitter matches the precision of the input mesh (e.g., float32)
    jitter = (radius * np.random.randn(N * K, dim)).astype(x.dtype)
    x_big += jitter

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
