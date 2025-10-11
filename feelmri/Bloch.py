import copy
import time
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity as Q_

from feelmri.BlochSimulator import solve_mri
from feelmri.Motion import POD
from feelmri.MPIUtilities import MPI_print, MPI_rank, MPI_comm
from feelmri.MRObjects import Scanner
from feelmri.Phantom import FEMPhantom


class SequenceBlock:
  def __init__(self, gradients=[], rf_pulses=[], dt_rf=Q_(0.01, 'ms'), dt_gr=Q_(-1, 'ms'), dt=Q_(10, 'ms'), dur=Q_(-1, 'ms'), empty=False, store_magnetization=False):
    self.gradients = gradients
    self.M_gradients = [g for g in self.gradients if g.axis == 0]
    self.P_gradients = [g for g in self.gradients if g.axis == 1]
    self.S_gradients = [g for g in self.gradients if g.axis == 2]
    self.rf_pulses = rf_pulses
    self.dt_rf = dt_rf
    self.dt_gr = dt_gr
    self.dt = dt
    self.dur = dur
    self.time_extent = self._get_extent()    
    self.discrete_times = self._discretization()
    self.Nb_times = len(self.discrete_times)
    self.empty = empty
    self.store_magnetization = store_magnetization

  def copy(self):
    return copy.deepcopy(self)

  def __call__(self, t):
    rf = np.sum([rf(t) for rf in self.rf_pulses], axis=0)
    m_gr = np.sum([g(t) for g in self.M_gradients], axis=0)
    p_gr = np.sum([g(t) for g in self.P_gradients], axis=0)
    s_gr = np.sum([g(t) for g in self.S_gradients], axis=0)
    return rf, (m_gr, p_gr, s_gr)

  def __repr__(self):
    return f"Sequence(gradients={self.gradients}, rf_pulses={self.rf_pulses}, dt_rf={self.dt_rf}, dt_gr={self.dt_gr})"

  def __str__(self):
    return f"Sequence with {len(self.gradients)} gradients and {len(self.rf_pulses)} RF pulses."

  def __len__(self):
    return len(self.gradients) + len(self.rf_pulses)

  def _get_extent(self):
    # Get time extent depending on gradient or RF timings
    # Get (t_min, t_max) for each gradient
    if self.gradients:
      time_extent_gr = Q_(np.array([(g.time.m, (g.time+g.dur).m) for g in self.gradients], dtype=np.float32), units=self.gradients[0].timings.u)
    else:
      time_extent_gr = Q_(np.array([(0, 0)], dtype=np.float32), units='ms')

    # Get (t_min, t_max) for each rf pulse
    if self.rf_pulses:
      time_extent_rf = Q_(np.array([((rf.time-rf.ref).m, (rf.time-rf.ref+rf.dur).m) for rf in self.rf_pulses], dtype=np.float32), units=self.rf_pulses[0].ref.u)
    else:
      time_extent_rf = Q_(np.array([(0, 0)], dtype=np.float32), units='ms')

    # Time extent
    t_min = np.min([time_extent_gr.m_as('ms').min(axis=0), time_extent_rf.m_as('ms').min(axis=0)])
    t_max = np.max([time_extent_gr.m_as('ms').max(axis=0), time_extent_rf.m_as('ms').max(axis=0)])
    if (t_max - t_min) < self.dur.m_as('ms'):
      t_max += self.dur.m_as('ms') - (t_max - t_min)

    # Update duration if dur is negative
    if self.dur.m_as('ms') < 0:
      self.dur = Q_(t_max - t_min, 'ms')

    return [Q_(t_min, 'ms'), Q_(t_max, 'ms')]

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
      steps = int(np.ceil((end - start)/self.dt_rf.m))
      t  = np.linspace(start, end, steps)
      rf_d.append((t, rf(t)))

    return rf_d, M_d_gr, P_d_gr, S_d_gr
  
  def _discretization(self):
    # Get gradient timings while considering the dt_gr
    if self.gradients:
        gr_timings = np.concatenate([g.timings.m for g in self.gradients])
        if self.dt_gr > 0:
            gr_timings = np.concatenate([np.arange(g.timings[0].m, g.timings[-1].m, self.dt_gr.m) for g in self.gradients] + [gr_timings])
    else:
        gr_timings = np.array([])

    # Get RF timings while considering the dt_rf
    if self.rf_pulses:
        rf_timings = np.concatenate([[(rf.time-rf.ref).m, (rf.time-rf.ref+rf.dur).m] for rf in self.rf_pulses])
        if self.dt_rf > 0:
            rf_timings = np.concatenate([np.arange((rf.time-rf.ref).m, (rf.time-rf.ref+rf.dur).m, self.dt_rf.m) for rf in self.rf_pulses] + [rf_timings])
    else:
        rf_timings = np.array([])

    # Sequence timings
    seq_timings = np.arange(self.time_extent[0].m, self.time_extent[1].m, self.dt.m)

    # Concatenate all timings, sort them and remove duplicates.
    all_timings = np.concatenate((gr_timings, rf_timings, seq_timings))
    all_timings = np.unique(np.sort(all_timings))

    return Q_(all_timings, units='ms')
  
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
  def __init__(self, blocks=[]):
    self.blocks = blocks
    self.Nb_blocks = len(self.blocks)
    self.time_extent = self._get_extent()
    self.dur = self.time_extent[1] - self.time_extent[0]
    self.non_empty = [~block.empty for block in self.blocks if block is not None]

  def add_block(self, block: SequenceBlock | Q_, dt: Q_ = Q_(10, 'ms')):
    # Add a block to the sequence
    if isinstance(block, SequenceBlock):
      block = block.copy()  # Ensure we work with a copy
      block.change_time(self.time_extent[-1].to('ms') - block.time_extent[0].to('ms'))
      self.blocks = [b for b in self.blocks + [block]]
      self.Nb_blocks = len(self.blocks)
      self.time_extent = self._get_extent()
      self.dur = self.time_extent[1] - self.time_extent[0]
      self.non_empty.append(not block.empty)
    elif isinstance(block, Q_):
      # If a duration is provided, create a new block with that duration
      if block > Q_(0, 'ms'):
        block = SequenceBlock(dur=block.to('ms'), dt=dt, empty=True, store_magnetization=False)
        block.change_time(self.time_extent[-1].to('ms'))
        self.blocks = [b for b in self.blocks + [block]]
        self.Nb_blocks = len(self.blocks)
        self.time_extent = self._get_extent()
        self.dur = self.time_extent[1] - self.time_extent[0]
        self.non_empty.append(not block.empty)
    else:
      warnings.warn("Only SequenceBlock or Quantity instances can be added to the sequence.")

  def update_block_references(self):
    # Update reference time for each block
    for i, block in enumerate(self.blocks):
      shift = block.time_extent[-1].to('ms') + i * self.dt_blocks.to('ms') + self.dt_prep.to('ms')
      block.change_time(shift)

  def _get_extent(self):
    # Get time extent depending on gradient or RF timings
    # Get (t_min, t_max) for each gradient
    time_extent_b = np.array([(b.time_extent[0].m, b.time_extent[1].m) for b in self.blocks if b is not None])

    # Time extent
    if time_extent_b.size == 0:
      # If no blocks, return zero extent
      t_min = 0.0
      t_max = 0.0
    else:
      t_min = np.min([time_extent_b.min(axis=0)])
      t_max = np.max([time_extent_b.max(axis=0)])

    return (Q_(t_min, 'ms'), Q_(t_max, 'ms'))
  
  def plot(self, blocks=None, tight_layout=True, figsize=None, export_to=None):
    if MPI_rank == 0:
      # Plot RF pulses and MR gradients
      titles = ['RF', 'M', 'P', 'S']
      if blocks is None: # Plot all
        discrete_blocks = [block._discrete_objects() for block in self.blocks]
        extents = [block.time_extent for block in self.blocks]
      else:            # Plot selected blocks
        discrete_blocks = [block._discrete_objects() for block in self.blocks[blocks]]
        extents = [block.time_extent for block in self.blocks[blocks]]


      fig, ax = plt.subplots(4, 1, figsize=figsize)
      for i, objects in enumerate(discrete_blocks):
        for j, obj in enumerate(objects):
          if titles[j] == 'RF':
            for t, amp in obj:
              ax[j].plot(t, np.real(amp), label='Real', color='b')
              ax[j].plot(t, np.imag(amp), label='Imaginary', color='r')
          else:
            for t, amp in obj:
              ax[j].plot(t, amp, color='b')
          ax[j].set_ylabel(titles[j])

        # Add vertical lines for block extents
        [ax[k].axvline(extents[i][0].m, color=mcolors.CSS4_COLORS['pink'], linestyle=':') for k in range(4)]
        [ax[k].axvline(extents[i][1].m, color=mcolors.CSS4_COLORS['pink'], linestyle='--') for k in range(4)]

      # Add horizontal lines at zero
      [ax[k].axhline(0, color=mcolors.CSS4_COLORS['gray'], linestyle='--') for k in range(4)]

      # Set x- and y-limits
      [ax[k].set_xlim([extents[0][0].m, extents[-1][1].m]) for k in range(4)]

      ax[0].legend(['Real', 'Imaginary'], loc='upper right')
      ax[-1].set_xlabel('Time (ms)')
      if tight_layout:
        plt.tight_layout()
      if export_to is not None:
        plt.savefig(export_to, bbox_inches='tight')
      plt.show()

    # Synchronize all processes
    MPI_comm.Barrier()


class BlochSolver:
  def __init__(self, sequence: Sequence, 
               phantom: FEMPhantom, 
               scanner: Scanner = Scanner(), 
               M0: np.ndarray | float = 1.0, 
               T1: Q_ = Q_(1000.0, 'ms'), 
               T2: Q_ = Q_(100.0, 'ms'), 
               delta_B: np.ndarray | float = 0.0,
               pod_trajectory: POD | None = None,
               initial_Mxy: np.ndarray | float = 0.0,
               initial_Mz: np.ndarray | float = None):
    ones = np.ones((phantom.local_nodes.shape[0], 1), dtype=np.float32)
    self.sequence = sequence
    self.scanner = scanner
    self.phantom = phantom
    self.M0 = M0
    self.T1 = Q_(T1.m * ones, T1.units)
    self.T2 = Q_(T2.m * ones, T2.units)
    self.delta_B = delta_B * ones
    self.initial_Mxy = initial_Mxy * ones.astype(np.complex64)
    self.initial_Mz = initial_Mz * ones if initial_Mz is not None else M0 * ones
    self.pod_trajectory = pod_trajectory

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

    # Stripe units of Bloch parameters just once
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
      discrete_times = block._discretization().m_as('ms')
      dt = np.diff(discrete_times, prepend=0)

      # Precompute RF and gradients
      rf_pulses = np.zeros((discrete_times.shape[0], 1), dtype=np.complex64)
      gradients = np.zeros((discrete_times.shape[0], 3), dtype=np.float32)
      rf, G = block(discrete_times)
      rf_pulses[:, 0] = rf
      gradients[:, 0] = G[0]
      gradients[:, 1] = G[1]
      gradients[:, 2] = G[2]

      # Indicator array
      regime_idx = np.abs(rf_pulses) != 0.0

      # Solve
      Mxy_, Mz_ = solve_mri(x, T1, T2, self.delta_B, self.M0, gamma, rf_pulses, gradients, dt, regime_idx, self.initial_Mxy, self.initial_Mz, self.pod_trajectory)

      # Update magnetizations
      Mxy[:, i] = Mxy_[:, -1]
      Mz[:, i]  = Mz_[:, -1]

      # Update the initial magnetization for the next block
      # TODO: I'm not sure if this is correct, it should be checked.
      if block.empty is True:
        self.initial_Mxy[:,0] = Mxy_[:, -1]
      else:
        # This is done because gradient or RF spoiling cannot be applied on coarse meshes. Therefore, we need to artificially spoil the magnetization.
        self.initial_Mxy[:,0] = 0*Mxy_[:, -1]
      self.initial_Mz[:,0] = Mz_[:, -1]

    # Print elapsed time
    MPI_print('[BlochSolver] Elapsed time for solving the sequence: {:.2f} s'.format(time.perf_counter() - t0))

    # # Reset POD trajectory time shift
    # if self.pod_trajectory is not None:
    #   self.pod_trajectory.update_timeshift(0.0)

    # Synchronize all processes
    MPI_comm.Barrier()

    return Mxy[:, store_indices], Mz[:, store_indices]