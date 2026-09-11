import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity as Q_

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.MPIUtilities import MPI_print, MPI_rank
from feelmri.MRObjects import RF, Scanner
from feelmri.Phantom import FEMPhantom

# Sub-voxel T2' by a spectral sub-ensemble.
#
# Dephasing has two parts. The IRREVERSIBLE part is T2 and is gone for good.
# The REVERSIBLE part comes from static field variation INSIDE a voxel: spins
# fan out because they sit at slightly different frequencies, and a refocusing
# pulse turns them around and brings them back. Together they give T2*.
#
# A single exponential cannot express the second part. exp(-t/T2*) decays
# monotonically from wherever it starts, so it keeps falling straight through
# the echo -- the one place the signal is supposed to come back. What is needed
# is a real sub-ensemble: every node carries K sub-spins at slightly different
# frequencies, and they rephase on their own because that is what the physics
# does.
#
# BlochSolver(t2_prime=...) does exactly that, and the ensemble PERSISTS across
# blocks, which is what lets a 180 played in one block undo dephasing that
# accumulated in another.
#
# The sub-spin frequencies are not sampled randomly. They are a quadrature rule
# for the intra-voxel lineshape, so the ensemble average converges on the
# lineshape's decay exponentially rather than as 1/sqrt(K): K = 16 is
# machine-precision for 'gaussian' and 'uniform'.
#
# K has to be sized against the longest UNREFOCUSED coherence, though. A finite
# set of frequencies is quasi-periodic, so it cannot stay cancelled forever and
# the decay revives once the phase spread wraps: the gaussian rule tracks its
# lineshape only out to tau/T2' = 0.2*K. Here the rim runs to 60 ms at
# T2' = 10 ms, i.e. 6 * T2', so K = 16 would show a spurious revival at ~55 ms
# and K = 32 (good to 7.86) is the right size.

# Enable fast mode for testing if the environment variable is set
FAST_MODE = os.getenv("FEELMRI_FAST_TEST", "0") == "1"

# Two tissues in one phantom: a well-shimmed centre and a rim near a
# susceptibility interface, where the intra-voxel field spread is four times
# worse. T2' is a per-node map, exactly like delta_B.
t2_prime_centre = 40.0
t2_prime_rim = 10.0

# Irreversible T2, the same everywhere. Deliberately long so that everything
# the echo recovers is unambiguously the REVERSIBLE part.
t2_ms = 400.0

# Echo time. The 180 is played at tau, so the echo forms at 2*tau.
tau_ms = 30.0
n_samples = 8 if FAST_MODE else 20
pulse_ms = 0.002

if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # 1. A wide, flat disc: 200 mm across and 50 mm thick.
  phantom = FEMPhantom(path=script_path/'phantoms/water_fat_P1_prism.xdmf',
                       scale_factor=0.01)
  phantom.set_assembler(voxel_size=1e3, lorder=1, horder=1,
                        nodal_approximation=True, lumped=True)

  # 2. The T2' map: good shim in the middle, poor at the rim.
  nodes = phantom.local_nodes.astype(np.float64)
  radius = np.hypot(nodes[:, 0], nodes[:, 1])
  u = (radius / np.abs(nodes[:, :2]).max())**2
  t2_prime = t2_prime_centre + (t2_prime_rim - t2_prime_centre) * u

  scanner = Scanner()

  # 3. 90 -- tau -- [180] -- tau, sampled all the way through so the whole
  # envelope is visible rather than just its endpoints.
  def hard_pulse(flip_deg):
    return RF(scanner=scanner, shape='hard',
              flip_angle=Q_(np.deg2rad(flip_deg), 'rad'),
              dur=Q_(pulse_ms, 'ms'), ref=Q_(0.0, 'ms'), time=Q_(0.0, 'ms'))

  def build(refocus):
    seq = Sequence()
    seq.add_block(SequenceBlock(rf_pulses=[hard_pulse(90.0)],
                                dur=Q_(pulse_ms, 'ms'), dt=Q_(1e-4, 'ms'),
                                empty=False, store_magnetization=True))
    step = tau_ms / n_samples
    for half in range(2):
      if half == 1 and refocus:
        seq.add_block(SequenceBlock(rf_pulses=[hard_pulse(180.0)],
                                    dur=Q_(pulse_ms, 'ms'), dt=Q_(1e-4, 'ms'),
                                    empty=False, store_magnetization=True))
      for _ in range(n_samples):
        seq.add_block(SequenceBlock(dur=Q_(step, 'ms'), dt=Q_(step, 'ms'),
                                    empty=True, store_magnetization=True))
    return seq

  def run(refocus, use_bins):
    # With the ensemble off, the only way to express T2' with a scalar is to
    # fold it into T2 -- which is exactly the model this example is about.
    t2 = t2_ms if use_bins else 1.0/(1.0/t2_ms + 1.0/t2_prime)  # (n,) per node
    extra = dict(t2_prime=Q_(t2_prime, 'ms'), spectral_bins=32,
                 lineshape='gaussian') if use_bins else {}
    solver = BlochSolver(build(refocus), phantom,
                         T1=Q_(1e9, 'ms'),
                         T2=Q_(t2, 'ms'),
                         initial_Mxy=0.0 + 0j,
                         initial_Mz=1.0,
                         perfect_spoiling=False,
                         dtype='float64',
                         scanner=scanner,
                         **extra)
    Mxy, _ = solver.solve()
    return np.abs(Mxy)

  fid = run(refocus=False, use_bins=True)
  echo = run(refocus=True, use_bins=True)
  scalar = run(refocus=True, use_bins=False)

  # 4. Report. Times are measured from the 90; the refocused runs carry one
  # extra stored column for the 180 block itself.
  step = tau_ms / n_samples
  t_fid = np.arange(fid.shape[1]) * step
  t_echo = np.concatenate([np.arange(n_samples + 1) * step,
                           np.arange(n_samples + 1) * step + tau_ms])

  inner = int(np.argmin(radius))
  outer = int(np.argmax(radius))
  MPI_print('T2\' map runs {:.0f} ms at the centre to {:.0f} ms at the rim; '
            'T2 is {:.0f} ms everywhere.'.format(
              t2_prime[inner], t2_prime[outer], t2_ms))
  for label, node in (('centre', inner), ('rim', outer)):
    MPI_print('  {:<7} FID at tau {:.4f} (closed form {:.4f})   '
              'echo at 2 tau {:.4f}   scalar model {:.4f}'.format(
                label, fid[node, n_samples],
                np.exp(-0.5*(tau_ms/t2_prime[node])**2) * np.exp(-tau_ms/t2_ms),
                echo[node, -1], scalar[node, -1]))
  MPI_print('The echo recovers to exp(-2 tau / T2) = {:.4f}: everything lost '
            'to T2\' comes back, everything lost to T2 does not.'.format(
              np.exp(-2*tau_ms/t2_ms)))

  # 5. Show it.
  if MPI_rank == 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for label, node, colour in (('centre, T2\' = {:.0f} ms'.format(t2_prime[inner]),
                                 inner, 'tab:blue'),
                                ('rim, T2\' = {:.0f} ms'.format(t2_prime[outer]),
                                 outer, 'tab:red')):
      ax.plot(t_fid, fid[node], 'o-', color=colour, markersize=3,
              label='{}: free induction'.format(label))
      ax.plot(t_fid, np.exp(-0.5*(t_fid/t2_prime[node])**2)
              * np.exp(-t_fid/t2_ms), '--', color=colour, linewidth=1,
              label='closed form')
    ax.set_xlabel('time after the 90 (ms)')
    ax.set_ylabel('|Mxy|')
    ax.set_title('Free induction: the ensemble reproduces its own lineshape')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t_echo, echo[inner], 'o-', color='tab:blue', markersize=3,
            label='sub-ensemble, centre')
    ax.plot(t_echo, echo[outer], 'o-', color='tab:red', markersize=3,
            label='sub-ensemble, rim')
    ax.plot(t_echo, scalar[outer], 's-', color='0.5', markersize=3,
            label='scalar T2* model, rim')
    ax.axhline(np.exp(-2*tau_ms/t2_ms), color='k', linestyle=':', linewidth=1,
               label='exp(-2 tau / T2), the irreversible floor')
    ax.axvline(tau_ms, color='k', linewidth=0.8, alpha=0.4)
    ax.annotate('180', xy=(tau_ms, 0.05), xytext=(tau_ms + 1.5, 0.05),
                fontsize=9)
    ax.set_xlabel('time after the 90 (ms)')
    ax.set_ylabel('|Mxy|')
    ax.set_title('Spin echo: the reversible part comes back, T2 does not')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper center')

    fig.tight_layout()
    plt.show()
