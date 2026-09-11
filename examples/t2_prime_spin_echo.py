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

# Dephasing has two parts. T2 is irreversible and gone for good. The rest comes
# from static field variation INSIDE a voxel: spins fan out because they sit at
# slightly different frequencies, and a refocusing pulse turns them around and
# brings them back. Together they give T2*.
#
# A single exponential cannot express the second part -- exp(-t/T2*) falls
# monotonically straight through the echo, which is the one place the signal is
# supposed to return. BlochSolver(t2_prime=...) instead gives every node a small
# ensemble of sub-spins at slightly different frequencies, which rephase on
# their own because that is what the physics does.
#
# This example plays 90 - tau - 180 - tau and compares three models: the scalar
# T2*, and the sub-ensemble under two lineshapes.
#
# The 'lorentzian' lineshape is the one that targets the conventional
# exp(-t/T2*), so before the 180 it lies almost on top of the scalar curve --
# and after it, unlike the scalar, it comes back. That is the whole gain, drawn
# in one figure. It is also the inaccurate option: exp(-t/T2*) is the Fourier
# transform of a Lorentzian, which has infinite variance, so no finite spin
# ensemble reproduces it. Even K = 512 is off by 2.6%.
#
# Its curve below is visibly RAGGED, and that is the rule, not a defect. The
# quantile rule puts a few bins far out in the tails -- the extreme offset grows
# from 20/T2' at K=32 to 326/T2' at K=512 -- and those very fast sub-spins leave
# a permanent ripple. Measured on a dense grid, the total upward wobble is 1.21
# at K=32 and 1.27 at K=512: raising K shrinks the ENVELOPE error but never the
# ripple. The gaussian rule is exactly monotone with zero wobble.

FAST_MODE = os.getenv("FEELMRI_FAST_TEST", "0") == "1"

T2_MS = 400.0        # irreversible: long, so what returns is unambiguous
T2_PRIME_MS = 15.0   # reversible: the intra-voxel spread
TAU_MS = 30.0        # the 180 is at tau, so the echo forms at 2 tau
N_SAMPLES = 8 if FAST_MODE else 20

# K = 32 for the gaussian rule rather than the default 16: a finite set of
# frequencies is quasi-periodic, so the decay revives past tau/T2' = 0.2*K and
# this run goes unrefocused to 2*tau = 4*T2'. The lorentzian rule needs far
# more and is still not exact -- over this range it is off by 29% at K=16,
# 6.2% at K=128 and 2.6% at K=512.
BINS = {'gaussian': 32, 'lorentzian': 32 if FAST_MODE else 128}

if __name__ == '__main__':

  script_path = Path(__file__).parent

  phantom = FEMPhantom(path=script_path/'phantoms/water_fat_P1_prism.xdmf',
                       scale_factor=0.01)
  phantom.set_assembler(voxel_size=1e3, lorder=1, horder=1,
                        nodal_approximation=True, lumped=True)
  scanner = Scanner()

  # 1. 90 -- tau -- 180 -- tau, storing the magnetization all the way through
  # so the whole envelope is visible and not just its endpoints.
  def build():
    def pulse(flip_deg):
      return RF(scanner=scanner, shape='hard',
                flip_angle=Q_(np.deg2rad(flip_deg), 'rad'), dur=Q_(0.002, 'ms'),
                ref=Q_(0.0, 'ms'), time=Q_(0.0, 'ms'))

    def rf_block(flip_deg):
      return SequenceBlock(rf_pulses=[pulse(flip_deg)], dur=Q_(0.002, 'ms'),
                           dt=Q_(1e-4, 'ms'), empty=False,
                           store_magnetization=True)

    seq = Sequence()
    seq.add_block(rf_block(90.0))
    step = TAU_MS / N_SAMPLES
    for half in range(2):
      if half == 1:
        seq.add_block(rf_block(180.0))
      for _ in range(N_SAMPLES):
        seq.add_block(SequenceBlock(dur=Q_(step, 'ms'), dt=Q_(step, 'ms'),
                                    empty=True, store_magnetization=True))
    return seq

  # 2. Two models of the same physics. Without the sub-ensemble the only way to
  # express T2' with a scalar is to fold it into T2 -- which is the T2* model.
  def run(lineshape):
    sub_ensemble = lineshape is not None
    t2 = T2_MS if sub_ensemble else 1.0/(1.0/T2_MS + 1.0/T2_PRIME_MS)
    extra = dict(t2_prime=Q_(T2_PRIME_MS, 'ms'), lineshape=lineshape,
                 spectral_bins=BINS[lineshape]) if sub_ensemble else {}
    Mxy, _ = BlochSolver(build(), phantom, T1=Q_(1e9, 'ms'), T2=Q_(t2, 'ms'),
                         initial_Mxy=0.0 + 0j, initial_Mz=1.0,
                         perfect_spoiling=False, dtype='float64',
                         scanner=scanner, **extra).solve()
    return np.abs(Mxy[0, :])

  gaussian = run('gaussian')
  lorentzian = run('lorentzian')
  scalar = run(None)

  # 3. Report. The echo can only recover the reversible part, so it must land
  # on exp(-2*tau/T2) whatever T2' was.
  step = TAU_MS / N_SAMPLES
  t = np.concatenate([np.arange(N_SAMPLES + 1) * step,
                      np.arange(N_SAMPLES + 1) * step + TAU_MS])
  floor = np.exp(-2*TAU_MS/T2_MS)
  MPI_print("T2 = {:.0f} ms, T2' = {:.0f} ms, echo at {:.0f} ms".format(
    T2_MS, T2_PRIME_MS, 2*TAU_MS))
  for label, mag in (('sub-ensemble, gaussian  ', gaussian),
                     ('sub-ensemble, lorentzian', lorentzian),
                     ('scalar T2* model        ', scalar)):
    MPI_print('  {}: {:.4f} at tau, {:.4f} at the echo'.format(
      label, mag[N_SAMPLES], mag[-1]))
  MPI_print('  the two ensembles reach exp(-2 tau/T2) = {:.4f}; the scalar '
            'model is monotone and never comes back'.format(floor))

  # How well the lorentzian ensemble reproduces the exponential it targets.
  # Before the 180 the scalar model IS exp(-t/T2*), so it doubles as the
  # reference; the gap is the price of using a finite ensemble for a Lorentzian.
  half = slice(0, N_SAMPLES + 1)
  MPI_print('  lorentzian at K = {}: worst departure from exp(-t/T2*) before '
            'the 180 is {:.3f}, and its decay is not monotone -- the ripple is '
            'inherent to the rule and does not shrink with K'.format(
              BINS['lorentzian'],
              float(np.abs(lorentzian[half] - scalar[half]).max())))

  # 4. Show it.
  if MPI_rank == 0:
    plt.figure(figsize=(7, 5))
    plt.plot(t, gaussian, 'o-', markersize=3,
             label="sub-ensemble, gaussian lineshape")
    plt.plot(t, lorentzian, '^-', markersize=3, color='tab:orange',
             label="sub-ensemble, lorentzian (targets exp(-t/T2*))")
    plt.plot(t, scalar, 's-', markersize=3, color='0.5',
             label='scalar T2* model, exp(-t/T2*)')
    plt.axhline(floor, color='k', linestyle=':', linewidth=1,
                label='exp(-2 tau / T2), the irreversible floor')
    plt.axvline(TAU_MS, color='k', linewidth=0.8, alpha=0.4)
    plt.annotate('180', xy=(TAU_MS, 0.02), xytext=(TAU_MS + 1.0, 0.02))
    plt.xlabel('time after the 90 (ms)')
    plt.ylabel('|Mxy|')
    plt.title("A spin echo recovers what T2' dephased; a scalar T2* cannot")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
