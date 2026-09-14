import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from pint import Quantity as Q_
from scipy.interpolate import griddata

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock, lineshape_bins
from feelmri.MPIUtilities import MPI_comm, MPI_print, MPI_rank
from feelmri.MRObjects import RF, Scanner
from feelmri.Phantom import FEMPhantom

# Sub-voxel T2' -- everything BlochSolver(t2_prime=...) does, in four panels.
#
# Transverse decay has two parts. T2 is irreversible and gone for good. The rest
# comes from static field variation INSIDE a voxel: spins fan out because they
# sit at slightly different frequencies. A refocusing pulse turns them around
# and brings that part back. Together they give T2*.
#
# A single exponential cannot express the second part -- exp(-t/T2*) falls
# monotonically straight through the echo, the one place the signal is supposed
# to return. `t2_prime` instead gives every node K sub-spins at slightly
# different frequencies, weighted by a quadrature rule for the intra-voxel
# lineshape, and keeps that ensemble alive ACROSS blocks so a 180 played in one
# block undoes dephasing accumulated in another.
#
# What this does NOT do: the ensemble is collapsed before the k-space assembler,
# so decay DURING a readout is still the assembler's own exp(-t/T2). This buys
# correct echo formation between blocks, not intra-readout lineshape. Pass T2 --
# not T2* -- to Phantom.set_static_fields when the solver models T2', or the two
# double-count.

FAST_MODE = os.getenv("FEELMRI_FAST_TEST", "0") == "1"

T2_MS = 400.0            # irreversible; long, so what returns is unambiguous
T2_PRIME_CENTRE = 40.0   # reversible, well-shimmed middle of the object
T2_PRIME_RIM = 10.0      # reversible, near a susceptibility interface
TAU_MS = 30.0            # the 180 sits at tau, so the echo forms at 2 tau
N_SAMPLES = 8 if FAST_MODE else 20

# K is sized against the longest UNREFOCUSED coherence. A finite set of
# frequencies is quasi-periodic, so the decay revives past tau/T2' = 0.2*K --
# panel 3 shows exactly that. The lorentzian rule needs far more bins and is
# still not exact, which is panel 1.
K_GOOD = 32
K_POOR = 8
K_LORENTZIAN = 32 if FAST_MODE else 128

if __name__ == '__main__':

  script_path = Path(__file__).parent

  # 1. A 200 mm disc, with a T2' MAP: t2_prime takes one value per node, like
  # delta_B. Good shim in the middle, poor at the rim.
  phantom = FEMPhantom(path=script_path/'phantoms/water_fat_P1_prism.xdmf',
                       scale_factor=0.01)
  phantom.set_assembler(voxel_size=1e3, lorder=1, horder=1,
                        nodal_approximation=True, lumped=True)
  scanner = Scanner()

  nodes = phantom.local_nodes.astype(np.float64)
  radius = np.hypot(nodes[:, 0], nodes[:, 1])
  # GLOBAL, not per-rank. `local_nodes` is this rank's slice, so a rank-local
  # maximum makes the T2' map a function of how the mesh was cut. At 8 ranks a
  # rank holding only the outer annulus has a small local max, `u` exceeds 1,
  # and T2' comes out NEGATIVE -- measured -1.266 ms, which the solver refuses.
  r_max = MPI_comm.allreduce(float(np.abs(nodes[:, :2]).max()), op=MPI.MAX)
  u = (radius / r_max)**2
  t2_prime = T2_PRIME_CENTRE + (T2_PRIME_RIM - T2_PRIME_CENTRE) * u

  # Every number reported below is read at the node nearest the axis or the
  # node furthest from it. Both are picked GLOBALLY: an argmin over one rank's
  # slice names a different node on each of them, which moved a reported
  # inversion by 17% at 8 ranks in the sibling example.
  def _extreme(key, take_max):
    i = int(np.argmax(key) if take_max else np.argmin(key))
    op = MPI.MAXLOC if take_max else MPI.MINLOC
    _, owner = MPI_comm.allreduce((float(key[i]), MPI_rank), op=op)
    return i, owner

  def _at(values, where):
    """That node's row, broadcast so every rank reports the same number."""
    i, owner = where
    return MPI_comm.bcast(values[i] if MPI_rank == owner else None, root=owner)

  centre, rim = _extreme(radius, False), _extreme(radius, True)

  # 2. 90 -- tau -- [180] -- tau, storing all the way through so the whole
  # envelope is visible and not just its endpoints.
  def build(refocus, n_halves=2):
    def rf_block(flip_deg):
      rf = RF(scanner=scanner, shape='hard',
              flip_angle=Q_(np.deg2rad(flip_deg), 'rad'), dur=Q_(0.002, 'ms'),
              ref=Q_(0.0, 'ms'), time=Q_(0.0, 'ms'))
      return SequenceBlock(rf_pulses=[rf], dur=Q_(0.002, 'ms'),
                           dt=Q_(1e-4, 'ms'), empty=False,
                           store_magnetization=True)

    seq = Sequence()
    seq.add_block(rf_block(90.0))
    step = TAU_MS / N_SAMPLES
    for half in range(n_halves):
      if half == 1 and refocus:
        seq.add_block(rf_block(180.0))
      for _ in range(N_SAMPLES):
        seq.add_block(SequenceBlock(dur=Q_(step, 'ms'), dt=Q_(step, 'ms'),
                                    empty=True, store_magnetization=True))
    return seq

  def run(seq, lineshape, bins=K_GOOD):
    """lineshape=None is the scalar T2* model: the only way to express T2'
    without an ensemble is to fold it into T2."""
    if lineshape is None:
      t2, extra = 1.0/(1.0/T2_MS + 1.0/t2_prime), {}
    else:
      t2 = T2_MS
      extra = dict(t2_prime=Q_(t2_prime, 'ms'), lineshape=lineshape,
                   spectral_bins=bins)
    Mxy, Mz = BlochSolver(seq, phantom, T1=Q_(1e9, 'ms'), T2=Q_(t2, 'ms'),
                          initial_Mxy=0.0 + 0j, initial_Mz=1.0,
                          perfect_spoiling=False, dtype='float64',
                          scanner=scanner, **extra).solve()
    return np.abs(Mxy)

  fid = build(refocus=False)
  echo = build(refocus=True)
  step = TAU_MS / N_SAMPLES
  t_fid = np.arange(N_SAMPLES * 2 + 1) * step
  t_echo = np.concatenate([np.arange(N_SAMPLES + 1) * step,
                           np.arange(N_SAMPLES + 1) * step + TAU_MS])

  # Closed forms each lineshape must reproduce, as a function of t/T2'.
  SHAPES = {
    'gaussian': lambda r: np.exp(-0.5 * r**2),
    'uniform': lambda r: np.abs(np.sinc(np.sqrt(3.0) * r / np.pi)),
    'lorentzian': lambda r: np.exp(-r),
  }

  fids = {s: run(fid, s, K_LORENTZIAN if s == 'lorentzian' else K_GOOD)
          for s in SHAPES}
  echoes = {s: run(echo, s, K_LORENTZIAN if s == 'lorentzian' else K_GOOD)
            for s in ('gaussian', 'lorentzian')}
  echoes['scalar'] = run(echo, None)
  revival = {K: run(fid, 'gaussian', K) for K in (K_POOR, K_GOOD)}

  # 3. Report. The two nodes' values are collected here, on every rank: the
  # figure below is drawn on rank 0 only, and a broadcast inside that branch
  # would leave the others waiting.
  t2c, t2r = float(_at(t2_prime, centre)), float(_at(t2_prime, rim))
  fids_c = {k: _at(v, centre) for k, v in fids.items()}
  ech_c = {k: _at(v, centre) for k, v in echoes.items()}
  ech_r = {k: _at(v, rim) for k, v in echoes.items()}
  rev_r = {k: _at(v, rim) for k, v in revival.items()}

  floor = np.exp(-2 * TAU_MS / T2_MS)
  MPI_print("T2 = {:.0f} ms everywhere; T2' runs {:.0f} ms at the centre to "
            "{:.0f} ms at the rim.".format(T2_MS, t2c,
                                           t2r))
  MPI_print('Free induction at the centre, against each lineshape\'s closed '
            'form:')
  for s, f in SHAPES.items():
    want = f(t_fid / t2c) * np.exp(-t_fid / T2_MS)
    MPI_print('  {:<11} K={:<4} worst departure {:.4f}'.format(
      s, K_LORENTZIAN if s == 'lorentzian' else K_GOOD,
      float(np.abs(fids_c[s] - want).max())))
  MPI_print('Spin echo (the echo can only recover the REVERSIBLE part, so it '
            'must land on exp(-2 tau/T2) = {:.4f}):'.format(floor))
  for label in ('gaussian', 'lorentzian', 'scalar'):
    MPI_print('  {:<11} centre {:.4f} -> {:.4f}   rim {:.4f} -> {:.4f}'.format(
      label, ech_c[label][N_SAMPLES], ech_c[label][-1],
      ech_r[label][N_SAMPLES], ech_r[label][-1]))
  MPI_print('  the rim loses 99% of its signal and gets all of it back; the '
            'scalar model is monotone and never does.')

  # The weights are a probability distribution summing to exactly 1. If they
  # were not, the collapsed equilibrium would be M0 * sum(w) and the whole
  # phantom would relax to the wrong level.
  recov = BlochSolver(
    Sequence(), phantom, M0=1.0, T1=Q_(100.0, 'ms'), T2=Q_(1e9, 'ms'),
    initial_Mxy=0.0 + 0j, initial_Mz=0.0, perfect_spoiling=False,
    dtype='float64', t2_prime=Q_(t2_prime, 'ms'), spectral_bins=K_GOOD)
  recov.sequence.add_block(SequenceBlock(dur=Q_(500.0, 'ms'), dt=Q_(10.0, 'ms'),
                                         empty=True, store_magnetization=True))
  _, Mz = recov.solve()
  mz_c = _at(Mz, centre)
  MPI_print('Weight normalisation: Mz after 5 T1 is {:.6f}, closed form '
            '{:.6f}'.format(float(mz_c[0]), 1.0 - np.exp(-5.0)))

  # Asserted, not just printed: an example that only reports its own error
  # exits 0 however wrong the physics has become.
  for s_name, f in SHAPES.items():
    want = f(t_fid / t2c) * np.exp(-t_fid / T2_MS)
    worst = float(np.abs(fids_c[s_name] - want).max())
    # The lorentzian rule cannot reach the exponential it targets -- that is
    # the point of showing it -- so it gets the bound its own rule predicts.
    limit = 0.25 if s_name == 'lorentzian' else 2e-3
    assert worst < limit, f'{s_name} FID departs from its closed form by {worst:.3f}'
  for label in ('gaussian', 'lorentzian'):
    assert abs(ech_r[label][-1] - floor) < 5e-3, (
      f'{label}: the echo should recover to exp(-2 tau/T2) = {floor:.4f}, got '
      f'{ech_r[label][-1]:.4f}')
  assert ech_r['scalar'][-1] < 0.1 * floor, (
    'the scalar T2* control should NOT recover at the echo')
  assert np.all(np.diff(ech_r['scalar']) <= 1e-9), (
    'the scalar T2* control must be monotone')
  assert abs(float(mz_c[0]) - (1.0 - np.exp(-5.0))) < 1e-6, (
    'the quadrature weights no longer sum to 1; the phantom relaxes to the '
    'wrong M0')

  # 4. What Stage 1 refuses, and why. Each guard names a measured cost rather
  # than being silently slow or wrong.
  MPI_print('Refused combinations:')
  for label, kwargs in (
      ('a per-node T1/T2 (kernel would take its per-node exp() path)',
       dict(T2=Q_(np.linspace(40.0, 60.0, nodes.shape[0]), 'ms'))),
      ('spectral_bins=1 (one bin is not an ensemble)',
       dict(T2=Q_(50.0, 'ms'), spectral_bins=1))):
    try:
      BlochSolver(fid, phantom, T1=Q_(1e9, 'ms'), initial_Mz=1.0,
                  perfect_spoiling=False, t2_prime=Q_(T2_PRIME_CENTRE, 'ms'),
                  **kwargs)
      MPI_print('  {:<62} accepted'.format(label))
    except (NotImplementedError, ValueError):
      MPI_print('  {:<62} refused'.format(label))

  # 5. Show it.
  if MPI_rank == 0:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    ax = axes[0, 0]
    for s in SHAPES:
      line, = ax.plot(t_fid, fids_c[s], 'o-', markersize=3, label=s)
      ax.plot(t_fid, SHAPES[s](t_fid / t2c)
              * np.exp(-t_fid / T2_MS), '--', linewidth=1,
              color=line.get_color(), alpha=0.6)
    ax.set_title('1. The lineshape sets the DECAY SHAPE\n'
                 '(dashed: closed form; lorentzian ripples at any K)')
    ax.set_xlabel('time after the 90 (ms)')
    ax.set_ylabel('|Mxy| at the centre')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for label, style, colour in (('gaussian', 'o-', 'tab:blue'),
                                 ('lorentzian', '^-', 'tab:orange'),
                                 ('scalar', 's-', '0.5')):
      ax.plot(t_echo, ech_r[label], style, markersize=3, color=colour,
              label='{}, rim'.format(label))
    ax.axhline(floor, color='k', linestyle=':', linewidth=1,
               label='exp(-2 tau / T2), the irreversible floor')
    ax.axvline(TAU_MS, color='k', linewidth=0.8, alpha=0.4)
    ax.annotate('180', xy=(TAU_MS, 0.02), xytext=(TAU_MS + 1.0, 0.02))
    ax.set_title('2. A 180 recovers ALL of it, whatever the lineshape\n'
                 '(a scalar T2* cannot: it is monotone)')
    ax.set_xlabel('time after the 90 (ms)')
    ax.set_ylabel('|Mxy| at the rim')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for K, style in ((K_POOR, '^-'), (K_GOOD, 'o-')):
      ax.plot(t_fid, rev_r[K], style, markersize=3,
              label='gaussian, K = {}'.format(K))
    ax.plot(t_fid, SHAPES['gaussian'](t_fid / t2r)
            * np.exp(-t_fid / T2_MS), 'k--', linewidth=1, label='closed form')
    ax.axvline(0.2 * K_POOR * t2r, color='tab:red', linewidth=0.8)
    ax.annotate('0.2*K*T2\' for K = {}'.format(K_POOR),
                xy=(0.2 * K_POOR * t2r, 0.5),
                xytext=(0.2 * K_POOR * t2r + 1.5, 0.5), fontsize=8,
                color='tab:red')
    ax.set_title('3. Size K, or the decay REVIVES\n'
                 "(a finite bin set is quasi-periodic)")
    ax.set_xlabel('time after the 90 (ms)')
    ax.set_ylabel('|Mxy| at the rim')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    grid = np.linspace(-np.abs(nodes[:, :2]).max(),
                       np.abs(nodes[:, :2]).max(), 200)
    gx, gy = np.meshgrid(grid, grid)
    image = griddata(nodes[:, :2], fids['gaussian'][:, N_SAMPLES], (gx, gy),
                     method='linear')
    mesh = ax.pcolormesh(gx, gy, image, cmap='inferno', shading='auto')
    fig.colorbar(mesh, ax=ax, label='|Mxy| at t = {:.0f} ms'.format(TAU_MS))
    ax.set_title("4. t2_prime is a per-node MAP\n"
                 "(T2' {:.0f} ms centre, {:.0f} ms rim)".format(
                   t2c, t2r))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')

    fig.tight_layout()
    plt.show()
