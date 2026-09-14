import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from pint import Quantity as Q_
from scipy.interpolate import griddata

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.MPIUtilities import MPI_comm, MPI_print, MPI_rank
from feelmri.MRObjects import RF, Scanner
from feelmri.Phantom import FEMPhantom

# Transmit (B1+) inhomogeneity. The transmit coil does not deliver the same
# field everywhere, so the flip angle a spin actually receives is the nominal
# one scaled by the local sensitivity: alpha(x) = |b1(x)| * alpha_nominal. A
# complex b1 additionally rotates the transverse axis the pulse tips onto.
#
# The consequence is NOT simply "more b1, more signal". After a single
# excitation the transverse magnetization is sin(alpha), which peaks at 90 deg
# and falls again beyond it -- so with a map that overshoots at the centre, the
# brightest signal sits on a RING where |b1| = 1, not where the transmit field
# is strongest. This example shows that ring, and the matching failure of a
# nominal 180 deg inversion everywhere off that ring.

# Enable fast mode for testing if the environment variable is set
FAST_MODE = os.getenv("FEELMRI_FAST_TEST", "0") == "1"

# Transmit map, as a function of in-plane radius normalised to the object:
# over-flipping at the centre, under-flipping at the rim, with a small
# quadratic transmit phase on top.
b1_centre = 1.30
b1_rim = 0.60
phase_rim = 0.80

# A hard pulse, so the delivered flip is gamma * B1 * dur with nothing else in
# it, and NO gradient. Measuring a flip with a slice-select lobe on samples the
# slice profile instead and reads far below nominal -- that is physics, not a
# defect, but it is not what this example is about.
pulse_duration = 0.5

if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # 1. A wide, flat disc: 200 mm across and 50 mm thick.
  phantom = FEMPhantom(path=script_path/'phantoms/water_fat_P1_prism.xdmf',
                       scale_factor=0.01)
  phantom.set_assembler(voxel_size=1e3, lorder=1, horder=1,
                        nodal_approximation=True, lumped=True)

  # 2. Build the map on the LOCAL nodes -- b1_map is indexed by local node, the
  # same convention delta_B uses, so under MPI each rank supplies its own share.
  nodes = phantom.local_nodes.astype(np.float64)
  radius = np.hypot(nodes[:, 0], nodes[:, 1])
  r_max = MPI_comm.allreduce(float(radius.max()), op=MPI.MAX)
  u = (radius / r_max)**2
  b1_map = (b1_centre + (b1_rim - b1_centre)*u) * np.exp(1j*phase_rim*u)

  scanner = Scanner()

  # 3. One nominal flip angle, delivered through the map.
  def excite(nominal_deg, b1):
    rf = RF(scanner=scanner, shape='hard',
            flip_angle=Q_(np.deg2rad(nominal_deg), 'rad'),
            dur=Q_(pulse_duration, 'ms'),
            ref=Q_(0.0, 'ms'), time=Q_(0.0, 'ms'))
    seq = Sequence()
    seq.add_block(SequenceBlock(rf_pulses=[rf],
                                dur=Q_(pulse_duration, 'ms'),
                                dt=Q_(0.005 if FAST_MODE else 0.001, 'ms'),
                                empty=False,
                                store_magnetization=True))
    solver = BlochSolver(seq, phantom,
                         T1=Q_(1e9, 'ms'),
                         T2=Q_(1e9, 'ms'),
                         initial_Mxy=0.0 + 0j,
                         initial_Mz=1.0,
                         perfect_spoiling=False,
                         dtype='float64',
                         scanner=scanner,
                         b1_map=b1)
    Mxy, Mz = solver.solve()
    return Mxy[:, 0], Mz[:, 0]

  Mxy_90, Mz_90 = excite(90.0, b1_map)
  _, Mz_180 = excite(180.0, b1_map)
  Mxy_ref, _ = excite(90.0, None)          # the nominal field, as a control

  # 4. Report. The delivered flip is read from the magnetization itself rather
  # than assumed, and compared against nominal x |b1|, which is exact for a
  # hard pulse.
  delivered = np.arctan2(np.abs(Mxy_90), Mz_90)
  expected = np.deg2rad(90.0) * np.abs(b1_map)
  worst = MPI_comm.allreduce(float(np.abs(delivered - expected).max()), op=MPI.MAX)
  MPI_print('Delivered flip vs nominal x |b1|: worst deviation {:.2e} rad'.format(worst))

  worst_ref = MPI_comm.allreduce(float(np.abs(np.abs(Mxy_ref) - 1.0).max()), op=MPI.MAX)
  MPI_print('Control (b1_map=None): |Mxy| departs from 1 by {:.2e}'.format(worst_ref))

  # Asserted, not just printed. The delivered flip is exact for a hard pulse,
  # so these tolerances are the raster's, not the physics'.
  assert worst < 1e-6, f'delivered flip departs from nominal x |b1| by {worst:.2e}'
  assert worst_ref < 1e-6, f'the b1_map=None control is off by {worst_ref:.2e}'

  # The transmit phase lands on the transverse magnetization.
  phase_error = np.abs(np.exp(1j*np.angle(Mxy_90))
                       - np.exp(1j*(np.angle(Mxy_ref) + np.angle(b1_map))))
  worst_phase = MPI_comm.allreduce(float(phase_error.max()), op=MPI.MAX)
  MPI_print('Transmit phase transferred to Mxy: worst deviation {:.2e}'.format(
    worst_phase))
  assert worst_phase < 1e-9, (
    f'arg(b1) should land on Mxy exactly; off by {worst_phase:.2e}')

  # The signal peak is NOT where the transmit field peaks. The node is picked
  # GLOBALLY: `radius` and `Mxy_90` are this rank's slice, so an argmax over
  # them names a different node on every rank -- which moved the reported
  # centre inversion below from -0.588 to -0.691 at 8 ranks.
  def _extreme(key, take_max):
    i = int(np.argmax(key) if take_max else np.argmin(key))
    op = MPI.MAXLOC if take_max else MPI.MINLOC
    _, owner = MPI_comm.allreduce((float(key[i]), MPI_rank), op=op)
    return i, owner

  def _at(values, where):
    i, owner = where
    return MPI_comm.bcast(values[i] if MPI_rank == owner else None, root=owner)

  peak_signal_r = float(_at(radius, _extreme(np.abs(Mxy_90), True)))
  MPI_print('|b1| runs {:.2f} at the centre to {:.2f} at the rim, so the flip '
            'runs {:.1f} to {:.1f} deg'.format(
              b1_centre, b1_rim, 90*b1_centre, 90*b1_rim))
  MPI_print('Brightest signal sits at r = {:.3f} m of {:.3f} m, where |b1| = 1 '
            'and the flip is 90 deg -- not at the centre'.format(
              peak_signal_r, r_max))
  # The point of the example: the signal peak is NOT the transmit peak.
  ring = r_max * np.sqrt((b1_centre - 1.0) / (b1_centre - b1_rim))
  assert abs(peak_signal_r - ring) < 0.15 * r_max, (
    f'the brightest signal should sit on the |b1| = 1 ring at r = {ring:.3f} m, '
    f'not at r = {peak_signal_r:.3f} m')
  MPI_print('Nominal 180 deg inversion: Mz runs {:+.3f} at the centre to '
            '{:+.3f} at the rim; only the |b1| = 1 ring inverts fully'.format(
              float(_at(Mz_180, _extreme(radius, False))),
              float(_at(Mz_180, _extreme(radius, True)))))

  # 5. Show it. Under MPI each rank holds a subset of the nodes, so this draws
  # rank 0's own share.
  if MPI_rank == 0:
    edge = float(np.abs(nodes[:, :2]).max())
    grid = np.linspace(-edge, edge, 240)
    gx, gy = np.meshgrid(grid, grid)

    def on_grid(values):
      return griddata(nodes[:, :2], values, (gx, gy), method='linear')

    # The delivered-flip map is deliberately NOT drawn: it is |b1| x 90 and
    # would be the first panel again in different units. The radial profile
    # below carries that information quantitatively instead.
    panels = [('Transmit map |b1|', np.abs(b1_map), 'viridis'),
              ('|Mxy| after a nominal 90 deg', np.abs(Mxy_90), 'inferno'),
              ('Mz after a nominal 180 deg', Mz_180, 'inferno')]

    # Where |b1| = 1: the excitation is on nominal here and nowhere else.
    ring = r_max * np.sqrt((b1_centre - 1.0) / (b1_centre - b1_rim))
    angle = np.linspace(0, 2*np.pi, 200)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax, (title, values, cmap) in zip(axes.flat, panels):
      image = on_grid(values)
      # Autoscaled to the data. A fixed 0-1 scale on |Mxy| would compress a
      # 0.81-1.00 range into the top fifth of the colour map and hide the ring,
      # which is the whole point of the panel.
      mesh = ax.pcolormesh(gx, gy, image, cmap=cmap, shading='auto')
      ax.set_title(title)
      ax.set_xlabel('x (m)')
      ax.set_ylabel('y (m)')
      ax.set_aspect('equal')
      fig.colorbar(mesh, ax=ax)
      # The |b1| = 1 ring on every panel: the contour of the transmit map that
      # the two magnetization panels turn into a bright and a fully inverted
      # ring respectively.
      ax.plot(ring*np.cos(angle), ring*np.sin(angle), 'w--', linewidth=1.2)

    # Radial profile. Binned into shells, since the phase is a function of
    # radius alone and the node cloud is unstructured.
    ax = axes.flat[3]
    bins = np.linspace(0.0, radius.max(), 30)
    centres = 0.5*(bins[:-1] + bins[1:])
    index = np.clip(np.digitize(radius, bins) - 1, 0, len(centres) - 1)

    def binned(values):
      return np.array([values[index == b].mean() if np.any(index == b) else np.nan
                       for b in range(len(centres))])

    ax.plot(centres, binned(np.abs(b1_map)), label='|b1|')
    ax.plot(centres, binned(np.abs(Mxy_90)), label='|Mxy|, nominal 90 deg')
    ax.plot(centres, binned(Mz_180), label='Mz, nominal 180 deg')
    ax.axvline(ring, color='k', linestyle='--', linewidth=1)
    ax.axhline(1.0, color='0.7', linewidth=0.8)
    ax.annotate('|b1| = 1', xy=(ring, -0.55), xytext=(ring + 0.004, -0.55),
                fontsize=9)
    ax.set_xlabel('in-plane radius (m)')
    ax.set_ylabel('magnetization / sensitivity')
    ax.set_title('Signal peaks where |b1| = 1, not where |b1| peaks')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)

    fig.tight_layout()
    plt.show()
