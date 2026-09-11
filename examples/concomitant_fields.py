import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from scipy.interpolate import griddata
from pint import Quantity as Q_

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.MPIUtilities import MPI_comm, MPI_print, MPI_rank
from feelmri.MRObjects import Gradient, Scanner
from feelmri.Phantom import FEMPhantom

# Concomitant (Maxwell) fields. Maxwell's equations forbid a gradient that is
# purely longitudinal, so every imaging gradient drags along transverse
# components. To lowest order they add
#
#   Bc = [(Gx^2 + Gy^2) z^2 + (Gz^2/4)(x^2 + y^2)
#         - Gx*Gz*x*z - Gy*Gz*y*z] / (2*B0)
#
# to Bz. It is QUADRATIC in position, so unlike the imaging gradient it cannot
# be undone by the encoding, and it is QUADRATIC in gradient amplitude, so
# reversing a lobe does not reverse it. It also scales as 1/B0, which is why it
# is a low-field problem.
#
# This example isolates it with a BIPOLAR pair on Gz: the two lobes have equal
# and opposite area, so the linear term G.x refocuses exactly and every radian
# left on the magnetization is concomitant. The same construction is what makes
# concomitant fields a real nuisance for phase contrast, where the velocity
# encoding IS a bipolar pair.

# SCOPE: this is a SOLVER term, and the readout is not solved. The k-space
# signal is synthesized by the assembler from the trajectory, exp(-t/T2) and a
# static off-resonance phi_dB0, none of which carries Bc -- so concomitant
# phase accumulated DURING an ADC window is NOT modelled. What IS carried is
# everything the solver integrates up to the magnetization snapshot: the prep,
# the slice select, the velocity-encoding lobes.
#
# The size of what is missed follows from which term survives an in-plane
# readout gradient. For Gz = 0 the expression collapses to (Gx^2 + Gy^2)z^2 /
# (2*B0), which is ZERO in a slice at isocentre and grows quadratically with
# slice offset. Integrated over the 103 ms echo train of tests/data/epi_v142.seq
# at 1.5 T: 0.000 rad at z = 0, -5.50 rad at z = 50 mm, -22.0 rad at z = 100 mm.

# Enable fast mode for testing if the environment variable is set
FAST_MODE = os.getenv("FEELMRI_FAST_TEST", "0") == "1"

# One lobe of the bipolar pair: a trapezoid of amplitude g_amp (mT/m) with
# g_rise ms ramps and a g_flat ms plateau. 30 mT/m over 0.2 ms is 150 mT/m/ms,
# inside the default 180 slew limit.
g_amp = 30.0
g_rise = 0.20
g_flat = 2.0

# Bc goes as 1/B0, so the comparison is between a clinical field and a
# low-field system.
field_strengths = [1.5, 0.55]

if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # 1. A wide, flat disc: 200 mm across and 50 mm thick. The term that survives
  # a pure Gz is (Gz^2/4)(x^2 + y^2), so what matters is how far the object
  # reaches in plane, not how thick it is.
  phantom = FEMPhantom(path=script_path/'phantoms/water_fat_P1_prism.xdmf',
                       scale_factor=0.01)
  phantom.set_assembler(voxel_size=1e3, lorder=1, horder=1,
                        nodal_approximation=True, lumped=True)

  # 2. The bipolar pair, written out as corner points: up, plateau, down,
  # then the mirror image. The amplitudes are user-defined rather than built
  # by make_bipolar because the point here is the waveform, not a VENC.
  timings = np.array([0.0,
                      g_rise,
                      g_rise + g_flat,
                      2*g_rise + g_flat,
                      3*g_rise + g_flat,
                      3*g_rise + 2*g_flat,
                      4*g_rise + 2*g_flat])
  amplitudes = np.array([0.0, g_amp, g_amp, 0.0, -g_amp, -g_amp, 0.0])
  duration = Q_(timings[-1], 'ms')

  scanner = Scanner()
  gradient = Gradient(timings=Q_(timings, 'ms'),
                      amplitudes=Q_(amplitudes, 'mT/m'),
                      scanner=scanner,
                      ref=Q_(0.0, 'ms'),
                      time=Q_(0.0, 'ms'),
                      axis=2)

  # The first moment is zero, which is what makes the control run meaningful.
  MPI_print('Net gradient area: {:.2e} mT/m*ms'.format(
    float(np.trapezoid(amplitudes, timings))))

  # 3. Evolve a fully transverse magnetization through the pair. Relaxation is
  # switched off so the phase is the only thing that moves, and spoiling is off
  # so nothing else touches Mxy.
  def phase_after_bipolar(concomitant, field_strength=1.5):
    seq = Sequence()
    seq.add_block(SequenceBlock(gradients=[gradient],
                                dur=duration,
                                dt=Q_(0.05 if FAST_MODE else 0.01, 'ms'),
                                empty=False,
                                store_magnetization=True))
    solver = BlochSolver(seq, phantom,
                         T1=Q_(1e9, 'ms'),
                         T2=Q_(1e9, 'ms'),
                         initial_Mxy=1.0 + 0j,
                         initial_Mz=0.0,
                         perfect_spoiling=False,
                         dtype='float64',
                         scanner=Scanner(field_strength=Q_(field_strength, 'T')),
                         concomitant_fields=concomitant)
    Mxy, _ = solver.solve()
    return np.angle(Mxy[:, 0])

  MPI_print('Note: Bc is carried through the blocks the solver integrates. '
            'It is NOT applied during an ADC window -- see the scope note at '
            'the top of this file.')

  control = phase_after_bipolar(concomitant=False)
  measured = {B0: phase_after_bipolar(True, B0) for B0 in field_strengths}

  # 4. The closed form. With Gx = Gy = 0 only the (Gz^2/4)(x^2 + y^2) term
  # survives, so the accumulated phase is
  #
  #   phi = -gamma * (r_perp^2 / 4) / (2*B0) * integral(Gz^2 dt)
  #
  # The SECOND moment is what enters, and a ramp contributes G^2*rise/3, not
  # the G^2*rise/2 a trapezoidal rule over the corner list would give -- G^2 is
  # quadratic in time along a ramp. On this waveform the two differ by 3.1%,
  # which is far larger than the solver's own error.
  second_moment = 2 * g_amp**2 * (g_flat + 2*g_rise/3.0)
  gamma = scanner.gamma.m_as('rad/ms/mT')
  nodes = phantom.local_nodes.astype(np.float64)
  radius = np.hypot(nodes[:, 0], nodes[:, 1])

  def analytical_phase(field_strength):
    B0 = Q_(field_strength, 'T').m_as('mT')
    return -gamma * 0.25 * radius**2 * second_moment / (2*B0)

  # 5. Report. The control is the proof that the bipolar pair really does
  # refocus the linear term; the deviations are measured as a distance between
  # unit phasors, so they are immune to 2*pi wrapping.
  worst_control = MPI_comm.allreduce(float(np.abs(control).max()), op=MPI.MAX)
  MPI_print('Concomitant OFF: worst residual phase {:.2e} rad '
            '(the bipolar pair refocuses the linear term)'.format(worst_control))

  # Asserted, not just printed: an example that only reports its own error
  # exits 0 however wrong the physics has become.
  assert worst_control < 1e-9, (
    f'the bipolar pair should refocus the linear term to ~0; got '
    f'{worst_control:.2e} rad')

  for B0 in field_strengths:
    reference = analytical_phase(B0)
    deviation = np.abs(np.exp(1j*measured[B0]) - np.exp(1j*reference))
    peak = MPI_comm.allreduce(float(np.abs(reference).max()), op=MPI.MAX)
    worst = MPI_comm.allreduce(float(deviation.max()), op=MPI.MAX)
    MPI_print('B0 = {:>4} T: peak concomitant phase {:5.3f} rad at the rim, '
              'worst deviation from the closed form {:.2e}'.format(B0, peak, worst))
    # Proportional to the phase, because that is what the error is: a
    # coarser raster makes a RELATIVE error on the accumulated phase, so a
    # fixed bound would pass at 1.5 T and fail at 0.55 T purely because the
    # phase there is 2.7x larger.
    #
    # Measured at 7.8e-5 per radian at both field strengths, and the same on
    # the coarsened FEELMRI_FAST_TEST raster: the solver sub-samples gradient
    # ramps itself when the concomitant term is on, because Bc goes as G^2 and
    # is therefore quadratic along a ramp, where no trapezoidal rule is exact.
    # Before it did, this read 1.95e-3 per radian at dt = 0.05 ms -- 25x worse,
    # and within 1.5x of the bound below.
    assert worst < 3e-4 * peak, (
      f'B0 = {B0} T: concomitant phase departs from the closed form by '
      f'{worst:.2e}, i.e. {worst / peak:.2e} per radian of a {peak:.3f} rad '
      f'phase')

  # 6. Show the phase across the slab. The nodes are an unstructured cloud, so
  # they are resampled onto a regular grid for display and drawn as a map with
  # iso-phase contours -- the bull's-eye those contours form is the signature
  # of a term quadratic in position. Under MPI each rank holds a subset of the
  # nodes, so this draws rank 0's own share.
  if MPI_rank == 0:
    edge = float(np.abs(nodes[:, :2]).max())
    grid = np.linspace(-edge, edge, 240)
    gx, gy = np.meshgrid(grid, grid)

    def on_grid(values):
      return griddata(nodes[:, :2], values, (gx, gy), method='linear')

    span = float(np.abs(analytical_phase(min(field_strengths))).max())
    levels = [lvl for lvl in np.arange(-np.ceil(span*4)/4, span + 0.25, 0.25)
              if abs(lvl) > 1e-9]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    panels = [('Concomitant fields off', control),
              *[('B0 = {} T'.format(B0), measured[B0]) for B0 in field_strengths]]
    for ax, (title, phase) in zip(axes.flat, panels):
      image = on_grid(phase)
      mesh = ax.pcolormesh(gx, gy, image, cmap='RdBu_r', vmin=-span, vmax=span,
                           shading='auto')
      # Iso-phase contours, but only where there is a field to contour: the
      # control panel sits at 1e-13 rad and contouring it draws pure noise.
      if np.nanmax(np.abs(image)) > 0.05:
        contours = ax.contour(gx, gy, image, levels=levels, colors='k',
                              linewidths=0.5, alpha=0.6)
        ax.clabel(contours, fmt='%.2f', fontsize=7)
      ax.set_title(title)
      ax.set_xlabel('x (m)')
      ax.set_ylabel('y (m)')
      ax.set_aspect('equal')
      fig.colorbar(mesh, ax=ax, label='phase (rad)')

    # Radial profile. The phase is a function of radius alone, so binning the
    # nodes by radius collapses the cloud onto a curve that can be read against
    # the closed form.
    ax = axes.flat[3]
    bins = np.linspace(0.0, radius.max(), 40)
    centres = 0.5*(bins[:-1] + bins[1:])
    index = np.clip(np.digitize(radius, bins) - 1, 0, len(centres) - 1)
    for B0 in field_strengths:
      binned = np.array([measured[B0][index == b].mean() if np.any(index == b)
                         else np.nan for b in range(len(centres))])
      line, = ax.plot(centres, binned, 'o', markersize=4,
                      label='B0 = {} T, simulated'.format(B0))
      order = np.argsort(radius)
      ax.plot(radius[order], analytical_phase(B0)[order], '-', linewidth=1.5,
              color=line.get_color(), alpha=0.7,
              label='B0 = {} T, closed form'.format(B0))
    ax.set_xlabel('in-plane radius (m)')
    ax.set_ylabel('phase (rad)')
    ax.set_title('Radially binned simulation against the closed form')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)

    fig.tight_layout()
    plt.show()
