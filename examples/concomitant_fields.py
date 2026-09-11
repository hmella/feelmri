import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
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

  # 1. A wide, flat slab: 200 x 200 x 50 mm. The term that survives a pure Gz
  # is (Gz^2/4)(x^2 + y^2), so what matters is how far the object reaches in
  # plane, not how thick it is.
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

  for B0 in field_strengths:
    reference = analytical_phase(B0)
    deviation = np.abs(np.exp(1j*measured[B0]) - np.exp(1j*reference))
    peak = MPI_comm.allreduce(float(np.abs(reference).max()), op=MPI.MAX)
    worst = MPI_comm.allreduce(float(deviation.max()), op=MPI.MAX)
    MPI_print('B0 = {:>4} T: peak concomitant phase {:5.3f} rad at the corner, '
              'worst deviation from the closed form {:.2e}'.format(B0, peak, worst))

  # 6. Show the phase across the slab. Under MPI each rank holds a subset of
  # the nodes, so this draws rank 0's own share.
  if MPI_rank == 0:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    panels = [('Concomitant fields off', control),
              *[('B0 = {} T'.format(B0), measured[B0]) for B0 in field_strengths]]
    for ax, (title, phase) in zip(axes.flat, panels):
      sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=phase, s=1,
                      cmap='twilight', vmin=-np.pi, vmax=np.pi)
      ax.set_title(title)
      ax.set_xlabel('x (m)')
      ax.set_ylabel('y (m)')
      ax.set_aspect('equal')
      fig.colorbar(sc, ax=ax, label='phase (rad)')

    ax = axes.flat[3]
    order = np.argsort(radius)
    for B0 in field_strengths:
      line, = ax.plot(radius[order], measured[B0][order], '.', markersize=1,
                      label='B0 = {} T, simulated'.format(B0))
      ax.plot(radius[order], analytical_phase(B0)[order], '-', linewidth=1,
              color=line.get_color(), alpha=0.6)
    ax.set_xlabel('in-plane radius (m)')
    ax.set_ylabel('phase (rad)')
    ax.set_title('Simulated (dots) against the closed form (lines)')
    ax.legend(loc='lower left')

    fig.tight_layout()
    plt.show()
