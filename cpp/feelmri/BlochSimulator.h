/**
 * @file BlochSimulator.h
 * @brief Bloch equation solver with selectable rotation operator.
 *
 * Three rotation operators are supported per time step:
 *
 *   order = 0  Cayley-Klein hard-pulse on end-of-step field. Equivalent to the
 *              historical FEelMRI solver and to KomaMRI's BlochMagnus1 in spinor
 *              form. First-order accurate in dt for smoothly varying fields.
 *
 *   order = 2  2nd-order Magnus expansion. Builds the rotation-angle vector
 *              from the trapezoidal average of the field at the start and end
 *              of the step. Second-order accurate in dt; commutator dropped.
 *
 *   order = 4  Magnus expansion with the linearly-interpolated commutator
 *              Omega_2 = -dt^2 / 12 * [Omega(t_old), Omega(t_new)] added on
 *              top of the order-2 trapezoidal rotation vector. Naming follows
 *              KomaMRI ("BlochMagnus4"); globally still O(dt^2) because the
 *              trapezoidal Omega_1 quadrature limits the order, but with a
 *              smaller error constant than order = 2 because the leading
 *              commutator correction is no longer dropped. The commutator
 *              vanishes for piecewise-constant fields and for on-resonance
 *              real RF with no gradient — in those regimes the three orders
 *              agree to FP rounding. A genuine O(dt^4) Magnus scheme would
 *              need Gauss-Legendre interior quadrature on Omega_1, which is
 *              not implemented here.
 *
 * Cayley-Klein spinor rotation and T1/T2 relaxation are shared across all
 * orders. Magnus orders carry per-node Bz_old and a shared scalar rf_old
 * between calls; the kernel returns them so the Python wrapper can stitch
 * blocks together with continuous field history.
 *
 * Optional deforming-mesh support: pre-computed POD modes and weights are
 * applied via Eigen GEMV (`r_curr = r0 + modes @ weights[i+1, :]`) once per
 * time step, with zero Python callbacks during the inner loop.
 *
 * Return contract. `solve_mri_f32` / `solve_mri_f64` return
 * `(Mxy, Mz, Bz_old_final, rf_old_final)`. By default `Mxy` and `Mz` have
 * shape `(n_pos, 1)` and hold only the final state: the magnetisation is
 * advanced in place through per-node rolling buffers, since the final column
 * is the only one BlochSolver consumes. Passing `store_history = true`
 * restores the full `(n_pos, n_time)` trace for the isochromat-dephasing
 * plotter and ad-hoc debugging. `M[:, -1]` is the final state under both
 * shapes, so callers need not branch on it.
 *
 * Steps with no RF take a reduced z-rotation path: with B1 = 0 the
 * Cayley-Klein off-diagonal beta is exactly zero, so the square root,
 * complex division and four complex products of the transverse algebra are
 * skipped. The dropped terms are exact zeros, so results are bit-identical
 * to the full path. The branch is per time step, not per node.
 *
 * Relaxation exponentials. When T1 and T2 are constant across nodes — the
 * case whenever a phantom is built from scalar relaxation times —
 * `exp(-dt/T1)` and `exp(-dt/T2)` are scalars and a change of time step
 * costs two `std::exp` calls rather than `2 * n_pos`. This matters because
 * RF rasters are generally not uniform in dt (an apodized-sinc excitation
 * block can carry >100 distinct dt values), so the per-node path would
 * re-exponentiate every node on most steps. The per-node path is kept for
 * genuine T1/T2 maps and stays on libm `std::exp` so it remains bit-identical
 * to the historical solver.
 */
#pragma once

#include <pybind11/eigen.h> // REQUIRED for zero-copy memory views
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

using namespace Eigen;
namespace py = pybind11;
