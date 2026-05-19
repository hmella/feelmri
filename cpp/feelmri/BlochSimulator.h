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
 */
#pragma once

#include <pybind11/eigen.h> // REQUIRED for zero-copy memory views
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

using namespace Eigen;
namespace py = pybind11;
