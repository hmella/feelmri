/**
 * @file BlochSimulator.h
 * @brief Hard-pulse Bloch equation solver with T1/T2 relaxation and optional
 *        deforming-mesh support via POD trajectory callbacks.
 *
 * The solver uses the Cayley-Klein (half-angle) parametrisation of the
 * hard-pulse rotation to advance the magnetisation at every finite-element
 * node through each time step.  Relaxation is applied after each rotation
 * using pre-computed exponentials.
 */
#pragma once

#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

using namespace Eigen;
namespace py = pybind11;

/**
 * @brief Output pair returned by @ref solve_mri.
 *
 * - ``first``  : complex transverse magnetisation Mxy, shape ``(n_pos, n_time)``
 * - ``second`` : longitudinal magnetisation Mz,    shape ``(n_pos, n_time)``
 *
 * @tparam T Floating-point scalar type (``float`` or ``double``).
 */
template <typename T>
using Magnetization = std::pair<
    Matrix<std::complex<T>, Dynamic, Dynamic>,
    Matrix<T, Dynamic, Dynamic>
    >;

/**
 * @brief Solve the Bloch equations for a deforming mesh (moving-tissue variant).
 *
 * Iterates over time steps applying a hard-pulse rotation followed by T1/T2
 * relaxation at each finite-element node.  At every step the node positions
 * are updated by adding the displacement returned by @p pod_trajectory.
 *
 * @tparam T Floating-point scalar type (``float`` or ``double``).
 *
 * @param r0            Reference node positions, shape ``(n_pos, 3)`` (m).
 * @param T1            Longitudinal relaxation times, shape ``(n_pos,)`` (ms).
 * @param T2            Transverse relaxation times,   shape ``(n_pos,)`` (ms).
 * @param delta_B       Off-resonance B0 field at each node, shape ``(n_pos,)`` (mT).
 * @param M0            Equilibrium magnetisation magnitude (a.u.).
 * @param gamma         Gyromagnetic ratio (rad/mT/ms).
 * @param rf_all        Complex RF envelope at each time step, shape ``(n_time,)`` (mT).
 * @param G_all         Gradient waveforms [Gx, Gy, Gz], shape ``(n_time, 3)`` (mT/m).
 * @param dt            Time-step durations, shape ``(n_time,)`` (ms).
 * @param regime_idx    Boolean mask reserved for regime-selection logic (currently unused).
 * @param Mxy_initial   Initial transverse magnetisation, shape ``(n_pos,)``.
 * @param Mz_initial    Initial longitudinal magnetisation, shape ``(n_pos,)``.
 * @param pod_trajectory Python callable ``f(t) -> ndarray(n_pos, 3)`` returning
 *                       nodal displacements (m) at time @p t (ms).
 *
 * @return Magnetization<T> containing Mxy ``(n_pos, n_time)`` and Mz ``(n_pos, n_time)``.
 */
template <typename T>
Magnetization<T> solve_mri(
    const Matrix<T, Dynamic, 3, RowMajor> &r0,
    const Matrix<T, Dynamic, 1> &T1,
    const Matrix<T, Dynamic, 1> &T2,
    const Matrix<T, Dynamic, 1> &delta_B,
    const T &M0,
    const T &gamma,
    const Matrix<std::complex<T>, Dynamic, 1> &rf_all,
    const Matrix<T, Dynamic, 3> &G_all,
    const Matrix<T, Dynamic, 1> &dt,
    const Matrix<bool, Dynamic, 1> &regime_idx,
    const Matrix<std::complex<T>, Dynamic, 1> &Mxy_initial,
    const Matrix<T, Dynamic, 1> &Mz_initial,
    const py::function &pod_trajectory
);

/**
 * @brief Solve the Bloch equations for static node positions (no mesh motion).
 *
 * Identical to the moving-mesh overload except that node positions are fixed at
 * @p r0 for all time steps — no trajectory callback is invoked.
 *
 * @tparam T Floating-point scalar type (``float`` or ``double``).
 *
 * @param r0            Node positions, shape ``(n_pos, 3)`` (m).
 * @param T1            Longitudinal relaxation times, shape ``(n_pos,)`` (ms).
 * @param T2            Transverse relaxation times,   shape ``(n_pos,)`` (ms).
 * @param delta_B       Off-resonance B0 field at each node, shape ``(n_pos,)`` (mT).
 * @param M0            Equilibrium magnetisation magnitude (a.u.).
 * @param gamma         Gyromagnetic ratio (rad/mT/ms).
 * @param rf_all        Complex RF envelope at each time step, shape ``(n_time,)`` (mT).
 * @param G_all         Gradient waveforms [Gx, Gy, Gz], shape ``(n_time, 3)`` (mT/m).
 * @param dt            Time-step durations, shape ``(n_time,)`` (ms).
 * @param regime_idx    Boolean mask reserved for regime-selection logic (currently unused).
 * @param Mxy_initial   Initial transverse magnetisation, shape ``(n_pos,)``.
 * @param Mz_initial    Initial longitudinal magnetisation, shape ``(n_pos,)``.
 * @param pod_trajectory Pass ``py::none()`` to indicate no mesh motion.
 *
 * @return Magnetization<T> containing Mxy ``(n_pos, n_time)`` and Mz ``(n_pos, n_time)``.
 */
template <typename T>
Magnetization<T> solve_mri(
    const Matrix<T, Dynamic, 3, RowMajor> &r0,
    const Matrix<T, Dynamic, 1> &T1,
    const Matrix<T, Dynamic, 1> &T2,
    const Matrix<T, Dynamic, 1> &delta_B,
    const T &M0,
    const T &gamma,
    const Matrix<std::complex<T>, Dynamic, 1> &rf_all,
    const Matrix<T, Dynamic, 3> &G_all,
    const Matrix<T, Dynamic, 1> &dt,
    const Matrix<bool, Dynamic, 1> &regime_idx,
    const Matrix<std::complex<T>, Dynamic, 1> &Mxy_initial,
    const Matrix<T, Dynamic, 1> &Mz_initial,
    const py::none &pod_trajectory
);

PYBIND11_MODULE(BlochSimulator, m) {
  // Define function overloads
  // Overload for the case with pod_trajectory
  m.def("solve_mri", py::overload_cast<
    const Matrix<float, Dynamic, 3, RowMajor> & ,
    const Matrix<float, Dynamic, 1> & ,
    const Matrix<float, Dynamic, 1> & ,
    const Matrix<float, Dynamic, 1> & ,
    const float & ,
    const float & ,
    const Matrix<std::complex<float>, Dynamic, 1> &,
    const Matrix<float, Dynamic, 3> & ,
    const Matrix<float, Dynamic, 1> & ,
    const Matrix<bool, Dynamic, 1> & ,
    const Matrix<std::complex<float>, Dynamic, 1> & ,
    const Matrix<float, Dynamic, 1> & ,
    const py::function &>(&solve_mri<float>));
  m.def("solve_mri", py::overload_cast<
    const Matrix<float, Dynamic, 3, RowMajor> & ,
    const Matrix<float, Dynamic, 1> & ,
    const Matrix<float, Dynamic, 1> & ,
    const Matrix<float, Dynamic, 1> & ,
    const float & ,
    const float & ,
    const Matrix<std::complex<float>, Dynamic, 1> &,
    const Matrix<float, Dynamic, 3> & ,
    const Matrix<float, Dynamic, 1> & ,
    const Matrix<bool, Dynamic, 1> & ,
    const Matrix<std::complex<float>, Dynamic, 1> & ,
    const Matrix<float, Dynamic, 1> & ,
    const py::none &>(&solve_mri<float>));
}