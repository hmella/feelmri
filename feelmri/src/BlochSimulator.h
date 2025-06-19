#pragma once
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

using namespace Eigen;
namespace py = pybind11;

// template <typename T>
// std::pair<
//     Matrix<std::complex<T>, Dynamic, Dynamic>,
//     Matrix<T, Dynamic, Dynamic>
// > solve_mri(
//     const Matrix<T, Dynamic, 3>& x,              // (n_pos, 3)
//     const Array<T, Dynamic, 1>& T1,              // (n_pos,)
//     const Array<T, Dynamic, 1>& T2,              // (n_pos,)
//     const Array<T, Dynamic, 1>& delta_B,         // (n_pos,)
//     const T &M0,                                 // Scalar
//     const T &gamma,                              // rad/ms/mT
//     const Array<std::complex<T>, Dynamic, 1>& rf_all,       // (n_time,)
//     const Matrix<T, Dynamic, 3>& G_all,          // (n_time, 3)
//     const Array<T, Dynamic, 1>& dt,              // (n_time,)
//     const VectorXi& regime_idx,                  // (n_time,)
//     const Vector<std::complex<T>, Dynamic> &Mxy_initial,
//     const Vector<T, Dynamic> &Mz_initial
// );


// PYBIND11_MODULE(BlochSimulator, m) {
//   m.def("solve_mri", py::overload_cast<
//     const Matrix<double, Dynamic, 3> &,
//     const Array<double, Dynamic, 1> &,
//     const Array<double, Dynamic, 1> &,
//     const Array<double, Dynamic, 1> &,
//     const double &,
//     const double &,
//     const Array<std::complex<double>, Dynamic, 1> &,
//     const Matrix<double, Dynamic, 3> &,
//     const Array<double, Dynamic, 1> &,
//     const VectorXi &,
//     const Vector<std::complex<double>, Dynamic> &,
//     const Vector<double, Dynamic> &
//   >(&solve_mri<double>));
//   m.def("solve_mri", py::overload_cast<
//     const Matrix<float, Dynamic, 3> &,
//     const Array<float, Dynamic, 1> &,
//     const Array<float, Dynamic, 1> &,
//     const Array<float, Dynamic, 1> &,
//     const float &,
//     const float &,
//     const Array<std::complex<float>, Dynamic, 1> &,
//     const Matrix<float, Dynamic, 3> &,
//     const Array<float, Dynamic, 1> &,
//     const VectorXi &,
//     const Vector<std::complex<float>, Dynamic> &,
//     const Vector<float, Dynamic> &
//   >(&solve_mri<float>));
// }