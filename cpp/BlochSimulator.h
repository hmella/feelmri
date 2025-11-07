#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

using namespace Eigen;
namespace py = pybind11;

template <typename T>
using Magnetization = std::pair<
    Matrix<std::complex<T>, Dynamic, Dynamic>,
    Matrix<T, Dynamic, Dynamic>
    >;

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