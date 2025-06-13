#pragma once
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

template <typename T>
Tensor<std::complex<T>, 4> SPAMM(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Tensor<T, 3> &t,
  const Matrix<T, Dynamic, Dynamic> &r0,
  const Matrix<std::complex<T>, Dynamic, Dynamic> &M_spamm,
  const Vector<T, Dynamic> &phi_dB0,
  const Vector<T, Dynamic> &T2
  );

template <typename T>
Tensor<std::complex<T>, 4> SPAMM(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Tensor<T, 3> &t,
  const Matrix<T, Dynamic, Dynamic> &r,
  const Matrix<std::complex<T>, Dynamic, Dynamic> &M_spamm,
  const Vector<T, Dynamic> &phi_dB0,
  const Vector<T, Dynamic> &T2,
  const py::function &pod_trajectory
  );

PYBIND11_MODULE(Tagging, m) {
    m.doc() = "Utilities for MR image generation";
    // Functions without pod_trajectory
    m.def("SPAMM", py::overload_cast<
      const int &,
      const SparseMatrix<double> &,
      const std::vector<Tensor<double, 3>> &,
      const Tensor<double, 3> &,
      const Matrix<double, Dynamic, Dynamic> &,
      const Matrix<std::complex<double>, Dynamic, Dynamic> &,
      const Vector<double, Dynamic> &,
      const Vector<double, Dynamic> &
    >(&SPAMM<double>));
    m.def("SPAMM", py::overload_cast<
      const int &,
      const SparseMatrix<float> &,
      const std::vector<Tensor<float, 3>> &,
      const Tensor<float, 3> &,
      const Matrix<float, Dynamic, Dynamic> &,
      const Matrix<std::complex<float>, Dynamic, Dynamic> &,
      const Vector<float, Dynamic> &,
      const Vector<float, Dynamic> &
    >(&SPAMM<float>));
    // Functions with pod_trajectory
    m.def("SPAMM", py::overload_cast<
      const int &,
      const SparseMatrix<double> &,
      const std::vector<Tensor<double, 3>> &,
      const Tensor<double, 3> &,
      const Matrix<double, Dynamic, Dynamic> &,
      const Matrix<std::complex<double>, Dynamic, Dynamic> &,
      const Vector<double, Dynamic> &,
      const Vector<double, Dynamic> &,
      const py::function &
    >(&SPAMM<double>));
    m.def("SPAMM", py::overload_cast<
      const int &,
      const SparseMatrix<float> &,
      const std::vector<Tensor<float, 3>> &,
      const Tensor<float, 3> &,
      const Matrix<float, Dynamic, Dynamic> &,
      const Matrix<std::complex<float>, Dynamic, Dynamic> &,
      const Vector<float, Dynamic> &,
      const Vector<float, Dynamic> &,
      const py::function &
    >(&SPAMM<float>));
}