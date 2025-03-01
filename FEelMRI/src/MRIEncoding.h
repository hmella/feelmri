#pragma once
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>


template <typename T>
Tensor<std::complex<T>, 4> WaterFat(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Tensor<T, 3> &t,
  const Matrix<T, Dynamic, Dynamic> &r0,
  const Vector<T, Dynamic> &phi_dB0,
  const Vector<T, Dynamic> &M0,
  const Vector<T, Dynamic> &T2,
  const Vector<std::complex<T>, Dynamic> &profile,
  const Vector<T, Dynamic> &rho_w,
  const Vector<T, Dynamic> &rho_f,
  const Vector<T, Dynamic> &chemical_shift  //
  );


PYBIND11_MODULE(MRIEncoding, m) {
    m.doc() = "Utilities for MR image generation";
    m.def("WaterFat", py::overload_cast<const int &,
      const SparseMatrix<double> &,
      const std::vector<Tensor<double, 3>> &,
      const Tensor<double, 3> &,
      const Matrix<double, Dynamic, Dynamic> &,
      const Vector<double, Dynamic> &,
      const Vector<double, Dynamic> &,
      const Vector<double, Dynamic> &,
      const Vector<std::complex<double>, Dynamic> &,
      const Vector<double, Dynamic> &,
      const Vector<double, Dynamic> &,
      const Vector<double, Dynamic> &>(&WaterFat<double>));
    m.def("WaterFat", py::overload_cast<const int &,
      const SparseMatrix<float> &,
      const std::vector<Tensor<float, 3>> &,
      const Tensor<float, 3> &,
      const Matrix<float, Dynamic, Dynamic> &,
      const Vector<float, Dynamic> &,
      const Vector<float, Dynamic> &,
      const Vector<float, Dynamic> &,
      const Vector<std::complex<float>, Dynamic> &,
      const Vector<float, Dynamic> &,
      const Vector<float, Dynamic> &,
      const Vector<float, Dynamic> &>(&WaterFat<float>));
}
