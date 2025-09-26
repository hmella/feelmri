#pragma once
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>


using namespace Eigen;
namespace py = pybind11;

// Definition of PI and complex unit
const float PI = M_PI;

template <typename T>
Tensor<std::complex<T>, 4, RowMajor> Signal(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Tensor<T, 3> &t,
  const Matrix<T, Dynamic, Dynamic> &r0,
  const Array<T, Dynamic, 1> &phi_dB0,
  const Array<T, Dynamic, 1> &T2,
  const Matrix<std::complex<T>, Dynamic, Dynamic> &Mxy,
  const py::function &pod_trajectory
);


template <typename T>
Tensor<std::complex<T>, 4, RowMajor> Signal(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Tensor<T, 3> &t,
  const Matrix<T, Dynamic, Dynamic> &r0,
  const Array<T, Dynamic, 1> &phi_dB0,
  const Array<T, Dynamic, 1> &T2,
  const Matrix<std::complex<T>, Dynamic, Dynamic> &Mxy,
  const py::none &pod_trajectory
);


PYBIND11_MODULE(MRI, m) {
    m.doc() = "Utilities for MR image generation";
    // Overloaded function for PC with different parameters      
    m.def("Signal", py::overload_cast<const int &,
      const SparseMatrix<float> &,
      const std::vector<Tensor<float, 3>> &,
      const Tensor<float, 3> &,
      const Matrix<float, Dynamic, Dynamic> &,
      const Array<float, Dynamic, 1> &,
      const Array<float, Dynamic, 1> &,
      const Matrix<std::complex<float>, Dynamic, Dynamic> &,
      const py::function &>(&Signal<float>));  
    m.def("Signal", py::overload_cast<const int &,
      const SparseMatrix<float> &,
      const std::vector<Tensor<float, 3>> &,
      const Tensor<float, 3> &,
      const Matrix<float, Dynamic, Dynamic> &,
      const Array<float, Dynamic, 1> &,
      const Array<float, Dynamic, 1> &,
      const Matrix<std::complex<float>, Dynamic, Dynamic> &,
      const py::none &>(&Signal<float>));  

}
