#pragma once
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

Tensor<std::complex<float>, 4> SPAMM(
  const int &MPI_rank,
  const SparseMatrix<float> &M,
  const std::vector<Tensor<float, 3>> &kloc,
  const MatrixXf &t,
  const MatrixXf &r,
  const MatrixXcf &M_spamm,
  const VectorXf &phi_dB0,
  const float &T2,
  const VectorXcf &profile
  );

PYBIND11_MODULE(Tagging, m) {
    m.doc() = "Utilities for MR image generation";
    m.def("SPAMM", &SPAMM);
}