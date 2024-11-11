#pragma once
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>


Tensor<cfloat, 4> WaterFat(
  const int &MPI_rank,
  const SparseMatrix<float> &M,
  const std::vector<Tensor<float, 3>> &kloc,
  const MatrixXf &t,
  const MatrixXf &r0,
  const VectorXf &phi_dB0,
  const VectorXf &M0,
  const VectorXf &T2,
  const VectorXcf &profile,
  const VectorXf &rho_w,
  const VectorXf &rho_f,
  const VectorXf &chemical_shift  //
  );


PYBIND11_MODULE(MRIEncoding, m) {
    m.doc() = "Utilities for MR image generation";
    m.def("WaterFat", &WaterFat);
}
