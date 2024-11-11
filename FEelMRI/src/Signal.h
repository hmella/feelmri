#pragma once
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

using namespace Eigen;
namespace py = pybind11;

// Datatypes
typedef std::complex<float> cfloat;    // Complex float datatype

// Definition of PI and complex unit
const float PI = M_PI;
const cfloat i1(0.0, 1.0);

/**
 * @brief Computes the Fourier encoding for a given set of parameters.
 *
 * This function calculates the Fourier encoding using the provided spatial coordinates,
 * k-space trajectories, and phase offsets. The result is a complex array representing
 * the Fourier encoded data.
 *
 * @param r A matrix of spatial coordinates (Nx3).
 * @param kx A 3D tensor representing the k-space trajectory in the x-direction.
 * @param ky A 3D tensor representing the k-space trajectory in the y-direction.
 * @param kz A 3D tensor representing the k-space trajectory in the z-direction.
 * @param phi_off A vector of phase offsets.
 * @param i The index for the first dimension of the k-space tensors.
 * @param j The index for the second dimension of the k-space tensors.
 * @param k The index for the third dimension of the k-space tensors.
 * @return A complex array representing the Fourier encoded data.
 */
ArrayXcf FourierEncoding(const MatrixXf &r, const Tensor<float, 3> kx, const Tensor<float, 3> ky, const Tensor<float, 3> kz, const VectorXf &phi_off, const uint i, const uint j, const uint k);

/**
 * @brief Computes the signal by applying a sparse matrix transformation, 
 *        element-wise product with a Fourier vector, and a decay factor.
 * 
 * @param M A sparse matrix of floats representing the transformation matrix.
 * @param Mxy A complex matrix where each column represents a different state.
 * @param fourier A complex vector representing the Fourier coefficients.
 * @param decay A float reference representing the decay factor to be applied.
 * @return cfloat The computed signal as a complex float.
 */
cfloat signal(const SparseMatrix<float> &M, const MatrixXcf &Mxy, const VectorXcf &fourier, const VectorXf &decay, const uint &enc_dir);

cfloat signal(const SparseMatrix<float> &M, const MatrixXcf &Mxy, const VectorXcf &fourier, const float &decay, const uint &enc_dir);

/**
 * @brief Updates the position of an object based on its initial position, velocity, and time step.
 * 
 * @param r0 Initial position matrix.
 * @param v0 Initial velocity matrix.
 * @param dt Time step.
 * @return MatrixXf Updated position matrix.
 */
MatrixXf UpdatePosition(const MatrixXf &r0, const MatrixXf &v0, const float &dt);

/**
 * @brief Updates the position of an object based on its initial position, velocity, acceleration, and time step.
 *
 * This function calculates the new position of an object using the initial position (r0), initial velocity (v0),
 * initial acceleration (a0), and the time step (dt). The calculation is based on the kinematic equation:
 * r = r0 + v0 * dt + 0.5 * a0 * dt^2.
 *
 * @param r0 The initial position matrix (MatrixXf).
 * @param v0 The initial velocity matrix (MatrixXf).
 * @param a0 The initial acceleration matrix (MatrixXf).
 * @param dt The time step (float).
 * @return MatrixXf The updated position matrix.
 */
MatrixXf UpdatePosition(const MatrixXf &r0, const MatrixXf &v0, const MatrixXf &a0, const float &dt);


// PYBIND11_MODULE(Signal, m) {
//     m.doc() = "Utilities for MR image generation";
//     m.def("FourierEncoding", &FourierEncoding);
//     m.def("signal", py::overload_cast<const SparseMatrix<float> &, const MatrixXcf &, const VectorXcf &, const VectorXf &, const uint &>(&signal));
//     m.def("signal", py::overload_cast<const SparseMatrix<float> &, const MatrixXcf &, const VectorXcf &, const float &, const uint &>(&signal));
//     m.def("UpdatePosition", py::overload_cast<const MatrixXf &, const MatrixXf &, const float &>(&UpdatePosition));
//     m.def("UpdatePosition", py::overload_cast<const MatrixXf &, const MatrixXf &, const MatrixXf &, const float &>(&UpdatePosition));
// }