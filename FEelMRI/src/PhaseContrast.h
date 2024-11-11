#pragma once
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

/**
 * @brief Computes the 3D flow image from k-space data.
 *
 * This function calculates the 3D flow image by simulating the k-space 
 * acquisition process. It takes into account the mass matrix, k-space 
 * trajectory, k-space timings, object initial position and velocity, 
 * encoded velocities, field inhomogeneity, T2 time of the blood, velocity 
 * encoding, and slice profile.
 *
 * @param MPI_rank The rank of the MPI process.
 * @param M The mass matrix.
 * @param kloc The k-space trajectory.
 * @param t The k-space timings.
 * @param r0 The initial position of the object.
 * @param v The initial velocity of the object.
 * @param phi_v The encoded velocities.
 * @param phi_dB0 The field inhomogeneity.
 * @param T2 The T2 time of the blood.
 * @param profile The slice profile.
 * @return A 4D tensor representing the k-space data.
 */
Tensor<cfloat, 4> PC(
  const int &MPI_rank,
  const SparseMatrix<float> &M,
  const std::vector<Tensor<float, 3>> &kloc,
  const MatrixXf &t,
  const MatrixXf &r0,
  const MatrixXf &v,
  const MatrixXf &phi_v,
  const VectorXf &phi_dB0,
  const VectorXf &T2,
  const VectorXcf &profile
  );

/**
 * @brief Computes the 3D flow image with fat signal contribution.
 * 
 * This function calculates the k-space values for a 3D flow image, taking into account the fat signal contribution. 
 * It uses the input parameters to compute the Fourier exponential, off-resonance phase, and T2* decay, and iterates 
 * over the k-space phase lines, readout points, and slices to generate the final k-space tensor.
 * 
 * @param MPI_rank The rank of the MPI process.
 * @param M Sparse matrix used in the computation.
 * @param kloc Vector of 3D tensors representing k-space locations.
 * @param t Matrix of time points.
 * @param r0 Initial positions of spins.
 * @param v Velocities of spins.
 * @param phi_v Matrix of velocity encoding phases.
 * @param phi_dB0 Vector of off-resonance frequencies.
 * @param T2 T2 relaxation time.
 * @param profile Vector of profile values.
 * @param rho_w Vector of water proton densities.
 * @param rho_f Vector of fat proton densities.
 * @param chemical_shift Vector of chemical shift values.
 * @return Tensor<cfloat, 4> The computed k-space tensor with dimensions (nb_meas, nb_lines, nb_kz, 3).
 */
Tensor<cfloat, 4> PCFat(
  const int &MPI_rank,
  const SparseMatrix<float> &M,
  const std::vector<Tensor<float, 3>> &kloc,
  const MatrixXf &t,
  const MatrixXf &r0,
  const MatrixXf &v,
  const MatrixXf &phi_v,
  const VectorXf &phi_dB0,
  const float &T2,
  const VectorXcf &profile,
  const VectorXf &rho_w,
  const VectorXf &rho_f,
  const VectorXf &chemical_shift  //
  );


/**
 * @brief Computes the 3D flow image with fat signal contribution.
 * 
 * This function calculates the k-space values for a 3D flow image, taking into account the fat signal contribution. 
 * It uses the input parameters to compute the Fourier exponential, off-resonance phase, and T2* decay, and iterates 
 * over the k-space phase lines, readout points, and slices to generate the final k-space tensor.
 * 
 * @param MPI_rank The rank of the MPI process.
 * @param M Sparse matrix used in the computation.
 * @param kloc Vector of 3D tensors representing k-space locations.
 * @param t Matrix of time points.
 * @param r0 Initial positions of spins.
 * @param v Velocities of spins.
 * @param phi_v Matrix of velocity encoding phases.
 * @param phi_dB0 Vector of off-resonance frequencies.
 * @param T2 T2 relaxation time.
 * @param VENC Matrix of velocity encoding values.
 * @param profile Vector of profile values.
 * @param rho_w Vector of water proton densities.
 * @param rho_f Vector of fat proton densities.
 * @param chemical_shift Vector of chemical shift values.
 * @return Tensor<cfloat, 4> The computed k-space tensor with dimensions (nb_meas, nb_lines, nb_kz, 3).
 */
Tensor<cfloat, 4> PCFat2(
  const int &MPI_rank,
  const SparseMatrix<float> &M,
  const std::vector<Tensor<float, 3>> &kloc,
  const MatrixXf &t,
  const MatrixXf &r0,
  const MatrixXf &v,
  const MatrixXf &phi_v,
  const VectorXf &phi_dB0,
  const VectorXf &M0,
  const VectorXf &T2,
  const VectorXcf &profile,
  const VectorXf &rho_w,
  const VectorXf &rho_f,
  const VectorXf &chemical_shift  //
  );

  /**
 * @brief Computes the 1D flow image for a given slice and phase line.
 * 
 * This function calculates the k-space values for a given slice and phase line
 * by updating the blood position, off-resonance phase, and Fourier exponential
 * at each readout point. It also applies T2* decay to the computed k-space values.
 * 
 * @param MPI_rank The rank of the MPI process.
 * @param M A sparse matrix representing the encoding matrix.
 * @param kloc A vector of 3D tensors representing the k-space locations.
 * @param t A matrix representing the time points.
 * @param r0 A matrix representing the initial positions of the spins.
 * @param v A matrix representing the velocities of the spins.
 * @param phi_v A matrix representing the phase encoding velocities.
 * @param phi_dB0 A vector representing the off-resonance frequencies.
 * @param T2 The T2 relaxation time constant.
 * @param VENC A matrix representing the velocity encoding.
 * @param profile A vector representing the profile.
 * @param j The phase line index.
 * @param k The slice index.
 * @return MatrixXcf The computed k-space values.
 */
MatrixXcf PC1D(
  const int &MPI_rank,
  const SparseMatrix<float> &M,
  const std::vector<Tensor<float, 3>> &kloc,
  const MatrixXf &t,
  const MatrixXf &r0,
  const MatrixXf &v,
  const MatrixXf &phi_v,
  const VectorXf &phi_dB0,
  const float &T2,
  const VectorXcf &profile,
  const int &j,
  const int &k
  );


PYBIND11_MODULE(PhaseContrast, m) {
    m.doc() = "Utilities for MR image generation";
    m.def("PC", &PC);
    m.def("PC1D", &PC1D);
    m.def("PCFat", &PCFat);
    m.def("PCFat2", &PCFat2);
}
