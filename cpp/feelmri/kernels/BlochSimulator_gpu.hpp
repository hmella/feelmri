/**
 * @file BlochSimulator_gpu.hpp
 * @brief Launch wrappers for the GPU Bloch kernel.
 *
 * Two extern "C" entry points cover the float32 and float64 paths; each
 * dispatches internally on `order` (0 = Cayley-Klein on end field, 2 =
 * 2nd-order Magnus, 4 = Magnus + commutator). The pybind11 binding TU
 * `cpp/feelmri/BlochSimulator.cpp` is compiled with the host CXX
 * compiler so the wrappers are plain-C; the kernel is compiled with
 * nvcc / hipcc and shares a single templated source in `.cu`.
 *
 * Magnus orders carry per-node `Bz_old` and a shared scalar `rf_old`
 * between calls. The host caller (Python BlochSolver) reads them from
 * `*_final` outputs and re-seeds them on the next block; the wrapper
 * accepts them via `Bz_old_init` and `rf_old_init`. For Order = 0 the
 * Magnus state arguments are unused but still required for signature
 * uniformity; the wrapper zeroes the `*_final` outputs.
 */
#pragma once

#include <complex>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

int feelmri_solve_mri_gpu_f32(
  const float* r0,
  const float* T1,
  const float* T2,
  const float* delta_B,
  float M0,
  float gamma,
  const std::complex<float>* rf_all,
  const float* G_all,
  const float* dt,
  const std::complex<float>* Mxy_initial,
  const float* Mz_initial,
  const float* modes_x,
  const float* modes_y,
  const float* modes_z,
  const float* weights,
  int has_traj,
  int n_modes,
  int n_pos,
  int n_time,
  int order,
  const float* Bz_old_init,
  std::complex<float> rf_old_init,
  std::complex<float>* Mxy_last,
  float* Mz_last,
  float* Bz_old_final,
  std::complex<float>* rf_old_final
);

int feelmri_solve_mri_gpu_f64(
  const double* r0,
  const double* T1,
  const double* T2,
  const double* delta_B,
  double M0,
  double gamma,
  const std::complex<double>* rf_all,
  const double* G_all,
  const double* dt,
  const std::complex<double>* Mxy_initial,
  const double* Mz_initial,
  const double* modes_x,
  const double* modes_y,
  const double* modes_z,
  const double* weights,
  int has_traj,
  int n_modes,
  int n_pos,
  int n_time,
  int order,
  const double* Bz_old_init,
  std::complex<double> rf_old_init,
  std::complex<double>* Mxy_last,
  double* Mz_last,
  double* Bz_old_final,
  std::complex<double>* rf_old_final
);

#ifdef __cplusplus
}
#endif
