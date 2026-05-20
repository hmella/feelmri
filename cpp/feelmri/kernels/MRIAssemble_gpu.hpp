/**
 * @file MRIAssemble_gpu.hpp
 * @brief Launch wrapper for the GPU MRI-signal accumulation kernel.
 *
 * One launch produces (n_samples, nv) complex signal samples from the per-
 * node static fields (positions, 1/T2, B0 phase), the per-node complex
 * magnetisation, and a flat k-space trajectory + sample-time vector. The
 * trajectory update (`r_curr = r_rest + modes @ weights[row]`) is fused
 * into the kernel so the displaced positions never round-trip through
 * global memory.
 *
 * Covers `SignalAssembler::signal_sum` and `signal_nodal` — same kernel,
 * the caller passes either `Mxy_nodes` or `M @ Mxy_nodes` as the
 * magnetisation pointer.
 *
 * M1 scope: T = float, nv = 1 (typical single-coil). nv > 1 is supported;
 * coil loop is internal to the kernel.
 */
#pragma once

#include <complex>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fused projection + signal kernel for the GPU quadrature path.
 *
 * Replaces the host-side `f_Mxy_ = S * Mxy_nodes` followed by
 * `Mxy_q_weighted = f_wq * f_Mxy_` pre-pass with two on-device kernels:
 *  1. Sparse-times-dense projection of the per-node magnetisation onto
 *     the quadrature points, scaled by f_wq, writing directly into the
 *     wrapper's persistent device-side Mxy_q_weighted buffer.
 *  2. The normal signal accumulation kernel, reading from that buffer.
 *
 * The CSR S matrix and f_wq are uploaded once per process via pointer
 * fingerprinting (they do not change after `set_assembler`).
 *
 * Inputs are the same as `feelmri_mri_signal_gpu_f32` except:
 *  - Pass `Mxy_nodes` (n_nodes x nv RowMajor) instead of a pre-projected
 *    Mxy_q_weighted at the quadrature points.
 *  - Pass the CSR triple (`S_row_ptr`, `S_col_idx`, `S_values`) + `S_nnz`
 *    + `S_n_nodes` describing the (n_qp x n_nodes) projection matrix.
 *  - Pass `fwq` (n_qp,) quadrature weights.
 *
 * `n_qp` in the inputs (passed as the same `n_nodes` parameter that
 * `feelmri_mri_signal_gpu_f32` uses to size the per-quadrature arrays)
 * is the destination row count for the projection.
 */
int feelmri_mri_signal_with_projection_gpu_f32(
  // Quadrature-point static fields (n_qp,)
  const float* nodes_x0,
  const float* nodes_x1,
  const float* nodes_x2,
  const float* invT2,
  const float* phi,
  // Per-node Mxy (n_nodes_per_solver, nv) RowMajor; the projection input.
  const std::complex<float>* Mxy_nodes,
  int S_n_nodes,
  // S_global_ in CSR format. Pointer fingerprint controls re-upload.
  const int*   S_row_ptr,    // (n_qp + 1,)
  const int*   S_col_idx,    // (S_nnz,)
  const float* S_values,     // (S_nnz,)
  int          S_nnz,
  // Quadrature weights (n_qp,). Same fingerprinted upload as the CSR.
  const float* fwq,
  // POD modes projected to quadrature points (n_qp, n_modes) RowMajor
  // or nullptr when has_traj == 0; same convention as the un-fused path.
  const float* modes_x,
  const float* modes_y,
  const float* modes_z,
  // Per-k-sample mode weights (n_samples, n_modes) RowMajor or nullptr.
  const float* weights,
  int has_traj,
  int n_modes,
  int n_qp,
  int nv,
  // Flat k-space trajectory and time vector (n_samples,)
  const float* kloc_x,
  const float* kloc_y,
  const float* kloc_z,
  const float* t,
  int n_samples,
  // Output: (n_samples, nv) RowMajor complex signal.
  std::complex<float>* signal_out
);

int feelmri_mri_signal_gpu_f32(
  // Per-node static fields (n_nodes,)
  const float* nodes_x0,
  const float* nodes_x1,
  const float* nodes_x2,
  const float* invT2,
  const float* phi,
  // Per-node complex magnetisation (n_nodes, nv) RowMajor
  const std::complex<float>* Mxy_nodes,
  // Per-node POD modes (n_nodes, n_modes) RowMajor or nullptr when has_traj == 0
  const float* modes_x,
  const float* modes_y,
  const float* modes_z,
  // Per-sample mode weights (n_samples, n_modes) RowMajor or nullptr
  const float* weights,
  int has_traj,
  int n_modes,
  int n_nodes,
  int nv,
  // Flat k-space trajectory and time vector (n_samples,)
  const float* kloc_x,
  const float* kloc_y,
  const float* kloc_z,
  const float* t,
  int n_samples,
  // Output: (n_samples, nv) RowMajor complex signal
  std::complex<float>* signal_out
);

#ifdef __cplusplus
}
#endif
