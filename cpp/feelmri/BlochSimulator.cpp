#include "BlochSimulator.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

using namespace Eigen;
namespace py = pybind11;

// Return tuple: (Mxy, Mz, Bz_old_final, rf_old_final).
//   Mxy            (n_pos, n_time) complex
//   Mz             (n_pos, n_time) real
//   Bz_old_final   (n_pos,) real         -- per-node Bz at the last time step, for cross-block Magnus state
//   rf_old_final   scalar complex        -- shared RF at the last time step
template <typename T>
using MagnetizationState = std::tuple<
  Matrix<std::complex<T>, Dynamic, Dynamic>,
  Matrix<T, Dynamic, Dynamic>,
  Matrix<T, Dynamic, 1>,
  std::complex<T>
>;

// Templated kernel.
//   Order = 0 -> Cayley-Klein hard-pulse (uses end-of-step field; current solver behavior).
//   Order = 2 -> 2nd-order Magnus (trapezoidal field average, commutator dropped).
//   Order = 4 -> 4th-order Magnus (M2 result plus -dt^2/12 commutator term).
template <typename T, int Order>
MagnetizationState<T> solve_mri_impl(
  Eigen::Ref<const Matrix<T, Dynamic, 3, RowMajor>> r0,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> T1,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> T2,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> delta_B,
  const T &M0,
  const T &gamma,
  Eigen::Ref<const Matrix<std::complex<T>, Dynamic, 1>> rf_all,
  Eigen::Ref<const Matrix<T, Dynamic, 3>> G_all,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> dt,
  Eigen::Ref<const Matrix<bool, Dynamic, 1>> regime_idx,
  Eigen::Ref<const Matrix<std::complex<T>, Dynamic, 1>> Mxy_initial,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> Mz_initial,
  Eigen::Ref<const Matrix<T, Dynamic, Dynamic>> modes_x,
  Eigen::Ref<const Matrix<T, Dynamic, Dynamic>> modes_y,
  Eigen::Ref<const Matrix<T, Dynamic, Dynamic>> modes_z,
  Eigen::Ref<const Matrix<T, Dynamic, Dynamic>> weights,
  bool has_traj,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> Bz_old_init,
  std::complex<T> rf_old_init
){
  (void)regime_idx; // reserved for regime-selection logic; currently unused

  using C = std::complex<T>;
  const C i1(T(0), T(1));

  const int n_pos = r0.rows();
  const int n_time = rf_all.size();

  Matrix<C, Dynamic, Dynamic> Mxy(n_pos, n_time);
  Matrix<T, Dynamic, Dynamic> Mz(n_pos, n_time);
  Mxy.col(0) = Mxy_initial;
  Mz.col(0) = Mz_initial;

  Eigen::Array<T, Eigen::Dynamic, 1> curr_x = r0.col(0).array();
  Eigen::Array<T, Eigen::Dynamic, 1> curr_y = r0.col(1).array();
  Eigen::Array<T, Eigen::Dynamic, 1> curr_z = r0.col(2).array();

  const Matrix<T, Dynamic, 1> invT1 = T1.cwiseInverse();
  const Matrix<T, Dynamic, 1> invT2 = T2.cwiseInverse();
  Matrix<T, Dynamic, 1> e1(n_pos);
  Matrix<T, Dynamic, 1> e2(n_pos);

  // Persisted Magnus state. For Order == 0 we never read it; for Order > 0 we
  // initialize from the caller-provided seed.
  Matrix<T, Dynamic, 1> Bz_old(n_pos);
  if constexpr (Order > 0) {
    Bz_old = Bz_old_init;
  } else {
    Bz_old.setZero();
  }
  C rf_old = rf_old_init;

  // Small-angle threshold for the Magnus spinor build. Below this we use a
  // Taylor series for cos(theta/2) and sin(theta/2)/theta to avoid 0/0.
  const T small_theta = (sizeof(T) == 4) ? T(1e-3) : T(1e-6);

  T dt_i = T(-1);
  for (int i = 0; i < n_time - 1; ++i) {

    // Recompute relaxation exponentials only when the step size changes.
    if (dt_i != dt[i + 1]) {
      for (int p = 0; p < n_pos; ++p) {
        e1(p) = std::exp(-dt[i + 1] * invT1(p));
        e2(p) = std::exp(-dt[i + 1] * invT2(p));
      }
    }
    dt_i = dt[i + 1];

    // Trajectory update via AVX2 GEMV (zero-copy).
    if (has_traj) {
      auto w = weights.row(i + 1).transpose();
      curr_x = r0.col(0).array() + (modes_x * w).array();
      curr_y = r0.col(1).array() + (modes_y * w).array();
      curr_z = r0.col(2).array() + (modes_z * w).array();
    }

    const C rf_new = rf_all[i + 1];
    const T Gx = G_all(i + 1, 0);
    const T Gy = G_all(i + 1, 1);
    const T Gz = G_all(i + 1, 2);

    // Per-order step prefactors (constant across nodes within one step).
    const T kappa    = -T(0.5) * gamma * dt_i;          // Order = 0
    const T m2_scale = -T(0.5) * gamma * dt_i;          // Order >= 2
    const T m4_scale =  gamma * gamma * dt_i * dt_i / T(12); // Order = 4

    for (int p = 0; p < n_pos; ++p) {

      const T Bz_new = curr_x(p)*Gx + curr_y(p)*Gy + curr_z(p)*Gz + delta_B(p);

      C alpha_p, beta_p;

      if constexpr (Order == 0) {
        // Cayley-Klein on end-of-step field (current solver, unchanged).
        const T rf2 = std::norm(rf_new);
        T Bnorm = std::sqrt(Bz_new*Bz_new + rf2);
        if (Bnorm < T(1e-12)) Bnorm = T(1e-12);
        const T nz = Bz_new / Bnorm;
        const T half_phi = Bnorm * kappa;
        const T c = std::cos(half_phi);
        const T s = std::sin(half_phi);
        alpha_p = C(c, -nz * s);
        const C nxy = rf_new / Bnorm;
        beta_p = -i1 * nxy * s;
      } else {
        // Magnus order >= 2: build rotation-angle vector (theta_xy, theta_z).
        const T Bz_o = Bz_old(p);

        // Order-2 trapezoidal terms.
        C theta_xy = m2_scale * (rf_old + rf_new);
        T theta_z  = m2_scale * (Bz_o + Bz_new);

        if constexpr (Order == 4) {
          // Commutator correction. In B-space: omega = -gamma * B so the
          // sign squares away in the bilinear cross product.
          theta_xy -= i1 * m4_scale * (rf_new * Bz_o - rf_old * Bz_new);
          theta_z  -= m4_scale * std::imag(std::conj(rf_old) * rf_new);
        }

        // Spinor build.
        const T theta_sq = std::norm(theta_xy) + theta_z * theta_z;
        const T theta    = std::sqrt(theta_sq);

        T c, half_sinc;
        if (theta < small_theta) {
          // Taylor: cos(theta/2)      ~ 1 - theta^2/8  + theta^4/384
          //         sin(theta/2)/theta ~ 1/2 - theta^2/48 + theta^4/3840
          const T t2 = theta_sq;
          c         = T(1)   - t2 * T(0.125)   + t2 * t2 / T(384);
          half_sinc = T(0.5) - t2 / T(48)      + t2 * t2 / T(3840);
        } else {
          const T half = theta * T(0.5);
          c = std::cos(half);
          half_sinc = std::sin(half) / theta;
        }
        alpha_p = C(c, -theta_z * half_sinc);
        beta_p = -i1 * theta_xy * half_sinc;
      }

      // Cayley-Klein rotation of M (identical across all orders).
      const C conj_a = std::conj(alpha_p);
      const T a2 = std::norm(alpha_p);
      const T b2 = std::norm(beta_p);

      const C Mxy_prev = Mxy(p, i);
      const T Mz_prev  = Mz(p, i);

      const C Mxy_new = T(2) * conj_a * beta_p * Mz_prev
                      + conj_a * conj_a * Mxy_prev
                      - beta_p * beta_p * std::conj(Mxy_prev);
      const T Mz_new  = (a2 - b2) * Mz_prev
                      - T(2) * std::real(alpha_p * beta_p * std::conj(Mxy_prev));

      // T1/T2 relaxation.
      Mxy(p, i + 1) = Mxy_new * e2(p);
      Mz (p, i + 1) = Mz_new  * e1(p) + (T(1) - e1(p)) * M0;

      if constexpr (Order > 0) {
        Bz_old(p) = Bz_new;
      }
    }

    if constexpr (Order > 0) {
      rf_old = rf_new;
    }
  }

  return std::make_tuple(std::move(Mxy), std::move(Mz), std::move(Bz_old), rf_old);
}

// Order dispatch helper.
template <typename T>
MagnetizationState<T> solve_mri_dispatch(
  int order,
  Eigen::Ref<const Matrix<T, Dynamic, 3, RowMajor>> r0,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> T1,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> T2,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> delta_B,
  const T &M0,
  const T &gamma,
  Eigen::Ref<const Matrix<std::complex<T>, Dynamic, 1>> rf_all,
  Eigen::Ref<const Matrix<T, Dynamic, 3>> G_all,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> dt,
  Eigen::Ref<const Matrix<bool, Dynamic, 1>> regime_idx,
  Eigen::Ref<const Matrix<std::complex<T>, Dynamic, 1>> Mxy_initial,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> Mz_initial,
  Eigen::Ref<const Matrix<T, Dynamic, Dynamic>> modes_x,
  Eigen::Ref<const Matrix<T, Dynamic, Dynamic>> modes_y,
  Eigen::Ref<const Matrix<T, Dynamic, Dynamic>> modes_z,
  Eigen::Ref<const Matrix<T, Dynamic, Dynamic>> weights,
  bool has_traj,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> Bz_old_init,
  std::complex<T> rf_old_init
){
  switch (order) {
    case 0:
      return solve_mri_impl<T, 0>(r0, T1, T2, delta_B, M0, gamma, rf_all, G_all, dt, regime_idx,
                                  Mxy_initial, Mz_initial, modes_x, modes_y, modes_z, weights,
                                  has_traj, Bz_old_init, rf_old_init);
    case 2:
      return solve_mri_impl<T, 2>(r0, T1, T2, delta_B, M0, gamma, rf_all, G_all, dt, regime_idx,
                                  Mxy_initial, Mz_initial, modes_x, modes_y, modes_z, weights,
                                  has_traj, Bz_old_init, rf_old_init);
    case 4:
      return solve_mri_impl<T, 4>(r0, T1, T2, delta_B, M0, gamma, rf_all, G_all, dt, regime_idx,
                                  Mxy_initial, Mz_initial, modes_x, modes_y, modes_z, weights,
                                  has_traj, Bz_old_init, rf_old_init);
    default:
      throw std::invalid_argument("solve_mri: order must be 0, 2, or 4");
  }
}

PYBIND11_MODULE(BlochSimulator, m) {
  using f32 = float;
  using f64 = double;

  using R0_f32     = Eigen::Ref<const Matrix<f32, Dynamic, 3, RowMajor>>;
  using Vec_f32    = Eigen::Ref<const Matrix<f32, Dynamic, 1>>;
  using CVec_f32   = Eigen::Ref<const Matrix<std::complex<f32>, Dynamic, 1>>;
  using Mat3_f32   = Eigen::Ref<const Matrix<f32, Dynamic, 3>>;
  using Bool_T     = Eigen::Ref<const Matrix<bool, Dynamic, 1>>;
  using MatDyn_f32 = Eigen::Ref<const Matrix<f32, Dynamic, Dynamic>>;

  using R0_f64     = Eigen::Ref<const Matrix<f64, Dynamic, 3, RowMajor>>;
  using Vec_f64    = Eigen::Ref<const Matrix<f64, Dynamic, 1>>;
  using CVec_f64   = Eigen::Ref<const Matrix<std::complex<f64>, Dynamic, 1>>;
  using Mat3_f64   = Eigen::Ref<const Matrix<f64, Dynamic, 3>>;
  using MatDyn_f64 = Eigen::Ref<const Matrix<f64, Dynamic, Dynamic>>;

  m.def("solve_mri_f32",
    [](R0_f32 r0, Vec_f32 T1, Vec_f32 T2, Vec_f32 delta_B,
       const f32 &M0, const f32 &gamma,
       CVec_f32 rf_all, Mat3_f32 G_all, Vec_f32 dt, Bool_T regime_idx,
       CVec_f32 Mxy_initial, Vec_f32 Mz_initial,
       MatDyn_f32 modes_x, MatDyn_f32 modes_y, MatDyn_f32 modes_z,
       MatDyn_f32 weights, bool has_traj,
       int order, Vec_f32 Bz_old_init, std::complex<f32> rf_old_init) {
      return solve_mri_dispatch<f32>(order, r0, T1, T2, delta_B, M0, gamma,
                                     rf_all, G_all, dt, regime_idx,
                                     Mxy_initial, Mz_initial,
                                     modes_x, modes_y, modes_z, weights, has_traj,
                                     Bz_old_init, rf_old_init);
    });

  m.def("solve_mri_f64",
    [](R0_f64 r0, Vec_f64 T1, Vec_f64 T2, Vec_f64 delta_B,
       const f64 &M0, const f64 &gamma,
       CVec_f64 rf_all, Mat3_f64 G_all, Vec_f64 dt, Bool_T regime_idx,
       CVec_f64 Mxy_initial, Vec_f64 Mz_initial,
       MatDyn_f64 modes_x, MatDyn_f64 modes_y, MatDyn_f64 modes_z,
       MatDyn_f64 weights, bool has_traj,
       int order, Vec_f64 Bz_old_init, std::complex<f64> rf_old_init) {
      return solve_mri_dispatch<f64>(order, r0, T1, T2, delta_B, M0, gamma,
                                     rf_all, G_all, dt, regime_idx,
                                     Mxy_initial, Mz_initial,
                                     modes_x, modes_y, modes_z, weights, has_traj,
                                     Bz_old_init, rf_old_init);
    });
}
