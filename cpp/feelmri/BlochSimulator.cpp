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
//   Mxy            (n_pos, n_out) complex
//   Mz             (n_pos, n_out) real
//   Bz_old_final   (n_pos,) real         -- per-node Bz at the last time step, for cross-block Magnus state
//   rf_old_final   scalar complex        -- shared RF at the last time step
//
// n_out is 1 by default: the magnetisation is advanced in place through two
// rolling per-node buffers and only the final state is materialised, because
// that is the only column BlochSolver ever reads. Pass store_history = true to
// get the historical (n_pos, n_time) trace back, for the isochromat-dephasing
// plotter or ad-hoc debugging. Either way the last column is the final state,
// so `M[:, -1]` is correct for both shapes.
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
//
// UniformRelax selects the relaxation-exponential strategy. When T1 and T2 are
// constant across nodes -- the case for every phantom built from scalar T1/T2 --
// exp(-dt/T1) and exp(-dt/T2) are scalars, so a change of time step costs two
// std::exp calls instead of 2 * n_pos of them. This matters because a Pulseq /
// apodized-sinc raster is not uniform in dt: the free-running imaging block has
// 142 distinct dt values over 294 steps, so the naive path re-exponentiates
// every node on two thirds of all steps.
template <typename T, int Order, bool UniformRelax>
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
  Eigen::Ref<const Matrix<T, Dynamic, Dynamic>> modes,
  Eigen::Ref<const Matrix<T, Dynamic, Dynamic, RowMajor>> weights,
  bool has_traj,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> Bz_old_init,
  std::complex<T> rf_old_init,
  bool store_history
){
  // The caller's rf!=0 mask is redundant: the kernel derives the rf-free
  // condition from rf_all itself, so a stale or wrong mask cannot corrupt
  // the solution. Kept in the signature for API compatibility.
  (void)regime_idx;

  using C = std::complex<T>;
  const C i1(T(0), T(1));

  const int n_pos = r0.rows();
  const int n_time = rf_all.size();
  const int n_out = store_history ? n_time : 1;

  Matrix<C, Dynamic, Dynamic> Mxy(n_pos, n_out);
  Matrix<T, Dynamic, Dynamic> Mz(n_pos, n_out);

  // Rolling state. Each node is advanced independently and in place: the old
  // Mxy/Mz are read into registers before either is overwritten, so a single
  // buffer suffices. Working set is 12 B/node instead of 12 B/node/step.
  Matrix<C, Dynamic, 1> Mxy_cur = Mxy_initial;
  Matrix<T, Dynamic, 1> Mz_cur  = Mz_initial;

  if (store_history) {
    Mxy.col(0) = Mxy_cur;
    Mz.col(0)  = Mz_cur;
  }

  // Current node positions, interleaved (x, y, z) per node so the node loop
  // reads one contiguous stream instead of three. The deformed positions are
  // produced by a single row-major GEMV over the (3 * n_pos, n_modes) mode
  // matrix. The mode matrix is column-major (Fortran order from NumPy): the
  // GEMV is then n_modes long axpy passes over 3 * n_pos contiguous floats,
  // which vectorises cleanly. Row-major would degenerate into an n_modes-long dot
  // product per output element, and is slower. Passing it pre-transposed also
  // avoids a pybind11 transpose on every call.
  Matrix<T, Dynamic, 3, RowMajor> curr = r0;
  Eigen::Map<Matrix<T, Dynamic, 1>> curr_flat(curr.data(), 3 * n_pos);
  Eigen::Map<const Matrix<T, Dynamic, 1>> r0_flat(r0.data(), 3 * n_pos);

  const Matrix<T, Dynamic, 1> invT1 = T1.cwiseInverse();
  const Matrix<T, Dynamic, 1> invT2 = T2.cwiseInverse();

  // Relaxation exponentials: scalars on the uniform path, per-node vectors
  // otherwise.
  const T invT1_0 = (n_pos > 0) ? invT1(0) : T(0);
  const T invT2_0 = (n_pos > 0) ? invT2(0) : T(0);
  T e1s = T(1), e2s = T(1);
  Matrix<T, Dynamic, 1> e1, e2;
  if constexpr (!UniformRelax) {
    e1.resize(n_pos);
    e2.resize(n_pos);
  }

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
      const T d = dt[i + 1];
      if constexpr (UniformRelax) {
        e1s = std::exp(-d * invT1_0);
        e2s = std::exp(-d * invT2_0);
      } else {
        // Deliberately std::exp and not Eigen's vectorised array exp(): with
        // EIGEN_FAST_MATH the latter is a polynomial approximation that drifts
        // ~1 ulp per call, which accumulates to ~1e-5 relative over a few
        // hundred float32 steps. Keeping libm here makes the per-node T1/T2
        // path bit-identical to the historical solver.
        for (int p = 0; p < n_pos; ++p) {
          e1(p) = std::exp(-d * invT1(p));
          e2(p) = std::exp(-d * invT2(p));
        }
      }
    }
    dt_i = dt[i + 1];

    // Trajectory update: r_curr = r0 + modes @ weights[i + 1, :], one GEMV.
    if (has_traj) {
      curr_flat.noalias() = r0_flat + modes * weights.row(i + 1).transpose();
    }

    const C rf_new = rf_all[i + 1];
    const T Gx = G_all(i + 1, 0);
    const T Gy = G_all(i + 1, 1);
    const T Gz = G_all(i + 1, 2);

    // Per-order step prefactors (constant across nodes within one step).
    const T kappa    = -T(0.5) * gamma * dt_i;          // Order = 0
    const T m2_scale = -T(0.5) * gamma * dt_i;          // Order >= 2
    const T m4_scale =  gamma * gamma * dt_i * dt_i / T(12); // Order = 4

    // With no B1 the effective field is purely longitudinal, the rotation
    // axis is z, and the Cayley-Klein off-diagonal beta is exactly zero. The
    // transverse algebra -- a square root, a complex division and four
    // complex products per node -- then contributes nothing, so it is
    // skipped. The dropped terms are exact zeros (x + 0 == x, a2 - 0 == a2),
    // which is why the reduced update still forms a2 the same way instead of
    // assuming |alpha|^2 == 1: the rotation is unitary in exact arithmetic
    // but a2 is not exactly 1 in floating point, and substituting 1 would
    // silently change every rf-free step. The condition is per time step,
    // not per node, so it is hoisted out of the node loop. Magnus orders
    // additionally need the *previous* step to be rf-free, since their
    // rotation vector straddles both ends of the interval.
    const bool rf_free = (Order == 0)
        ? (rf_new == C(0))
        : (rf_new == C(0) && rf_old == C(0));

    auto advance_nodes = [&]<bool RfFree>() {
      for (int p = 0; p < n_pos; ++p) {

        const T Bz_new = curr(p, 0)*Gx + curr(p, 1)*Gy + curr(p, 2)*Gz + delta_B(p);

        C alpha_p, beta_p;

        if constexpr (Order == 0) {
          // Cayley-Klein on end-of-step field.
          T Bnorm;
          if constexpr (RfFree) {
            Bnorm = std::abs(Bz_new);          // == sqrt(Bz^2 + 0)
          } else {
            Bnorm = std::sqrt(Bz_new*Bz_new + std::norm(rf_new));
          }
          if (Bnorm < T(1e-12)) Bnorm = T(1e-12);
          const T nz = Bz_new / Bnorm;
          const T half_phi = Bnorm * kappa;
          const T c = std::cos(half_phi);
          const T s = std::sin(half_phi);
          alpha_p = C(c, -nz * s);
          if constexpr (RfFree) {
            beta_p = C(0);
          } else {
            const C nxy = rf_new / Bnorm;
            beta_p = -i1 * nxy * s;
          }
        } else {
          // Magnus order >= 2: build rotation-angle vector (theta_xy, theta_z).
          const T Bz_o = Bz_old(p);

          // Order-2 trapezoidal terms.
          C theta_xy = m2_scale * (rf_old + rf_new);
          T theta_z  = m2_scale * (Bz_o + Bz_new);

          if constexpr (Order == 4) {
            // Commutator correction. In B-space: omega = -gamma * B so the
            // sign squares away in the bilinear cross product. Both RF
            // endpoints are zero on the RfFree path, so the whole term drops.
            if constexpr (!RfFree) {
              theta_xy -= i1 * m4_scale * (rf_new * Bz_o - rf_old * Bz_new);
              theta_z  -= m4_scale * std::imag(std::conj(rf_old) * rf_new);
            }
          }

          // Spinor build.
          T theta_sq;
          if constexpr (RfFree) {
            theta_sq = theta_z * theta_z;      // norm(theta_xy) is exactly 0
          } else {
            theta_sq = std::norm(theta_xy) + theta_z * theta_z;
          }
          const T theta = std::sqrt(theta_sq);

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
          if constexpr (RfFree) {
            beta_p = C(0);
          } else {
            beta_p = -i1 * theta_xy * half_sinc;
          }
        }

        // Cayley-Klein rotation of M (identical across all orders).
        const C conj_a = std::conj(alpha_p);
        const T a2 = std::norm(alpha_p);

        const C Mxy_prev = Mxy_cur(p);
        const T Mz_prev  = Mz_cur(p);

        C Mxy_new;
        T Mz_new;
        if constexpr (RfFree) {
          // beta == 0: only the z-rotation of Mxy survives, Mz is untouched
          // apart from the |alpha|^2 factor the full expression also applies.
          Mxy_new = conj_a * conj_a * Mxy_prev;
          Mz_new  = a2 * Mz_prev;
        } else {
          const T b2 = std::norm(beta_p);
          Mxy_new = T(2) * conj_a * beta_p * Mz_prev
                  + conj_a * conj_a * Mxy_prev
                  - beta_p * beta_p * std::conj(Mxy_prev);
          Mz_new  = (a2 - b2) * Mz_prev
                  - T(2) * std::real(alpha_p * beta_p * std::conj(Mxy_prev));
        }

        // T1/T2 relaxation.
        T e1p, e2p;
        if constexpr (UniformRelax) {
          e1p = e1s;
          e2p = e2s;
        } else {
          e1p = e1(p);
          e2p = e2(p);
        }
        Mxy_cur(p) = Mxy_new * e2p;
        Mz_cur (p) = Mz_new  * e1p + (T(1) - e1p) * M0;

        if constexpr (Order > 0) {
          Bz_old(p) = Bz_new;
        }
      }
    };

    if (rf_free) {
      advance_nodes.template operator()<true>();
    } else {
      advance_nodes.template operator()<false>();
    }

    if constexpr (Order > 0) {
      rf_old = rf_new;
    }

    if (store_history) {
      Mxy.col(i + 1) = Mxy_cur;
      Mz.col(i + 1)  = Mz_cur;
    }
  }

  if (!store_history) {
    Mxy.col(0) = Mxy_cur;
    Mz.col(0)  = Mz_cur;
  }

  return std::make_tuple(std::move(Mxy), std::move(Mz), std::move(Bz_old), rf_old);
}

// Order / relaxation-path dispatch helper.
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
  Eigen::Ref<const Matrix<T, Dynamic, Dynamic>> modes,
  Eigen::Ref<const Matrix<T, Dynamic, Dynamic, RowMajor>> weights,
  bool has_traj,
  Eigen::Ref<const Matrix<T, Dynamic, 1>> Bz_old_init,
  std::complex<T> rf_old_init,
  bool store_history
){
  // Constant T1/T2 across nodes is the common case (phantoms built from scalar
  // relaxation times); it lets the relaxation exponentials stay in registers.
  const bool uniform_relax =
      (T1.size() > 0) && (T2.size() > 0)
      && (T1.array() == T1(0)).all()
      && (T2.array() == T2(0)).all();

  #define FEELMRI_DISPATCH(ORDER, UNIFORM)                                     \
    return solve_mri_impl<T, ORDER, UNIFORM>(                                  \
        r0, T1, T2, delta_B, M0, gamma, rf_all, G_all, dt, regime_idx,         \
        Mxy_initial, Mz_initial, modes, weights,                               \
        has_traj, Bz_old_init, rf_old_init, store_history)

  switch (order) {
    case 0:
      if (uniform_relax) { FEELMRI_DISPATCH(0, true); }
      else               { FEELMRI_DISPATCH(0, false); }
    case 2:
      if (uniform_relax) { FEELMRI_DISPATCH(2, true); }
      else               { FEELMRI_DISPATCH(2, false); }
    case 4:
      if (uniform_relax) { FEELMRI_DISPATCH(4, true); }
      else               { FEELMRI_DISPATCH(4, false); }
    default:
      throw std::invalid_argument("solve_mri: order must be 0, 2, or 4");
  }

  #undef FEELMRI_DISPATCH
}

PYBIND11_MODULE(BlochSimulator, m) {
  using f32 = float;
  using f64 = double;

  using R0_f32     = Eigen::Ref<const Matrix<f32, Dynamic, 3, RowMajor>>;
  using Vec_f32    = Eigen::Ref<const Matrix<f32, Dynamic, 1>>;
  using CVec_f32   = Eigen::Ref<const Matrix<std::complex<f32>, Dynamic, 1>>;
  using Mat3_f32   = Eigen::Ref<const Matrix<f32, Dynamic, 3>>;
  using Bool_T     = Eigen::Ref<const Matrix<bool, Dynamic, 1>>;
  using Modes_f32  = Eigen::Ref<const Matrix<f32, Dynamic, Dynamic>>;
  using MatDyn_f32 = Eigen::Ref<const Matrix<f32, Dynamic, Dynamic, RowMajor>>;

  using R0_f64     = Eigen::Ref<const Matrix<f64, Dynamic, 3, RowMajor>>;
  using Vec_f64    = Eigen::Ref<const Matrix<f64, Dynamic, 1>>;
  using CVec_f64   = Eigen::Ref<const Matrix<std::complex<f64>, Dynamic, 1>>;
  using Mat3_f64   = Eigen::Ref<const Matrix<f64, Dynamic, 3>>;
  using Modes_f64  = Eigen::Ref<const Matrix<f64, Dynamic, Dynamic>>;
  using MatDyn_f64 = Eigen::Ref<const Matrix<f64, Dynamic, Dynamic, RowMajor>>;

  m.def("solve_mri_f32",
    [](R0_f32 r0, Vec_f32 T1, Vec_f32 T2, Vec_f32 delta_B,
       const f32 &M0, const f32 &gamma,
       CVec_f32 rf_all, Mat3_f32 G_all, Vec_f32 dt, Bool_T regime_idx,
       CVec_f32 Mxy_initial, Vec_f32 Mz_initial,
       Modes_f32 modes, MatDyn_f32 weights, bool has_traj,
       int order, Vec_f32 Bz_old_init, std::complex<f32> rf_old_init,
       bool store_history) {
      return solve_mri_dispatch<f32>(order, r0, T1, T2, delta_B, M0, gamma,
                                     rf_all, G_all, dt, regime_idx,
                                     Mxy_initial, Mz_initial,
                                     modes, weights, has_traj,
                                     Bz_old_init, rf_old_init, store_history);
    },
    py::arg("r0"), py::arg("T1"), py::arg("T2"), py::arg("delta_B"),
    py::arg("M0"), py::arg("gamma"), py::arg("rf_all"), py::arg("G_all"),
    py::arg("dt"), py::arg("regime_idx"), py::arg("Mxy_initial"),
    py::arg("Mz_initial"), py::arg("modes"), py::arg("weights"),
    py::arg("has_traj"),
    py::arg("order"), py::arg("Bz_old_init"), py::arg("rf_old_init"),
    py::arg("store_history") = false);

  m.def("solve_mri_f64",
    [](R0_f64 r0, Vec_f64 T1, Vec_f64 T2, Vec_f64 delta_B,
       const f64 &M0, const f64 &gamma,
       CVec_f64 rf_all, Mat3_f64 G_all, Vec_f64 dt, Bool_T regime_idx,
       CVec_f64 Mxy_initial, Vec_f64 Mz_initial,
       Modes_f64 modes, MatDyn_f64 weights, bool has_traj,
       int order, Vec_f64 Bz_old_init, std::complex<f64> rf_old_init,
       bool store_history) {
      return solve_mri_dispatch<f64>(order, r0, T1, T2, delta_B, M0, gamma,
                                     rf_all, G_all, dt, regime_idx,
                                     Mxy_initial, Mz_initial,
                                     modes, weights, has_traj,
                                     Bz_old_init, rf_old_init, store_history);
    },
    py::arg("r0"), py::arg("T1"), py::arg("T2"), py::arg("delta_B"),
    py::arg("M0"), py::arg("gamma"), py::arg("rf_all"), py::arg("G_all"),
    py::arg("dt"), py::arg("regime_idx"), py::arg("Mxy_initial"),
    py::arg("Mz_initial"), py::arg("modes"), py::arg("weights"),
    py::arg("has_traj"),
    py::arg("order"), py::arg("Bz_old_init"), py::arg("rf_old_init"),
    py::arg("store_history") = false);
}
