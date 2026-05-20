// GPU implementation of the FEelMRI Bloch kernel.
//
// One templated `__global__` covers six instantiations:
//   T in {float, double} x Order in {0, 2, 4}.
//
// Order = 0: Cayley-Klein on end-of-step field (matches the historical
//            single-order GPU kernel byte-for-byte).
// Order = 2: 2nd-order Magnus (trapezoidal omega average).
// Order = 4: 2nd-order Magnus + linear-interpolation Omega_2 commutator
//            correction (KomaMRI's "BlochMagnus4" naming; globally still
//            O(dt^2)).
//
// The CPU kernel at cpp/feelmri/BlochSimulator.cpp is the math source of
// truth; this file mirrors that algebra line-for-line so the numerical
// agreement test in tests/test_bloch_gpu.py holds to float32 ulps for
// Order = 0 and to within 1e-12 in float64.
//
// One thread = one mesh node. The per-time-step loop runs serially
// inside the thread; the trajectory update (r = r_rest + modes @
// weights[i]) is fused into the loop so displaced positions never
// round-trip through global memory. For Order > 0 the per-thread
// `Bz_old` and shared `rf_old` are carried across the time loop in
// registers and written back at kernel exit so the Python BlochSolver
// can stitch them across sequence blocks.

#include "BlochSimulator_gpu.hpp"
#include "../runtime/device.hpp"
#include "../runtime/device_buffer.hpp"
#include "../runtime/device_init.hpp"

#include <complex>
#include <cstring>

namespace {

using feelmri::gpu::complex_t;
using feelmri::gpu::make_complex;

// Type-dispatched math intrinsics. The kernel uses these instead of the
// CUDA / HIP runtime functions directly so float and double share one
// source. Selection is at compile time via specialisation.
template <typename T> struct fp_math;
template <> struct fp_math<float> {
  __device__ static float  exp_(float x)  { return expf(x); }
  __device__ static float  cos_(float x)  { return cosf(x); }
  __device__ static float  sin_(float x)  { return sinf(x); }
  __device__ static float  sqrt_(float x) { return sqrtf(x); }
  __device__ static void   sincos_(float x, float* sn, float* cs) { __sincosf(x, sn, cs); }
  static constexpr float SMALL_THETA = 1.0e-3f;
};
template <> struct fp_math<double> {
  __device__ static double exp_(double x)  { return exp(x); }
  __device__ static double cos_(double x)  { return cos(x); }
  __device__ static double sin_(double x)  { return sin(x); }
  __device__ static double sqrt_(double x) { return sqrt(x); }
  __device__ static void   sincos_(double x, double* sn, double* cs) { sincos(x, sn, cs); }
  static constexpr double SMALL_THETA = 1.0e-6;
};

// Complex helpers — templated equivalents of the original cf_* funcs.
// `complex_t<T>` is feelmri::gpu::complex_f32_t / complex_f64_t (both
// POD { T x; T y; } structs from the cuComplex / hipComplex headers).

template <typename T>
__device__ __forceinline__ complex_t<T> cx_make(T re, T im) {
  return make_complex<T>(re, im);
}

template <typename T>
__device__ __forceinline__ complex_t<T> cx_conj(complex_t<T> a) {
  return make_complex<T>(a.x, -a.y);
}

template <typename T>
__device__ __forceinline__ T cx_norm2(complex_t<T> a) {
  return a.x * a.x + a.y * a.y;
}

template <typename T>
__device__ __forceinline__ complex_t<T> cx_mul(complex_t<T> a, complex_t<T> b) {
  return make_complex<T>(a.x * b.x - a.y * b.y,
                          a.x * b.y + a.y * b.x);
}

template <typename T>
__device__ __forceinline__ complex_t<T> cx_mul_real(complex_t<T> a, T s) {
  return make_complex<T>(a.x * s, a.y * s);
}

// =====================================================================
// Main kernel
// =====================================================================

template <typename T, int Order>
__global__ void bloch_kernel(
  const T*               __restrict__ r0,             // (n_pos, 3) row-major
  const T*               __restrict__ T1,
  const T*               __restrict__ T2,
  const T*               __restrict__ delta_B,
  T                                   M0,
  T                                   gamma,
  const complex_t<T>*    __restrict__ rf_all,         // (n_time,)
  const T*               __restrict__ G_all,          // (n_time, 3) row-major
  const T*               __restrict__ dt,             // (n_time,)
  const complex_t<T>*    __restrict__ Mxy_initial,
  const T*               __restrict__ Mz_initial,
  const T*               __restrict__ modes_x,        // (n_pos, n_modes) row-major
  const T*               __restrict__ modes_y,
  const T*               __restrict__ modes_z,
  const T*               __restrict__ weights,        // (n_time, n_modes) row-major
  int                                 has_traj,
  int                                 n_modes,
  int                                 n_pos,
  int                                 n_time,
  const T*               __restrict__ Bz_old_init,    // (n_pos,) or unused if Order==0
  complex_t<T>                        rf_old_init,
  complex_t<T>*          __restrict__ Mxy_last,       // (n_pos,)
  T*                     __restrict__ Mz_last,        // (n_pos,)
  T*                     __restrict__ Bz_old_final,   // (n_pos,) or unused if Order==0
  complex_t<T>*          __restrict__ rf_old_final    // scalar; only thread 0 writes
) {
  using FM = fp_math<T>;
  using C  = complex_t<T>;

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_pos) {
    return;
  }

  C Mxy_prev = Mxy_initial[p];
  T Mz_prev  = Mz_initial[p];

  const T r0x = r0[3 * p + 0];
  const T r0y = r0[3 * p + 1];
  const T r0z = r0[3 * p + 2];
  const T inv_T1 = T(1) / T1[p];
  const T inv_T2 = T(1) / T2[p];
  const T dB     = delta_B[p];

  // Magnus state. For Order == 0 these are not read.
  T Bz_old_p = (Order > 0) ? Bz_old_init[p] : T(0);
  C rf_old   = rf_old_init;

  for (int i = 0; i < n_time - 1; ++i) {
    const T dt_i = dt[i + 1];
    const T e1   = FM::exp_(-dt_i * inv_T1);
    const T e2   = FM::exp_(-dt_i * inv_T2);

    // Trajectory update (mode-weight GEMV folded into the per-node thread).
    T cx = r0x, cy = r0y, cz = r0z;
    if (has_traj) {
      const T* wrow  = weights + (i + 1) * n_modes;
      const T* mxrow = modes_x + p * n_modes;
      const T* myrow = modes_y + p * n_modes;
      const T* mzrow = modes_z + p * n_modes;
      for (int m = 0; m < n_modes; ++m) {
        const T wm = wrow[m];
        cx += mxrow[m] * wm;
        cy += myrow[m] * wm;
        cz += mzrow[m] * wm;
      }
    }

    const C rf_new = rf_all[i + 1];
    const T Gx = G_all[3 * (i + 1) + 0];
    const T Gy = G_all[3 * (i + 1) + 1];
    const T Gz = G_all[3 * (i + 1) + 2];
    const T Bz_new = cx * Gx + cy * Gy + cz * Gz + dB;

    // Build spinor (alpha, beta) for the rotation operator.
    C alpha, beta;

    if constexpr (Order == 0) {
      // Cayley-Klein on end-of-step field (unchanged from M1).
      const T rf2     = cx_norm2<T>(rf_new);
      T Bnorm         = FM::sqrt_(Bz_new * Bz_new + rf2);
      if (Bnorm < T(1e-12)) {
        Bnorm = T(1e-12);
      }
      const T nz       = Bz_new / Bnorm;
      const T kappa    = -T(0.5) * gamma * dt_i;
      const T half_phi = Bnorm * kappa;
      const T c        = FM::cos_(half_phi);
      const T s        = FM::sin_(half_phi);
      alpha = cx_make<T>(c, -nz * s);
      // nxy = rf_new / Bnorm, beta = -i * nxy * s.
      // -i * (a + ib) * s = (b*s) - i*(a*s)  =>  (b*s, -a*s)
      beta = cx_make<T>(rf_new.y * s / Bnorm, -rf_new.x * s / Bnorm);
    } else {
      // Magnus orders 2 / 4: build the rotation-angle vector
      // theta = (theta_xy_real, theta_xy_imag, theta_z) from the field.
      const T m2_scale = -T(0.5) * gamma * dt_i;
      C theta_xy = cx_make<T>(
        m2_scale * (rf_old.x + rf_new.x),
        m2_scale * (rf_old.y + rf_new.y));
      T theta_z = m2_scale * (Bz_old_p + Bz_new);

      if constexpr (Order == 4) {
        // Linear-interpolation Omega_2 commutator:
        //   theta_xy -= i * gamma^2 * dt^2 / 12 * (rf_new * Bz_old - rf_old * Bz_new)
        //   theta_z  -= gamma^2 * dt^2 / 12 * Im(conj(rf_old) * rf_new)
        const T m4_scale = gamma * gamma * dt_i * dt_i / T(12);
        const T corr_x = m4_scale * (rf_new.x * Bz_old_p - rf_old.x * Bz_new);
        const T corr_y = m4_scale * (rf_new.y * Bz_old_p - rf_old.y * Bz_new);
        // theta_xy -= i * (corr_x + i*corr_y) = theta_xy - (i*corr_x - corr_y)
        //          = theta_xy.x + corr_y + i*(theta_xy.y - corr_x)
        theta_xy.x = theta_xy.x + corr_y;
        theta_xy.y = theta_xy.y - corr_x;
        // Im(conj(rf_old) * rf_new) = rf_old.x*rf_new.y - rf_old.y*rf_new.x
        const T im_corr = m4_scale * (rf_old.x * rf_new.y - rf_old.y * rf_new.x);
        theta_z -= im_corr;
      }

      // Spinor build. Use Taylor for sin(theta/2)/theta near zero to
      // avoid 0/0 — matches the CPU kernel's small-angle branch.
      const T theta_sq = theta_xy.x * theta_xy.x
                       + theta_xy.y * theta_xy.y
                       + theta_z   * theta_z;
      const T theta    = FM::sqrt_(theta_sq);
      T c, half_sinc;
      if (theta < FM::SMALL_THETA) {
        const T t2 = theta_sq;
        c         = T(1)   - t2 * T(0.125)  + t2 * t2 / T(384);
        half_sinc = T(0.5) - t2 / T(48)     + t2 * t2 / T(3840);
      } else {
        const T half = theta * T(0.5);
        c         = FM::cos_(half);
        half_sinc = FM::sin_(half) / theta;
      }
      alpha = cx_make<T>(c, -theta_z * half_sinc);
      // beta = -i * theta_xy * half_sinc
      beta  = cx_make<T>( theta_xy.y * half_sinc, -theta_xy.x * half_sinc);
    }

    // Cayley-Klein rotation of M (identical across all orders).
    const C conj_a = cx_conj<T>(alpha);
    const T a2     = cx_norm2<T>(alpha);
    const T b2     = cx_norm2<T>(beta);
    const C Mxy_prev_conj = cx_conj<T>(Mxy_prev);

    // Mxy_new = 2 conj(alpha) beta Mz + conj(alpha)^2 Mxy - beta^2 conj(Mxy)
    C term1 = cx_mul<T>(conj_a, beta);
    term1 = cx_mul_real<T>(term1, T(2) * Mz_prev);
    const C term2 = cx_mul<T>(cx_mul<T>(conj_a, conj_a), Mxy_prev);
    const C term3 = cx_mul<T>(cx_mul<T>(beta, beta), Mxy_prev_conj);
    C Mxy_new = cx_make<T>(term1.x - term3.x + term2.x,
                            term1.y - term3.y + term2.y);

    // Mz_new = (|alpha|^2 - |beta|^2) Mz - 2 Re(alpha * beta * conj(Mxy))
    const C re_prod = cx_mul<T>(cx_mul<T>(alpha, beta), Mxy_prev_conj);
    const T Mz_new  = (a2 - b2) * Mz_prev - T(2) * re_prod.x;

    // T1 / T2 relaxation.
    Mxy_new = cx_mul_real<T>(Mxy_new, e2);
    const T Mz_relaxed = Mz_new * e1 + (T(1) - e1) * M0;

    Mxy_prev = Mxy_new;
    Mz_prev  = Mz_relaxed;

    if constexpr (Order > 0) {
      Bz_old_p = Bz_new;
      rf_old   = rf_new;
    }
  }

  // Final-state write (single column).
  Mxy_last[p] = Mxy_prev;
  Mz_last [p] = Mz_prev;
  if constexpr (Order > 0) {
    Bz_old_final[p] = Bz_old_p;
    if (p == 0) {
      *rf_old_final = rf_old;
    }
  }
}

// =====================================================================
// Templated host-side launch implementation
// =====================================================================

template <typename T>
int solve_mri_gpu_impl(
  const T* r0,
  const T* T1,
  const T* T2,
  const T* delta_B,
  T M0,
  T gamma,
  const std::complex<T>* rf_all,
  const T* G_all,
  const T* dt,
  const std::complex<T>* Mxy_initial,
  const T* Mz_initial,
  const T* modes_x,
  const T* modes_y,
  const T* modes_z,
  const T* weights,
  int has_traj,
  int n_modes,
  int n_pos,
  int n_time,
  int order,
  const T* Bz_old_init_host,
  std::complex<T> rf_old_init,
  std::complex<T>* Mxy_last,
  T* Mz_last,
  T* Bz_old_final_host,
  std::complex<T>* rf_old_final
) {
  try {
    using feelmri::DeviceBuffer;
    using feelmri::check_device;
    using C = complex_t<T>;

    const C* rf_all_cf       = reinterpret_cast<const C*>(rf_all);
    const C* Mxy_initial_cf  = reinterpret_cast<const C*>(Mxy_initial);
    C*       Mxy_last_cf     = reinterpret_cast<C*>(Mxy_last);

    static DeviceBuffer<T>  d_r0, d_T1, d_T2, d_dB;
    static DeviceBuffer<C>  d_rf, d_Mxy_init;
    static DeviceBuffer<T>  d_G, d_dt, d_Mz_init;
    static DeviceBuffer<T>  d_mx, d_my, d_mz, d_w;
    static DeviceBuffer<C>  d_Mxy_last;
    static DeviceBuffer<T>  d_Mz_last;
    // Magnus state buffers (allocated only when order > 0 callers run).
    static DeviceBuffer<T>  d_Bz_old;
    static DeviceBuffer<C>  d_rf_old_final;

    d_r0.resize(3 * n_pos);
    d_T1.resize(n_pos);
    d_T2.resize(n_pos);
    d_dB.resize(n_pos);
    d_rf.resize(n_time);
    d_G.resize(3 * n_time);
    d_dt.resize(n_time);
    d_Mxy_init.resize(n_pos);
    d_Mz_init.resize(n_pos);
    d_Mxy_last.resize(n_pos);
    d_Mz_last.resize(n_pos);

    const std::size_t mode_count = static_cast<std::size_t>(n_pos) * n_modes;
    const std::size_t w_count    = static_cast<std::size_t>(n_time) * n_modes;
    d_mx.resize(has_traj ? mode_count : 0);
    d_my.resize(has_traj ? mode_count : 0);
    d_mz.resize(has_traj ? mode_count : 0);
    d_w .resize(has_traj ? w_count    : 0);

    if (order > 0) {
      d_Bz_old.resize(n_pos);
      d_rf_old_final.resize(1);
    }

    auto stage = [](auto& buf, const auto* src, std::size_t n) {
      if (n == 0) return;
      using ElemT = typename std::remove_pointer<decltype(buf.host())>::type;
      std::memcpy(buf.host(), src, n * sizeof(ElemT));
      buf.upload_async(0);
    };

    stage(d_r0,       r0,             3 * n_pos);
    stage(d_T1,       T1,             n_pos);
    stage(d_T2,       T2,             n_pos);
    stage(d_dB,       delta_B,        n_pos);
    stage(d_rf,       rf_all_cf,      n_time);
    stage(d_G,        G_all,          3 * n_time);
    stage(d_dt,       dt,             n_time);
    stage(d_Mxy_init, Mxy_initial_cf, n_pos);
    stage(d_Mz_init,  Mz_initial,     n_pos);
    if (has_traj) {
      stage(d_mx, modes_x, mode_count);
      stage(d_my, modes_y, mode_count);
      stage(d_mz, modes_z, mode_count);
      stage(d_w,  weights, w_count);
    }
    if (order > 0) {
      stage(d_Bz_old, Bz_old_init_host, n_pos);
    }

    constexpr int kBlock = 256;
    const int grid = (n_pos + kBlock - 1) / kBlock;

    const C rf_old_init_dev = reinterpret_cast<const C&>(rf_old_init);

    // Order dispatch: compile-time selection of the right template
    // instantiation. nvcc emits all three kernels regardless; the
    // switch costs nothing at runtime.
    switch (order) {
      case 0:
        bloch_kernel<T, 0><<<grid, kBlock>>>(
          d_r0.device(), d_T1.device(), d_T2.device(), d_dB.device(),
          M0, gamma,
          d_rf.device(), d_G.device(), d_dt.device(),
          d_Mxy_init.device(), d_Mz_init.device(),
          d_mx.device(), d_my.device(), d_mz.device(), d_w.device(),
          has_traj, n_modes, n_pos, n_time,
          /*Bz_old_init=*/nullptr, rf_old_init_dev,
          d_Mxy_last.device(), d_Mz_last.device(),
          /*Bz_old_final=*/nullptr, /*rf_old_final=*/nullptr);
        break;
      case 2:
        bloch_kernel<T, 2><<<grid, kBlock>>>(
          d_r0.device(), d_T1.device(), d_T2.device(), d_dB.device(),
          M0, gamma,
          d_rf.device(), d_G.device(), d_dt.device(),
          d_Mxy_init.device(), d_Mz_init.device(),
          d_mx.device(), d_my.device(), d_mz.device(), d_w.device(),
          has_traj, n_modes, n_pos, n_time,
          d_Bz_old.device(), rf_old_init_dev,
          d_Mxy_last.device(), d_Mz_last.device(),
          d_Bz_old.device(), d_rf_old_final.device());
        break;
      case 4:
        bloch_kernel<T, 4><<<grid, kBlock>>>(
          d_r0.device(), d_T1.device(), d_T2.device(), d_dB.device(),
          M0, gamma,
          d_rf.device(), d_G.device(), d_dt.device(),
          d_Mxy_init.device(), d_Mz_init.device(),
          d_mx.device(), d_my.device(), d_mz.device(), d_w.device(),
          has_traj, n_modes, n_pos, n_time,
          d_Bz_old.device(), rf_old_init_dev,
          d_Mxy_last.device(), d_Mz_last.device(),
          d_Bz_old.device(), d_rf_old_final.device());
        break;
      default:
        throw std::runtime_error(
          "solve_mri_gpu: order must be 0, 2, or 4");
    }

    check_device(feelmri::gpu::get_last_error(), "bloch_kernel launch");
    d_Mxy_last.download_async(0);
    d_Mz_last.download_async(0);
    if (order > 0) {
      d_Bz_old.download_async(0);
      d_rf_old_final.download_async(0);
    }
    check_device(feelmri::gpu::stream_synchronize(0), "stream sync");

    std::memcpy(Mxy_last_cf, d_Mxy_last.host(),
                static_cast<std::size_t>(n_pos) * sizeof(C));
    std::memcpy(Mz_last, d_Mz_last.host(),
                static_cast<std::size_t>(n_pos) * sizeof(T));
    if (order > 0) {
      std::memcpy(Bz_old_final_host, d_Bz_old.host(),
                  static_cast<std::size_t>(n_pos) * sizeof(T));
      std::memcpy(rf_old_final, d_rf_old_final.host(), sizeof(C));
    } else {
      // Match the CPU path's "order=0 returns zero state" convention.
      std::memset(Bz_old_final_host, 0,
                  static_cast<std::size_t>(n_pos) * sizeof(T));
      *rf_old_final = std::complex<T>(0, 0);
    }
    return 0;
  } catch (const std::exception& e) {
    static thread_local std::string msg;
    msg = e.what();
    return -1;
  }
}

}  // namespace

// =====================================================================
// extern "C" entry points
// =====================================================================

extern "C" int feelmri_solve_mri_gpu_f32(
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
) {
  return solve_mri_gpu_impl<float>(
    r0, T1, T2, delta_B, M0, gamma, rf_all, G_all, dt,
    Mxy_initial, Mz_initial, modes_x, modes_y, modes_z, weights,
    has_traj, n_modes, n_pos, n_time, order,
    Bz_old_init, rf_old_init,
    Mxy_last, Mz_last, Bz_old_final, rf_old_final);
}

extern "C" int feelmri_solve_mri_gpu_f64(
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
) {
  return solve_mri_gpu_impl<double>(
    r0, T1, T2, delta_B, M0, gamma, rf_all, G_all, dt,
    Mxy_initial, Mz_initial, modes_x, modes_y, modes_z, weights,
    has_traj, n_modes, n_pos, n_time, order,
    Bz_old_init, rf_old_init,
    Mxy_last, Mz_last, Bz_old_final, rf_old_final);
}
