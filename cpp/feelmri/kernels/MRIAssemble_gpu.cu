// GPU implementation of the FEelMRI MRI signal accumulation kernel.
// Mirrors the CPU `SignalAssembler::signal_sum` / `signal_nodal` math
// line-for-line so numerical agreement between device='cpu' and
// device='gpu' is exact up to float32 round-off.
//
// One thread = one k-space sample. The per-node reduction runs serially
// inside the thread; nodes are read with the broadcast pattern (all
// threads in a warp read the same node index simultaneously, hitting L1
// after the first miss). The trajectory update r = r_rest + modes @
// weights[row] is fused into the inner loop so the displaced positions
// never round-trip through global memory.

#include "MRIAssemble_gpu.hpp"
#include "../runtime/device.hpp"
#include "../runtime/device_buffer.hpp"
#include "../runtime/device_init.hpp"

#include <complex>
#include <cstring>

// Vendor-neutral aliases (see kernels/BlochSimulator_gpu.cu for the
// matching pattern). Resolve to CUDA primitives under the default build
// and HIP under -DFEELMRI_GPU_BACKEND=hip.
using cuFloatComplex = feelmri::gpu::complex_f32_t;
#define mk_cf(re, im) feelmri::gpu::make_complex_f32((re), (im))

namespace {

  constexpr float kTwoPi = 6.28318530717958647692f;

  // Tunable parameters for the shared-memory tiled kernel below.
  //   BLOCK_THREADS = k-samples processed per block (one thread each).
  //   TILE_NODES    = quadrature/node points staged into shared mem per
  //                   inner iteration; cooperatively loaded by the first
  //                   TILE_NODES threads of the block.
  //   MAX_MODES     = compile-time bound on the POD mode count for the
  //                   shared-mem allocation; runtime n_modes <= MAX_MODES.
  // Shared memory budget (TILE_NODES=64, MAX_MODES=32):
  //   modes_x/y/z      3 * 64 * 32 * 4 =  24576 B
  //   nodes_x/y/z      3 * 64 *  1 * 4 =    768 B
  //   invT2, phi       2 * 64 *  1 * 4 =    512 B
  //   Mxy_re, Mxy_im   2 * 64 *  1 * 4 =    512 B
  //   total                              ~26.4 KB
  // Fits in the 48 KB default shared-mem carveout on Ada; with the
  // higher carveout set via cudaFuncSetAttribute, two blocks per SM
  // become reachable for higher occupancy.
  constexpr int kTileBlockThreads = 128;
  constexpr int kTileNodes        = 64;
  constexpr int kTileMaxModes     = 32;
  // Upper bound on the coil / velocity-encoding count handled by the
  // tiled fast path. Covers 4D-flow workloads (4 velocity encodings)
  // and multi-coil up to 4 channels. nv > 4 falls back to the original
  // atomic-add kernel.
  constexpr int kTileMaxNV        = 4;

  // ===================================================================
  // Sparse projection kernel: Mxy_q_weighted = f_wq * (S_csr * Mxy_nodes)
  // One thread per quadrature row; walks the row's CSR nonzeros (~4 for
  // tetra meshes) and accumulates per coil. The result is scaled by the
  // quadrature weight in place so the downstream signal kernel can use
  // the existing Mxy_q_weighted code path unchanged.
  // ===================================================================
  __global__ void mri_project_mxy_kernel_f32(
    const int*   __restrict__ row_ptr,        // (Q + 1,)
    const int*   __restrict__ col_idx,        // (nnz,)
    const float* __restrict__ values,         // (nnz,)
    const float* __restrict__ fwq,            // (Q,)
    const cuFloatComplex* __restrict__ Mxy_nodes,  // (N, nv) RowMajor
    int Q,
    int nv,
    cuFloatComplex* __restrict__ Mxy_q_weighted    // (Q, nv) RowMajor
  ) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q) {
      return;
    }
    const int rp     = row_ptr[q];
    const int rp_end = row_ptr[q + 1];
    const float w    = fwq[q];

    // Per-coil accumulator in registers (kTileMaxNV upper bound).
    float re[kTileMaxNV];
    float im[kTileMaxNV];
    #pragma unroll
    for (int v = 0; v < kTileMaxNV; ++v) {
      re[v] = 0.0f;
      im[v] = 0.0f;
    }

    for (int i = rp; i < rp_end; ++i) {
      const int   n = col_idx[i];
      const float c = values[i];
      for (int v = 0; v < nv; ++v) {
        const cuFloatComplex m = Mxy_nodes[n * nv + v];
        re[v] = fmaf(c, m.x, re[v]);
        im[v] = fmaf(c, m.y, im[v]);
      }
    }

    for (int v = 0; v < nv; ++v) {
      Mxy_q_weighted[q * nv + v] = mk_cf(w * re[v], w * im[v]);
    }
  }

  // ===================================================================
  // Shared-memory tiled signal kernel (nv up to kTileMaxNV coils /
  // velocity encodings).
  //
  // Each block of kTileBlockThreads threads handles kTileBlockThreads
  // independent k-samples. Inside the inner loop, the block
  // cooperatively stages a kTileNodes-chunk of quadrature data into
  // shared memory; every thread then iterates the tile locally, so
  // each global modes_q value is read once per BLOCK rather than once
  // per warp. Brings the dominant DRAM term on workloads where
  // modes_q does not fit in L2 (Q * n_modes >> 12 M scalars) down by
  // roughly kTileBlockThreads / warp_size.
  //
  // nv > 1 is handled by widening the shared-mem Mxy buffer to
  // (kTileNodes, kTileMaxNV) and accumulating per-coil into a
  // register-resident vector of complex; the per-coil multiply-add is
  // independent per coil so the GPU lane utilisation stays high. The
  // 4D-flow case (nv = 4 velocity encodings) is the dominant
  // multi-coil workload at FEelMRI's scale.
  // ===================================================================
  __global__ void mri_signal_tiled_kernel_f32(
    const float* __restrict__ nodes_x0,
    const float* __restrict__ nodes_x1,
    const float* __restrict__ nodes_x2,
    const float* __restrict__ invT2,
    const float* __restrict__ phi,
    const cuFloatComplex* __restrict__ Mxy_nodes,  // (n_nodes, nv) RowMajor
    const float* __restrict__ modes_x,
    const float* __restrict__ modes_y,
    const float* __restrict__ modes_z,
    const float* __restrict__ weights,
    int has_traj,
    int n_modes,
    int n_nodes,
    int nv,
    const float* __restrict__ kloc_x,
    const float* __restrict__ kloc_y,
    const float* __restrict__ kloc_z,
    const float* __restrict__ t,
    int n_samples,
    cuFloatComplex* __restrict__ signal_out
  ) {
    __shared__ float s_x0[kTileNodes];
    __shared__ float s_x1[kTileNodes];
    __shared__ float s_x2[kTileNodes];
    __shared__ float s_invT2[kTileNodes];
    __shared__ float s_phi  [kTileNodes];
    // Mxy interleaved per quadrature point: layout [tile][coil].
    // For nv = 1 only the first column is touched; the size is
    // bounded at compile time by kTileMaxNV so shared-mem allocation
    // stays static.
    __shared__ float s_Mxy_re[kTileNodes * kTileMaxNV];
    __shared__ float s_Mxy_im[kTileNodes * kTileMaxNV];
    __shared__ float s_mx[kTileNodes * kTileMaxModes];
    __shared__ float s_my[kTileNodes * kTileMaxModes];
    __shared__ float s_mz[kTileNodes * kTileMaxModes];

    const int row = blockIdx.x * kTileBlockThreads + threadIdx.x;
    const int tid = threadIdx.x;
    const bool valid = (row < n_samples);

    // Per-thread row constants. Cache weights[row] in registers so the
    // inner mode-add loop only touches shared memory.
    float tij = 0.0f, kx = 0.0f, ky = 0.0f, kz = 0.0f;
    float w_local[kTileMaxModes];
    if (valid) {
      tij = t[row];
      kx  = kTwoPi * kloc_x[row];
      ky  = kTwoPi * kloc_y[row];
      kz  = kTwoPi * kloc_z[row];
      if (has_traj) {
        const float* wr = weights + row * n_modes;
        for (int m = 0; m < n_modes; ++m) {
          w_local[m] = wr[m];
        }
      }
    }

    // Per-coil accumulator in registers (split into real / imag arrays
    // so nvcc keeps them in registers rather than spilling).
    float accum_re[kTileMaxNV];
    float accum_im[kTileMaxNV];
    #pragma unroll
    for (int v = 0; v < kTileMaxNV; ++v) {
      accum_re[v] = 0.0f;
      accum_im[v] = 0.0f;
    }

    for (int q_start = 0; q_start < n_nodes; q_start += kTileNodes) {
      const int q_count = min(kTileNodes, n_nodes - q_start);

      // Cooperative load: only the first q_count threads of the block
      // participate (kTileNodes <= kTileBlockThreads).
      if (tid < q_count) {
        const int g = q_start + tid;
        s_x0[tid]    = nodes_x0[g];
        s_x1[tid]    = nodes_x1[g];
        s_x2[tid]    = nodes_x2[g];
        s_invT2[tid] = invT2[g];
        s_phi[tid]   = phi[g];
        // Load nv coils of Mxy for this tile slot. Indices follow the
        // host RowMajor layout: Mxy_nodes[g * nv + v].
        for (int v = 0; v < nv; ++v) {
          const cuFloatComplex mvg = Mxy_nodes[g * nv + v];
          s_Mxy_re[tid * kTileMaxNV + v] = mvg.x;
          s_Mxy_im[tid * kTileMaxNV + v] = mvg.y;
        }
        if (has_traj) {
          const float* mxg = modes_x + g * n_modes;
          const float* myg = modes_y + g * n_modes;
          const float* mzg = modes_z + g * n_modes;
          for (int m = 0; m < n_modes; ++m) {
            s_mx[tid * kTileMaxModes + m] = mxg[m];
            s_my[tid * kTileMaxModes + m] = myg[m];
            s_mz[tid * kTileMaxModes + m] = mzg[m];
          }
        }
      }
      __syncthreads();

      if (valid) {
        for (int q = 0; q < q_count; ++q) {
          float cx = s_x0[q], cy = s_x1[q], cz = s_x2[q];
          if (has_traj) {
            const float* mxq = s_mx + q * kTileMaxModes;
            const float* myq = s_my + q * kTileMaxModes;
            const float* mzq = s_mz + q * kTileMaxModes;
            for (int m = 0; m < n_modes; ++m) {
              const float wm = w_local[m];
              cx = fmaf(mxq[m], wm, cx);
              cy = fmaf(myq[m], wm, cy);
              cz = fmaf(mzq[m], wm, cz);
            }
          }
          const float phase = s_phi[q] * tij - (kx * cx + ky * cy + kz * cz);
          const float mag   = expf(-tij * s_invT2[q]);
          float sn, cs;
          __sincosf(phase, &sn, &cs);
          const float fr = mag * cs;
          const float fi = mag * sn;
          // accum[v] += fourier * Mxy[q, v] for each active coil.
          const float* qre = s_Mxy_re + q * kTileMaxNV;
          const float* qim = s_Mxy_im + q * kTileMaxNV;
          for (int v = 0; v < nv; ++v) {
            const float mvre = qre[v];
            const float mvim = qim[v];
            accum_re[v] = fmaf(fr, mvre, accum_re[v]) - fi * mvim;
            accum_im[v] = fmaf(fr, mvim, accum_im[v]) + fi * mvre;
          }
        }
      }
      __syncthreads();
    }

    if (valid) {
      for (int v = 0; v < nv; ++v) {
        signal_out[row * nv + v] = mk_cf(accum_re[v], accum_im[v]);
      }
    }
  }

  __global__ void mri_signal_kernel_f32(
    const float* __restrict__ nodes_x0,
    const float* __restrict__ nodes_x1,
    const float* __restrict__ nodes_x2,
    const float* __restrict__ invT2,
    const float* __restrict__ phi,
    const cuFloatComplex* __restrict__ Mxy_nodes,
    const float* __restrict__ modes_x,
    const float* __restrict__ modes_y,
    const float* __restrict__ modes_z,
    const float* __restrict__ weights,
    int has_traj,
    int n_modes,
    int n_nodes,
    int nv,
    const float* __restrict__ kloc_x,
    const float* __restrict__ kloc_y,
    const float* __restrict__ kloc_z,
    const float* __restrict__ t,
    int n_samples,
    cuFloatComplex* __restrict__ signal_out
  ) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_samples) {
      return;
    }

    const float tij = t[row];
    const float kx  = kTwoPi * kloc_x[row];
    const float ky  = kTwoPi * kloc_y[row];
    const float kz  = kTwoPi * kloc_z[row];

    // Up to 4 coils accumulated in registers; for nv > 4 fall back to
    // global-memory writes per coil (rare path in MRI workloads).
    constexpr int kMaxRegCoils = 4;
    cuFloatComplex accum[kMaxRegCoils];
    for (int v = 0; v < kMaxRegCoils && v < nv; ++v) {
      accum[v] = mk_cf(0.0f, 0.0f);
    }

    // Pointer to this row's weights (used per node when has_traj).
    const float* w_row = has_traj ? (weights + row * n_modes) : nullptr;

    for (int p = 0; p < n_nodes; ++p) {
      // Trajectory update.
      float cx = nodes_x0[p];
      float cy = nodes_x1[p];
      float cz = nodes_x2[p];
      if (has_traj) {
        const float* mxrow = modes_x + p * n_modes;
        const float* myrow = modes_y + p * n_modes;
        const float* mzrow = modes_z + p * n_modes;
        for (int m = 0; m < n_modes; ++m) {
          const float wm = w_row[m];
          cx = fmaf(mxrow[m], wm, cx);
          cy = fmaf(myrow[m], wm, cy);
          cz = fmaf(mzrow[m], wm, cz);
        }
      }

      const float phase = phi[p] * tij - (kx * cx + ky * cy + kz * cz);
      const float mag   = expf(-tij * invT2[p]);
      float sn, cs;
      __sincosf(phase, &sn, &cs);
      const cuFloatComplex fourier = mk_cf(mag * cs, mag * sn);

      // Per-coil multiply-accumulate.
      if (nv <= kMaxRegCoils) {
        for (int v = 0; v < nv; ++v) {
          const cuFloatComplex mv = Mxy_nodes[p * nv + v];
          accum[v].x = fmaf(fourier.x, mv.x, accum[v].x) - fourier.y * mv.y;
          accum[v].y = fmaf(fourier.x, mv.y, accum[v].y) + fourier.y * mv.x;
        }
      } else {
        // Multi-coil slow path: directly accumulate into global memory.
        for (int v = 0; v < nv; ++v) {
          const cuFloatComplex mv = Mxy_nodes[p * nv + v];
          const float ax = fourier.x * mv.x - fourier.y * mv.y;
          const float ay = fourier.x * mv.y + fourier.y * mv.x;
          cuFloatComplex& out = signal_out[row * nv + v];
          atomicAdd(&out.x, ax);
          atomicAdd(&out.y, ay);
        }
      }
    }

    if (nv <= kMaxRegCoils) {
      for (int v = 0; v < nv; ++v) {
        signal_out[row * nv + v] = accum[v];
      }
    }
  }

}  // namespace

// =====================================================================
// Fused projection + signal entry point.
//
// Independent device-state from feelmri_mri_signal_gpu_f32; this wrapper
// is the one used by the quadrature path. The CSR triple + f_wq stay
// resident across calls (they are derived from set_assembler /
// set_static_fields and never change during a phantom's lifetime); the
// per-call Mxy_nodes uploads its ~17 MB and the projection kernel
// builds the (Q, nv) weighted magnetisation directly on the device.
// =====================================================================
extern "C" int feelmri_mri_signal_with_projection_gpu_f32(
  const float* nodes_x0,
  const float* nodes_x1,
  const float* nodes_x2,
  const float* invT2,
  const float* phi,
  const std::complex<float>* Mxy_nodes,
  int S_n_nodes,
  const int*   S_row_ptr,
  const int*   S_col_idx,
  const float* S_values,
  int          S_nnz,
  const float* fwq,
  const float* modes_x,
  const float* modes_y,
  const float* modes_z,
  const float* weights,
  int has_traj,
  int n_modes,
  int n_qp,
  int nv,
  const float* kloc_x,
  const float* kloc_y,
  const float* kloc_z,
  const float* t,
  int n_samples,
  std::complex<float>* signal_out
) {
  try {
    using feelmri::DeviceBuffer;
    using feelmri::check_device;

    const cuFloatComplex* Mxy_nodes_cf =
        reinterpret_cast<const cuFloatComplex*>(Mxy_nodes);
    cuFloatComplex* signal_out_cf =
        reinterpret_cast<cuFloatComplex*>(signal_out);

    // Pointer-fingerprint cache for resident state. The CSR triple +
    // f_wq are derived from set_assembler / set_static_fields and stay
    // valid for the assembler's lifetime — uploading once per process
    // is the right granularity. Static fields (nodes / invT2 / phi /
    // modes) also fingerprint here; only Mxy_nodes / kspace / weights
    // / time are per-call.
    struct ProjCache {
      const void* nodes_x0 = nullptr;
      const void* nodes_x1 = nullptr;
      const void* nodes_x2 = nullptr;
      const void* invT2 = nullptr;
      const void* phi = nullptr;
      const void* modes_x = nullptr;
      const void* modes_y = nullptr;
      const void* modes_z = nullptr;
      const void* row_ptr = nullptr;
      const void* col_idx = nullptr;
      const void* values  = nullptr;
      const void* fwq     = nullptr;
      std::size_t n_qp     = 0;
      std::size_t S_nnz    = 0;
      int         n_modes  = 0;
    };
    static ProjCache cache;

    static DeviceBuffer<float>          d_nx0, d_nx1, d_nx2;
    static DeviceBuffer<float>          d_invT2, d_phi;
    static DeviceBuffer<int>            d_row_ptr, d_col_idx;
    static DeviceBuffer<float>          d_values, d_fwq;
    static DeviceBuffer<cuFloatComplex> d_Mxy_nodes;
    static DeviceBuffer<cuFloatComplex> d_Mxy;          // Mxy_q_weighted on device
    static DeviceBuffer<float>          d_mx, d_my, d_mz, d_w;
    static DeviceBuffer<float>          d_kx, d_ky, d_kz, d_t;
    static DeviceBuffer<cuFloatComplex> d_sig;

    d_nx0.resize(n_qp);
    d_nx1.resize(n_qp);
    d_nx2.resize(n_qp);
    d_invT2.resize(n_qp);
    d_phi.resize(n_qp);
    d_row_ptr.resize(static_cast<std::size_t>(n_qp) + 1);
    d_col_idx.resize(S_nnz);
    d_values.resize(S_nnz);
    d_fwq.resize(n_qp);
    d_Mxy_nodes.resize(static_cast<std::size_t>(S_n_nodes) * nv);
    d_Mxy.resize(static_cast<std::size_t>(n_qp) * nv);
    d_kx.resize(n_samples);
    d_ky.resize(n_samples);
    d_kz.resize(n_samples);
    d_t .resize(n_samples);
    d_sig.resize(static_cast<std::size_t>(n_samples) * nv);

    const std::size_t mode_count = static_cast<std::size_t>(n_qp) * n_modes;
    const std::size_t w_count    = static_cast<std::size_t>(n_samples) * n_modes;
    d_mx.resize(has_traj ? mode_count : 0);
    d_my.resize(has_traj ? mode_count : 0);
    d_mz.resize(has_traj ? mode_count : 0);
    d_w .resize(has_traj ? w_count    : 0);

    auto stage = [](auto& buf, const auto* src, std::size_t n) {
      if (n == 0) return;
      using ElemT = typename std::remove_pointer<decltype(buf.host())>::type;
      std::memcpy(buf.host(), src, n * sizeof(ElemT));
      buf.upload_async(0);
    };
    auto stage_if_changed = [&](auto& buf, const auto* src, std::size_t n,
                                  const void*& fingerprint) {
      const void* p = static_cast<const void*>(src);
      if (n == 0) {
        fingerprint = nullptr;
        return;
      }
      if (p == fingerprint && n == buf.size()) {
        return;
      }
      stage(buf, src, n);
      fingerprint = p;
    };

    // Invalidate fingerprints when the geometry size changes.
    if (cache.n_qp != static_cast<std::size_t>(n_qp) || cache.S_nnz != static_cast<std::size_t>(S_nnz)
        || cache.n_modes != n_modes) {
      cache.nodes_x0 = cache.nodes_x1 = cache.nodes_x2 = nullptr;
      cache.invT2 = cache.phi = nullptr;
      cache.row_ptr = cache.col_idx = cache.values = nullptr;
      cache.fwq = nullptr;
      cache.modes_x = cache.modes_y = cache.modes_z = nullptr;
    }
    cache.n_qp    = n_qp;
    cache.S_nnz   = S_nnz;
    cache.n_modes = n_modes;

    stage_if_changed(d_nx0,     nodes_x0,  n_qp,           cache.nodes_x0);
    stage_if_changed(d_nx1,     nodes_x1,  n_qp,           cache.nodes_x1);
    stage_if_changed(d_nx2,     nodes_x2,  n_qp,           cache.nodes_x2);
    stage_if_changed(d_invT2,   invT2,     n_qp,           cache.invT2);
    stage_if_changed(d_phi,     phi,       n_qp,           cache.phi);
    stage_if_changed(d_row_ptr, S_row_ptr, n_qp + 1,       cache.row_ptr);
    stage_if_changed(d_col_idx, S_col_idx, S_nnz,          cache.col_idx);
    stage_if_changed(d_values,  S_values,  S_nnz,          cache.values);
    stage_if_changed(d_fwq,     fwq,       n_qp,           cache.fwq);
    if (has_traj) {
      stage_if_changed(d_mx, modes_x, mode_count, cache.modes_x);
      stage_if_changed(d_my, modes_y, mode_count, cache.modes_y);
      stage_if_changed(d_mz, modes_z, mode_count, cache.modes_z);
    }

    // Per-call dynamic inputs.
    stage(d_Mxy_nodes, Mxy_nodes_cf,
          static_cast<std::size_t>(S_n_nodes) * nv);
    stage(d_kx, kloc_x,  n_samples);
    stage(d_ky, kloc_y,  n_samples);
    stage(d_kz, kloc_z,  n_samples);
    stage(d_t,  t,       n_samples);
    if (has_traj) stage(d_w, weights, w_count);

    // 1. Projection kernel: Mxy_q_weighted = f_wq * (S * Mxy_nodes).
    {
      constexpr int kProjBlock = 128;
      const int grid = (n_qp + kProjBlock - 1) / kProjBlock;
      mri_project_mxy_kernel_f32<<<grid, kProjBlock>>>(
        d_row_ptr.device(), d_col_idx.device(), d_values.device(),
        d_fwq.device(),
        d_Mxy_nodes.device(),
        n_qp, nv,
        d_Mxy.device());
      check_device(feelmri::gpu::get_last_error(),
                    "mri_project_mxy_kernel_f32 launch");
    }

    // 2. Signal kernel: same path as feelmri_mri_signal_gpu_f32 but with
    //    d_Mxy already populated by the projection above.
    const bool use_tiled = (nv <= kTileMaxNV) && (n_modes <= kTileMaxModes);
    if (use_tiled) {
      const int grid = (n_samples + kTileBlockThreads - 1) / kTileBlockThreads;
      mri_signal_tiled_kernel_f32<<<grid, kTileBlockThreads>>>(
        d_nx0.device(), d_nx1.device(), d_nx2.device(),
        d_invT2.device(), d_phi.device(),
        d_Mxy.device(),
        d_mx.device(), d_my.device(), d_mz.device(),
        d_w.device(),
        has_traj, n_modes, n_qp, nv,
        d_kx.device(), d_ky.device(), d_kz.device(), d_t.device(),
        n_samples,
        d_sig.device());
    } else {
      constexpr int kBlock = 128;
      const int grid = (n_samples + kBlock - 1) / kBlock;
      if (nv > kTileMaxNV) {
        check_device(feelmri::gpu::memset_async(
                       d_sig.device(), 0,
                       static_cast<std::size_t>(n_samples) * nv
                         * sizeof(cuFloatComplex), 0),
                      "fused signal: zero output");
      }
      mri_signal_kernel_f32<<<grid, kBlock>>>(
        d_nx0.device(), d_nx1.device(), d_nx2.device(),
        d_invT2.device(), d_phi.device(),
        d_Mxy.device(),
        d_mx.device(), d_my.device(), d_mz.device(),
        d_w.device(),
        has_traj, n_modes, n_qp, nv,
        d_kx.device(), d_ky.device(), d_kz.device(), d_t.device(),
        n_samples,
        d_sig.device());
    }
    check_device(feelmri::gpu::get_last_error(),
                  "signal kernel launch (fused)");
    d_sig.download_async(0);
    check_device(feelmri::gpu::stream_synchronize(0), "fused stream sync");

    std::memcpy(signal_out_cf, d_sig.host(),
                static_cast<std::size_t>(n_samples) * nv * sizeof(cuFloatComplex));
    return 0;
  } catch (const std::exception& e) {
    static thread_local std::string msg;
    msg = e.what();
    return -1;
  }
}

extern "C" int feelmri_mri_signal_gpu_f32(
  const float* nodes_x0,
  const float* nodes_x1,
  const float* nodes_x2,
  const float* invT2,
  const float* phi,
  const std::complex<float>* Mxy_nodes,
  const float* modes_x,
  const float* modes_y,
  const float* modes_z,
  const float* weights,
  int has_traj,
  int n_modes,
  int n_nodes,
  int nv,
  const float* kloc_x,
  const float* kloc_y,
  const float* kloc_z,
  const float* t,
  int n_samples,
  std::complex<float>* signal_out
) {
  try {
    using feelmri::DeviceBuffer;
    using feelmri::check_device;

    const cuFloatComplex* Mxy_nodes_cf =
        reinterpret_cast<const cuFloatComplex*>(Mxy_nodes);
    cuFloatComplex* signal_out_cf =
        reinterpret_cast<cuFloatComplex*>(signal_out);

    // Persistent device buffers. Function-scope statics survive across
    // calls and grow to the high-water-mark via DeviceBuffer::resize.
    // For static fields (nodes, invT2, phi, modes) the wrapper also
    // fingerprints the caller's host pointer + element count to skip
    // the H2D stage when the input is unchanged from the previous call
    // — this matters during spamm.py's 20-frame mri_signal loop where
    // the static fields are identical across all calls while only
    // Mxy_nodes (per-frame) and the k-space trajectory change.
    struct StaticCache {
      const void* nodes_x0  = nullptr;
      const void* nodes_x1  = nullptr;
      const void* nodes_x2  = nullptr;
      const void* invT2     = nullptr;
      const void* phi       = nullptr;
      const void* modes_x   = nullptr;
      const void* modes_y   = nullptr;
      const void* modes_z   = nullptr;
      std::size_t n_nodes   = 0;
      int         n_modes   = 0;
    };
    static StaticCache cache;

    static DeviceBuffer<float>          d_nx0, d_nx1, d_nx2;
    static DeviceBuffer<float>          d_invT2, d_phi;
    static DeviceBuffer<cuFloatComplex> d_Mxy;
    static DeviceBuffer<float>          d_mx, d_my, d_mz, d_w;
    static DeviceBuffer<float>          d_kx, d_ky, d_kz, d_t;
    static DeviceBuffer<cuFloatComplex> d_sig;

    d_nx0.resize(n_nodes);
    d_nx1.resize(n_nodes);
    d_nx2.resize(n_nodes);
    d_invT2.resize(n_nodes);
    d_phi.resize(n_nodes);
    d_Mxy.resize(static_cast<std::size_t>(n_nodes) * nv);
    d_kx.resize(n_samples);
    d_ky.resize(n_samples);
    d_kz.resize(n_samples);
    d_t .resize(n_samples);
    d_sig.resize(static_cast<std::size_t>(n_samples) * nv);

    const std::size_t mode_count = static_cast<std::size_t>(n_nodes) * n_modes;
    const std::size_t w_count    = static_cast<std::size_t>(n_samples) * n_modes;
    d_mx.resize(has_traj ? mode_count : 0);
    d_my.resize(has_traj ? mode_count : 0);
    d_mz.resize(has_traj ? mode_count : 0);
    d_w .resize(has_traj ? w_count    : 0);

    auto stage = [](auto& buf, const auto* src, std::size_t n) {
      if (n == 0) return;
      using ElemT = typename std::remove_pointer<decltype(buf.host())>::type;
      std::memcpy(buf.host(), src, n * sizeof(ElemT));
      buf.upload_async(0);
    };

    auto stage_if_changed = [&](auto& buf, const auto* src, std::size_t n,
                                 const void*& fingerprint) {
      const void* p = static_cast<const void*>(src);
      if (n == 0) {
        fingerprint = nullptr;
        return;
      }
      if (p == fingerprint && n == buf.size()) {
        return;  // identical to last call; device-side copy is current
      }
      stage(buf, src, n);
      fingerprint = p;
    };

    // Static fields: skip H2D when host pointer + size match the cache.
    const bool nodes_size_changed = (cache.n_nodes != static_cast<std::size_t>(n_nodes));
    const bool modes_layout_changed = (cache.n_modes != n_modes);
    if (nodes_size_changed || modes_layout_changed) {
      // Invalidate static fingerprints when geometry changes (e.g. a
      // different SignalAssembler instance reuses the global cache).
      cache.nodes_x0 = cache.nodes_x1 = cache.nodes_x2 = nullptr;
      cache.invT2 = cache.phi = nullptr;
      cache.modes_x = cache.modes_y = cache.modes_z = nullptr;
    }
    cache.n_nodes = n_nodes;
    cache.n_modes = n_modes;

    stage_if_changed(d_nx0,   nodes_x0,  n_nodes, cache.nodes_x0);
    stage_if_changed(d_nx1,   nodes_x1,  n_nodes, cache.nodes_x1);
    stage_if_changed(d_nx2,   nodes_x2,  n_nodes, cache.nodes_x2);
    stage_if_changed(d_invT2, invT2,     n_nodes, cache.invT2);
    stage_if_changed(d_phi,   phi,       n_nodes, cache.phi);
    if (has_traj) {
      stage_if_changed(d_mx, modes_x, mode_count, cache.modes_x);
      stage_if_changed(d_my, modes_y, mode_count, cache.modes_y);
      stage_if_changed(d_mz, modes_z, mode_count, cache.modes_z);
    }

    // Dynamic per-call inputs: always upload.
    stage(d_Mxy, Mxy_nodes_cf, static_cast<std::size_t>(n_nodes) * nv);
    stage(d_kx,  kloc_x,       n_samples);
    stage(d_ky,  kloc_y,       n_samples);
    stage(d_kz,  kloc_z,       n_samples);
    stage(d_t,   t,            n_samples);
    if (has_traj) stage(d_w, weights, w_count);

    // Atomic-add path in the multi-coil branch needs a zero-initialised
    // output. The single-coil fast path overwrites the buffer fully, so
    // the cudaMemsetAsync is cheap and avoids a kernel-level branch.
    if (nv > 4) {
      check_device(feelmri::gpu::memset_async(d_sig.device(), 0,
                                  static_cast<std::size_t>(n_samples) * nv
                                    * sizeof(cuFloatComplex), 0),
                  "MRIAssemble_gpu: signal out zero");
    }

    // Pick the kernel: the shared-memory tiled fast path applies when
    // the mode count and coil count both fit in the compile-time
    // shared-memory budget. The dominant workloads (POD-driven moving
    // tissue with n_modes in the 10-30 range, nv = 1 single-coil or
    // nv = 4 velocity encodings for 4D flow) hit this branch.
    const bool use_tiled = (nv <= kTileMaxNV) && (n_modes <= kTileMaxModes);
    if (use_tiled) {
      const int grid = (n_samples + kTileBlockThreads - 1) / kTileBlockThreads;
      mri_signal_tiled_kernel_f32<<<grid, kTileBlockThreads>>>(
        d_nx0.device(), d_nx1.device(), d_nx2.device(),
        d_invT2.device(), d_phi.device(),
        d_Mxy.device(),
        d_mx.device(), d_my.device(), d_mz.device(),
        d_w.device(),
        has_traj, n_modes, n_nodes, nv,
        d_kx.device(), d_ky.device(), d_kz.device(), d_t.device(),
        n_samples,
        d_sig.device());
    } else {
      constexpr int kBlock = 128;
      const int grid = (n_samples + kBlock - 1) / kBlock;
      mri_signal_kernel_f32<<<grid, kBlock>>>(
        d_nx0.device(), d_nx1.device(), d_nx2.device(),
        d_invT2.device(), d_phi.device(),
        d_Mxy.device(),
        d_mx.device(), d_my.device(), d_mz.device(),
        d_w.device(),
        has_traj, n_modes, n_nodes, nv,
        d_kx.device(), d_ky.device(), d_kz.device(), d_t.device(),
        n_samples,
        d_sig.device());
    }

    check_device(feelmri::gpu::get_last_error(), "mri_signal_kernel_f32 launch");
    d_sig.download_async(0);
    check_device(feelmri::gpu::stream_synchronize(0), "mri stream sync");

    std::memcpy(signal_out_cf, d_sig.host(),
                static_cast<std::size_t>(n_samples) * nv * sizeof(cuFloatComplex));
    return 0;
  } catch (const std::exception& e) {
    static thread_local std::string msg;
    msg = e.what();
    return -1;
  }
}
