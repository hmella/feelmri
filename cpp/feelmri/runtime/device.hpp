/**
 * @file device.hpp
 * @brief Vendor-neutral GPU primitives for FEelMRI kernels.
 *
 * All `.cu` / `.hip` translation units in this project include this header
 * and use the `feelmri::gpu::*` namespace instead of calling `cudaMalloc` /
 * `hipMalloc` directly. The mapping to a concrete backend is selected at
 * build time by the `FEELMRI_GPU_BACKEND` CMake option:
 *
 *   - Default: CUDA. Includes `<cuda_runtime.h>` and `<cuComplex.h>`.
 *   - `-DFEELMRI_GPU_BACKEND=hip`: HIP. Includes `<hip/hip_runtime.h>` and
 *     `<hip/hip_complex.h>`. Production-quality on AMD GPUs and on NVIDIA
 *     via the `__HIP_PLATFORM_NVIDIA__` translation layer.
 *
 * The two paths expose identical surface: types (`error_t`, `stream_t`,
 * `complex_f32_t`), constants (`SUCCESS`, `MEMCPY_H2D` / `MEMCPY_D2H`),
 * and inline wrappers (`malloc_`, `free_`, `host_alloc`, `host_free`,
 * `memcpy_async`, ...). Kernel arithmetic helpers (complex ops) ride on
 * top of `complex_f32_t` since both `cuFloatComplex` and `hipFloatComplex`
 * are layout-compatible `{ float x, y; }` structs.
 *
 * This file is the M2 milestone foundation: porting Bloch / MRIAssemble
 * to AMD becomes a CMake flag flip rather than a kernel rewrite once a
 * ROCm toolchain is in the loop.
 */
#pragma once

#include <cstddef>

#if defined(FEELMRI_GPU_BACKEND_HIP)
  #include <hip/hip_runtime.h>
  #include <hip/hip_complex.h>
#else
  #include <cuda_runtime.h>
  #include <cuComplex.h>
#endif

namespace feelmri {
namespace gpu {

#if defined(FEELMRI_GPU_BACKEND_HIP)
  using error_t       = hipError_t;
  using stream_t      = hipStream_t;
  using complex_f32_t = hipFloatComplex;
  using complex_f64_t = hipDoubleComplex;
  constexpr error_t SUCCESS    = hipSuccess;
  constexpr int     MEMCPY_H2D = hipMemcpyHostToDevice;
  constexpr int     MEMCPY_D2H = hipMemcpyDeviceToHost;
#else
  using error_t       = cudaError_t;
  using stream_t      = cudaStream_t;
  using complex_f32_t = cuFloatComplex;
  using complex_f64_t = cuDoubleComplex;
  constexpr error_t SUCCESS    = cudaSuccess;
  constexpr int     MEMCPY_H2D = cudaMemcpyHostToDevice;
  constexpr int     MEMCPY_D2H = cudaMemcpyDeviceToHost;
#endif

// ---------------------------------------------------------------------------
// Memory: device and pinned-host allocation, async transfer / memset.
// ---------------------------------------------------------------------------

inline error_t malloc_(void** p, std::size_t n) {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipMalloc(p, n);
#else
  return cudaMalloc(p, n);
#endif
}

inline error_t free_(void* p) {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipFree(p);
#else
  return cudaFree(p);
#endif
}

inline error_t host_alloc(void** p, std::size_t n) {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipHostMalloc(p, n);
#else
  return cudaHostAlloc(p, n, cudaHostAllocDefault);
#endif
}

inline error_t host_free(void* p) {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipHostFree(p);
#else
  return cudaFreeHost(p);
#endif
}

inline error_t memcpy_async(void* dst, const void* src, std::size_t n,
                              int kind, stream_t s) {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipMemcpyAsync(dst, src, n, static_cast<hipMemcpyKind>(kind), s);
#else
  return cudaMemcpyAsync(dst, src, n, static_cast<cudaMemcpyKind>(kind), s);
#endif
}

inline error_t memset_async(void* p, int v, std::size_t n, stream_t s) {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipMemsetAsync(p, v, n, s);
#else
  return cudaMemsetAsync(p, v, n, s);
#endif
}

// ---------------------------------------------------------------------------
// Stream / device control.
// ---------------------------------------------------------------------------

inline error_t stream_synchronize(stream_t s) {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipStreamSynchronize(s);
#else
  return cudaStreamSynchronize(s);
#endif
}

inline error_t get_last_error() {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipGetLastError();
#else
  return cudaGetLastError();
#endif
}

inline error_t set_device(int i) {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipSetDevice(i);
#else
  return cudaSetDevice(i);
#endif
}

inline error_t get_device_count(int* n) {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipGetDeviceCount(n);
#else
  return cudaGetDeviceCount(n);
#endif
}

inline error_t device_synchronize() {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipDeviceSynchronize();
#else
  return cudaDeviceSynchronize();
#endif
}

inline error_t device_reset() {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipDeviceReset();
#else
  return cudaDeviceReset();
#endif
}

inline const char* get_error_string(error_t e) {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return hipGetErrorString(e);
#else
  return cudaGetErrorString(e);
#endif
}

// ---------------------------------------------------------------------------
// Complex helpers. `cuFloatComplex` and `hipFloatComplex` are both POD
// `{ float x; float y; }` structs, so the maker and arithmetic helpers
// have the same layout under both backends.
// ---------------------------------------------------------------------------

__host__ __device__ inline complex_f32_t make_complex_f32(float re, float im) {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return make_hipFloatComplex(re, im);
#else
  return make_cuFloatComplex(re, im);
#endif
}

__host__ __device__ inline complex_f64_t make_complex_f64(double re, double im) {
#if defined(FEELMRI_GPU_BACKEND_HIP)
  return make_hipDoubleComplex(re, im);
#else
  return make_cuDoubleComplex(re, im);
#endif
}

// Templated complex maker / type lookup for kernels that want to be
// generic over T in {float, double}.
template <typename T> struct complex_for;
template <> struct complex_for<float>  { using type = complex_f32_t; };
template <> struct complex_for<double> { using type = complex_f64_t; };
template <typename T> using complex_t = typename complex_for<T>::type;

template <typename T>
__host__ __device__ inline complex_t<T> make_complex(T re, T im);

template <>
__host__ __device__ inline complex_t<float> make_complex<float>(float re, float im) {
  return make_complex_f32(re, im);
}
template <>
__host__ __device__ inline complex_t<double> make_complex<double>(double re, double im) {
  return make_complex_f64(re, im);
}

}  // namespace gpu
}  // namespace feelmri
