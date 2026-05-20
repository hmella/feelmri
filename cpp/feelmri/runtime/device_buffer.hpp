/**
 * @file device_buffer.hpp
 * @brief Pinned-host + device-resident buffer pair with async deep_copy.
 *
 * Uses the vendor-neutral primitives in `device.hpp` so the buffer works
 * identically against CUDA and HIP backends. The default upload/download
 * path is `memcpy_async` over pinned-host memory. Unified memory is
 * intentionally NOT the default for FEelMRI's streaming access pattern.
 *
 * Only included by `.cu` / `.hip` translation units. Pybind11-binding
 * TUs use the extern "C" launch wrappers in `kernels/*_gpu.hpp` instead.
 */
#pragma once

#include "device.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace feelmri {

  inline void check_device(gpu::error_t err, const char* where) {
    if (err != gpu::SUCCESS) {
      throw std::runtime_error(std::string(where) + ": "
                                + gpu::get_error_string(err));
    }
  }

  /**
   * One contiguous run of `T` elements with a host (pinned) and a device
   * mirror. Movable but not copyable.
   *
   * Typical lifecycle:
   *
   *   DeviceBuffer<float> rf(n_steps);
   *   std::memcpy(rf.host(), src, n_steps * sizeof(float));
   *   rf.upload_async(stream);
   *   launch_kernel<<<..., stream>>>(rf.device(), ...);
   *   rf.download_async(stream);
   *   gpu::stream_synchronize(stream);
   */
  template <typename T>
  class DeviceBuffer {
   public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t n) { resize(n); }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : host_(other.host_), device_(other.device_), n_(other.n_) {
      other.host_ = nullptr;
      other.device_ = nullptr;
      other.n_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
      if (this != &other) {
        release();
        host_ = other.host_;
        device_ = other.device_;
        n_ = other.n_;
        other.host_ = nullptr;
        other.device_ = nullptr;
        other.n_ = 0;
      }
      return *this;
    }

    void resize(std::size_t n) {
      if (n == n_) {
        return;
      }
      release();
      n_ = n;
      if (n_ == 0) {
        return;
      }
      check_device(gpu::host_alloc(reinterpret_cast<void**>(&host_),
                                     n_ * sizeof(T)),
                    "DeviceBuffer::host_alloc");
      check_device(gpu::malloc_(reinterpret_cast<void**>(&device_),
                                  n_ * sizeof(T)),
                    "DeviceBuffer::malloc");
    }

    /** Stage host_ -> device_. */
    void upload_async(gpu::stream_t stream = 0) {
      if (n_ == 0) {
        return;
      }
      check_device(gpu::memcpy_async(device_, host_, n_ * sizeof(T),
                                       gpu::MEMCPY_H2D, stream),
                    "DeviceBuffer::upload_async");
    }

    /** Stage device_ -> host_. */
    void download_async(gpu::stream_t stream = 0) {
      if (n_ == 0) {
        return;
      }
      check_device(gpu::memcpy_async(host_, device_, n_ * sizeof(T),
                                       gpu::MEMCPY_D2H, stream),
                    "DeviceBuffer::download_async");
    }

    T* host() { return host_; }
    const T* host() const { return host_; }
    T* device() { return device_; }
    const T* device() const { return device_; }
    std::size_t size() const { return n_; }

   private:
    void release() {
      if (device_ != nullptr) {
        gpu::free_(device_);
        device_ = nullptr;
      }
      if (host_ != nullptr) {
        gpu::host_free(host_);
        host_ = nullptr;
      }
      n_ = 0;
    }

    T* host_ = nullptr;
    T* device_ = nullptr;
    std::size_t n_ = 0;
  };

}  // namespace feelmri
