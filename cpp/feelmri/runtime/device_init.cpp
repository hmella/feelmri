// Backend-neutral implementation of the feelmri device-runtime API.
// The vendor primitives (set_device, malloc, etc.) come from device.hpp
// and resolve to cuda* under the default backend and hip* when built
// with -DFEELMRI_GPU_BACKEND=hip. The extern "C" surface in
// device_init.hpp does not change between backends.

#include "device.hpp"
#include "device_init.hpp"

#include <cstring>
#include <mutex>
#include <string>

namespace {

  struct DeviceState {
    int device_index = -1;
    int num_local_ranks = 0;
    bool initialized = false;
    std::string last_error;
    std::mutex mu;
  };

  DeviceState& state() {
    static DeviceState s;
    return s;
  }

  void record_device_error(feelmri::gpu::error_t err, const char* where) {
    if (err == feelmri::gpu::SUCCESS) {
      state().last_error.clear();
      return;
    }
    state().last_error.assign(where);
    state().last_error.append(": ");
    state().last_error.append(feelmri::gpu::get_error_string(err));
  }

  int visible_device_count_unlocked() {
    int n = 0;
    feelmri::gpu::error_t err = feelmri::gpu::get_device_count(&n);
    if (err != feelmri::gpu::SUCCESS) {
      // "no usable device" rather than a hard failure.
      record_device_error(err, "get_device_count");
      return 0;
    }
    return n;
  }

}  // namespace

extern "C" {

int feelmri_device_init(int local_rank, int num_local_ranks) {
  std::lock_guard<std::mutex> lock(state().mu);

  if (state().initialized) {
    if (local_rank == state().device_index
        && num_local_ranks == state().num_local_ranks) {
      return 0;
    }
    state().last_error = "feelmri_device_init: already initialised with "
                         "different (local_rank, num_local_ranks)";
    return 1;
  }

  const int n = visible_device_count_unlocked();
  if (n <= 0) {
    state().last_error = "feelmri_device_init: no devices visible";
    return 2;
  }

  const int idx = local_rank % n;
  feelmri::gpu::error_t err = feelmri::gpu::set_device(idx);
  if (err != feelmri::gpu::SUCCESS) {
    record_device_error(err, "set_device");
    return 3;
  }
  // Force lazy context creation now so subsequent kernel launches don't
  // pay the first-call latency under a wall-clock benchmark.
  void* dummy = nullptr;
  err = feelmri::gpu::malloc_(&dummy, 1);
  if (err != feelmri::gpu::SUCCESS) {
    record_device_error(err, "malloc(1) [context prime]");
    return 4;
  }
  feelmri::gpu::free_(dummy);

  state().device_index = idx;
  state().num_local_ranks = num_local_ranks;
  state().initialized = true;
  state().last_error.clear();
  return 0;
}

void feelmri_device_shutdown(void) {
  std::lock_guard<std::mutex> lock(state().mu);
  if (!state().initialized) {
    return;
  }
  feelmri::gpu::device_reset();
  state().device_index = -1;
  state().num_local_ranks = 0;
  state().initialized = false;
  state().last_error.clear();
}

int feelmri_device_count(void) {
  std::lock_guard<std::mutex> lock(state().mu);
  return visible_device_count_unlocked();
}

int feelmri_device_current(void) {
  std::lock_guard<std::mutex> lock(state().mu);
  return state().device_index;
}

int feelmri_device_is_available(void) {
  return feelmri_device_count() > 0 ? 1 : 0;
}

const char* feelmri_device_last_error_string(void) {
  std::lock_guard<std::mutex> lock(state().mu);
  return state().last_error.c_str();
}

int feelmri_device_synchronize(void) {
  feelmri::gpu::error_t err = feelmri::gpu::device_synchronize();
  if (err != feelmri::gpu::SUCCESS) {
    std::lock_guard<std::mutex> lock(state().mu);
    record_device_error(err, "device_synchronize");
    return 1;
  }
  return 0;
}

}  // extern "C"
