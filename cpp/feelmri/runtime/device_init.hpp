/**
 * @file device_init.hpp
 * @brief Vendor-agnostic device-runtime control surface for FEelMRI.
 *
 * The four pybind11 extension modules (BlochSimulator, MRIAssemble, Assemble,
 * PODHelper) all link against a single `feelmri_runtime` shared library that
 * exports the functions declared here. That single point of ownership avoids
 * the classic singleton-init problem where each module would otherwise hold
 * its own copy of the device-runtime state.
 *
 * The current implementation talks to the CUDA runtime; the M2 milestone
 * swaps in HIP. The header is intentionally free of vendor types so the
 * pybind11 modules and Python wrapper code never need to change.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Pick a device for this process and prime the runtime. Idempotent: calling
 * twice with the same arguments is a no-op; calling with different arguments
 * after the first init is an error.
 *
 * @param local_rank        Rank index inside this node (0-based).
 * @param num_local_ranks   Number of ranks sharing this node.
 * @return 0 on success, non-zero on error (see device_last_error_string()).
 */
int feelmri_device_init(int local_rank, int num_local_ranks);

/** Tear down the device runtime; safe to call even if init never succeeded. */
void feelmri_device_shutdown(void);

/** Number of devices visible to this process (0 if none). */
int feelmri_device_count(void);

/** Index of the device this process is bound to, or -1 if not initialised. */
int feelmri_device_current(void);

/** 1 if at least one device is visible, 0 otherwise. */
int feelmri_device_is_available(void);

/**
 * Last error message from a runtime API call. The returned pointer is owned
 * by the runtime and remains valid until the next runtime call from this
 * process. Never returns NULL; an empty string indicates no error.
 */
const char* feelmri_device_last_error_string(void);

/** Block until all queued device work has completed. */
int feelmri_device_synchronize(void);

#ifdef __cplusplus
}
#endif
