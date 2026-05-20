"""Device runtime control surface for FEelMRI.

The library defaults to the CPU compute path; the optional GPU compute
path is enabled by building with ``-DFEELMRI_ENABLE_GPU=ON``. This
module is the small Python facade users interact with when they want
to query availability or bind a particular rank to a device.

The public API is intentionally minimal:

* :func:`is_gpu_available` — returns ``True`` when the build includes
  the GPU backend AND at least one device is visible.
* :func:`device_count` — number of devices visible to this process.
* :func:`device_init_for_rank` — bind the current MPI rank to a device
  based on its local rank index. Idempotent.
* :func:`_require_gpu` — raise a clear :class:`ImportError`-shaped
  :class:`RuntimeError` when a feature explicitly requested ``device='gpu'``
  but the GPU backend is not available, mirroring the
  ``_require_pypulseq`` pattern in :mod:`feelmri.PulseqAdapter`.

Aliases ``'cuda'`` and ``'hip'`` are accepted by the public
``device=`` keyword on :class:`feelmri.Bloch.BlochSolver` and resolve
to the same code path; FEelMRI selects the right vendor based on what
the C++ backend was compiled for.
"""
from __future__ import annotations

from typing import Optional

try:
  from feelmri.BlochSimulator import gpu_available as _GPU_BUILT
except ImportError:  # pragma: no cover (build-time concern)
  _GPU_BUILT = False


_DEVICE_ALIASES = {
  'cpu':    'cpu',
  'gpu':    'gpu',
  'cuda':   'gpu',
  'hip':    'gpu',
  'rocm':   'gpu',
}


def normalize_device(device: str) -> str:
  """Canonicalize a user-supplied ``device`` string.

  Returns ``'cpu'`` or ``'gpu'``. Raises ``ValueError`` for anything
  else so typos do not silently fall back to CPU.
  """
  if not isinstance(device, str):
    raise TypeError(f"device must be str, got {type(device).__name__}")
  key = device.lower()
  if key not in _DEVICE_ALIASES:
    raise ValueError(
      f"unknown device {device!r}; expected one of "
      f"{sorted(set(_DEVICE_ALIASES))}"
    )
  return _DEVICE_ALIASES[key]


def is_gpu_built() -> bool:
  """True when the C++ extension was compiled with GPU support."""
  return bool(_GPU_BUILT)


def is_gpu_available() -> bool:
  """True when the build includes GPU support AND a device is visible."""
  if not _GPU_BUILT:
    return False
  try:
    from feelmri.BlochSimulator import device_is_available
  except ImportError:  # pragma: no cover (build-time concern)
    return False
  return bool(device_is_available())


def device_count() -> int:
  """Number of devices visible to this process, or 0 if no GPU build."""
  if not _GPU_BUILT:
    return 0
  from feelmri.BlochSimulator import device_count as _count
  return int(_count())


def device_init_for_rank(local_rank: int, num_local_ranks: int) -> None:
  """Bind the current process to a GPU based on its local-rank index.

  Idempotent on a successful initial call. Raises if invoked with
  different ``(local_rank, num_local_ranks)`` after the first call, or
  if the GPU build is absent.
  """
  if not _GPU_BUILT:
    raise RuntimeError(
      "feelmri.runtime.device_init_for_rank: this build was compiled "
      "without GPU support. Rebuild with "
      "'pip install -e . --config-settings=cmake.define."
      "FEELMRI_ENABLE_GPU=ON' to enable the GPU compute path."
    )
  from feelmri.BlochSimulator import (
    device_init,
    device_last_error,
  )
  rc = int(device_init(int(local_rank), int(num_local_ranks)))
  if rc != 0:
    raise RuntimeError(
      f"feelmri.runtime.device_init_for_rank: device runtime returned "
      f"{rc}: {device_last_error()}"
    )


def current_device() -> int:
  """Index of the device this process is bound to, or -1 if unbound."""
  if not _GPU_BUILT:
    return -1
  from feelmri.BlochSimulator import device_current
  return int(device_current())


def _require_gpu(feature: str) -> None:
  """Raise a clear RuntimeError if the GPU backend is not available.

  Mirrors the shape of ``_require_pypulseq`` in
  :mod:`feelmri.PulseqAdapter` so the user gets a consistent error
  message across optional features.
  """
  if not _GPU_BUILT:
    raise RuntimeError(
      f"{feature} requires the optional GPU compute backend. Rebuild "
      f"with 'pip install -e . --config-settings=cmake.define."
      f"FEELMRI_ENABLE_GPU=ON' to enable it."
    )
  if not is_gpu_available():
    raise RuntimeError(
      f"{feature} requires a visible GPU. The build includes the GPU "
      f"backend but no device is visible to this process. Check "
      f"`nvidia-smi` / `rocminfo` and that CUDA_VISIBLE_DEVICES is set "
      f"appropriately."
    )
