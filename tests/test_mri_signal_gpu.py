"""GPU/CPU agreement tests for the MRI signal-assembly kernel (M3).

The GPU path is opt-in via ``FEMPhantom.set_assembler(..., device='gpu')``
and currently covers the ``signal_sum`` and ``signal_nodal`` variants
(the per-node summed integrators used by spamm.py's fast-mode mri_signal
path). The quadrature variant (``signal`` / ``signal_full``) stays on
the host until a follow-up milestone implements its sparse projection
on the GPU.

These tests build a small disk phantom + a synthetic k-space trajectory,
run the same workload on CPU and GPU, and assert agreement to within
float32 precision. The whole file is auto-skipped when the build does
not include the GPU backend or when no device is visible.
"""
from __future__ import annotations

import numpy as np
import pytest

from feelmri import FEMPhantom
from feelmri import runtime as feelmri_runtime

from _phantom_fixtures import make_2d_disk_mesh


pytestmark = [
  pytest.mark.gpu,
  pytest.mark.skipif(
    not feelmri_runtime.is_gpu_available(),
    reason="GPU backend not available (build flag off or no device visible)",
  ),
]


def _to_3d_inputs(kx, ky, kz, t):
  n = kx.size
  shape = (n, 1, 1)
  pts = [
    np.ascontiguousarray(kx.reshape(shape), dtype=np.float32),
    np.ascontiguousarray(ky.reshape(shape), dtype=np.float32),
    np.ascontiguousarray(kz.reshape(shape), dtype=np.float32),
  ]
  return pts, np.ascontiguousarray(t.reshape(shape), dtype=np.float32)


@pytest.fixture(scope='module')
def disk_phantom(tmp_path_factory):
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  mesh_dir = tmp_path_factory.mktemp('mri_signal_gpu')
  mesh_path = mesh_dir / 'disk.vtu'
  make_2d_disk_mesh(mesh_path, radius=5e-3, n_radial=4, n_angular=12, thickness=1e-4)
  return FEMPhantom(path=str(mesh_path))


def _run(phantom, device, *, nodal_approximation=False):
  """Configure ``phantom`` for the requested device and return the
  signal at a small synthetic k-space trajectory."""
  phantom.set_assembler(
    voxel_size=5e-4,
    lorder=1, horder=1,
    nodal_approximation=nodal_approximation,
    lumped=True,
    device=device,
  )
  n = phantom.local_nodes.shape[0]
  phantom.set_static_fields(
    T2=np.full(n, 100.0, dtype=np.float32),
    phi_dB0=np.zeros(n, dtype=np.float32),
  )
  rng = np.random.default_rng(seed=0)
  Mxy = (rng.standard_normal(n).astype(np.float32)
         + 1j * rng.standard_normal(n).astype(np.float32))
  phantom.update_magnetization(Mxy.reshape(-1, 1))

  n_samples = 32
  kx = rng.uniform(-200.0, 200.0, n_samples).astype(np.float32)
  ky = rng.uniform(-200.0, 200.0, n_samples).astype(np.float32)
  kz = np.zeros(n_samples, dtype=np.float32)
  t  = rng.uniform(0.0, 5.0, n_samples).astype(np.float32)
  pts, t3 = _to_3d_inputs(kx, ky, kz, t)

  S = phantom.mri_signal(pts, t3, None)
  return np.asarray(S).reshape(-1)


def test_signal_quadrature_cpu_gpu_agreement(disk_phantom):
  """Quadrature ``signal()`` path: highest-accuracy integrator used when
  ``nodal_approximation=False`` and (per Phantom.mri_signal) for the
  large-element assembler even when ``nodal_approximation=True``. The
  host-side projection S_global_ * modes runs before each kernel
  launch; the kernel itself is the same one used by signal_sum /
  signal_nodal, fed quadrature-point arrays."""
  cpu = _run(disk_phantom, device='cpu', nodal_approximation=False)
  gpu = _run(disk_phantom, device='gpu', nodal_approximation=False)
  # Absolute tolerance scaled to the signal magnitude — the kernel
  # accumulates ~10^3 terms, each ~10^-1 magnitude, so 1e-4 absolute is
  # well above float32 round-off for this scale. The quadrature path
  # uses --use_fast_math sin/cos intrinsics; a few extra ulps over the
  # node-based variants is expected.
  np.testing.assert_allclose(gpu, cpu, atol=5e-4, rtol=5e-4)


def test_signal_nodal_cpu_gpu_agreement(disk_phantom):
  """signal_nodal path: mass-matrix-projected integrator used by
  spamm.py for the small-element assembler."""
  cpu = _run(disk_phantom, device='cpu', nodal_approximation=True)
  gpu = _run(disk_phantom, device='gpu', nodal_approximation=True)
  np.testing.assert_allclose(gpu, cpu, atol=2e-4, rtol=2e-4)


def test_set_device_round_trip(disk_phantom):
  """Switching device flag back and forth must give identical CPU
  results both times (no state leaks from the GPU path)."""
  cpu_a = _run(disk_phantom, device='cpu', nodal_approximation=True)
  _    = _run(disk_phantom, device='gpu', nodal_approximation=True)
  cpu_b = _run(disk_phantom, device='cpu', nodal_approximation=True)
  np.testing.assert_array_equal(cpu_a, cpu_b)


def test_unsupported_device_raises(disk_phantom):
  with pytest.raises(ValueError):
    disk_phantom.set_assembler(voxel_size=5e-4, device='moonbeam')


def test_quadrature_with_pod_trajectory_cpu_gpu_agreement(disk_phantom):
  """has_traj = True path: a POD trajectory drives the modes/weights
  feed-through, so this hits the fused projection's S × modes -> modes_q
  cache + per-step trajectory update inside the kernel. Catches drift
  that has_traj = False tests do not exercise."""
  from feelmri.Motion import POD as PODMotion

  n = disk_phantom.local_nodes.shape[0]
  rng = np.random.default_rng(seed=11)

  # Build a small POD trajectory: 8 time samples of synthetic 3-D
  # displacements over the local mesh.
  pod_times = np.linspace(0.0, 8.0, 8, dtype=np.float32)
  pod_data = rng.standard_normal((n, 3, 8)).astype(np.float32) * 1e-4
  pod = PODMotion(
    times=pod_times,
    data=pod_data,
    global_to_local=None,
    n_modes=4,
    is_periodic=False,
    interpolation_method='Pchip',
  )

  def _run(device):
    disk_phantom.set_assembler(
      voxel_size=5e-4,
      lorder=1, horder=1,
      nodal_approximation=False,
      lumped=True,
      device=device,
    )
    disk_phantom.set_static_fields(
      T2=np.full(n, 100.0, dtype=np.float32),
      phi_dB0=np.zeros(n, dtype=np.float32),
    )
    Mxy = (rng.standard_normal((n, 1)).astype(np.float32)
           + 1j * rng.standard_normal((n, 1)).astype(np.float32))
    disk_phantom.update_magnetization(Mxy)
    n_samples = 32
    kx = rng.uniform(-200.0, 200.0, n_samples).astype(np.float32)
    ky = rng.uniform(-200.0, 200.0, n_samples).astype(np.float32)
    kz = np.zeros(n_samples, dtype=np.float32)
    t  = rng.uniform(0.0, 5.0, n_samples).astype(np.float32)
    pts, t3 = _to_3d_inputs(kx, ky, kz, t)
    return np.asarray(disk_phantom.mri_signal(pts, t3, pod))

  cpu = _run('cpu')
  gpu = _run('gpu')
  # Trajectory tightens the tolerance slightly because the per-sample
  # k-vector now also picks up POD-driven displacement; agreement is
  # still bounded by the GPU's `--use_fast_math` sincos intrinsics.
  np.testing.assert_allclose(gpu, cpu, atol=1e-3, rtol=1e-3)


def test_quadrature_nv_above_tile_max_falls_back_cpu_gpu_agreement(disk_phantom):
  """nv > kTileMaxNV (= 4) triggers the atomic-add fallback inside the
  GPU signal kernel. Verify it still matches CPU when a large coil
  count is used (e.g. 8-coil array). This path bypasses the per-coil
  register accumulator and uses global-memory atomics instead."""
  def _run_multicoil(device, nv):
    disk_phantom.set_assembler(
      voxel_size=5e-4,
      lorder=1, horder=1,
      nodal_approximation=False,
      lumped=True,
      device=device,
    )
    n = disk_phantom.local_nodes.shape[0]
    disk_phantom.set_static_fields(
      T2=np.full(n, 100.0, dtype=np.float32),
      phi_dB0=np.zeros(n, dtype=np.float32),
    )
    rng = np.random.default_rng(seed=99)
    Mxy = (rng.standard_normal((n, nv)).astype(np.float32)
           + 1j * rng.standard_normal((n, nv)).astype(np.float32))
    disk_phantom.update_magnetization(Mxy)
    n_samples = 32
    kx = rng.uniform(-200.0, 200.0, n_samples).astype(np.float32)
    ky = rng.uniform(-200.0, 200.0, n_samples).astype(np.float32)
    kz = np.zeros(n_samples, dtype=np.float32)
    t  = rng.uniform(0.0, 5.0, n_samples).astype(np.float32)
    pts, t3 = _to_3d_inputs(kx, ky, kz, t)
    return np.asarray(disk_phantom.mri_signal(pts, t3, None))
  cpu = _run_multicoil('cpu', 8)
  gpu = _run_multicoil('gpu', 8)
  np.testing.assert_allclose(gpu, cpu, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize('nv', [2, 4])
def test_quadrature_multi_coil_cpu_gpu_agreement(disk_phantom, nv):
  """Multi-coil / multi-velocity-encoding agreement for the quadrature
  signal() path. Exercises the tiled GPU kernel's per-coil accumulator
  branch — the path 4D-flow scripts hit (nv = 4 velocity encodings)."""
  def _run_multicoil(device):
    disk_phantom.set_assembler(
      voxel_size=5e-4,
      lorder=1, horder=1,
      nodal_approximation=False,
      lumped=True,
      device=device,
    )
    n = disk_phantom.local_nodes.shape[0]
    disk_phantom.set_static_fields(
      T2=np.full(n, 100.0, dtype=np.float32),
      phi_dB0=np.zeros(n, dtype=np.float32),
    )
    rng = np.random.default_rng(seed=42)
    Mxy = (rng.standard_normal((n, nv)).astype(np.float32)
           + 1j * rng.standard_normal((n, nv)).astype(np.float32))
    disk_phantom.update_magnetization(Mxy)
    n_samples = 32
    kx = rng.uniform(-200.0, 200.0, n_samples).astype(np.float32)
    ky = rng.uniform(-200.0, 200.0, n_samples).astype(np.float32)
    kz = np.zeros(n_samples, dtype=np.float32)
    t  = rng.uniform(0.0, 5.0, n_samples).astype(np.float32)
    pts, t3 = _to_3d_inputs(kx, ky, kz, t)
    S = disk_phantom.mri_signal(pts, t3, None)
    return np.asarray(S)
  cpu = _run_multicoil('cpu')
  gpu = _run_multicoil('gpu')
  np.testing.assert_allclose(gpu, cpu, atol=5e-4, rtol=5e-4)
