"""GPU/CPU agreement tests for the FEelMRI Bloch kernel (M1).

The GPU path is opt-in via ``BlochSolver(device='gpu')`` and currently
covers the ``method='cayley_klein'`` (order = 0) + ``dtype='float32'``
combination. These tests run the same closed-form Bloch scenarios used
by ``tests/test_bloch_physics.py`` on both backends and assert that the
two outputs agree to within float32 precision. The whole file is
auto-skipped when the build does not include the GPU backend or when
no device is visible.
"""
from __future__ import annotations

import numpy as np
import pytest
from pint import Quantity

from feelmri import (
  BlochSolver,
  FEMPhantom,
  Scanner,
)
from feelmri import runtime as feelmri_runtime

from _phantom_fixtures import make_minimal_tet_mesh
from _seq_fixtures import (
  make_empty_block,
  make_hard_pulse_block,
  make_single_block_sequence,
)


pytestmark = [
  pytest.mark.gpu,
  pytest.mark.skipif(
    not feelmri_runtime.is_gpu_available(),
    reason="GPU backend not available (build flag off or no device visible)",
  ),
]


@pytest.fixture(scope='module')
def minimal_phantom(tmp_path_factory):
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  mesh_dir = tmp_path_factory.mktemp('gpu_mesh')
  mesh_path = mesh_dir / 'tet.vtu'
  make_minimal_tet_mesh(mesh_path)
  return FEMPhantom(path=str(mesh_path))


def _run_pair(phantom, build_seq, **solver_kwargs):
  """Solve the same sequence on CPU and GPU; return both magnetisations."""
  cpu_solver = BlochSolver(build_seq(), phantom, device='cpu', **solver_kwargs)
  cpu_Mxy, cpu_Mz = cpu_solver.solve()
  gpu_solver = BlochSolver(build_seq(), phantom, device='gpu', **solver_kwargs)
  gpu_Mxy, gpu_Mz = gpu_solver.solve()
  return (cpu_Mxy, cpu_Mz), (gpu_Mxy, gpu_Mz)


def test_t1_recovery_agrees(minimal_phantom):
  T1_ms = 200.0
  dur_ms = 0.5 * T1_ms
  build = lambda: make_single_block_sequence(make_empty_block(dur_ms, dt_ms=1.0))
  cpu, gpu = _run_pair(
    minimal_phantom, build,
    M0=1.0,
    T1=Quantity(T1_ms, 'ms'),
    T2=Quantity(50.0, 'ms'),
    initial_Mxy=0.0,
    initial_Mz=0.0,
    perfect_spoiling=False,
  )
  np.testing.assert_allclose(gpu[0], cpu[0], atol=1e-5)
  np.testing.assert_allclose(gpu[1], cpu[1], atol=1e-5)


def test_t2_decay_agrees(minimal_phantom):
  T2_ms = 50.0
  dur_ms = 2.0 * T2_ms
  build = lambda: make_single_block_sequence(make_empty_block(dur_ms, dt_ms=1.0))
  cpu, gpu = _run_pair(
    minimal_phantom, build,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(T2_ms, 'ms'),
    initial_Mxy=1.0 + 0.0j,
    initial_Mz=1.0,
    perfect_spoiling=False,
  )
  np.testing.assert_allclose(np.abs(gpu[0]), np.abs(cpu[0]), atol=1e-5)
  np.testing.assert_allclose(gpu[1], cpu[1], atol=1e-5)


def test_free_precession_agrees(minimal_phantom):
  T_ms = 5.0
  dB0_mT = 1e-3
  build = lambda: make_single_block_sequence(make_empty_block(T_ms, dt_ms=0.05))
  cpu, gpu = _run_pair(
    minimal_phantom, build,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    delta_B=dB0_mT,
    initial_Mxy=1.0 + 0.0j,
    initial_Mz=1.0,
    perfect_spoiling=False,
  )
  np.testing.assert_allclose(np.angle(gpu[0]), np.angle(cpu[0]), atol=2e-4)
  np.testing.assert_allclose(np.abs(gpu[0]), np.abs(cpu[0]), atol=1e-5)


def test_hard_90_agrees(minimal_phantom):
  build = lambda: make_single_block_sequence(make_hard_pulse_block(np.pi / 2, dur_ms=0.2))
  cpu, gpu = _run_pair(
    minimal_phantom, build,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    initial_Mxy=0.0,
    initial_Mz=1.0,
    perfect_spoiling=False,
  )
  np.testing.assert_allclose(np.abs(gpu[0]), np.abs(cpu[0]), atol=5e-5)
  np.testing.assert_allclose(gpu[1], cpu[1], atol=5e-5)


def test_hard_180_agrees(minimal_phantom):
  build = lambda: make_single_block_sequence(make_hard_pulse_block(np.pi, dur_ms=0.2))
  cpu, gpu = _run_pair(
    minimal_phantom, build,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    initial_Mxy=0.0,
    initial_Mz=1.0,
    perfect_spoiling=False,
  )
  np.testing.assert_allclose(np.abs(gpu[0]), np.abs(cpu[0]), atol=5e-5)
  np.testing.assert_allclose(gpu[1], cpu[1], atol=5e-5)


_METHOD_DTYPE_MATRIX = [
  ('cayley_klein', 'float32'),
  ('cayley_klein', 'float64'),
  ('magnus2',      'float32'),
  ('magnus2',      'float64'),
  ('magnus4',      'float32'),
  ('magnus4',      'float64'),
]


@pytest.mark.parametrize('method,dtype', _METHOD_DTYPE_MATRIX)
def test_hard_pulse_dtype_method_matrix(minimal_phantom, method, dtype):
  """Hard 90 across the full (dtype, method) GPU support matrix.
  Asserts CPU/GPU agreement at the precision floor for each dtype."""
  build = lambda: make_single_block_sequence(make_hard_pulse_block(np.pi / 2, dur_ms=0.2))
  kwargs = dict(
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    initial_Mxy=0.0,
    initial_Mz=1.0,
    perfect_spoiling=False,
    method=method,
    dtype=dtype,
  )
  cpu_solver = BlochSolver(build(), minimal_phantom, device='cpu', **kwargs)
  cpu_Mxy, cpu_Mz = cpu_solver.solve()
  gpu_solver = BlochSolver(build(), minimal_phantom, device='gpu', **kwargs)
  gpu_Mxy, gpu_Mz = gpu_solver.solve()
  # float32 tolerance is loose because the GPU build enables
  # --use_fast_math (sin / cos intrinsics drop ~1 ulp of precision vs
  # std::cos / std::sin on the host). 5e-5 still pins the rotation
  # within physical-accuracy bounds; float64 keeps near-exact agreement.
  atol = 5e-5 if dtype == 'float32' else 1e-12
  np.testing.assert_allclose(np.abs(gpu_Mxy), np.abs(cpu_Mxy), atol=atol)
  np.testing.assert_allclose(gpu_Mz, cpu_Mz, atol=atol)


@pytest.mark.parametrize('method,dtype', _METHOD_DTYPE_MATRIX)
def test_free_precession_dtype_method_matrix(minimal_phantom, method, dtype):
  """Free precession with non-zero delta_B0 across the matrix. The
  Magnus orders exercise their Bz_old / rf_old state-carry path here
  (the CPU and GPU branches must seed identically)."""
  T_ms = 5.0
  dB0_mT = 5e-4
  build = lambda: make_single_block_sequence(make_empty_block(T_ms, dt_ms=0.05))
  kwargs = dict(
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    delta_B=dB0_mT,
    initial_Mxy=1.0 + 0.0j,
    initial_Mz=1.0,
    perfect_spoiling=False,
    method=method,
    dtype=dtype,
  )
  cpu_solver = BlochSolver(build(), minimal_phantom, device='cpu', **kwargs)
  cpu_Mxy, cpu_Mz = cpu_solver.solve()
  gpu_solver = BlochSolver(build(), minimal_phantom, device='gpu', **kwargs)
  gpu_Mxy, gpu_Mz = gpu_solver.solve()
  atol = 2e-4 if dtype == 'float32' else 1e-12
  np.testing.assert_allclose(np.angle(gpu_Mxy), np.angle(cpu_Mxy), atol=atol)
  np.testing.assert_allclose(np.abs(gpu_Mxy), np.abs(cpu_Mxy), atol=atol)


def test_unknown_device_string_raises(minimal_phantom):
  base = lambda: make_single_block_sequence(make_empty_block(1.0, dt_ms=0.1))
  with pytest.raises(ValueError):
    BlochSolver(base(), minimal_phantom, device='moonbeam')
