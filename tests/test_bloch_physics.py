"""Closed-form Bloch-equation validation on a tiny phantom.

For a single isochromat with no RF and no gradient,

  M_z(t)  = M0 + (M_z(0) - M0) * exp(-t / T1)
  M_xy(t) = M_xy(0) * exp(-t / T2) * exp(i * 2*pi * gamma * dB0 * t)

These tests assert that ``BlochSolver.solve()`` reproduces those
closed-form expressions on a 2-tetrahedron mesh. The mesh has no
spatial extent that matters (no gradient is applied), so every node
sees the same expected magnetization."""
from __future__ import annotations

import numpy as np
import pytest
from pint import Quantity

from feelmri import (
  BlochSolver,
  FEMPhantom,
  Scanner,
)

from _phantom_fixtures import make_minimal_tet_mesh
from _seq_fixtures import (
  make_empty_block,
  make_hard_pulse_block,
  make_single_block_sequence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def minimal_phantom(tmp_path_factory):
  pytest.importorskip('mpi4py')
  pytest.importorskip('pymetis')
  pytest.importorskip('meshio')
  mesh_dir = tmp_path_factory.mktemp('phys_mesh')
  mesh_path = mesh_dir / 'tet.vtu'
  make_minimal_tet_mesh(mesh_path)
  return FEMPhantom(path=str(mesh_path))


# ---------------------------------------------------------------------------
# T1 recovery
# ---------------------------------------------------------------------------

def test_t1_recovery_pure_relaxation(minimal_phantom):
  """Starting from Mz=0 with M0=1, after t = T1/2 of free relaxation
  Mz should reach 1 - exp(-1/2) = 0.3935..."""
  T1_ms = 200.0
  dur_ms = 0.5 * T1_ms
  expected_Mz = 1.0 - np.exp(-0.5)

  seq = make_single_block_sequence(make_empty_block(dur_ms, dt_ms=1.0))
  solver = BlochSolver(
    seq, minimal_phantom,
    M0=1.0,
    T1=Quantity(T1_ms, 'ms'),
    T2=Quantity(50.0, 'ms'),
    initial_Mxy=0.0,
    initial_Mz=0.0,
    perfect_spoiling=False,
  )
  Mxy, Mz = solver.solve()
  assert Mz.shape[1] == 1
  np.testing.assert_allclose(Mz[:, 0], expected_Mz, atol=5e-3)


# ---------------------------------------------------------------------------
# T2 decay
# ---------------------------------------------------------------------------

def test_t2_decay_pure_relaxation(minimal_phantom):
  """Starting from |Mxy|=1 and no RF, after t = 2 T2
  |Mxy| should be exp(-2) = 0.1353..."""
  T2_ms = 50.0
  dur_ms = 2.0 * T2_ms
  expected_abs = np.exp(-2.0)

  seq = make_single_block_sequence(make_empty_block(dur_ms, dt_ms=1.0))
  solver = BlochSolver(
    seq, minimal_phantom,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(T2_ms, 'ms'),
    initial_Mxy=1.0 + 0.0j,
    initial_Mz=1.0,
    perfect_spoiling=False,
  )
  Mxy, _ = solver.solve()
  np.testing.assert_allclose(np.abs(Mxy[:, 0]), expected_abs, atol=5e-3)


# ---------------------------------------------------------------------------
# Free precession with B0 offset
# ---------------------------------------------------------------------------

def test_free_precession_accumulates_b0_phase(minimal_phantom):
  """With delta_B = 1 microtesla and T = 5 ms,
  the accumulated phase is  Delta_phi = 2 * pi * gammabar * dB0 * T.
  gammabar = 42.576 MHz/T -> dB0 = 1e-6 T -> 42.576 Hz precession.
  Delta_phi mod 2*pi at 5 ms = 2*pi * 42.576 * 5e-3 = 1.3376 rad.
  """
  T_ms = 5.0
  dB0_mT = 1e-3   # = 1 microtesla
  gammabar = 42.576e6   # Hz / T (matches Scanner default).
  # FEelMRI's Bloch kernel follows the standard rotating-frame
  # convention Mxy(t) = Mxy(0) * exp(-i * 2*pi * gammabar * dB0 * t),
  # so a positive dB0 accumulates a negative phase angle.
  expected_phase = -2.0 * np.pi * gammabar * (dB0_mT * 1e-3) * (T_ms * 1e-3)
  expected_phase = np.angle(np.exp(1j * expected_phase))   # wrap to (-pi, pi]

  seq = make_single_block_sequence(make_empty_block(T_ms, dt_ms=0.05))
  solver = BlochSolver(
    seq, minimal_phantom,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    delta_B=dB0_mT,
    initial_Mxy=1.0 + 0.0j,
    initial_Mz=1.0,
    perfect_spoiling=False,
  )
  Mxy, _ = solver.solve()
  measured = np.angle(Mxy[:, 0])
  # All nodes see the same field; their phases must agree to within FP.
  np.testing.assert_allclose(measured, expected_phase, atol=2e-2)


# ---------------------------------------------------------------------------
# Hard 90 deg pulse
# ---------------------------------------------------------------------------

def test_hard_90_excitation(minimal_phantom):
  """Hard 90 deg pulse on resonance, starting from M_z = M0.
  After the pulse: M_z -> 0 and |M_xy| -> M0."""
  pulse_dur_ms = 0.2
  block = make_hard_pulse_block(np.pi / 2, dur_ms=pulse_dur_ms)
  seq = make_single_block_sequence(block)
  solver = BlochSolver(
    seq, minimal_phantom,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    initial_Mxy=0.0,
    initial_Mz=1.0,
    perfect_spoiling=False,
  )
  Mxy, Mz = solver.solve()
  np.testing.assert_allclose(np.abs(Mxy[:, 0]), 1.0, atol=5e-2)
  np.testing.assert_allclose(Mz[:, 0], 0.0, atol=5e-2)


# ---------------------------------------------------------------------------
# Hard 180 deg inversion
# ---------------------------------------------------------------------------

def test_hard_180_inversion(minimal_phantom):
  """Hard 180 deg inversion: M_z = M0 -> -M0 and |M_xy| stays near zero."""
  pulse_dur_ms = 0.2
  block = make_hard_pulse_block(np.pi, dur_ms=pulse_dur_ms)
  seq = make_single_block_sequence(block)
  solver = BlochSolver(
    seq, minimal_phantom,
    M0=1.0,
    T1=Quantity(1e6, 'ms'),
    T2=Quantity(1e6, 'ms'),
    initial_Mxy=0.0,
    initial_Mz=1.0,
    perfect_spoiling=False,
  )
  Mxy, Mz = solver.solve()
  np.testing.assert_allclose(Mz[:, 0], -1.0, atol=5e-2)
  np.testing.assert_allclose(np.abs(Mxy[:, 0]), 0.0, atol=5e-2)
