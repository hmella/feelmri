"""
The MPI leg of the example runner.

``test_examples_run.py`` runs each example serially; this file runs the
subset that supports it under ``mpirun -n 2``, so a rank-dependent failure
(an ungated collective, a rank-0-only array) surfaces as a non-zero exit.
The physics equivalence between rank counts is a different question and is
tested directly in ``test_mpi_equivalence.py``.

These are subprocess tests and take a large share of the suite's wall
clock. They are marked ``slow`` so ``-m "not slow"`` is a fast suite, and
``requires_mpi`` because they need ``mpirun`` on PATH.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# Each script runs a full Bloch + signal-assembly pipeline in a subprocess
# with a 180 s budget, well past the global 30 s pytest-timeout default.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_mpi,
    pytest.mark.timeout(240),
]


# The four scripts that ASSERT are the ones worth running here, and three of
# them were missing: an assertion read at a rank-local `argmin` over
# `local_nodes` is only correct at one rank. `t2_prime_spin_echo.py`
# normalised its own T2' map that way and built a NEGATIVE T2' of -1.266 ms at
# 8 ranks; `b1_inhomogeneity.py` reported a centre inversion of -0.691 against
# -0.588. Neither was run under mpirun, which is why nothing caught them.
@pytest.mark.parametrize("script", [
    "4dflow.py",
    "b1_inhomogeneity.py",
    "coil_sensitivity.py",
    "concomitant_fields.py",
    "t2_prime_spin_echo.py",
    "free_running.py",
    "gradient_orientation.py",
    "gradient_spoiling.py",
    "phase_contrast.py",
    "test_all_recon.py",
    "pod.py",
    "pvsm_parameters.py",
    "trajectories.py",    
    "spamm.py",
    "water_and_fat.py",
])
def test_example_parallel(script):
    """Run one example under ``mpirun -n 2`` and require a clean exit."""
    if shutil.which('mpirun') is None:
        pytest.skip('mpirun not on PATH')

    script_path = EXAMPLES_DIR / script
    assert script_path.exists(), f"Example script not found: {script_path}"

    # Set environment variable to enable fast mode in the example
    env = dict(os.environ,
              FEELMRI_FAST_TEST="1",
              MPLBACKEND="Agg",
              COVERAGE_PROCESS_START=str(Path(__file__).resolve().parent.parent / ".coveragerc"),
              COVERAGE_FILE=str(Path(__file__).resolve().parent.parent / f".coverage.{script}")
              )

    # Run the temporary script in the examples directory
    result = subprocess.run(
        ["mpirun", "-n", "2", 
        sys.executable,
        "-m", "coverage",
        "run", "--parallel-mode",
        str(script_path)],
        cwd=EXAMPLES_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=180,
    )

    # Ensure script finishes successfully
    assert result.returncode == 0, (
        f"Example {script} failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )