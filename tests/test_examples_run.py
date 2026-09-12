"""
Functional tests that verify FEelMRI examples run successfully.

These tests ensure that example scripts (e.g., 4dflow.py, phase_contrast.py)
execute without errors — a form of regression testing for the full workflow.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# Each script runs a full pipeline in a subprocess with a 180 s budget,
# well past the global 30 s pytest-timeout default. Measured, the 16 of
# them are 97 s -- half the suite's wall clock -- so they carry `slow`
# and leave `-m "not slow"` a fast suite. CI does not deselect them.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.timeout(240),
]


@pytest.mark.parametrize("script", [
    "4dflow.py",
    "b1_inhomogeneity.py",
    "coil_sensitivity.py",
    "concomitant_fields.py",
    "free_running.py",
    "gradient_orientation.py",
    "gradient_spoiling.py",
    "phase_contrast.py",
    "test_all_recon.py",
    "pod.py",
    "pvsm_parameters.py",
    "trajectories.py",    
    "spamm.py",
    "t2_prime_spin_echo.py",
    "water_and_fat.py",
    # Both exit 0 with a message when pypulseq is absent, so they are safe to
    # list unconditionally. The writer redirects its .seq under
    # FEELMRI_FAST_TEST so it cannot overwrite the tracked fixture the
    # adapter tests read.
    "pulseq_write_epi_tagging.py",
    "pulseq_run_epi_tagging.py",
])
def test_example_runs(script):
    """
    Run each example script using the system Python interpreter.

    This test passes if the script runs without throwing an exception.
    The stdout/stderr are captured for debugging if it fails.
    """
    script_path = EXAMPLES_DIR / script
    assert script_path.exists(), f"Example script not found: {script_path}"

    # Set environment variable to enable fast mode in the example
    env = dict(os.environ,
              FEELMRI_FAST_TEST="1",
              MPLBACKEND="Agg",
              COVERAGE_PROCESS_START=str(Path(__file__).resolve().parent.parent / ".coveragerc"),
              COVERAGE_FILE=str(Path(__file__).resolve().parent.parent / f".coverage.{script}")
              )

    # Set unique coverage file for each example
    env["COVERAGE_FILE"] = str(Path(__file__).resolve().parent.parent / f".coverage.{script}")

    # Run the temporary script in the examples directory
    result = subprocess.run(
        [sys.executable, 
        "-m", "coverage", 
        "run", "-p",
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

    # Optionally verify that the script produced some output files
    out_dir = EXAMPLES_DIR / "output"
    if out_dir.exists():
        assert any(out_dir.iterdir()), f"{script} ran but produced no output files"