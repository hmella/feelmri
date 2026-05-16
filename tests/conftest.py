"""Global pytest configuration for FEelMRI tests.

Adds the tests/ directory to ``sys.path`` so the shared helper modules
(``_phantom_fixtures``, ``_seq_fixtures``) can be imported by test
files via ``from _phantom_fixtures import …``."""

import os
import sys

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


def pytest_configure(config):
    config.addinivalue_line("markers", "mpi: marks tests that use MPI")
    config.addinivalue_line(
        "markers",
        "requires_mpi: marks tests that require mpirun on PATH",
    )
    config.addinivalue_line(
        "markers",
        "pulseq: marks tests that depend on the optional 'pypulseq' package "
        "(opt-out in CI via '-m \"not pulseq\"')",
    )