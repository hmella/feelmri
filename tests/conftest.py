"""
Global pytest configuration for FEelMRI tests.
"""

import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "mpi: marks tests that use MPI")