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


def skip_if_pypulseq_too_old(seq_path):
    """Skip when the installed pypulseq cannot read this .seq file's format.

    pypulseq reads only up to its own format version, and the v1.5 layout is
    not backward readable -- 1.4.x mis-parses a v1.5 file and dies inside
    calculate_kspace. FEelMRI's own reader handles both, so the combination is
    unsupported rather than broken, and CI exercises both versions.
    """
    pytest.importorskip("pypulseq")
    from feelmri.PulseqAdapter import read_seq, pypulseq_can_read, pypulseq_version

    file_version = read_seq(str(seq_path)).DEF.get("PulseqVersion")
    if file_version is None:
        return
    if not pypulseq_can_read(file_version):
        installed = pypulseq_version()
        pytest.skip(
            f"{os.path.basename(str(seq_path))} is Pulseq v{file_version.major}."
            f"{file_version.minor} and the installed pypulseq is "
            f"{installed.major}.{installed.minor}, which cannot read it"
        )