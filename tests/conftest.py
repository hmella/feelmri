"""Global pytest configuration for FEelMRI tests.

Adds the tests/ directory to ``sys.path`` so the shared helper modules
(``_phantom_fixtures``, ``_seq_fixtures``) can be imported by test
files via ``from _phantom_fixtures import …``.

Also owns the canonical ``.seq`` fixture list and the two cached readers
every Pulseq test file shares -- see ``SEQ_FILES`` below."""

import functools
import os
import sys
from pathlib import Path

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


# ---------------------------------------------------------------------------
# The .seq fixture list, and the two readers shared across every Pulseq test
# file.
#
# Reading a sequence twice gives the same answer, so the results are cached for
# the whole session. Before this, each test file rebuilt its own module-scoped
# pypulseq cache and called import_pulseq fresh inside every test body: 15
# fixtures x 14 test functions was about 210 imports of the same files, with
# the two 231-block EPI files parsed 14 times each.
#
# The cached objects are shared, so a test that MUTATES an import would corrupt
# every later one. Copy first (PulseqImport.copy_block does) or build your own.
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / 'data'
EXAMPLES_SEQ_DIR = Path(__file__).parent.parent / 'examples' / 'pulseq'

SEQ_FILES = sorted(DATA_DIR.glob('*.seq')) + sorted(EXAMPLES_SEQ_DIR.glob('*.seq'))


def seq_ids(paths):
    """Parametrisation ids: the file stem, so a failure names the fixture."""
    return [p.stem for p in paths]


@functools.lru_cache(maxsize=None)
def _read_feelmri(seq_path):
    from feelmri.PulseqAdapter import import_pulseq
    return import_pulseq(seq_path)


@functools.lru_cache(maxsize=None)
def _read_pypulseq(seq_path):
    import pypulseq as pp

    seq = pp.Sequence()
    try:
        seq.read(str(seq_path), detect_rf_use=False)
    except Exception:
        return None
    return seq


@pytest.fixture(scope='session')
def pulseq_import():
    """``pulseq_import(path)`` -> a cached ``PulseqImport``. Treat as read-only."""
    pytest.importorskip('pypulseq')
    return _read_feelmri


@pytest.fixture(scope='session')
def pypulseq_ref():
    """``pypulseq_ref(path)`` -> a cached ``pp.Sequence``, or skip.

    Two distinct reasons pypulseq may refuse a file: it is too old for the
    format (say so precisely), or the file uses an extension it does not
    implement at all -- ROTATIONS, which it has no support for. The file is
    valid either way, so this skips rather than fails.
    """
    pytest.importorskip('pypulseq')

    def get(seq_path):
        seq = _read_pypulseq(seq_path)
        if seq is None:
            skip_if_pypulseq_too_old(seq_path)
            pytest.skip(f'pypulseq does not implement an extension used by '
                        f'{Path(seq_path).name}; no reference available')
        return seq

    return get


def pypulseq_block_durations_ms(pp_seq):
    """Block durations in ms. ``block_durations`` is keyed from 1 in pypulseq 1.5."""
    import numpy as np

    n = len(pp_seq.block_durations)
    return np.array([pp_seq.block_durations[i + 1] for i in range(n)]) * 1e3
