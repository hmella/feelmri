import numpy as np
import pytest
from pathlib import Path

from feelmri.PulseqAdapter import import_pulseq

# Timings of the imported sequence must match the Pulseq file. pypulseq is the
# reference: its block_durations, duration() and adc_times() are read directly
# and compared against the feelmri blocks.

pytestmark = pytest.mark.pulseq

DATA_DIR = Path(__file__).parent / 'data'
EXAMPLES_DIR = Path(__file__).parent.parent / 'examples' / 'pulseq'

SEQ_FILES = sorted(DATA_DIR.glob('*.seq')) + sorted(EXAMPLES_DIR.glob('*.seq'))

# Time tolerance in ms. The block raster is 10 us, so anything at this level is
# floating point noise rather than a timing difference.
TOL_MS = 1e-9


def _ids(paths):
    return [p.stem for p in paths]


@pytest.fixture(scope='module')
def pulseq_sequences():
    # One file is skipped rather than read: tests/data/rotation_minimal.seq is
    # hand written to exercise the adapter's ROTATIONS parsing and pypulseq
    # itself cannot read it, so there is no reference to compare against.
    pp = pytest.importorskip('pypulseq')
    out = {}
    for path in SEQ_FILES:
        seq = pp.Sequence()
        try:
            seq.read(str(path), detect_rf_use=False)
        except Exception:
            continue
        out[path] = seq
    return out


def _reference(sequences, seq_path):
    if seq_path not in sequences:
        pytest.skip(f'pypulseq cannot read {seq_path.name}; no reference available')
    return sequences[seq_path]


def _durations_ms(pp_seq):
    # block_durations is a dict keyed from 1 in pypulseq 1.5
    n = len(pp_seq.block_durations)
    return np.array([pp_seq.block_durations[i + 1] for i in range(n)]) * 1e3


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_block_count_matches(seq_path, pulseq_sequences):
    ref = _reference(pulseq_sequences, seq_path)
    imp = import_pulseq(seq_path)
    assert len(imp.feelmri_seq.blocks) == len(ref.block_durations)


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_block_durations_match(seq_path, pulseq_sequences):
    ref = _reference(pulseq_sequences, seq_path)
    imp = import_pulseq(seq_path)
    expected = _durations_ms(ref)
    got = np.array([b.dur.m_as('ms') for b in imp.feelmri_seq.blocks])
    assert np.abs(expected - got).max() < TOL_MS


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_block_start_times_match(seq_path, pulseq_sequences):
    # The absolute start of every block, which is where drift accumulates
    ref = _reference(pulseq_sequences, seq_path)
    imp = import_pulseq(seq_path)
    expected = np.concatenate(([0.0], np.cumsum(_durations_ms(ref))[:-1]))
    got = np.array([b.time_extent[0].m_as('ms') for b in imp.feelmri_seq.blocks])
    assert np.abs(expected - got).max() < TOL_MS


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_total_duration_matches(seq_path, pulseq_sequences):
    ref = _reference(pulseq_sequences, seq_path)
    imp = import_pulseq(seq_path)
    expected = ref.duration()[0] * 1e3
    got = imp.feelmri_seq.dur.m_as('ms')
    assert abs(expected - got) < TOL_MS


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_every_block_is_integrated(seq_path):
    # A block needs at least two raster points, otherwise the kernel takes no
    # step and the block evolves the magnetization not at all. Delays with no
    # events are the usual case, and in a spin echo they carry the T2 weighting.
    imp = import_pulseq(seq_path)
    unstepped = [i for i, b in enumerate(imp.feelmri_seq.blocks)
                 if b.dur.m_as('ms') > 0 and len(b.discrete_times) < 2]
    assert not unstepped, (
        f'{len(unstepped)} block(s) of nonzero duration get no integration step: '
        f'{unstepped[:10]}')


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_adc_sample_times_match(seq_path, pulseq_sequences):
    ref = _reference(pulseq_sequences, seq_path)
    imp = import_pulseq(seq_path)
    if not imp.readouts:
        pytest.skip('no ADC in this sequence')
    got = np.concatenate([r.times for r in imp.readouts])
    expected = ref.adc_times()[0] * 1e3   # (t_adc, freq/phase offsets)
    assert got.size == expected.size
    assert np.abs(np.sort(got) - np.sort(expected)).max() < 1e-6
