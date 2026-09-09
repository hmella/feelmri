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
    # tests/data/rotation_minimal.seq is skipped: it uses the ROTATIONS
    # extension, which pypulseq does not implement at all, so there is no
    # reference to compare against. The file itself is valid.
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
        pytest.skip(f'pypulseq does not implement an extension used by '
                    f'{seq_path.name}; no reference available')
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


def _pp_gradients_hz_per_m(wf, t_ms, t0, t1):
    # waveforms_and_times concatenates the shape pieces of every block with no
    # padding, so interpolating the whole array bridges the gaps between them.
    # Restrict to the samples of one block, where the pieces are contiguous.
    out = np.zeros((3, t_ms.size))
    for axis in range(3):
        w = np.asarray(wf[axis], dtype=float)
        if w.size == 0:
            continue
        t_s = w[0] * 1e3
        inside = (t_s >= t0 - 1e-9) & (t_s <= t1 + 1e-9)
        if not np.any(inside):
            continue
        out[axis] = np.interp(t_ms, t_s[inside], w[1][inside], left=0.0, right=0.0)
    return out


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_gradient_waveforms_match(seq_path, pulseq_sequences):
    # The gradient the solver integrates must be the gradient pypulseq plays.
    # This is what catches a shaped gradient laid down half a raster off, one
    # missing the first/last boundary samples of a v1.5 file, or a gamma that
    # does not match the one the solver multiplies by.
    from feelmri.MRObjects import Scanner
    gammabar = Scanner().gammabar.m_as('Hz/T')

    ref = _reference(pulseq_sequences, seq_path)
    imp = import_pulseq(seq_path)
    wf = ref.waveforms_and_times()[0]
    pp_times_ms = [np.asarray(np.asarray(w)[0]) * 1e3 for w in wf if np.asarray(w).size]

    worst = 0.0
    peak = 1.0
    for block in imp.feelmri_seq.blocks:
        t0 = block.time_extent[0].m_as('ms')
        t1 = block.time_extent[1].m_as('ms')
        # Corners of both representations, then midpoints: a piecewise-linear
        # function is pinned by its value between adjacent corners, and
        # midpoints avoid the ambiguity of landing exactly on one.
        corners = [block.discrete_times.m_as('ms')]
        for t_s in pp_times_ms:
            corners.append(t_s[(t_s >= t0) & (t_s <= t1)])
        grid = np.unique(np.concatenate(corners))
        # The same corner reached from the two sides differs in the last bit,
        # which would leave a zero-width interval whose midpoint falls on the
        # corner itself. Collapse anything below a picosecond.
        grid = grid[np.concatenate(([True], np.diff(grid) > 1e-9))]
        if grid.size < 2:
            continue
        t_ms = 0.5 * (grid[:-1] + grid[1:])

        _rf, G, _mask = block(t_ms)
        # An axis with no gradient sums to a scalar zero, not an array.
        got = np.vstack([np.broadcast_to(np.asarray(G[a], dtype=float), t_ms.shape)
                         for a in range(3)])
        got = got * 1e-3 * gammabar
        expected = _pp_gradients_hz_per_m(wf, t_ms, t0, t1)
        worst = max(worst, float(np.abs(got - expected).max()))
        peak = max(peak, float(np.abs(expected).max()))

    assert worst < 1e-6 * peak, (
        f'max |delta| = {worst:.4g} Hz/m against a peak of {peak:.4g}')


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_block_adc_times_match(seq_path, pulseq_sequences):
    # ReadoutWindow.times comes from pypulseq, so it cannot catch a mistake in
    # the in-house read_ADC. The block's own ADC does: its dwell and delay
    # convention (half a dwell into the first sample) has to agree.
    ref = _reference(pulseq_sequences, seq_path)
    imp = import_pulseq(seq_path)
    got = []
    for block in imp.feelmri_seq.blocks:
        if block.adc is None:
            continue
        got.append(block.time_extent[0].m_as('ms') + block.adc.times.m_as('ms'))
    if not got:
        pytest.skip('no ADC in this sequence')
    got = np.sort(np.concatenate(got))
    expected = np.sort(ref.adc_times()[0] * 1e3)
    assert got.size == expected.size
    assert np.abs(got - expected).max() < 1e-6


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_rf_waveforms_match(seq_path, pulseq_sequences):
    # The last leg of the in-house-parser-vs-pypulseq comparison. RF has the
    # same half-raster convention that the gradients do: read_RF shifts the
    # delay by dt_rf/2 for a uniform raster, and this is what pins it.
    from feelmri.MRObjects import Scanner
    gammabar = Scanner().gammabar.m_as('Hz/T')

    ref = _reference(pulseq_sequences, seq_path)
    imp = import_pulseq(seq_path)

    n_checked = 0
    for i in range(1, len(ref.block_durations) + 1):
        rf = getattr(ref.get_block(i), 'rf', None)
        if rf is None:
            continue
        block = imp.feelmri_seq.blocks[i - 1]
        assert block.rf_pulses, f'block {i - 1} lost its RF pulse'
        got = block.rf_pulses[0]
        t0 = block.time_extent[0].m_as('ms') * 1e-3
        t_got = got.timings.m_as('ms') * 1e-3 - t0
        # feelmri carries B1 in mT, pypulseq in Hz.
        a_got = got.waveform.m_as('mT') * 1e-3 * gammabar
        t_expected = np.asarray(rf.delay + rf.t, dtype=float)
        a_expected = np.asarray(rf.signal, dtype=complex)

        assert t_got.size == t_expected.size
        assert np.abs(t_got - t_expected).max() < 1e-12
        peak = max(float(np.abs(a_expected).max()), 1.0)
        assert np.abs(a_got - a_expected).max() < 1e-9 * peak
        n_checked += 1

    if n_checked == 0:
        pytest.skip('no RF in this sequence')
