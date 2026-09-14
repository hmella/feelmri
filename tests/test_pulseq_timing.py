import numpy as np
import pytest

from conftest import SEQ_FILES, pypulseq_block_durations_ms, seq_ids

# The parser gate: what the in-house reader makes of a .seq file, against what
# pypulseq makes of the SAME file. Four properties, one test each, every one
# parametrised over every bundled fixture, so a failure names both the property
# and the file.
#
# Four, not nine. Block count, absolute start times and total duration are the
# length, the cumulative sum and the sum of the per-block duration array, so
# they cannot fail while it passes -- they are folded into one test rather than
# run as three more sweeps over fifteen files.
#
# What this gate CANNOT catch is an error the two readers share; that is what
# test_pulseq_native_equivalence.py and test_pulseq_analytical.py are for.

pytestmark = pytest.mark.pulseq

# Time tolerance in ms. The block raster is 10 us, so anything at this level is
# floating point noise rather than a timing difference.
TOL_MS = 1e-9


def _timing_grid_matches(seq_path, pulseq_import, pypulseq_ref):
    """Block count, per-block duration, absolute start and total duration.

    The four are one property. `time_extent[0]` is the running sum of the
    durations and `seq.dur` their total, so asserting them separately over
    fifteen fixtures re-runs the same comparison three more times; asserting
    them together still names which of the four broke.
    """
    ref = pypulseq_ref(seq_path)
    imp = pulseq_import(seq_path)
    blocks = imp.feelmri_seq.blocks

    assert len(blocks) == len(ref.block_durations)

    expected = pypulseq_block_durations_ms(ref)
    got = np.array([b.dur.m_as('ms') for b in blocks])
    assert np.abs(expected - got).max() < TOL_MS, 'per-block duration'

    # The absolute start of every block, which is where drift accumulates.
    expected_start = np.concatenate(([0.0], np.cumsum(expected)[:-1]))
    got_start = np.array([b.time_extent[0].m_as('ms') for b in blocks])
    assert np.abs(expected_start - got_start).max() < TOL_MS, 'block start time'

    assert abs(ref.duration()[0] * 1e3
               - imp.feelmri_seq.dur.m_as('ms')) < TOL_MS, 'total duration'


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


def _gradient_waveforms_match(seq_path, pulseq_import, pypulseq_ref):
    # The gradient the solver integrates must be the gradient pypulseq plays.
    # This is what catches a shaped gradient laid down half a raster off, one
    # missing the first/last boundary samples of a v1.5 file, or a gamma that
    # does not match the one the solver multiplies by.
    from feelmri.MRObjects import Scanner
    gammabar = Scanner().gammabar.m_as('Hz/T')

    ref = pypulseq_ref(seq_path)
    imp = pulseq_import(seq_path)
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


def _adc_times_match(seq_path, pulseq_import, pypulseq_ref):
    """The in-house read_ADC, and the readout-window partition over it.

    Two different claims, and only the first tests our own parser.
    ReadoutWindow.times is sliced out of calculate_kspace, so comparing it
    against adc_times() compares pypulseq with itself. It still says something
    the block-level check does not -- that the windows between them cover every
    ADC sample exactly once, with none dropped or double-counted by the
    grouping in _identify_readout_groups.
    """
    ref = pypulseq_ref(seq_path)
    imp = pulseq_import(seq_path)
    expected = np.sort(ref.adc_times()[0] * 1e3)

    # 1. The block's own ADC: its dwell and delay convention (half a dwell into
    #    the first sample) has to agree with the file.
    got = [block.time_extent[0].m_as('ms') + block.adc.times.m_as('ms')
           for block in imp.feelmri_seq.blocks if block.adc is not None]
    if not got:
        pytest.skip('no ADC in this sequence')
    got = np.sort(np.concatenate(got))
    assert got.size == expected.size, 'in-house ADC sample count'
    assert np.abs(got - expected).max() < 1e-6, 'in-house ADC sample times'

    # 2. The windows partition those samples.
    windowed = np.sort(np.concatenate([r.times for r in imp.readouts]))
    assert windowed.size == expected.size, 'readout windows drop or repeat samples'
    assert np.abs(windowed - expected).max() < 1e-6, 'readout window coverage'


def _rf_waveforms_match(seq_path, pulseq_import, pypulseq_ref):
    # The last leg of the in-house-parser-vs-pypulseq comparison. RF has the
    # same half-raster convention that the gradients do: read_RF shifts the
    # delay by dt_rf/2 for a uniform raster, and this is what pins it.
    #
    # Pinning the samples pointwise at 1e-9 of peak also pins every functional
    # of them -- the flip angle gamma*INT(B1 dt) included, which is why there is
    # no separate flip-angle round trip here. The delivered flip through the
    # SOLVER is a different question, and lives in test_pulseq_analytical.py.
    from feelmri.MRObjects import Scanner
    gammabar = Scanner().gammabar.m_as('Hz/T')

    ref = pypulseq_ref(seq_path)
    imp = pulseq_import(seq_path)

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

@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=seq_ids(SEQ_FILES))
def test_the_in_house_parser_agrees_with_pypulseq_on_every_fixture(seq_path, pulseq_import, pypulseq_ref):
  """Every quantity the in-house parser and pypulseq must agree on.

  Four legs of ONE comparison -- timing grid, gradients, ADC sample times, RF
  waveforms -- over the same fixture, the same import and the same reference.
  Sweeping fifteen files four times re-read nothing; each helper keeps its own
  tolerances and messages, so a failure still names which leg broke.
  """
  checks = (
      (_timing_grid_matches, (seq_path, pulseq_import, pypulseq_ref,)),
      (_gradient_waveforms_match, (seq_path, pulseq_import, pypulseq_ref,)),
      (_adc_times_match, (seq_path, pulseq_import, pypulseq_ref,)),
      (_rf_waveforms_match, (seq_path, pulseq_import, pypulseq_ref,)),
  )
  skipped = []
  for fn, fn_args in checks:
    try:
      fn(*fn_args)
    except pytest.skip.Exception as exc:
      # Per CHECK, not per test. A skip inside one helper would otherwise
      # abort the others, so a fixture with no ADC would silently stop being
      # checked for everything else -- a coverage loss disguised as a skip.
      skipped.append(f'{fn.__name__}: {exc}')
  if len(skipped) == len(checks):
    pytest.skip('; '.join(skipped))
