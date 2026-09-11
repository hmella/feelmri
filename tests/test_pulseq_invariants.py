import numpy as np
import pytest
from pathlib import Path

from conftest import skip_if_pypulseq_too_old
from feelmri.PulseqAdapter import import_pulseq

# Invariants an imported sequence must satisfy for the dual-path workflow to be
# sound. These need no Bloch solve, so they run on every bundled fixture.
#
# The one that matters most is the anchor: pypulseq's calculate_kspace resets
# k=0 at the excitation and integrates every gradient after it, so the k handed
# to mri_signal already contains the prephaser and slice-rephaser moments. The
# magnetization snapshot handed alongside is taken at the END of a block, so it
# carries whatever moment has accumulated by then. The two must be measured
# from the same instant or the assembler applies that moment twice.

pytestmark = pytest.mark.pulseq

DATA_DIR = Path(__file__).parent / 'data'
EXAMPLES_DIR = Path(__file__).parent.parent / 'examples' / 'pulseq'

SEQ_FILES = sorted(DATA_DIR.glob('*.seq')) + sorted(EXAMPLES_DIR.glob('*.seq'))

# 1/m. A gradient moment this small cannot wind an appreciable phase across any
# phantom: 1e-3 1/m is one cycle per kilometre.
TOL_INV_M = 1e-3


def _ids(paths):
    return [p.stem for p in paths]


@pytest.fixture(scope='module')
def pulseq_sequences():
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
        skip_if_pypulseq_too_old(seq_path)
        pytest.skip(f'pypulseq does not implement an extension used by '
                    f'{seq_path.name}; no reference available')
    return sequences[seq_path]


def _gradient_moment(waveforms, t_from_s, t_to_s, n=200000):
    """Integrate each gradient axis over [t_from, t_to], in 1/m.

    Interpolate onto a fine uniform grid rather than masking the waveform's own
    samples: waveforms_and_times stores a trapezoid as four corners, so masking
    at an arbitrary t silently drops the partial interval between t and the
    next corner. That mistake reads 2.5 1/m where the true moment is 252.5.
    """
    tt = np.linspace(t_from_s, t_to_s, n)
    out = []
    for axis in range(3):
        w = np.asarray(waveforms[axis], dtype=float)
        if w.size == 0:
            out.append(0.0)
            continue
        g = np.interp(tt, w[0], w[1], left=0.0, right=0.0)
        out.append(float(np.trapezoid(g, tt)))
    return np.array(out)


def _governing_excitation(t_excitation, t_end_s):
    """The excitation the snapshot at ``t_end_s`` belongs to: the latest one at
    or before it. calculate_kspace measures k from this instant."""
    te = np.atleast_1d(np.asarray(t_excitation)).ravel()
    earlier = te[te <= t_end_s + 1e-12]
    return float(earlier[-1]) if earlier.size else float(te[0])


def _forward_reference_applies(ref, t_anchor_s):
    """Whether a forward integral of ``waveforms_and_times`` from the governing
    excitation is a valid reference for the moment carried at ``t_anchor_s``.

    It is not, in two cases, and both are properties of the REFERENCE rather
    than of the adapter:

    * **A refocusing pulse at or before the anchor.** ``calculate_kspace``
      negates the accumulated k at a 180, so a plain integral overshoots by
      twice whatever was accumulated before it -- exactly the 920 = 2 x 460 1/m
      seen on ``se_v15``.
    * **A shaped (non-trapezoid) gradient.** ``waveforms_and_times``
      concatenates shape pieces with no padding between them, so interpolating
      across it bridges the gaps and reads a moment the sequence never played.

    The adapter is unaffected by both: it works backwards from the first ADC
    sample of the file's own trajectory, inheriting ``calculate_kspace``'s
    handling for free. Where this reference does not apply, the invariant is
    covered end-to-end instead by ``test_pulseq_analytical.py`` -- the cube
    box-transform for the encoding and the spin echo for the refocusing path.
    """
    t_ref = np.atleast_1d(np.asarray(t_refocusing_of(ref))).ravel()
    if t_ref.size and np.any(t_ref <= t_anchor_s + 1e-12):
        return False, 'a refocusing pulse precedes the anchor'
    for i in range(1, len(ref.block_durations) + 1):
        block = ref.get_block(i)
        for axis in ('gx', 'gy', 'gz'):
            g = getattr(block, axis, None)
            if g is not None and getattr(g, 'type', None) != 'trap':
                return False, 'the sequence carries a shaped gradient'
    return True, ''


def t_refocusing_of(ref):
    _k, _kf, _te, t_ref, _ta = ref.calculate_kspace()
    return t_ref if np.size(t_ref) else np.empty(0)


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_readout_kspace_is_measured_from_the_snapshot(seq_path, pulseq_sequences):
    """Each window's k-space must be the file's trajectory MINUS the gradient
    moment already carried by the magnetization at its anchor.

    Otherwise the assembler winds that moment a second time. On the bundled
    files the anchor is the RF block, whose end sits after the whole
    slice-select lobe: kz there is +252 to +460 1/m, which is 1-2 full cycles
    across a 5 mm slice.

    The decomposition itself is asserted on every fixture; the independent
    forward-integral reference only where it is valid -- see
    ``_forward_reference_applies``.
    """
    ref = _reference(pulseq_sequences, seq_path)
    imp = import_pulseq(seq_path)
    if not imp.readouts:
        pytest.skip('no ADC in this sequence')

    k_raw, _k_full, t_exc, _t_ref, t_adc = ref.calculate_kspace()
    k_raw = np.asarray(k_raw, dtype=float)
    t_adc = np.asarray(t_adc, dtype=float)
    waveforms = ref.waveforms_and_times()[0]

    for rw in imp.readouts:
        anchor = imp.feelmri_seq.blocks[rw.m_storage_block]
        t_end = anchor.time_extent[1].m_as('ms') * 1e-3

        # The window carries the raw trajectory, the moment removed, and the
        # instant it was measured from -- and they must reconstruct each other.
        assert rw.t_anchor == pytest.approx(t_end * 1e3, abs=1e-9), (
            f'window {rw.first_block}-{rw.last_block}: t_anchor is '
            f'{rw.t_anchor} ms, but the anchor block ends at {t_end * 1e3} ms')
        assert np.abs((rw.kspace_file - np.asarray(rw.k_at_anchor)) - rw.kspace).max()             < TOL_INV_M

        # The samples of this window, in the file's own trajectory.
        t0 = rw.times.min() * 1e-3
        t1 = rw.times.max() * 1e-3
        mask = (t_adc >= t0 - 1e-12) & (t_adc <= t1 + 1e-12)
        assert rw.kspace_file.shape == k_raw[:, mask].T.shape
        assert np.abs(rw.kspace_file - k_raw[:, mask].T).max() < TOL_INV_M

        applies, why = _forward_reference_applies(ref, t_end)
        if not applies:
            continue
        k_anchor = _gradient_moment(
            waveforms, _governing_excitation(t_exc, t_end), t_end)
        expected = k_raw[:, mask].T - k_anchor
        assert rw.kspace.shape == expected.shape
        worst = float(np.abs(rw.kspace - expected).max())
        assert worst < TOL_INV_M, (
            f'window {rw.first_block}-{rw.last_block}: k-space is off by '
            f'{worst:.4g} 1/m; the anchor carries a moment of '
            f'({k_anchor[0]:.2f}, {k_anchor[1]:.2f}, {k_anchor[2]:.2f}) 1/m '
            f'that is being applied twice')


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_k_at_anchor_is_recorded(seq_path, pulseq_sequences):
    """The moment subtracted above is kept on the window, so a caller can see
    what was removed and recover the file's own trajectory."""
    ref = _reference(pulseq_sequences, seq_path)
    imp = import_pulseq(seq_path)
    if not imp.readouts:
        pytest.skip('no ADC in this sequence')
    if not hasattr(imp.readouts[0], 'k_at_anchor'):
        pytest.fail('ReadoutWindow has no k_at_anchor field')

    _k, _kf, t_exc, _tr, _ta = ref.calculate_kspace()
    waveforms = ref.waveforms_and_times()[0]
    for rw in imp.readouts:
        anchor = imp.feelmri_seq.blocks[rw.m_storage_block]
        t_end = anchor.time_extent[1].m_as('ms') * 1e-3
        # kspace_file keeps the trajectory as the file wrote it.
        assert np.abs((rw.kspace_file - rw.kspace)
                      - np.asarray(rw.k_at_anchor)).max() < TOL_INV_M
        applies, _why = _forward_reference_applies(ref, t_end)
        if not applies:
            continue
        expected = _gradient_moment(
            waveforms, _governing_excitation(t_exc, t_end), t_end)
        assert np.abs(np.asarray(rw.k_at_anchor) - expected).max() < TOL_INV_M


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_raster_spans_every_block(seq_path):
    """The integration raster must cover each block exactly, and carry no
    zero-length steps.

    A half-open arange used to drop each block's final interval, leaving
    25-39% of sequence time unintegrated. The fix appends the block end, which
    can introduce a duplicate of a timing already contributed by a gradient
    corner -- the same instant computed two ways, differing by an ulp until a
    later shift collapses them.
    """
    # This is the one test here that needs no pypulseq reference, so it never
    # passes through _reference() and has to do its own version gate: under
    # pypulseq 1.4 a v1.5 file cannot be read at all and import_pulseq raises
    # out of the trajectory step.
    skip_if_pypulseq_too_old(seq_path)
    imp = import_pulseq(seq_path)
    seq = imp.feelmri_seq
    integrated = 0.0
    for i, b in enumerate(seq.blocks):
        t = b.discrete_times.m_as('ms')
        lo = b.time_extent[0].m_as('ms')
        hi = b.time_extent[1].m_as('ms')
        if b.dur.m_as('ms') > 0:
            assert t.size >= 2, f'block {i} of nonzero duration gets no step'
        if t.size < 2:
            continue
        assert abs(t[0] - lo) < 1e-9, f'block {i} raster starts at {t[0]}, not {lo}'
        assert abs(t[-1] - hi) < 1e-9, f'block {i} raster ends at {t[-1]}, not {hi}'
        steps = np.diff(t)
        assert steps.min() > 0.0, (
            f'block {i} has {int((steps <= 0).sum())} zero-length step(s); '
            f'the kernel re-exponentiates every node at each one')
        integrated += t[-1] - t[0]

    total = seq.dur.m_as('ms')
    assert abs(integrated - total) < 1e-6 * max(total, 1.0), (
        f'{integrated:.4f} of {total:.4f} ms integrated '
        f'({100 * integrated / total:.2f}%)')


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=_ids(SEQ_FILES))
def test_rf_flip_angle_round_trip(seq_path, pulseq_sequences):
    """gamma * INT(B1 dt) through the import equals what pypulseq put in the file.

    Integrate the SIGNED waveform. A sinc's side lobes are negative, so
    INT(|B1|) reads about 10% high on an apodized sinc and exact on a block
    pulse -- which looks like a shaped-pulse bug and is not one.
    """
    from feelmri.MRObjects import Scanner
    gamma = Scanner().gamma.m_as('rad/ms/mT')

    ref = _reference(pulseq_sequences, seq_path)
    imp = import_pulseq(seq_path)

    checked = 0
    for i in range(1, len(ref.block_durations) + 1):
        rf_pp = getattr(ref.get_block(i), 'rf', None)
        if rf_pp is None:
            continue
        got = imp.feelmri_seq.blocks[i - 1].rf_pulses[0]
        flip = gamma * np.trapezoid(
            np.real(got.waveform.m_as('mT')), got.timings.m_as('ms'))
        # pypulseq's own view of the same pulse: B1 in Hz, times in s.
        expected = 2 * np.pi * np.trapezoid(np.real(rf_pp.signal), rf_pp.t)
        assert abs(flip - expected) < 1e-6 * max(abs(expected), 1e-3), (
            f'block {i - 1}: {np.degrees(flip):.4f} deg imported vs '
            f'{np.degrees(expected):.4f} deg in the file')
        checked += 1

    if checked == 0:
        pytest.skip('no RF in this sequence')
