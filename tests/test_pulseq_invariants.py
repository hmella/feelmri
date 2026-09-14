import numpy as np
import pytest

from conftest import SEQ_FILES, seq_ids, skip_if_pypulseq_too_old

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

# 1/m. A gradient moment this small cannot wind an appreciable phase across any
# phantom: 1e-3 1/m is one cycle per kilometre.
TOL_INV_M = 1e-3


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


def t_refocusing_of(ref):
    _k, _kf, _te, t_ref, _ta = ref.calculate_kspace()
    return t_ref if np.size(t_ref) else np.empty(0)


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


def _readout_anchor_invariant(seq_path, pulseq_import, pypulseq_ref):
    """Each window's k-space must be the file's trajectory MINUS the gradient
    moment already carried by the magnetization at its anchor.

    Otherwise the assembler winds that moment a second time. On the bundled
    files the anchor is the RF block, whose end sits after the whole
    slice-select lobe: kz there is +252 to +460 1/m, which is 1-2 full cycles
    across a 5 mm slice.

    Three fields carry the decomposition and each must reconstruct the others:
    ``kspace_file`` as the file wrote it, ``k_at_anchor`` the moment removed,
    and ``kspace`` what a caller hands to ``mri_signal``. The identity is
    asserted on every fixture; the independent forward-integral reference only
    where it is valid -- see ``_forward_reference_applies``.
    """
    ref = pypulseq_ref(seq_path)
    imp = pulseq_import(seq_path)
    if not imp.readouts:
        pytest.skip('no ADC in this sequence')

    k_raw, _k_full, t_exc, _t_ref, t_adc = ref.calculate_kspace()
    k_raw = np.asarray(k_raw, dtype=float)
    t_adc = np.asarray(t_adc, dtype=float)
    waveforms = ref.waveforms_and_times()[0]

    for rw in imp.readouts:
        anchor = imp.feelmri_seq.blocks[rw.m_storage_block]
        t_end = anchor.time_extent[1].m_as('ms') * 1e-3

        assert rw.t_anchor == pytest.approx(t_end * 1e3, abs=1e-9), (
            f'window {rw.first_block}-{rw.last_block}: t_anchor is '
            f'{rw.t_anchor} ms, but the anchor block ends at {t_end * 1e3} ms')
        assert np.abs((rw.kspace_file - np.asarray(rw.k_at_anchor))
                      - rw.kspace).max() < TOL_INV_M, 'kf - k_at_anchor != k'

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
        # The moment the adapter removed is the moment actually played.
        assert np.abs(np.asarray(rw.k_at_anchor) - k_anchor).max() < TOL_INV_M
        expected = k_raw[:, mask].T - k_anchor
        assert rw.kspace.shape == expected.shape
        worst = float(np.abs(rw.kspace - expected).max())
        assert worst < TOL_INV_M, (
            f'window {rw.first_block}-{rw.last_block}: k-space is off by '
            f'{worst:.4g} 1/m; the anchor carries a moment of '
            f'({k_anchor[0]:.2f}, {k_anchor[1]:.2f}, {k_anchor[2]:.2f}) 1/m '
            f'that is being applied twice')


def _raster_spans_every_block(seq_path, pulseq_import):
    """The integration raster must cover each block exactly, and carry no
    zero-length steps.

    A half-open arange used to drop each block's final interval, leaving
    25-39% of sequence time unintegrated. The fix appends the block end, which
    can introduce a duplicate of a timing already contributed by a gradient
    corner -- the same instant computed two ways, differing by an ulp until a
    later shift collapses them.

    The `t.size >= 2` leg is the one that matters most on its own: a block with
    fewer points gets no integration step at all and evolves the magnetization
    not at all. Event-free delays are the usual case, and in a spin echo they
    carry the T2 weighting.
    """
    # This is the one test here that needs no pypulseq reference, so it never
    # asks for one and has to do its own version gate: under pypulseq 1.4 a
    # v1.5 file cannot be read at all and import_pulseq raises out of the
    # trajectory step.
    skip_if_pypulseq_too_old(seq_path)
    seq = pulseq_import(seq_path).feelmri_seq
    integrated = 0.0
    unstepped = []
    for i, b in enumerate(seq.blocks):
        t = b.discrete_times.m_as('ms')
        lo = b.time_extent[0].m_as('ms')
        hi = b.time_extent[1].m_as('ms')
        if b.dur.m_as('ms') > 0 and t.size < 2:
            unstepped.append(i)
        if t.size < 2:
            continue
        assert abs(t[0] - lo) < 1e-9, f'block {i} raster starts at {t[0]}, not {lo}'
        assert abs(t[-1] - hi) < 1e-9, f'block {i} raster ends at {t[-1]}, not {hi}'
        steps = np.diff(t)
        assert steps.min() > 0.0, (
            f'block {i} has {int((steps <= 0).sum())} zero-length step(s); '
            f'the kernel re-exponentiates every node at each one')
        integrated += t[-1] - t[0]

    assert not unstepped, (
        f'{len(unstepped)} block(s) of nonzero duration get no integration '
        f'step: {unstepped[:10]}')

    total = seq.dur.m_as('ms')
    assert abs(integrated - total) < 1e-6 * max(total, 1.0), (
        f'{integrated:.4f} of {total:.4f} ms integrated '
        f'({100 * integrated / total:.2f}%)')


def _no_rf_plays_inside_a_readout_window(seq_path, pulseq_import):
    """The dual path solves the sequence ONCE and then synthesizes each readout
    from its trajectory. That is equivalent to evolving through the readout only
    while no RF plays between the magnetization snapshot and the last sample:
    the equivalence rests on `2*pi k.r = gamma integral(G.r dt)`, which is a
    statement about precession under a longitudinal field alone.

    An RF pulse inside a window would break it silently -- the assembler has no
    B1 channel at all, so the signal would simply be computed as though the
    pulse had not happened. Nothing checked this, on any fixture.

    `_identify_readout_groups` is what keeps it true: a window ends at the last
    ADC attributed to one coherence anchor, and the next RF starts a new one.
    This asserts the property the grouping is supposed to deliver.
    """
    skip_if_pypulseq_too_old(seq_path)
    imp = pulseq_import(seq_path)
    if not imp.readouts:
        pytest.skip('no ADC in this sequence')

    blocks = imp.feelmri_seq.blocks
    checked = 0
    for rw in imp.readouts:
        if rw.m_storage_idx < 0:
            continue
        # Strictly after the anchor block, up to and including the last block
        # the window covers.
        for i in range(rw.m_storage_block + 1, rw.last_block + 1):
            pulses = getattr(blocks[i], 'rf_pulses', None) or []
            assert not pulses, (
                f'block {i} carries RF inside the readout window '
                f'{rw.first_block}-{rw.last_block}, whose snapshot is at block '
                f'{rw.m_storage_block}; the assembler has no B1 channel and '
                f'would ignore it')
        checked += 1
    assert checked > 0, 'no window had a usable anchor, so nothing was checked'

@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=seq_ids(SEQ_FILES))
def test_the_import_invariants_hold_on_every_fixture(seq_path, pulseq_import, pypulseq_ref):
  """Three invariants of ONE import, on one fixture.

  They shared a fixture, an import and a parametrisation and differed only in
  which property they read off it, so sweeping fifteen files three times
  re-imported nothing and asserted three unrelated things separately. Each
  helper keeps its own assertions and messages, so a failure still names which
  invariant broke -- only the test COUNT changes.
  """
  checks = (
      (_readout_anchor_invariant, (seq_path, pulseq_import, pypulseq_ref,)),
      (_raster_spans_every_block, (seq_path, pulseq_import,)),
      (_no_rf_plays_inside_a_readout_window, (seq_path, pulseq_import,)),
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
