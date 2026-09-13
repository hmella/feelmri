"""Channel combination in feelmri.Recon.

`Recon.py` had no tests at all. `_combine_channels` is pure numpy, so the
matched filter can be checked as an identity rather than a tolerance: with
`I_c = S_c * m` the Roemer estimator must return `m` exactly.
"""
import numpy as np
import pytest

from feelmri.Recon import _combine_channels


def _fields(n_coils=4, shape=(6, 5), seed=7):
    """A complex map and a complex object, both with structure in the PHASE.

    The map MUST be complex. A real map makes `conj(S)` and `S` the same thing,
    so dropping the conjugate -- which doubles the object's phase error -- is
    invisible; that is the one mistake this combination is prone to.
    """
    rng = np.random.default_rng(seed)
    S = (rng.normal(size=(n_coils, *shape))
         + 1j * rng.normal(size=(n_coils, *shape)))
    m = (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    return S, m


@pytest.mark.parametrize('n_coils', [1, 2, 8])
def test_roemer_inverts_a_known_sensitivity_exactly(n_coils):
    """`I_c = S_c m` is what the forward model produces, so the matched filter
    returns `m` itself -- magnitude shading and coil phase both removed."""
    S, m = _fields(n_coils)
    got = _combine_channels(S * m[None], 'roemer', S)
    assert np.abs(got - m.astype(np.complex64)).max() < 1e-5 * np.abs(m).max()


def test_roemer_needs_the_conjugate_of_the_map():
    """Combining with `S` instead of `conj(S)` leaves the coil phase in.

    Measured on this map: 3.03 rad of worst-case phase error against 3.5e-08
    for the correct form, plus 52% on the magnitude. A REAL map shows neither,
    since `conj(S) == S` there -- which is why the fixture is complex, and why
    a real-valued test of this estimator would pin nothing.
    """
    S, m = _fields(3)
    right = _combine_channels(S * m[None], 'roemer', S)
    wrong = np.sum(S * (S * m[None]), axis=0) / np.sum(np.abs(S) ** 2, axis=0)
    err_right = np.abs(np.angle(right / m)).max()
    err_wrong = np.abs(np.angle(wrong / m)).max()
    assert err_right < 1e-5
    assert err_wrong > 1.0, (
        f'dropping the conjugate costs only {err_wrong:.3f} rad here, so this '
        f'map cannot discriminate the two')


def test_rss_recovers_the_magnitude_and_discards_the_phase():
    """Which is the whole trade: no map needed, no phase kept."""
    S, m = _fields(4)
    got = _combine_channels(S * m[None], 'rss', None)
    expected = np.sqrt(np.sum(np.abs(S) ** 2, axis=0)) * np.abs(m)
    assert np.abs(np.abs(got) - expected).max() < 1e-4 * expected.max()
    assert np.abs(np.angle(got)).max() < 1e-6, 'rss returned a phase'
    assert np.abs(np.angle(m)).max() > 1.0, 'the object had no phase to lose'


def test_a_single_channel_is_unwrapped_by_none_and_rss_but_divided_by_roemer():
    """The three differ at one channel, and the difference is deliberate:
    `None` and `'rss'` keep their pre-existing behaviour, while the matched
    filter still divides the shading out."""
    S, m = _fields(1)
    img = S * m[None]
    assert np.array_equal(_combine_channels(img, None, None), img[0])
    assert np.array_equal(_combine_channels(img, 'rss', None), img[0])
    divided = _combine_channels(img, 'roemer', S)
    assert np.abs(divided - m.astype(np.complex64)).max() < 1e-5 * np.abs(m).max()
    # ... and that IS a change: the map varies by 37x across this grid, so the
    # uncombined single-coil image is off by 2.1x the object's own peak.
    assert np.abs(img[0] - m).max() > 0.5 * np.abs(m).max()


def test_none_keeps_every_channel():
    S, m = _fields(3)
    img = S * m[None]
    kept = _combine_channels(img, None, None)
    assert kept.shape == img.shape
    assert np.array_equal(kept, img)


def test_roemer_zeroes_the_region_no_coil_can_see():
    """Where the denominator is at the working-precision floor the division is
    meaningless rather than merely noisy, so it returns zero instead of a large
    complex number. Above the floor it is left alone -- the 1/|S| amplification
    is the estimator's physics and the caller's choice of map."""
    S, m = _fields(2)
    S[:, 0, 0] = 0.0
    out = _combine_channels(S * m[None], 'roemer', S)
    assert out[0, 0] == 0
    assert np.abs(out[1, 1]) > 0


@pytest.mark.parametrize('kwargs,match', [
    (dict(combine='roemer', sensitivities=None), 'needs `sensitivities`'),
    (dict(combine='wombat', sensitivities=None), 'unknown combine'),
])
def test_bad_combine_arguments_are_refused(kwargs, match):
    S, m = _fields(2)
    with pytest.raises(ValueError, match=match):
        _combine_channels(S * m[None], kwargs['combine'], kwargs['sensitivities'])


def test_a_mismatched_map_is_refused():
    """Channel for channel and voxel for voxel: a map of the wrong shape would
    otherwise broadcast into a plausible, wrong image."""
    S, m = _fields(3)
    with pytest.raises(ValueError, match='against images'):
        _combine_channels(S * m[None], 'roemer', S[:2])
