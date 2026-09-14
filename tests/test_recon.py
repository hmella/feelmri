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


# `n_coils=1` is not in this sweep: `test_none_and_rss_pass_a_single_channel
# _through_where_roemer_divides` asserts the same roemer line verbatim, plus
# the two claims only the single-channel case can make.
@pytest.mark.parametrize('n_coils', [2, 8])
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


def test_none_and_rss_pass_a_single_channel_through_where_roemer_divides():
    """The three differ at one channel, and the difference is deliberate:
    `None` and `'rss'` keep their pre-existing behaviour, while the matched
    filter still divides the shading out.

    The multi-channel pass-through is asserted here too, so `None` keeping
    every channel is covered at one channel and at three by one test."""
    S, m = _fields(1)
    img = S * m[None]
    assert np.array_equal(_combine_channels(img, None, None), img[0])
    assert np.array_equal(_combine_channels(img, 'rss', None), img[0])
    divided = _combine_channels(img, 'roemer', S)
    assert np.abs(divided - m.astype(np.complex64)).max() < 1e-5 * np.abs(m).max()
    # ... and that IS a change: the map varies by 37x across this grid, so the
    # uncombined single-coil image is off by 2.1x the object's own peak.
    assert np.abs(img[0] - m).max() > 0.5 * np.abs(m).max()

    # And at more than one channel `None` is a pass-through of the whole stack.
    S3, m3 = _fields(3)
    img3 = S3 * m3[None]
    kept = _combine_channels(img3, None, None)
    assert kept.shape == img3.shape
    assert np.array_equal(kept, img3)


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


def _radial_traj(n_ro=16, n_spokes=9, n_slices=1, kmax=0.4):
    """Golden-ish radial spokes, tiny. `n_slices > 1` repeats the same in-plane
    spokes at Cartesian kz, which is what makes `reconstruct_nufft` take the
    hybrid branch instead of the full 3-D one."""
    r = np.linspace(-kmax, kmax, n_ro)
    ang = np.pi * np.arange(n_spokes) / n_spokes
    kx = np.outer(r, np.cos(ang))
    ky = np.outer(r, np.sin(ang))
    kx = np.repeat(kx[:, :, None], n_slices, axis=2)
    ky = np.repeat(ky[:, :, None], n_slices, axis=2)
    if n_slices == 1:
        kz = np.zeros_like(kx)
    else:
        kzv = (np.arange(n_slices) - n_slices // 2) / float(n_slices)
        kz = np.broadcast_to(kzv[None, None, :], kx.shape).copy()
    return kx, ky, kz


@pytest.mark.parametrize('n_slices,branch', [(1, 'full_3d'), (4, 'hybrid')])
def test_the_combine_arguments_reach_both_dispatch_branches(monkeypatch,
                                                            n_slices, branch):
    """`reconstruct_nufft` picks between a hybrid stack-of-X path and a full
    3-D one, and `combine` / `sensitivities` are threaded separately through
    each. Neither had a test.

    What is checked is the plumbing: the channel axis must arrive LEADING, the
    map must arrive unchanged, and the image shape must match it -- which is
    what `_combine_channels` then requires of them.
    """
    from feelmri import Recon

    img_shape = (8, 8, 4) if n_slices > 1 else (8, 8)
    kx, ky, kz = _radial_traj(n_slices=n_slices)
    n_ch = 3
    rng = np.random.default_rng(3)
    kdata = (rng.normal(size=(*kx.shape, n_ch))
             + 1j * rng.normal(size=(*kx.shape, n_ch))).astype(np.complex64)
    S = (rng.normal(size=(n_ch, *img_shape))
         + 1j * rng.normal(size=(n_ch, *img_shape))).astype(np.complex64)

    seen = {}
    real = Recon._combine_channels

    def spy(img, combine, sensitivities):
        seen['shape'] = img.shape
        seen['combine'] = combine
        seen['map_is'] = sensitivities is S
        return real(img, combine, sensitivities)

    monkeypatch.setattr(Recon, '_combine_channels', spy)
    out = Recon.reconstruct_nufft(kdata, (kx, ky, kz), img_shape,
                                  auto_dcw=None, combine='roemer',
                                  sensitivities=S)

    assert seen['combine'] == 'roemer'
    assert seen['map_is'], 'the map did not reach the combine unchanged'
    assert seen['shape'] == (n_ch, *img_shape), (
        f'{branch}: the channel axis arrived as {seen["shape"]} against a map '
        f'of {S.shape}')
    assert out.shape == img_shape


@pytest.mark.parametrize('n_slices', [1, 4])
def test_combining_inside_the_recon_equals_combining_after_it(n_slices):
    """The reconstruction is linear per channel, so collapsing the channels
    inside it must give exactly what collapsing its `combine=None` output
    gives. That pins the ORDER of the two operations, which is the only thing
    the caller cannot check for themselves."""
    from feelmri import Recon

    img_shape = (8, 8, 4) if n_slices > 1 else (8, 8)
    kx, ky, kz = _radial_traj(n_slices=n_slices)
    n_ch = 2
    rng = np.random.default_rng(11)
    kdata = (rng.normal(size=(*kx.shape, n_ch))
             + 1j * rng.normal(size=(*kx.shape, n_ch))).astype(np.complex64)
    S = (rng.normal(size=(n_ch, *img_shape))
         + 1j * rng.normal(size=(n_ch, *img_shape))).astype(np.complex64)

    channels = Recon.reconstruct_nufft(kdata, (kx, ky, kz), img_shape,
                                       auto_dcw=None, combine=None)
    assert channels.shape == (n_ch, *img_shape)
    inside = Recon.reconstruct_nufft(kdata, (kx, ky, kz), img_shape,
                                     auto_dcw=None, combine='roemer',
                                     sensitivities=S)
    after = _combine_channels(channels, 'roemer', S)
    scale = float(np.abs(after).max())
    assert float(np.abs(inside - after).max()) < 1e-5 * scale
    # Not vacuous: the combine has to be doing something to the channels.
    assert float(np.abs(channels[0] - after).max()) > 1e-3 * scale
