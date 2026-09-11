import numpy as np
import pytest
from pathlib import Path
from pint import Quantity as Q_

from conftest import skip_if_pypulseq_too_old

# Closed-form checks that join the two halves of the dual-path workflow: a
# solver-produced magnetization, an ADC-derived trajectory, and an answer known
# on paper. Nothing else in the suite does this. Every other Pulseq test is
# relative -- FEelMRI against pypulseq's reading of the same file, or native
# against imported -- so both arms can be wrong identically and stay green.
#
# The fixtures are built by tests/data/generate_seq_fixtures.py with zero dead
# time and hard (non-selective) pulses, so the transverse magnetization a pulse
# creates is spatially uniform and the algebra below has no hidden terms.

pytestmark = [pytest.mark.pulseq, pytest.mark.timeout(240)]

DATA_DIR = Path(__file__).parent / 'data'

T1_MS = 800.0      # must match generate_seq_fixtures.py
T2_MS = 60.0
TAU_MS = 5.0


def _require(name):
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f'run tests/data/generate_seq_fixtures.py to build {name}')
    skip_if_pypulseq_too_old(path)
    return path


@pytest.fixture(scope='module')
def cube(tmp_path_factory):
    """A cube of exactly known volume, integrated at high quadrature order.

    Same assembler settings as
    test_fe_element_ordering.py::test_cube_signal_at_k0_equals_volume, which
    pins Re S(0) == volume to 1e-4 -- but bypasses the sequence entirely.
    """
    pytest.importorskip('mpi4py')
    pytest.importorskip('pymetis')
    try:
        from feelmri.Phantom import FEMPhantom
    except ImportError as exc:
        pytest.skip(f'feelmri C++ extensions not available: {exc}')
    from _phantom_fixtures import make_cube_mesh

    path = tmp_path_factory.mktemp('an') / 'cube.vtu'
    _p, volume = make_cube_mesh(path, 'tetra', n=2, scale=2e-3)
    phantom = FEMPhantom(path=str(path))
    phantom.set_assembler(voxel_size=0.0, lorder=2, horder=4,
                          nodal_approximation=False, lumped=False)
    return phantom, float(volume)


def _static(phantom, t2_ms, dw_rad_per_ms=0.0):
    n = phantom.local_nodes.shape[0]
    phantom.set_static_fields(
        T2=np.full(n, t2_ms, dtype=np.float32),
        phi_dB0=np.full(n, dw_rad_per_ms, dtype=np.float32))


def _anchor_end_ms(imp, rw):
    return imp.feelmri_seq.blocks[rw.m_storage_block].time_extent[1].m_as('ms')


def test_t1_recovery_through_the_import(cube):
    """A hard 90 leaves Mz = 0; a delay of exactly T1 recovers it to
    M0(1 - e^-1).

    Relaxation under a constant field composes exactly step by step, so this is
    insensitive to how coarse the raster is -- what it does catch is raster
    that is MISSING. A half-open arange used to drop each block's final
    interval, leaving a quarter to a third of the sequence unintegrated, which
    lands straight on this number.
    """
    from feelmri.Bloch import BlochSolver
    from feelmri.PulseqAdapter import import_pulseq

    phantom, _volume = cube
    imp = import_pulseq(_require('t1_v15.seq'))
    seq = imp.feelmri_seq
    # The file has no ADC, so the import flags nothing; take the snapshot at
    # the end of the delay ourselves.
    seq.blocks[-1].store_magnetization = True

    solver = BlochSolver(sequence=seq, phantom=phantom, M0=1.0,
                         T1=Q_(T1_MS, 'ms'), T2=Q_(1e9, 'ms'),
                         dtype='float64')
    _Mxy, Mz = solver.solve()

    expected = 1.0 - np.exp(-1.0)
    got = float(np.real(Mz[:, -1]).mean())
    assert abs(got - expected) < 5e-4, (
        f'Mz after one T1 is {got:.6f}, expected {expected:.6f}')


def test_fid_decays_as_exp_minus_t_over_t2(cube):
    """A hard 90 then a gradient-free readout: the trajectory sits at k = 0, so
    every sample is the whole object integral and decays purely as exp(-t/T2).

    S(t) = V * exp(-(t - t_anchor)/T2), with t measured from the instant the
    magnetization was snapshotted. This pins the ABSOLUTE level, which is what
    makes it sensitive to the time origin: handing the assembler times measured
    from the start of the .seq file applies a spurious exp(-t_abs/T2) that a
    ratio-only check would cancel out.
    """
    from feelmri.PulseqAdapter import simulate_pulseq

    phantom, volume = cube
    _static(phantom, T2_MS)
    path = _require('fid_v15.seq')

    sim = simulate_pulseq(path, phantom, M0=1.0, T1=Q_(1e9, 'ms'),
                          T2=Q_(T2_MS, 'ms'), dtype='float64')
    assert len(sim.kspace) == 1
    rw = sim.imp.readouts[0]
    t = rw.times - _anchor_end_ms(sim.imp, rw)          # ms since the snapshot
    got = np.abs(sim.kspace[0][:, 0, 0, 0])
    expected = volume * np.exp(-t / T2_MS)

    worst = float(np.abs(got - expected).max() / expected.max())
    assert worst < 2e-3, (
        f'FID departs from V*exp(-t/T2) by {worst:.3e} relative; '
        f'first sample {got[0]:.6g} vs {expected[0]:.6g}, '
        f'last {got[-1]:.6g} vs {expected[-1]:.6g}')


def test_gradient_echo_follows_the_box_transform(cube):
    """A rewound readout over a cube: |S(k)| = V * |sinc(k L)|.

    The cube spans [0, L]^3, so its Fourier transform along the readout axis is
    V sinc(k L) times a linear phase from the offset; the magnitude is the
    closed form. k passes through zero at the plateau centre, where the signal
    must be the exact volume.
    """
    from feelmri.PulseqAdapter import simulate_pulseq

    phantom, volume = cube
    _static(phantom, 1e9)                                # no relaxation
    path = _require('gre_an_v15.seq')
    L = 4e-3                                             # n=2 * scale=2e-3

    sim = simulate_pulseq(path, phantom, M0=1.0, T1=Q_(1e9, 'ms'),
                          T2=Q_(1e9, 'ms'), dtype='float64')
    rw = sim.imp.readouts[0]
    kx = rw.kspace[:, 0].astype(float)
    got = np.abs(sim.kspace[0][:, 0, 0, 0])
    expected = volume * np.abs(np.sinc(kx * L))          # np.sinc is sin(pi x)/(pi x)

    centre = int(np.argmin(np.abs(kx)))
    assert abs(kx[centre]) < 1e-2, 'no sample sits at k = 0'
    assert abs(got[centre] - volume) < 1e-3 * volume, (
        f'signal at k=0 is {got[centre]:.6g}, cube volume is {volume:.6g}')

    worst = float(np.abs(got - expected).max() / volume)
    assert worst < 5e-3, (
        f'|S(k)| departs from V|sinc(kL)| by {worst:.3e} of the volume; '
        f'kL spans {abs(kx).max() * L:.3f}')


def test_spin_echo_refocuses_off_resonance(cube):
    """90 - tau - 180 - tau: at the echo the off-resonance phase is refocused
    whatever its value.

    A uniform off-resonance costs no magnitude -- every spin turns together --
    so the statement lives in the phase: arg S at the echo is independent of
    dB0, while away from the echo it is not. The second half is the negative
    control, without which the test would pass on a solver that simply ignored
    the field.
    """
    from feelmri.PulseqAdapter import simulate_pulseq

    phantom, _volume = cube
    path = _require('se_an_v15.seq')
    # The off-resonance has to reach the SOLVER, as delta_B in mT: only a field
    # the pulses actually precess in can be refocused by one. The phantom's
    # phi_dB0 is an assembler-side demodulation term applied after the fact, so
    # a 180 cannot touch it -- driving this test through phi_dB0 measures
    # nothing but exp(i phi t).
    dB_mT = 1e-3
    # The same field must be described to BOTH halves: delta_B in mT for the
    # solver, phi_dB0 = gamma*delta_B in rad/ms for the assembler, which
    # continues the precession through the readout. Give it to only one and the
    # signal stops turning at the snapshot.
    from feelmri.MRObjects import Scanner
    gamma = Scanner().gamma.m_as('rad/ms/mT')

    args = {}
    for label, dB in (('on', 0.0), ('off', dB_mT)):
        _static(phantom, T2_MS, gamma * dB)
        sim = simulate_pulseq(path, phantom, M0=1.0, T1=Q_(1e9, 'ms'),
                              T2=Q_(T2_MS, 'ms'), delta_B=dB, dtype='float64')
        rw = sim.imp.readouts[0]
        args[label] = (rw.times.copy(), np.angle(sim.kspace[0][:, 0, 0, 0]))

    t, phase_on = args['on']
    _t, phase_off = args['off']

    # The echo sits at 2*t_refocusing - t_excitation. Take both from pypulseq
    # rather than re-deriving them from the fixture's block durations.
    pp = pytest.importorskip('pypulseq')
    ref = pp.Sequence()
    ref.read(str(path), detect_rf_use=False)
    _k, _kf, t_exc, t_ref, _ta = ref.calculate_kspace()
    t_echo = (2 * float(np.atleast_1d(t_ref).ravel()[0])
              - float(np.atleast_1d(t_exc).ravel()[0])) * 1e3
    assert t.min() <= t_echo <= t.max(), (
        f'the echo at {t_echo:.3f} ms is outside the readout '
        f'[{t.min():.3f}, {t.max():.3f}] ms')
    echo = int(np.argmin(np.abs(t - t_echo)))
    far = int(np.argmax(np.abs(t - t_echo)))

    def wrap(d):
        return float(abs(np.angle(np.exp(1j * d))))

    at_echo = wrap(phase_off[echo] - phase_on[echo])
    away = wrap(phase_off[far] - phase_on[far])
    assert at_echo < 5e-2, (
        f'off-resonance phase is not refocused at the echo: {at_echo:.4f} rad')
    assert away > 5 * max(at_echo, 1e-3), (
        f'phase away from the echo ({away:.4f} rad) is not distinguishable '
        f'from the echo ({at_echo:.4f} rad) -- the field may be ignored')


@pytest.mark.parametrize('name,nominal_deg', [
    ('gre_v15.seq', 10.0), ('se_v15.seq', 90.0),
    ('t1_v15.seq', 90.0), ('fid_v15.seq', 90.0),
])
def test_imported_rf_delivers_its_nominal_flip(cube, name, nominal_deg):
    """An imported pulse must rotate M by the angle the .seq file specifies.

    The gradients are dropped first: with the slice-select lobe on, a phantom of
    finite thickness samples the slice PROFILE, and the mean Mz then reads far
    below the nominal flip -- 8.57 of 10 deg on gre_v15, 73.7 of 90 on se_v15.
    That is correct physics and not what this test is about; measuring it
    without dropping the gradients reads those numbers as a 14-18% error.

    Two raster defects showed here and neither is visible in
    test_pulseq_invariants.py's waveform round trip, which integrates the pulse
    on its own timings rather than on the raster the solver steps over:

    * The dt_rf grid was laid on [0, dur] while an imported pulse lives on
      [delay, delay + dur] -- 0.52% of flip.
    * The kernel charges each interval the field at its END, so the interval
      arriving at a pulse edge was billed full RF amplitude however long it
      was; on ppm_v15 the raster jumped 0 -> 0.1 ms onto the pulse start,
      +1.25%.
    """
    from feelmri.Bloch import BlochSolver, Sequence
    from feelmri.PulseqAdapter import import_pulseq

    phantom, _volume = cube
    imp = import_pulseq(_require(name))
    block = next(b for b in imp.feelmri_seq.blocks if b.rf_pulses).copy()
    block.gradients = []
    block.store_magnetization = True

    seq = Sequence()
    seq.add_block(block)
    _Mxy, Mz = BlochSolver(sequence=seq, phantom=phantom, M0=1.0,
                           T1=Q_(1e9, 'ms'), T2=Q_(1e9, 'ms'),
                           dtype='float64', perfect_spoiling=False).solve()

    got = np.degrees(np.arccos(np.clip(float(np.real(Mz[:, -1]).mean()), -1.0, 1.0)))
    assert abs(got - nominal_deg) < 5e-3 * nominal_deg, (
        f'{name}: the solver delivers {got:.4f} deg against a nominal '
        f'{nominal_deg:.4f} deg ({100 * (got / nominal_deg - 1):+.3f}%)')


def _column_along_z(path, half=8e-3, n=240, width=2e-4):
    """A thin column of spins along z, centred on z = 0.

    The slice axis has to be resolved to see a slice profile, and the shared
    `cube` fixture is 4 mm across a 5 mm nominal slice -- too coarse, which is
    why `test_imported_rf_delivers_its_nominal_flip` strips the gradients
    instead of resolving the profile.
    """
    import meshio
    z = np.linspace(-half, half, n + 1)
    pts, cells = [], []
    for zz in z:
        for (dx, dy) in ((0, 0), (width, 0), (0, width), (width, width)):
            pts.append([dx, dy, zz])
    for k in range(n):
        b, t = 4 * k, 4 * (k + 1)
        cells += [[b, b + 1, b + 2, t], [b + 1, b + 2, b + 3, t],
                  [b + 1, b + 3, t, t + 2]]
    meshio.write(str(path), meshio.Mesh(np.array(pts), [("tetra", np.array(cells))]))
    return path


@pytest.mark.slow
def test_slice_profile_matches_the_small_tip_transform(tmp_path):
    """A sinc plus a slice-select lobe must excite the profile the pulse's
    Fourier transform predicts.

    In the small-tip regime `Mxy(z) ~ FT{B1}(gammabar * Gz * z)`. This is the
    single most load-bearing untested behaviour of an imported `.seq`: it is
    what makes a slice a slice, and getting it wrong is invisible to every
    other test in the suite -- `test_imported_rf_delivers_its_nominal_flip`
    deletes the gradients precisely so the profile cannot contaminate its mean,
    and `test_bloch_magnus.py` compares two numerical methods to a fine-dt
    self-reference rather than to an analytic profile.

    It also pins the rephaser: without it the profile is there but the phase
    across the slice is not refocused, and |Mxy| integrated over z collapses.
    """
    pytest.importorskip('mpi4py')
    pytest.importorskip('pymetis')
    pytest.importorskip('meshio')
    from feelmri.Bloch import BlochSolver, Sequence
    from feelmri.MRObjects import Scanner
    from feelmri.Phantom import FEMPhantom
    from feelmri.PulseqAdapter import import_pulseq

    path = _column_along_z(tmp_path / 'column.vtu')
    phantom = FEMPhantom(path=str(path))
    phantom.set_assembler(voxel_size=0.0, lorder=2, horder=2,
                          nodal_approximation=False, lumped=False)

    imp = import_pulseq(_require('gre_v15.seq'))
    seq = Sequence()
    seq.add_block(imp.feelmri_seq.blocks[0].copy())       # sinc + slice select
    seq.add_block(imp.feelmri_seq.blocks[1].copy())       # slice rephaser
    seq.blocks[-1].store_magnetization = True
    Mxy, _Mz = BlochSolver(sequence=seq, phantom=phantom, M0=1.0,
                           T1=Q_(1e9, 'ms'), T2=Q_(1e9, 'ms'),
                           dtype='float64', perfect_spoiling=False).solve()

    zc = phantom.local_nodes[:, 2]
    zu, inv = np.unique(np.round(zc, 9), return_inverse=True)
    prof = np.abs(Mxy[:, -1])
    profile = np.array([prof[inv == i].mean() for i in range(zu.size)])

    rf = imp.feelmri_seq.blocks[0].rf_pulses[0]
    ts = rf.timings.m_as('ms')
    b1 = rf.waveform.m_as('mT')
    gz = float(np.abs(
        imp.feelmri_seq.blocks[0].gradients[0].amplitudes.m_as('mT/m')).max())
    gb_hz_per_mT = Scanner().gammabar.m_as('Hz/T') * 1e-3
    f_of_z = gb_hz_per_mT * gz * zu                       # Hz at each z
    tc = (ts - ts.mean()) * 1e-3                          # s, centred on the pulse
    predicted = np.abs(np.array(
        [np.trapezoid(b1 * np.exp(-2j * np.pi * f * tc), tc) for f in f_of_z]))
    predicted *= profile.max() / predicted.max()

    worst = float(np.abs(profile - predicted).max() / profile.max())
    assert worst < 5e-3, (
        f'the excited profile departs from the small-tip transform by '
        f'{worst:.3e} of peak')

    # The peak of a small-tip profile is sin(alpha) -- gre_v15 is a 10 deg pulse.
    assert abs(profile.max() - np.sin(np.radians(10.0))) < 5e-3, (
        f'peak |Mxy| is {profile.max():.5f}, expected sin(10 deg) = '
        f'{np.sin(np.radians(10.0)):.5f}')

    # And it must be a slice, not a slab: far off-resonance is not excited.
    fwhm = zu[profile >= 0.5 * profile.max()]
    thickness = float(fwhm.max() - fwhm.min())
    assert 3e-3 < thickness < 7e-3, f'slice is {thickness * 1e3:.3f} mm wide'
    assert profile[np.abs(zu) > 5e-3].max() < 0.05 * profile.max()


def test_spoiled_steady_state_matches_the_closed_form(cube):
    """A long TR train with ideal spoiling converges to
    `Mz = M0 (1 - E1) / (1 - E1 cos a)`.

    Exercises flip angle, T1 and TR together over 80 repetitions, which no
    other test does -- the next longest fixture is 4 TR, nowhere near steady
    state at TR/T1 = 20/500.
    """
    from feelmri.Bloch import BlochSolver
    from feelmri.PulseqAdapter import import_pulseq

    phantom, _volume = cube
    T1_ms, TR_ms, alpha_deg = 500.0, 20.0, 20.0

    imp = import_pulseq(_require('flash_tr_v15.seq'))
    seq = imp.feelmri_seq
    seq.blocks[-1].store_magnetization = True
    _Mxy, Mz = BlochSolver(sequence=seq, phantom=phantom, M0=1.0,
                           T1=Q_(T1_ms, 'ms'), T2=Q_(50.0, 'ms'),
                           dtype='float64', perfect_spoiling=True).solve()

    E1 = np.exp(-TR_ms / T1_ms)
    expected = (1.0 - E1) / (1.0 - E1 * np.cos(np.radians(alpha_deg)))
    got = float(np.real(Mz[:, -1]).mean())
    assert abs(got - expected) < 5e-3 * expected, (
        f'steady-state Mz is {got:.6f}, closed form {expected:.6f} '
        f'({100 * (got / expected - 1):+.3f}%)')


def test_cpmg_echo_train_decays_as_exp_minus_t_over_t2(cube):
    """Successive spin echoes fall as `exp(-n*TE/T2)`.

    A CPMG train refocuses static off-resonance at every echo, so what is left
    is pure T2 -- and it is the only fixture with more than one refocusing
    pulse per excitation, the case where a forward gradient-moment reference
    from the excitation is invalid because `calculate_kspace` negates k at each
    180.
    """
    from feelmri.PulseqAdapter import simulate_pulseq

    phantom, volume = cube
    _static(phantom, T2_MS)
    sim = simulate_pulseq(_require('cpmg_v15.seq'), phantom, M0=1.0,
                          T1=Q_(1e9, 'ms'), T2=Q_(T2_MS, 'ms'), dtype='float64')
    assert len(sim.kspace) >= 3, 'the CPMG fixture should give several echoes'

    peaks, centres = [], []
    for k, rw in zip(sim.kspace, sim.imp.readouts):
        mag = np.abs(np.asarray(k).reshape(-1))
        peaks.append(mag.max())
        centres.append(float(rw.times[int(np.argmax(mag))]))

    peaks = np.array(peaks)
    centres = np.array(centres)
    # Referenced to the first echo, so the excitation flip drops out.
    expected = peaks[0] * np.exp(-(centres - centres[0]) / T2_MS)
    worst = float(np.abs(peaks - expected).max() / peaks[0])
    assert worst < 2e-2, (
        f'echo amplitudes depart from exp(-t/T2) by {worst:.3e} of the first '
        f'echo; got {np.round(peaks / peaks[0], 4)}, '
        f'expected {np.round(expected / peaks[0], 4)}')


def test_pod_translation_obeys_the_shift_theorem(cube):
    """A rigid translation must appear in k-space as exactly a linear phase.

    `S_moved(k) = S_static(k) * exp(-i 2 pi k . dx)` with |S| unchanged. This is
    the only coverage of motion through the Pulseq path -- `simulate_pulseq`
    and `BlochSolver` both take a trajectory and no test passes one.
    """
    from feelmri.Motion import POD
    from feelmri.PulseqAdapter import import_pulseq, simulate_pulseq

    phantom, _volume = cube
    _static(phantom, 1e9)
    path = _require('gre_an_v15.seq')

    rw0 = import_pulseq(path).readouts[0]
    dx = 0.4e-3
    n_t = 6
    times = np.linspace(0.0, float(rw0.times.max()) + 1.0, n_t)
    data = np.zeros([phantom.global_shape[0], 3, n_t])
    data[:, 0, :] = dx                                    # constant shift along x
    pod = POD(times=times, data=data, n_modes=1)

    kw = dict(M0=1.0, T1=Q_(1e9, 'ms'), T2=Q_(1e9, 'ms'), dtype='float64')
    static = simulate_pulseq(path, phantom, **kw)
    moved = simulate_pulseq(path, phantom, pod=pod, **kw)

    s0 = np.asarray(static.kspace[0]).reshape(-1)
    s1 = np.asarray(moved.kspace[0]).reshape(-1)
    kx = static.imp.readouts[0].kspace[:, 0].astype(float)
    predicted = s0 * np.exp(-2j * np.pi * kx * dx)

    scale = np.abs(s0).max()
    assert np.abs(np.abs(s1) - np.abs(s0)).max() / scale < 1e-4, \
        'a rigid translation must not change |S(k)|'
    worst = float(np.abs(s1 - predicted).max() / scale)
    # Ignoring the trajectory entirely reads 1.7e-1 here, so this discriminates.
    assert worst < 1e-4, (
        f'k-space departs from the shift theorem by {worst:.3e} of peak')
