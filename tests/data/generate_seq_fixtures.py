import numpy as np
import pypulseq as pp
from pathlib import Path

# Writes the small Pulseq v1.5 fixtures used by the adapter tests. They are kept
# deliberately short so the tests stay fast, but each one carries a feature the
# v1.4 EPI fixture does not:
#
#   gre_v15.seq  RF 'use' labels and a bare delay block with no events
#   se_v15.seq   a refocusing pulse, so excitation and refocusing anchors differ
#   arb_v15.seq  arbitrary gradients: one on the regular raster, carrying the
#                first/last boundary samples, and one extended trapezoid
#   ppm_v15.seq  RF and ADC ppm offsets and a per-sample ADC phase shape
#
#   python3 tests/data/generate_seq_fixtures.py

# Get path of this script to allow running from any directory
script_path = Path(__file__).parent

# 1. Scanner limits, shared by both sequences
system = pp.Opts(
    max_grad=32, grad_unit='mT/m',
    max_slew=130, slew_unit='T/m/s',
    rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6,
)

fov = 200e-3
Nx, Ny = 8, 4
slice_thickness = 5e-3


def readout_events(flat_time=1.6e-3):
    gx = pp.make_trapezoid('x', flat_area=Nx / fov, flat_time=flat_time, system=system)
    adc = pp.make_adc(Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid('x', area=-gx.area / 2, duration=1e-3, system=system)
    return gx, adc, gx_pre


# 2. Gradient echo. The last block of every TR is a bare delay, which is the case
# that gives an event-free block no integration steps if dt is left at default.
seq = pp.Sequence(system=system)
rf, gz, gz_reph = pp.make_sinc_pulse(
    flip_angle=np.deg2rad(10), duration=1e-3, slice_thickness=slice_thickness,
    apodization=0.5, time_bw_product=4, system=system, use='excitation',
    return_gz=True)
gx, adc, gx_pre = readout_events()
phase_areas = (np.arange(Ny) - Ny / 2) / fov
gz_spoil = pp.make_trapezoid('z', area=4 / slice_thickness, duration=1e-3, system=system)

TR = 20e-3
for i in range(Ny):
    seq.add_block(rf, gz)
    gy_pre = pp.make_trapezoid('y', area=phase_areas[i], duration=1e-3, system=system)
    seq.add_block(gx_pre, gy_pre, gz_reph)
    seq.add_block(gx, adc)
    seq.add_block(gz_spoil)
    used = sum(seq.block_durations[b] for b in list(seq.block_durations)[-4:])
    seq.add_block(pp.make_delay(round((TR - used) / system.grad_raster_time)
                                * system.grad_raster_time))

ok, err = seq.check_timing()
print('gre_v15  check_timing:', 'OK' if ok else err)
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre')
seq.write(str(script_path / 'gre_v15.seq'))

# 3. Spin echo. The refocusing pulse carries use='refocusing', so the excitation
# and refocusing anchors are distinguishable.
seq = pp.Sequence(system=system)
rf90, gz90, gz90_reph = pp.make_sinc_pulse(
    flip_angle=np.deg2rad(90), duration=1e-3, slice_thickness=slice_thickness,
    apodization=0.5, time_bw_product=4, system=system, use='excitation',
    return_gz=True)
rf180, gz180, _ = pp.make_sinc_pulse(
    flip_angle=np.deg2rad(180), duration=1e-3, slice_thickness=slice_thickness,
    apodization=0.5, time_bw_product=4, system=system, use='refocusing',
    return_gz=True)
gx, adc, gx_pre = readout_events()

for i in range(Ny):
    seq.add_block(rf90, gz90)
    gy_pre = pp.make_trapezoid('y', area=phase_areas[i], duration=1e-3, system=system)
    seq.add_block(gx_pre, gy_pre, gz90_reph)
    seq.add_block(pp.make_delay(2e-3))
    seq.add_block(rf180, gz180)
    seq.add_block(pp.make_delay(2e-3))
    seq.add_block(gx, adc)
    seq.add_block(gz_spoil)

ok, err = seq.check_timing()
print('se_v15   check_timing:', 'OK' if ok else err)
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'se')
seq.write(str(script_path / 'se_v15.seq'))


# 4. Arbitrary gradients. The trapezoids above never exercise the shaped-
# gradient path, where the samples sit at raster centres and the amplitudes at
# the block boundaries come from the file's first/last columns.
seq = pp.Sequence(system=system)
n_samples = 200
t = np.arange(n_samples) * system.grad_raster_time
wave = 8e-3 * system.gamma * np.sin(2 * np.pi * t / (n_samples * system.grad_raster_time))
gx_arb = pp.make_arbitrary_grad('x', waveform=wave, system=system,
                                delay=system.grad_raster_time)

times = np.array([0.0, 0.4e-3, 1.2e-3, 1.6e-3])
amps = np.array([0.0, 6e-3, 6e-3, 0.0]) * system.gamma
gy_ext = pp.make_extended_trapezoid('y', amplitudes=amps, times=times, system=system)

adc_arb = pp.make_adc(64, duration=1.6e-3, delay=2e-4, system=system)

for i in range(2):
    seq.add_block(rf90, gz90)
    seq.add_block(gx_arb)
    seq.add_block(gy_ext, adc_arb)
    seq.add_block(gz_spoil)

ok, err = seq.check_timing()
print('arb_v15  check_timing:', 'OK' if ok else err)
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'arb')
seq.write(str(script_path / 'arb_v15.seq'))


# 5. PPM offsets and a per-sample ADC phase shape. The ppm columns and the ADC
# phase_id are v1.5 only, and nothing else here sets them.
seq = pp.Sequence(system=system)
rf_sat = pp.make_block_pulse(
    flip_angle=np.deg2rad(90), duration=8e-3, system=system, use='saturation',
    freq_ppm=-3.3, phase_ppm=0.25)
gx, adc, gx_pre = readout_events()
adc_mod = pp.make_adc(Nx, duration=gx.flat_time, delay=gx.rise_time, system=system,
                      freq_ppm=1.5, phase_ppm=-0.5,
                      phase_modulation=np.linspace(0.0, np.pi, Nx))

for i in range(2):
    seq.add_block(rf_sat)
    seq.add_block(gz_spoil)
    seq.add_block(rf90, gz90)
    seq.add_block(gx_pre, gz90_reph)
    seq.add_block(gx, adc_mod)

ok, err = seq.check_timing()
print('ppm_v15  check_timing:', 'OK' if ok else err)
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'ppm')
seq.write(str(script_path / 'ppm_v15.seq'))


# 6. Fixtures with closed-form answers. Dead times and ringdown are zero so the
# analytical expressions carry no hidden offsets, and every RF is a hard pulse
# so the transverse magnetization it creates is spatially uniform.
an = pp.Opts(max_grad=40, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s',
             rf_ringdown_time=0, rf_dead_time=0, adc_dead_time=0)

T1_MS = 800.0      # matches the T1 the analytical test solves with
TAU_MS = 5.0       # spin-echo half-echo time


def hard_pulse(flip_deg, use, dur=2e-4):
    return pp.make_block_pulse(flip_angle=np.deg2rad(flip_deg), duration=dur,
                               system=an, use=use)


# 6a. Free induction decay: one hard 90, then a long gradient-free readout.
# With no gradient the trajectory sits at k=0, so the signal is the whole
# object and decays purely as exp(-t/T2).
# The 50 ms of dead time BEFORE the excitation is deliberate. It is what
# separates times measured from the snapshot from times measured from the start
# of the file: the snapshot lands at 50.2 ms, so the two readings of the same
# readout differ by exp(-50.2/T2). A delay placed after the excitation would
# not do this -- it falls inside the interval either way.
seq = pp.Sequence(system=an)
seq.add_block(pp.make_delay(50e-3))
seq.add_block(hard_pulse(90, 'excitation'))
seq.add_block(pp.make_adc(64, duration=40e-3, system=an))
ok, err = seq.check_timing()
print('fid_v15  check_timing:', 'OK' if ok else err)
seq.set_definition('Name', 'fid')
seq.write(str(script_path / 'fid_v15.seq'))

# 6b. T1 recovery: a hard 90 leaves Mz = 0, then a delay of exactly T1.
# No ADC, so the import needs no trajectory.
seq = pp.Sequence(system=an)
seq.add_block(hard_pulse(90, 'excitation'))
seq.add_block(pp.make_delay(T1_MS * 1e-3))
ok, err = seq.check_timing()
print('t1_v15   check_timing:', 'OK' if ok else err)
seq.set_definition('Name', 't1')
seq.write(str(script_path / 't1_v15.seq'))

# 6c. Gradient echo with a rewound readout. The prephaser is exactly half the
# readout area, so k passes through zero mid-plateau and the signal there is
# the full object integral.
seq = pp.Sequence(system=an)
# 51 samples of 40 us: an odd count on the ADC raster, so one sample sits
# exactly at the plateau centre -- which for a symmetric trapezoid rewound by
# half its area is exactly k = 0.
gx_an = pp.make_trapezoid('x', flat_area=204.0, flat_time=2.04e-3, system=an)
adc_an = pp.make_adc(51, duration=gx_an.flat_time, delay=gx_an.rise_time, system=an)
seq.add_block(hard_pulse(90, 'excitation'))
seq.add_block(pp.make_trapezoid('x', area=-gx_an.area / 2, duration=1e-3, system=an))
seq.add_block(gx_an, adc_an)
ok, err = seq.check_timing()
print('gre_an_v15 check_timing:', 'OK' if ok else err)
seq.set_definition('Name', 'gre_analytical')
seq.write(str(script_path / 'gre_an_v15.seq'))

# 6d. Spin echo: 90 - tau - 180 - tau - readout. At the echo the off-resonance
# phase is refocused whatever its value, and only T2 remains.
seq = pp.Sequence(system=an)
seq.add_block(hard_pulse(90, 'excitation'))
seq.add_block(pp.make_delay(TAU_MS * 1e-3))
seq.add_block(hard_pulse(180, 'refocusing'))
seq.add_block(pp.make_delay(TAU_MS * 1e-3))
seq.add_block(pp.make_adc(16, duration=1e-3, system=an))
ok, err = seq.check_timing()
print('se_an_v15 check_timing:', 'OK' if ok else err)
seq.set_definition('Name', 'se_analytical')
seq.write(str(script_path / 'se_an_v15.seq'))

# ---------------------------------------------------------------------------
# 7. Regimes the suite had no fixture for at all (added 2026-09-10)
# ---------------------------------------------------------------------------

# 7a. CPMG: one excitation, several refocusing pulses. Pins exp(-n*TE/T2)
# across an echo train, and is the only fixture with more than one 180 per
# excitation -- the case where a forward gradient-moment integral from the
# excitation is invalid, since calculate_kspace negates k at every refocusing.
N_ECHO = 4
seq = pp.Sequence(system=an)
seq.add_block(hard_pulse(90, 'excitation'))
for _ in range(N_ECHO):
    seq.add_block(pp.make_delay(TAU_MS * 1e-3))
    seq.add_block(hard_pulse(180, 'refocusing'))
    seq.add_block(pp.make_delay(TAU_MS * 1e-3))
    seq.add_block(pp.make_adc(8, duration=0.5e-3, system=an))
ok, err = seq.check_timing()
print('cpmg_v15 check_timing:', 'OK' if ok else err)
seq.set_definition('Name', 'cpmg')
seq.set_definition('EchoSpacing_ms', 2.0 * TAU_MS)
seq.write(str(script_path / 'cpmg_v15.seq'))

# 7b. A TR train long enough to reach the spoiled steady state, so
# M0 (1 - E1) / (1 - E1 cos a) can be checked against the closed form. The
# previous longest fixture was 4 TR, nowhere near steady state.
FLASH_TR_MS, FLASH_ALPHA, FLASH_NTR = 20.0, 20.0, 80
seq = pp.Sequence(system=an)
for _ in range(FLASH_NTR):
    seq.add_block(hard_pulse(FLASH_ALPHA, 'excitation'))
    seq.add_block(pp.make_delay(round((FLASH_TR_MS * 1e-3 - 200e-6) / 1e-5) * 1e-5))
ok, err = seq.check_timing()
print('flash_tr_v15 check_timing:', 'OK' if ok else err)
seq.set_definition('Name', 'flash_tr')
seq.set_definition('TR_ms', FLASH_TR_MS)
seq.set_definition('FlipAngle_deg', FLASH_ALPHA)
seq.write(str(script_path / 'flash_tr_v15.seq'))

# 7c. A trapezoid whose ramps differ: rise 130 us against fall 50 us. Every
# other fixture is symmetric, so the A*(rise - fall)/2 error of an
# end-of-interval quadrature cancels and stays invisible.
#
# Note a .seq file CANNOT carry the harder case -- ramps that are not integer
# multiples of the gradient raster -- because Pulseq requires every time on that
# raster. That case is reachable only for a natively built Gradient, and is
# covered by tests/test_sequence_concat.py.
seq = pp.Sequence(system=an)
asym_amp = 5.0e-3 * 42.576e6          # 5 mT/m in Hz/m; 100 T/m/s on the 50 us fall
asym = pp.make_extended_trapezoid(
    'x',
    amplitudes=np.array([0.0, asym_amp, asym_amp, 0.0]),
    times=np.array([0.0, 130e-6, 1.13e-3, 1.18e-3]),               # rise 130 us, fall 50 us
    system=an)
seq.add_block(hard_pulse(90, 'excitation'))
seq.add_block(asym)
# 20 samples over 0.5 ms is a 25 us dwell, a clean multiple of the 100 ns ADC
# raster; 16 samples would give 31.25 us and fail check_timing.
seq.add_block(pp.make_adc(20, duration=0.5e-3, system=an))
ok, err = seq.check_timing()
print('asym_ramp_v15 check_timing:', 'OK' if ok else err)
seq.set_definition('Name', 'asym_ramp')
seq.write(str(script_path / 'asym_ramp_v15.seq'))
