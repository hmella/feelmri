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
