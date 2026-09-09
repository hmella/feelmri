import numpy as np
import pypulseq as pp
from pathlib import Path

# Writes the small Pulseq v1.5 fixtures used by the adapter tests. They are kept
# deliberately short so the tests stay fast, but each one carries a feature the
# v1.4 EPI fixture does not:
#
#   gre_v15.seq  RF 'use' labels and a bare delay block with no events
#   se_v15.seq   a refocusing pulse, so excitation and refocusing anchors differ
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
