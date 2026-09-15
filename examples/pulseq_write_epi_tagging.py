"""
Demo low-performance EPI sequence without ramp-sampling.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
  import pypulseq as pp
except ImportError:
  print(
    "[pulseq_write_epi_tagging] Skipped: this example builds a sequence with "
    "pypulseq, which is an optional dependency of FEelMRI. Install it with "
    "'pip install pypulseq' (or 'pip install feelmri[pulseq]') to run this "
    "example.",
    file=sys.stderr,
  )
  sys.exit(0)

from pathlib import Path

from feelmri.MRImaging import PositionEncoding
from feelmri.MRObjects import Gradient, Scanner
from feelmri.Parameters import ParameterHandler, PVSMParser
from pint import Quantity as Q_


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = 'epi_pypulseq.seq',
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 64,
    n_y: int = 64,
    slice_thickness: float = 8e-3,
    n_slices: int = 1,
    ramp_sampling: bool = True,
):
    """Create an EPI sequence with a ramp-sampled readout.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is 'epi_pypulseq.seq'.
    fov : float or tuple of float, optional
        Field of view in meters. If a single value, it is used for both x and y.
        If a tuple, it is (fov_x, fov_y). Default is 220e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    n_y : int, optional
        Number of phase encoding steps. Default is 64.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 3e-3.
    n_slices : int, optional
        Number of slices. Default is 3.
    ramp_sampling : bool, optional
        Sample during the readout ramps instead of only the flat top. The
        ramps then contribute k-space coverage rather than dead time, which
        shortens the echo train by about 40%, worth having when the train
        is long against T2*. The samples are no longer evenly spaced in k,
        which the NUFFT reconstruction handles. Default True; False restores
        the flat-top-only readout.

    Returns
    -------
    seq : pypulseq.Sequence
        The EPI sequence object.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    print(fov_x, fov_y, slice_thickness * n_slices)

    # Set system limits
    system = pp.Opts(
        max_grad=32,
        grad_unit='mT/m',
        max_slew=180,
        slew_unit='T/m/s',
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
    )

    seq = pp.Sequence(system)

    # Create tagging prepulse objects
    rf_prep1 = pp.make_block_pulse(
        flip_angle=np.deg2rad(90),
        system=system,
        duration=1000e-6,
        time_bw_product=4,
        use='preparation',
        delay=system.rf_dead_time
    )

    # Closing pulses, one per SPAMM module. Each is built with the flip angle
    # it needs: make_block_pulse computes rf.signal at construction, so
    # assigning rf.flip_angle afterwards does NOT recompute it and both
    # modules would silently close with the same pulse.
    rf_prep2 = pp.make_block_pulse(
        flip_angle=np.deg2rad(-90),
        system=system,
        duration=1000e-6,
        time_bw_product=4,
        use='preparation',
        delay=system.rf_dead_time
    )

    rf_prep3 = pp.make_block_pulse(
        flip_angle=np.deg2rad(90),
        system=system,
        duration=1000e-6,
        time_bw_product=4,
        use='preparation',
        delay=system.rf_dead_time
    )

    # Create tagging gradients. The area of the tagging gradients determines the spacing of the tag lines, which is typically around 8 mm for cardiac imaging.
    tag_spacing = 16e-3 # m
    tag_frequency = 1 / tag_spacing # 1/m
    area = tag_frequency
    g_tag_x = pp.make_trapezoid(channel='x', system=system, area=area)
    g_tag_y = pp.make_trapezoid(channel='y', system=system, area=area)

    # Create 90 degree slice selection pulse and gradient
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(8),
        system=system,
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        return_gz=True,
        delay=system.rf_dead_time,
        use='excitation',
    )

    # Define other gradients and ADC events
    delta_kx = 1 / fov_x
    delta_ky = 1 / fov_y
    delta_kz = 1 / (slice_thickness * n_slices)
    k_width = n_x * delta_kx
    if ramp_sampling:
        # Shortest trapezoid carrying the required k-space area, then an ADC
        # spanning all of it. The ramps read k-space instead of idling, so the
        # echo is set by the gradient area and the hardware limits rather than
        # by the dwell time.
        gx = pp.make_trapezoid(channel='x', system=system, area=k_width)
        gx_duration = gx.rise_time + gx.flat_time + gx.fall_time
        adc_dwell = np.floor(gx_duration / n_x / system.adc_raster_time) \
            * system.adc_raster_time
        adc_duration = n_x * adc_dwell
        adc = pp.make_adc(
            num_samples=n_x,
            duration=adc_duration,
            delay=(gx_duration - adc_duration) / 2,
        )
    else:
        adc_dwell = 4e-6
        adc_duration = n_x * adc_dwell
        gx_flat_time = adc_duration
        gx_flat_time = np.ceil(gx_flat_time * 1e5) * 1e-5  # Round-up to the gradient raster
        gx = pp.make_trapezoid(
            channel='x',
            system=system,
            amplitude=k_width / adc_duration,
            flat_time=gx_flat_time,
        )
        adc = pp.make_adc(
            num_samples=n_x,
            duration=adc_duration,
            delay=gx.rise_time + gx_flat_time / 2 - (adc_duration - adc_dwell) / 2,
        )

    # Pre-phasing gradients
    pre_time = 8e-4
    gx_pre = pp.make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=pre_time)
    gz_reph = pp.make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=pre_time)
    gy_pre = pp.make_trapezoid(channel='y', system=system, area=-n_y / 2 * delta_ky, duration=pre_time)

    # Phase blip in the shortest possible time. Let make_trapezoid pick the
    # duration: the closed form 2*sqrt(area/slew) ignores the gradient raster
    # that rise and fall are rounded onto, so it under-estimates and fails the
    # area assertion whenever the slew limit is raised.
    gy = pp.make_trapezoid(channel='y', system=system, area=delta_ky)

    # Gradient spoiling
    f = 2
    gx_spoil = pp.make_trapezoid(channel='x', area=f * 2 * n_x * delta_kx, system=system)
    gy_spoil = pp.make_trapezoid(channel='y', area=f * 2 * n_y * delta_ky, system=system)
    gz_spoil = pp.make_trapezoid(channel='z', area=f * 4 / slice_thickness, system=system)

    # Tagging preparation: 90 - tag gradient - 90, then a spoiler.
    seq.add_block(rf_prep1, pp.make_label(type='SET', label='SET', value=0))
    # seq.add_block(pp.make_delay(system.rf_dead_time))
    seq.add_block(g_tag_x, pp.make_label(type='SET', label='SET', value=0))
    # seq.add_block(pp.make_delay(system.rf_dead_time))
    seq.add_block(rf_prep2, pp.make_label(type='SET', label='SET', value=0))
    # seq.add_block(pp.make_delay(system.rf_dead_time))
    seq.add_block(gx_spoil, gy_spoil, gz_spoil, pp.make_label(type='SET', label='SET', value=100))

    # Second SPAMM module, along y, applied to the already-tagged Mz. The two
    # together give Mz ~ cos(kx x) cos(ky y): a tag grid.
    seq.add_block(rf_prep1, pp.make_label(type='SET', label='SET', value=1))
    seq.add_block(g_tag_y, pp.make_label(type='SET', label='SET', value=1))
    seq.add_block(rf_prep3, pp.make_label(type='SET', label='SET', value=1))
    seq.add_block(gx_spoil, gy_spoil, gz_spoil, pp.make_label(type='SET', label='SET', value=100))

    for i_slice in range(n_slices):
        rf.freq_offset = gz.amplitude * slice_thickness * (i_slice - (n_slices - 1) / 2)
        # The slice rephaser is played on its own, and the in-plane prephasers
        # follow it as SET=4. That puts a block boundary where the gradient
        # moment measured from the excitation is exactly zero on all three
        # axes, which is the only instant at which the magnetization can be
        # snapshotted for the k-space integral: calculate_kspace resets k=0 at
        # the RF, so any moment already accumulated when the snapshot is taken
        # gets applied a second time by the assembler. Playing gz_reph and the
        # prephasers together, as one block, leaves no such boundary.
        seq.add_block(rf, gz, pp.make_label(type='SET', label='SET', value=2))
        seq.add_block(gz_reph, pp.make_label(type='SET', label='SET', value=2))
        seq.add_block(gx_pre, gy_pre, pp.make_label(type='SET', label='SET', value=4))
        for _ in range(n_y):
            seq.add_block(gx, adc, pp.make_label(type='SET', label='SET', value=3))  # Read one line of k-space
            seq.add_block(gy, pp.make_label(type='SET', label='SET', value=3))  # Phase blip
            gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient
        seq.add_block(gx_spoil, gy_spoil, gz_spoil, pp.make_label(type='SET', label='SET', value=100))

    # Check timings
    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

        # Calculate trajectory for visualization
        k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

        fig, ax = plt.subplots(1, 2)
        ax[0].plot(k_traj_adc[0, :], k_traj_adc[1, :], '-')
        ax[0].set_title('k-space trajectory (ADC)')
        ax[0].set_xlabel('k_x (1/m)')
        ax[0].set_ylabel('k_y (1/m)')
        ax[0].axis('equal')
        ax[1].plot(k_traj[0, :], k_traj[1, :], '-')
        ax[1].set_title('k-space trajectory (all events)')
        ax[1].set_xlabel('k_x (1/m)')
        ax[1].set_ylabel('k_y (1/m)')
        ax[1].axis('equal')
        plt.tight_layout()
        plt.show()

    seq.set_definition(key='FOV', value=[fov_x, fov_y, slice_thickness * n_slices])
    seq.set_definition(key='Name', value='epi')

    if write_seq:
      seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    
    # Get path of this script to allow running from any directory
    script_path = Path(__file__).parent

    # Import imaging parameters
    parameters = ParameterHandler(script_path/'parameters/spamm_pulseq.yaml')

    # Import MRI slice planning
    planning = PVSMParser(script_path/parameters.Formatting.planning,
                            box_name='Box1',
                            transform_name='Transform1',
                            length_units=parameters.Formatting.units)

    # The checked-in epi_pypulseq.seq is a v1.4.2 fixture the adapter tests
    # read, so overwriting it from a test run would silently change what they
    # measure. Write next to it instead when running under the test harness.
    if os.getenv('FEELMRI_FAST_TEST', '0') == '1':
      seq_out = script_path/'output'/'epi_pypulseq_generated.seq'
      seq_out.parent.mkdir(parents=True, exist_ok=True)
    else:
      seq_out = script_path/'pulseq/epi_pypulseq.seq'

    main(plot=os.getenv('FEELMRI_FAST_TEST', '0') != '1',
         write_seq=True,
         seq_filename=seq_out,
         fov = tuple(2*planning.FOV[:-1].m_as('m')),
         n_x = parameters.Imaging.RES[0],
         n_y = parameters.Imaging.RES[1],
         slice_thickness = planning.FOV[-1].m_as('m'),
         n_slices = parameters.Imaging.RES[2])
