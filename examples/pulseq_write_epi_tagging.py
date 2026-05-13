"""
Demo low-performance EPI sequence without ramp-sampling.
"""

import numpy as np

import pypulseq as pp
from pathlib import Path

from feelmri.MRObjects import Gradient, Scanner
from feelmri.MRImaging import PositionEncoding
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
):
    """Create a basic EPI sequence without ramp-sampling.

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
        max_slew=120,
        slew_unit='T/m/s',
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
    )

    seq_prep_x = pp.Sequence(system)
    seq_prep_y = pp.Sequence(system)
    seq_ex = pp.Sequence(system)
    seq_ro = pp.Sequence(system)
    seq_spoil = pp.Sequence(system)

    # Create tagging prepulse objects
    rf_prep1 = pp.make_block_pulse(
        flip_angle=np.deg2rad(90),
        system=system,
        duration=1000e-6,
        time_bw_product=4,
        use='preparation',
        delay=system.rf_dead_time
    )

    rf_prep2 = pp.make_block_pulse(
        flip_angle=np.deg2rad(-90),
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

    # Phase blip in the shortest possible time
    gy_blip_duration = 2 * np.sqrt(delta_ky / system.max_slew)
    gy_blip_duration = np.ceil(gy_blip_duration / 10e-6) * 10e-6
    gy = pp.make_trapezoid(channel='y', system=system, area=delta_ky, duration=gy_blip_duration)

    # Gradient spoiling
    f = 2
    gx_spoil = pp.make_trapezoid(channel='x', area=f * 2 * n_x * delta_kx, system=system)
    gy_spoil = pp.make_trapezoid(channel='y', area=f * 2 * n_y * delta_ky, system=system)
    gz_spoil = pp.make_trapezoid(channel='z', area=f * 4 / slice_thickness, system=system)

    # Loop over slices
    seq_prep_x.add_block(rf_prep1)
    # seq_prep_x.add_block(pp.make_delay(system.rf_dead_time))
    seq_prep_x.add_block(g_tag_x)
    # seq_prep_x.add_block(pp.make_delay(system.rf_dead_time))
    seq_prep_x.add_block(rf_prep2)
    # seq_prep_x.add_block(pp.make_delay(system.rf_dead_time))
    # Crush the transverse residual left by the second SPAMM 90 deg pulse.
    # A 1-1 SPAMM prep leaves Mz = -M0*cos(phi(x)) AND My = +M0*sin(phi(x));
    # the unwanted My residual aliases the desired Mz modulation in the
    # first imaged frame unless it is spoiled before the next excitation.

    rf_prep2.flip_angle = np.deg2rad(90)
    seq_prep_y.add_block(rf_prep1)
    # seq_prep_x.add_block(pp.make_delay(system.rf_dead_time))
    seq_prep_y.add_block(g_tag_y)
    # seq_prep_x.add_block(pp.make_delay(system.rf_dead_time))
    seq_prep_y.add_block(rf_prep2)
    # seq_prep_x.add_block(pp.make_delay(system.rf_dead_time))

    for i_slice in range(n_slices):
        rf.freq_offset = gz.amplitude * slice_thickness * (i_slice - (n_slices - 1) / 2)
        seq_ex.add_block(rf, gz)
        seq_ex.add_block(gx_pre, gy_pre, gz_reph)
        for _ in range(n_y):
            seq_ro.add_block(gx, adc)  # Read one line of k-space
            seq_ro.add_block(gy)  # Phase blip
            gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient
        seq_spoil.add_block(gx_spoil, gy_spoil, gz_spoil)

    for i, seq in enumerate([seq_prep_x, seq_prep_y, seq_ex, seq_ro, seq_spoil]):
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

      seq.set_definition(key='FOV', value=[fov_x, fov_y, slice_thickness * n_slices])
      seq.set_definition(key='Name', value='epi')

      if write_seq:
          # Add i before .seq extension
          path = Path(seq_filename)
          seq_filename_i = path.with_name(f"{path.stem}_{i}{path.suffix}")
          seq.write(seq_filename_i)

    return (seq_prep_x, seq_prep_y, seq_ex, seq_ro, seq_spoil)


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

    main(plot=False,
         write_seq=True,
         seq_filename=script_path/'pulseq/epi_pypulseq.seq',
         fov = tuple(2*planning.FOV[:-1].m_as('m')),
         n_x = parameters.Imaging.RES[0],
         n_y = parameters.Imaging.RES[1],
         slice_thickness = planning.FOV[-1].m_as('m'),
         n_slices = parameters.Imaging.RES[2])
