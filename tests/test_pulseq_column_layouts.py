"""The v1.4 and v1.5 RF and ADC column layouts, read side by side.

`read_RF` and `read_ADC` each branch on the file's format version, and the two
arms take the same quantities from different column indices: v1.4 RF is 7
columns `(amp, mag_id, ph_id, time_id, delay, freq, phase)` while v1.5 is 11,
inserting `center` before the delay and `freq_ppm` / `phase_ppm` before the
frequency, and appending a `use` code. ADC goes from 5 columns to 8 the same
way.

The v1.4 arm was executed but never asserted. Both bundled v1.4 fixtures,
`epi_v142.seq` and `rotation_minimal.seq`, declare `freq = 0` and `phase = 0`
on every RF row, and no test reads those fields back, so the column indices in
that arm were free to be wrong. Adding 137 Hz to the v1.4 frequency column left
the suite fully green while the same edit to the v1.5 column failed nine cases,
and 137 Hz under a slice-select gradient moves the slice.

Both layouts describe ONE physical event here, so each arm is checked against
the literal values it was handed, and then against the other arm. The literal
check is what makes this more than a tautology: two arms broken the same way
agree with each other perfectly.

These call the format reader only. pypulseq is not involved, so the file
carries no `pulseq` marker and runs in the CI job that measures coverage.
"""
from __future__ import annotations

import numpy as np
import pytest

from feelmri.PulseqFile import (ADC, RF, Version, compress_shape, read_ADC,
                                read_RF)

V14 = Version(1, 4, 0)
V15 = Version(1, 5, 0)

# Distinct, non-zero, and different from each other, so that a frequency read
# out of the phase column (or either read from `center` or a ppm column) lands
# on a value no assertion below accepts.
RF_AMP = 12.5
RF_DELAY_S = 1.0e-4
RF_FREQ_HZ = 1703.2
RF_PHASE_RAD = 0.6
RF_CENTRE_S = 5.0e-5
DT_RF_S = 1.0e-6

ADC_NUM = 20
ADC_DWELL_S = 2.5e-5
ADC_DELAY_S = 3.0e-4
ADC_FREQ_HZ = -812.5
ADC_PHASE_RAD = 0.4


def _rf_library(version):
  """One RF row in the layout `version` declares."""
  if version >= V15:
    data = [RF_AMP, 1, 2, 0, RF_CENTRE_S, RF_DELAY_S,
            0.0, 0.0, RF_FREQ_HZ, RF_PHASE_RAD, 'e']
  else:
    data = [RF_AMP, 1, 2, 0, RF_DELAY_S, RF_FREQ_HZ, RF_PHASE_RAD]
  return {1: {'data': data}}


def _shape_library():
  """A magnitude ramp and a flat zero phase.

  The magnitude is deliberately not constant: a constant waveform is invariant
  under several ways of mis-reading a shape, so it cannot distinguish them.
  """
  mag = np.linspace(0.2, 1.0, 5)
  phase = np.zeros(5)
  return {1: compress_shape(mag), 2: compress_shape(phase)}


def _adc_library(version):
  """One ADC row in the layout `version` declares."""
  if version >= V15:
    data = [ADC_NUM, ADC_DWELL_S, ADC_DELAY_S, 0.0, 0.0,
            ADC_FREQ_HZ, ADC_PHASE_RAD, 0]
  else:
    data = [ADC_NUM, ADC_DWELL_S, ADC_DELAY_S, ADC_FREQ_HZ, ADC_PHASE_RAD]
  return {1: {'data': data}}


@pytest.mark.parametrize('version', [V14, V15], ids=['v1.4', 'v1.5'])
def test_each_rf_column_layout_reads_the_pulse_it_was_given(version):
  """Frequency, phase, delay and duration, against the numbers written in.

  `phase` is not a field on `RF`: it is folded into the complex waveform as
  `amp * mag * exp(i(2 pi rf_phi + phase))`, and the phase shape here is zero,
  so the argument of every sample IS the phase offset. `delay` carries half a
  raster because `time_shape_id` is 0, the uniform-raster case.
  """
  rf = read_RF(_rf_library(version), _shape_library(), DT_RF_S, 1, version)

  assert rf.df == pytest.approx(RF_FREQ_HZ), 'frequency column'
  assert rf.delay == pytest.approx(RF_DELAY_S + DT_RF_S / 2.0), 'delay column'
  assert rf.T == pytest.approx(4 * DT_RF_S), 'duration from the magnitude shape'

  np.testing.assert_allclose(np.angle(rf.waveform), RF_PHASE_RAD, atol=1e-12,
                             err_msg='phase column')
  np.testing.assert_allclose(np.abs(rf.waveform),
                             RF_AMP * np.linspace(0.2, 1.0, 5), rtol=1e-12)

  # The functional label exists only from v1.5; below it the reader must say so
  # rather than invent one, since the anchor selection keys off this.
  assert rf.use == ('excitation' if version >= V15 else 'undefined')


@pytest.mark.parametrize('version', [V14, V15], ids=['v1.4', 'v1.5'])
def test_each_adc_column_layout_reads_the_window_it_was_given(version):
  """Sample count, dwell, delay, and the two demodulation offsets.

  All three ADC offsets are demodulation parameters: the solver never samples
  the ADC, so a wrong column here is invisible until an image comes out with a
  shifted or rotated object.
  """
  adc = read_ADC(_adc_library(version), 1, version)

  assert adc.num == ADC_NUM
  assert adc.T == pytest.approx((ADC_NUM - 1) * ADC_DWELL_S), 'dwell column'
  assert adc.delay == pytest.approx(ADC_DELAY_S + ADC_DWELL_S / 2.0), 'delay'
  assert adc.df == pytest.approx(ADC_FREQ_HZ), 'frequency column'
  assert adc.phase == pytest.approx(ADC_PHASE_RAD), 'phase column'


def test_the_two_column_layouts_describe_the_same_event():
  """One physical RF and one physical ADC, written both ways, must parse alike.

  This catches a column that drifted on one side only, which the per-layout
  checks above would also catch, but it additionally pins the fields those
  checks do not name. It is deliberately NOT the only test here: two arms
  wrong in the same way would satisfy this one and nothing else.
  """
  shapes = _shape_library()
  rf14 = read_RF(_rf_library(V14), shapes, DT_RF_S, 1, V14)
  rf15 = read_RF(_rf_library(V15), shapes, DT_RF_S, 1, V15)

  np.testing.assert_allclose(rf14.waveform, rf15.waveform, rtol=1e-12)
  assert rf14.T == pytest.approx(rf15.T)
  assert rf14.df == pytest.approx(rf15.df)
  assert rf14.delay == pytest.approx(rf15.delay)

  adc14 = read_ADC(_adc_library(V14), 1, V14)
  adc15 = read_ADC(_adc_library(V15), 1, V15)

  for field in ('num', 'T', 'delay', 'df', 'phase'):
    assert getattr(adc14, field) == pytest.approx(getattr(adc15, field)), field

  # `center`, the v1.5 column with no v1.4 counterpart, is read and discarded:
  # FEelMRI has no pulse-centre anchor. It must not reach the delay.
  assert rf15.delay == pytest.approx(RF_DELAY_S + DT_RF_S / 2.0)


def test_a_row_of_the_wrong_width_is_refused_by_name():
  """A v1.5 row read as v1.4, or the reverse, must fail loudly.

  The column count is the only thing distinguishing the two layouts, so a
  version misdeclared in the file header would otherwise read a `use` code as
  a phase, or a phase as a delay.
  """
  for version, wrong in ((V14, V15), (V15, V14)):
    with pytest.raises(AssertionError, match='columns'):
      read_RF(_rf_library(wrong), _shape_library(), DT_RF_S, 1, version)
    with pytest.raises(AssertionError, match='columns'):
      read_ADC(_adc_library(wrong), 1, version)
