import json
import os
import numpy as np
import pytest
from pathlib import Path

from conftest import skip_if_pypulseq_too_old
from feelmri.PulseqAdapter import import_pulseq

# Absolute regression on what each bundled .seq imports to. The timing gate in
# test_pulseq_timing.py is a RELATIVE check -- it compares our reading of a
# file against pypulseq's -- so it cannot see a change that moves both the
# same way, such as a pypulseq upgrade that alters calculate_kspace. This one
# pins the numbers themselves: durations, event counts, TE-defining ADC
# placement, k-space extent and gradient maxima.
#
# Regenerate after a deliberate change:
#     FEELMRI_UPDATE_GOLDEN=1 pytest tests/test_pulseq_golden.py
# and read the diff before committing it.

pytestmark = pytest.mark.pulseq

DATA_DIR = Path(__file__).resolve().parent / 'data'
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / 'examples' / 'pulseq'
GOLDEN_DIR = DATA_DIR / 'golden'

SEQ_FILES = sorted(DATA_DIR.glob('*.seq')) + sorted(EXAMPLES_DIR.glob('*.seq'))

RTOL = 1e-9


def _summary(seq_path):
  imp = import_pulseq(seq_path)
  seq = imp.feelmri_seq

  n_rf = sum(len(b.rf_pulses) for b in seq.blocks)
  n_grad = [0, 0, 0]
  peak_grad = [0.0, 0.0, 0.0]
  for b in seq.blocks:
    for g in b.gradients:
      n_grad[g.axis] += 1
      peak_grad[g.axis] = max(peak_grad[g.axis],
                              float(np.abs(g.amplitudes.m_as('mT/m')).max()))

  k = np.concatenate([r.kspace for r in imp.readouts]) if imp.readouts \
      else np.zeros((0, 3))
  t = np.concatenate([r.times for r in imp.readouts]) if imp.readouts \
      else np.zeros((0,))

  return {
    'n_blocks': len(seq.blocks),
    'duration_ms': float(seq.dur.m_as('ms')),
    'n_rf': n_rf,
    'n_gradients': n_grad,
    'peak_gradient_mT_per_m': peak_grad,
    'n_readouts': len(imp.readouts),
    'n_adc_samples': int(t.size),
    'first_adc_ms': float(t.min()) if t.size else 0.0,
    'last_adc_ms': float(t.max()) if t.size else 0.0,
    'kspace_min_1_per_m': k.min(axis=0).tolist() if k.size else [0.0, 0.0, 0.0],
    'kspace_max_1_per_m': k.max(axis=0).tolist() if k.size else [0.0, 0.0, 0.0],
    'n_stored_columns': sum(1 for b in seq.blocks if b.store_magnetization),
    'timing_errors': len(imp.timing_errors),
  }


def _compare(got, want, path):
  assert set(got) == set(want), (
    f'{path}: summary keys changed; regenerate the golden files')
  for key in sorted(want):
    a, b = got[key], want[key]
    if isinstance(b, (int, str)):
      assert a == b, f'{path}: {key} is {a}, golden says {b}'
    else:
      assert np.allclose(np.asarray(a, dtype=float),
                         np.asarray(b, dtype=float), rtol=RTOL, atol=0.0), (
        f'{path}: {key} is {a}, golden says {b}')


@pytest.mark.parametrize('seq_path', SEQ_FILES, ids=lambda p: p.stem)
def test_import_matches_golden(seq_path):
  skip_if_pypulseq_too_old(seq_path)
  golden = GOLDEN_DIR / f'{seq_path.stem}.json'
  summary = _summary(seq_path)

  if os.getenv('FEELMRI_UPDATE_GOLDEN', '0') == '1':
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    pytest.skip(f'regenerated {golden.name}')

  assert golden.exists(), (
    f'no golden for {seq_path.name}; run '
    f'FEELMRI_UPDATE_GOLDEN=1 pytest {Path(__file__).name}')
  _compare(summary, json.loads(golden.read_text()), seq_path.name)
