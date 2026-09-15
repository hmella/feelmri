"""Block and MR-object labels, and the sidecar that persists them.

The library's labelling is read-only and per block: `import_pulseq` folds the
file's LABELSET and LABELINC extensions into a running dict, `filter_blocks`
queries it, and nothing writes one back. This adds writing and a per-object
layer, so the tests that matter are the ones showing the addition does not
change what the existing selection path returns.

Labels are tied to a sequence by block index and nothing else, which is why
the sidecar carries a checksum: a sidecar applied to the wrong file does not
fail, it mislabels.
"""
from __future__ import annotations

import pytest

from feelmri.gui.model.labels import (LabelStore, StaleSidecarError,
                                      sequence_checksum)


def test_labels_are_range_checked_because_an_index_is_all_they_have():
  store = LabelStore(n_blocks=3)
  store.set_block(0, 'SET', 2)
  assert store.labels_on(0) == {'SET': 2}
  store.set_block(-1, 'SET', 9)                 # negative wraps
  assert store.labels_on(2) == {'SET': 9}
  for bad in (3, -4, 99):
    with pytest.raises(IndexError, match='out of range'):
      store.set_block(bad, 'SET', 1)


def test_a_mismatched_block_count_is_refused_at_construction():
  with pytest.raises(ValueError, match='for 3 blocks'):
    LabelStore(n_blocks=3, block_labels=[{}, {}])
  with pytest.raises(ValueError, match='n_blocks must be'):
    LabelStore(n_blocks=-1)


def test_object_labels_are_keyed_by_kind_and_ordinal():
  store = LabelStore(n_blocks=2)
  store.set_object(0, 'gradient', 1, 'role', 'prephaser')
  store.set_object(0, 'rf', 0, 'role', 'excitation')
  assert store.labels_on_object(0, 'gradient', 1) == {'role': 'prephaser'}
  assert store.labels_on_object(0, 'rf', 0) == {'role': 'excitation'}
  assert store.labels_on_object(0, 'gradient', 0) == {}      # a different object

  with pytest.raises(ValueError, match='kind must be one of'):
    store.set_object(0, 'spoiler', 0, 'role', 'x')
  with pytest.raises(ValueError, match='ordinal must be'):
    store.set_object(0, 'rf', -1, 'role', 'x')


def test_clearing_removes_only_what_was_named():
  store = LabelStore(n_blocks=1)
  store.set_block(0, 'SET', 3)
  store.set_block(0, 'LIN', 7)
  store.clear_block(0, 'SET')
  assert store.labels_on(0) == {'LIN': 7}
  store.clear_block(0)
  assert store.labels_on(0) == {}

  store.set_object(0, 'adc', 0, 'a', 1)
  store.set_object(0, 'adc', 0, 'b', 2)
  store.clear_object(0, 'adc', 0, 'a')
  assert store.labels_on_object(0, 'adc', 0) == {'b': 2}
  store.clear_object(0, 'adc', 0)
  assert store.labels_on_object(0, 'adc', 0) == {}


def test_the_block_layer_wins_when_the_two_collide():
  """An object label is the finer annotation and must not override the block.

  Other tools read the block layer, including the file's own declared
  convention, so promoting an object label over it would silently change what
  `filter_blocks` returns for a sequence the user never relabelled.
  """
  store = LabelStore(n_blocks=2)
  store.set_block(0, 'SET', 3)
  store.set_object(0, 'rf', 0, 'SET', 99)        # collides
  store.set_object(1, 'rf', 0, 'ROLE', 'prep')   # does not

  merged = store.to_block_labels()
  assert merged[0]['SET'] == 3, 'the object label overrode the block label'
  assert merged[1]['ROLE'] == 'prep', 'the object label was not promoted'

  # The underlying layers are untouched by merging.
  assert store.labels_on_object(0, 'rf', 0) == {'SET': 99}
  assert store.labels_on(0) == {'SET': 3}


def test_select_has_the_same_and_semantics_as_filter_blocks():
  store = LabelStore(n_blocks=5)
  store.set_blocks([0, 1], 'SET', 1)
  store.set_blocks([2, 3], 'SET', 2)
  store.set_block(1, 'SLC', 7)

  assert store.select(SET=1) == [0, 1]
  assert store.select(SET=1, SLC=7) == [1]        # AND
  assert store.select(SET=9) == []
  assert store.select(SLC=7) == [1]               # block 4 has no SLC at all
  assert store.select() == [0, 1, 2, 3, 4]        # no filter selects everything


def test_names_lists_every_label_in_use():
  store = LabelStore(n_blocks=2)
  store.set_block(0, 'SET', 1)
  store.set_object(1, 'gradient', 0, 'ROLE', 'blip')
  assert store.names() == ['ROLE', 'SET']


def test_a_sidecar_round_trips(tmp_path):
  seq = tmp_path / 'fake.seq'
  seq.write_bytes(b'[VERSION]\nmajor 1\nminor 5\nrevision 0\n')

  store = LabelStore(n_blocks=3, source=str(seq),
                     checksum=sequence_checksum(seq))
  store.set_blocks([0, 2], 'SET', 4)
  store.set_object(1, 'gradient', 2, 'ROLE', 'readout')

  path = LabelStore.sidecar_path(seq)
  assert str(path).endswith('.seq.labels.yaml')
  store.save(path)

  back = LabelStore.load(path, seq)
  assert back.n_blocks == 3
  assert back.block_labels == store.block_labels
  assert back.object_labels == store.object_labels
  assert back.select(SET=4) == [0, 2]
  # The tuple key survived the string round trip.
  assert back.labels_on_object(1, 'gradient', 2) == {'ROLE': 'readout'}


def test_a_sidecar_written_for_another_sequence_is_refused(tmp_path):
  """The failure this guard exists for is silent mislabelling, not a crash."""
  seq = tmp_path / 'a.seq'
  seq.write_bytes(b'original contents')
  store = LabelStore(n_blocks=2, source=str(seq), checksum=sequence_checksum(seq))
  store.set_block(0, 'SET', 1)
  path = LabelStore.sidecar_path(seq)
  store.save(path)

  # Loading against the file it was written for is fine.
  assert LabelStore.load(path, seq).labels_on(0) == {'SET': 1}

  seq.write_bytes(b'the sequence has been edited since')
  with pytest.raises(StaleSidecarError, match='label the wrong blocks'):
    LabelStore.load(path, seq)

  # And the escape hatch works, for inspecting a sidecar known to be stale.
  assert LabelStore.load(path, seq, check=False).labels_on(0) == {'SET': 1}


def test_a_malformed_sidecar_is_refused_by_name(tmp_path):
  import yaml
  path = tmp_path / 'bad.labels.yaml'

  path.write_text(yaml.safe_dump({'version': 99, 'n_blocks': 1}))
  with pytest.raises(ValueError, match='sidecar version'):
    LabelStore.load(path)

  path.write_text(yaml.safe_dump(
    {'version': 1, 'n_blocks': 1, 'block_labels': [{}],
     'object_labels': {'0/rf': {'a': 1}}}))
  with pytest.raises(ValueError, match='malformed object key'):
    LabelStore.load(path)

  path.write_text(yaml.safe_dump(
    {'version': 1, 'n_blocks': 1, 'block_labels': [{}],
     'object_labels': {'0/spoiler/0': {'a': 1}}}))
  with pytest.raises(ValueError, match='unknown kind'):
    LabelStore.load(path)


@pytest.mark.pulseq
def test_a_real_import_seeds_and_agrees_with_filter_blocks(pulseq_import):
  """The addition must not change what the existing selection path returns.

  `epi_pypulseq.seq` carries the writer's SET convention, so this checks both
  that the convention arrives intact and that `select` reproduces
  `filter_blocks` for every value in the file, before and after adding an
  object label that does not collide.
  """
  from conftest import EXAMPLES_SEQ_DIR, skip_if_pypulseq_too_old
  path = EXAMPLES_SEQ_DIR / 'epi_pypulseq.seq'
  if not path.exists():
    pytest.skip('epi_pypulseq.seq not present')
  skip_if_pypulseq_too_old(path)

  imp = pulseq_import(path)
  store = LabelStore.from_import(imp, path)
  assert store.n_blocks == len(imp.block_labels)
  assert store.checksum == sequence_checksum(path)

  values = sorted({d['SET'] for d in imp.block_labels if 'SET' in d})
  assert values, 'the fixture carries no SET labels, so this proves nothing'

  for v in values:
    assert store.select(SET=v) == imp.filter_blocks(SET=v), (
      f'select and filter_blocks disagree for SET={v}')

  # An object label on a name the blocks do not use must not move any selection.
  store.set_object(0, 'rf', 0, 'ROLE', 'prep')
  for v in values:
    assert store.select(SET=v) == imp.filter_blocks(SET=v), (
      f'adding an object label changed the SET={v} selection')
  assert 'ROLE' in store.names()
