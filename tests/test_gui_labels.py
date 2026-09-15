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


# -- writing labels back into a .seq ----------------------------------------

def test_the_declared_format_version_is_read_from_the_header():
  """It decides whether the file can be written at all, so it has to be
  answerable before anything tries."""
  from feelmri.gui.model.labels import seq_format_version
  assert seq_format_version('examples/pulseq/epi_pypulseq.seq') == (1, 5, 0)
  assert seq_format_version('tests/data/epi_v142.seq') == (1, 4, 2)


def test_a_v14_file_is_refused_by_name_rather_than_failing_obscurely(tmp_path):
  """pypulseq can READ v1.4 and cannot WRITE it: `seq.write` raises a bare
  `KeyError: 1`, which says nothing about the cause. Refuse up front."""
  from feelmri.gui.model.labels import write_labelled_seq
  imp_labels = [{} for _ in range(231)]
  with pytest.raises(NotImplementedError, match=r'v1\.4\.2'):
    write_labelled_seq('tests/data/epi_v142.seq', tmp_path / 'out.seq',
                       imp_labels)


@pytest.mark.pulseq
def test_writing_the_labels_a_file_already_has_reproduces_them(pulseq_import,
                                                               tmp_path):
  """The identity case, which anything that writes must satisfy first."""
  from feelmri.gui.model.labels import write_labelled_seq
  from conftest import EXAMPLES_SEQ_DIR, skip_if_pypulseq_too_old
  path = EXAMPLES_SEQ_DIR / 'epi_pypulseq.seq'
  if not path.exists():
    pytest.skip('epi_pypulseq.seq not present')
  skip_if_pypulseq_too_old(path)

  from feelmri.PulseqAdapter import import_pulseq
  original = [dict(d) for d in pulseq_import(path).block_labels]
  out = write_labelled_seq(path, tmp_path / 'same.seq', original)
  assert [dict(d) for d in import_pulseq(out).block_labels] == original


@pytest.mark.pulseq
def test_everything_except_the_labels_survives_the_rewrite(pulseq_import,
                                                           tmp_path):
  """A label editor that quietly altered the physics would be far worse than
  one that refused to write at all.

  Only `[BLOCKS]`, whose extension ids are renumbered, and `[SIGNATURE]`,
  which is a recomputed hash the library never verifies, may differ.
  """
  import numpy as np

  from feelmri.PulseqAdapter import import_pulseq
  from feelmri.gui.model.labels import write_labelled_seq
  from conftest import EXAMPLES_SEQ_DIR, skip_if_pypulseq_too_old
  path = EXAMPLES_SEQ_DIR / 'epi_pypulseq.seq'
  if not path.exists():
    pytest.skip('epi_pypulseq.seq not present')
  skip_if_pypulseq_too_old(path)

  original = [dict(d) for d in pulseq_import(path).block_labels]
  out = write_labelled_seq(path, tmp_path / 'same.seq', original)

  def sections(p):
    found, name = {}, None
    for line in open(p):
      token = line.strip()
      if token.startswith('[') and token.endswith(']'):
        name = token
        found[name] = []
      elif name and token:
        found[name].append(token)
    return found

  before, after = sections(path), sections(out)
  assert set(before) == set(after)
  for name in before:
    if name in ('[BLOCKS]', '[SIGNATURE]'):
      continue
    assert before[name] == after[name], f'{name} changed'

  a, b = import_pulseq(path), import_pulseq(out)
  assert len(a.feelmri_seq.blocks) == len(b.feelmri_seq.blocks)
  assert len(a.readouts) == len(b.readouts)
  for ra, rb in zip(a.readouts, b.readouts):
    np.testing.assert_allclose(ra.kspace, rb.kspace, atol=0.0)


@pytest.mark.pulseq
def test_an_edited_label_reaches_the_file_and_comes_back(pulseq_import,
                                                         tmp_path):
  """The point of the writer, and the case a sidecar cannot serve."""
  from feelmri.PulseqAdapter import import_pulseq
  from feelmri.gui.model.labels import write_labelled_seq
  from conftest import EXAMPLES_SEQ_DIR, skip_if_pypulseq_too_old
  path = EXAMPLES_SEQ_DIR / 'epi_pypulseq.seq'
  if not path.exists():
    pytest.skip('epi_pypulseq.seq not present')
  skip_if_pypulseq_too_old(path)

  imp = pulseq_import(path)
  store = LabelStore.from_import(imp, path)
  # A label the file does not use, so it cannot pass by coincidence.
  half = store.n_blocks // 2
  store.set_blocks(range(0, half), 'SLC', 0)
  store.set_blocks(range(half, store.n_blocks), 'SLC', 1)

  out = write_labelled_seq(path, tmp_path / 'edited.seq',
                           store.to_block_labels())
  back = import_pulseq(out)
  assert [d.get('SLC') for d in back.block_labels] == \
         [0] * half + [1] * (store.n_blocks - half)

  # The file's own SET convention must be untouched by the addition.
  for value in sorted({d['SET'] for d in imp.block_labels if 'SET' in d}):
    assert back.filter_blocks(SET=value) == imp.filter_blocks(SET=value)


@pytest.mark.pulseq
def test_the_written_file_selects_the_same_blocks_the_gui_did(pulseq_import,
                                                              tmp_path):
  """M4's acceptance: a SET map made in the GUI must mean the same thing to
  `filter_blocks` after a round trip through the file.

  Built by hand rather than seeded from the import, so it is the GUI's map
  being checked and not the file's own.
  """
  from feelmri.PulseqAdapter import import_pulseq
  from feelmri.gui.model.labels import write_labelled_seq
  from conftest import EXAMPLES_SEQ_DIR, skip_if_pypulseq_too_old
  path = EXAMPLES_SEQ_DIR / 'epi_pypulseq.seq'
  if not path.exists():
    pytest.skip('epi_pypulseq.seq not present')
  skip_if_pypulseq_too_old(path)

  imp = pulseq_import(path)
  store = LabelStore(n_blocks=len(imp.block_labels))
  groups = {0: range(0, 10), 1: range(10, 40), 2: range(40, 41)}
  groups[3] = range(41, store.n_blocks)
  for value, blocks in groups.items():
    store.set_blocks(blocks, 'SET', value)

  out = write_labelled_seq(path, tmp_path / 'byhand.seq',
                           store.to_block_labels())
  back = import_pulseq(out)
  for value, blocks in groups.items():
    assert back.filter_blocks(SET=value) == list(blocks), (
      f'SET={value} did not survive the round trip')
    assert store.select(SET=value) == back.filter_blocks(SET=value)


@pytest.mark.pulseq
def test_a_label_that_disappears_mid_sequence_is_refused(tmp_path):
  """LABELSET can only SET a value, never unset one, so a running state that
  drops a label is not expressible and must say so rather than write a file
  that means something else."""
  from feelmri.gui.model.labels import write_labelled_seq
  from conftest import EXAMPLES_SEQ_DIR, skip_if_pypulseq_too_old
  path = EXAMPLES_SEQ_DIR / 'epi_pypulseq.seq'
  if not path.exists():
    pytest.skip('epi_pypulseq.seq not present')
  skip_if_pypulseq_too_old(path)

  from feelmri.PulseqAdapter import import_pulseq
  n = len(import_pulseq(path).block_labels)
  labels = [{'SET': 1} for _ in range(n)]
  labels[n // 2] = {}                       # SET vanishes here
  with pytest.raises(ValueError, match='never unset'):
    write_labelled_seq(path, tmp_path / 'bad.seq', labels)


@pytest.mark.pulseq
def test_a_name_pulseq_does_not_know_is_refused_with_the_supported_list(
    tmp_path):
  """A made-up label would be dropped silently by the writer otherwise."""
  from feelmri.gui.model.labels import write_labelled_seq
  from conftest import EXAMPLES_SEQ_DIR, skip_if_pypulseq_too_old
  path = EXAMPLES_SEQ_DIR / 'epi_pypulseq.seq'
  if not path.exists():
    pytest.skip('epi_pypulseq.seq not present')
  skip_if_pypulseq_too_old(path)

  from feelmri.PulseqAdapter import import_pulseq
  n = len(import_pulseq(path).block_labels)
  with pytest.raises(ValueError, match='not Pulseq labels'):
    write_labelled_seq(path, tmp_path / 'bad.seq',
                       [{'MYTAG': 1} for _ in range(n)])


@pytest.mark.pulseq
def test_a_wrong_block_count_is_refused(tmp_path):
  from feelmri.gui.model.labels import write_labelled_seq
  from conftest import EXAMPLES_SEQ_DIR, skip_if_pypulseq_too_old
  path = EXAMPLES_SEQ_DIR / 'epi_pypulseq.seq'
  if not path.exists():
    pytest.skip('epi_pypulseq.seq not present')
  skip_if_pypulseq_too_old(path)
  with pytest.raises(ValueError, match='label dicts for'):
    write_labelled_seq(path, tmp_path / 'bad.seq', [{'SET': 0}])


@pytest.mark.pulseq
def test_extensions_that_are_not_labels_survive_the_rewrite(tmp_path):
  """A file may carry TRIGGERS beside its labels, and rewriting the labels
  must not drop them.

  No bundled fixture has a non-label extension, so one is built here. That
  gap is the reason this test exists: discarding the non-label chain passed
  the whole suite before it.
  """
  import numpy as np
  pp = pytest.importorskip('pypulseq')

  from feelmri.gui.model.labels import write_labelled_seq

  system = pp.Opts()
  seq = pp.Sequence(system=system)
  rf = pp.make_block_pulse(flip_angle=np.pi / 2, duration=0.4e-3,
                           system=system)
  trigger = pp.make_trigger(channel='physio1', duration=100e-6, system=system)
  seq.add_block(rf, pp.make_label(label='SET', type='SET', value=1), trigger)
  seq.add_block(pp.make_delay(1e-3),
                pp.make_label(label='SET', type='SET', value=2))
  seq.add_block(pp.make_delay(1e-3))
  source = tmp_path / 'trigger.seq'
  seq.write(str(source))

  def extension_lines(path):
    lines, section = [], None
    for raw in open(path):
      token = raw.strip()
      if token.startswith('[') and token.endswith(']'):
        section = token
      elif section == '[EXTENSIONS]' and token and not token.startswith('#'):
        lines.append(token)
    return lines

  assert any(line.startswith('extension TRIGGERS')
             for line in extension_lines(source))

  def triggered_blocks(path):
    """Blocks whose extension chain actually REACHES a TRIGGERS entry.

    Reading the `[EXTENSIONS]` text is not enough: pypulseq writes the whole
    library whether or not any block references it, so the declaration and
    the trigger row survive even when every link to them has been dropped.
    Discarding the non-label chain passed a text-based version of this test.
    """
    reread = pp.Sequence()
    reread.read(str(path), detect_rf_use=False)
    trigger_type = None
    for numeric, name in zip(reread.extension_numeric_idx,
                             reread.extension_string_idx):
      if name == 'TRIGGERS':
        trigger_type = numeric
    assert trigger_type is not None, 'no TRIGGERS type in the file'
    found = []
    for block_id, events in reread.block_events.items():
      extension = int(events[6])
      while extension:
        type_id, _ref, following = reread.extensions_library.data[extension]
        if int(type_id) == trigger_type:
          found.append(block_id)
          break
        extension = int(following)
    return sorted(found)

  linked_before = triggered_blocks(source)
  assert linked_before, 'the fixture must link a block to the trigger'

  out = write_labelled_seq(source, tmp_path / 'relabelled.seq',
                           [{'SET': 7}, {'SET': 8}, {'SET': 8}])

  assert triggered_blocks(out) == linked_before, (
    'the block lost its link to the trigger')

  from feelmri.PulseqAdapter import import_pulseq
  assert [dict(d) for d in import_pulseq(out).block_labels] == \
         [{'SET': 7}, {'SET': 8}, {'SET': 8}]


@pytest.mark.pulseq
def test_loading_a_sequence_keeps_the_labels_the_file_already_has():
  """Opening a labelled `.seq` must not show a blank slate.

  `Session.set_sequence` takes a `Sequence`, which no longer knows its
  LABELSET state, so a caller that imports and then calls it directly loses
  the file's own convention silently. `load_sequence` is what keeps the two
  together, and it lives in the model precisely so this is testable.
  """
  from feelmri.gui.model.session import Session
  from conftest import EXAMPLES_SEQ_DIR, skip_if_pypulseq_too_old
  path = EXAMPLES_SEQ_DIR / 'epi_pypulseq.seq'
  if not path.exists():
    pytest.skip('epi_pypulseq.seq not present')
  skip_if_pypulseq_too_old(path)

  session = Session()
  imported = session.load_sequence(path)

  assert session.labels is not None
  assert session.labels.to_block_labels() == [dict(d)
                                              for d in imported.block_labels]
  values = sorted({d['SET'] for d in imported.block_labels if 'SET' in d})
  assert values, 'the fixture carries no SET labels, so this proves nothing'
  for value in values:
    assert session.labels.select(SET=value) == imported.filter_blocks(SET=value)

  # And the control: the blank path really would lose them, which is what
  # makes this more than a restatement of from_import.
  blank = Session()
  blank.set_sequence(imported.feelmri_seq, path=path)
  assert blank.labels.to_block_labels() != session.labels.to_block_labels()


# -- the per-block view and the running view are different things -----------

def test_a_running_state_carries_each_value_forward():
  """`LABELSET` is sticky and the format cannot unset a value, so the running
  view is the only faithful reading of what a written file would mean."""
  store = LabelStore(n_blocks=5)
  store.set_block(1, 'SET', 3)
  store.set_block(3, 'SET', 7)

  assert store.to_block_labels() == [{}, {'SET': 3}, {}, {'SET': 7}, {}]
  assert store.to_running_labels() == [
    {}, {'SET': 3}, {'SET': 3}, {'SET': 7}, {'SET': 7}]


def test_the_two_views_agree_for_a_store_seeded_from_a_file():
  """A file's own state was already read as a running one, so nothing moves.

  This is what makes the distinction safe to introduce: it changes the
  meaning of nothing that came from a `.seq`.
  """
  store = LabelStore(n_blocks=4,
                     block_labels=[{'SET': 1}, {'SET': 1}, {'SET': 2},
                                   {'SET': 2}])
  assert store.to_running_labels() == store.to_block_labels()


def test_an_object_label_carries_forward_in_the_running_view():
  """An object label merges down onto its block, and then behaves like any
  other block label -- which for a file means from that block onward."""
  store = LabelStore(n_blocks=4)
  store.set_object(1, 'gradient', 0, 'SEG', 5)
  assert store.to_block_labels() == [{}, {'SEG': 5}, {}, {}]
  assert store.to_running_labels() == [{}, {'SEG': 5}, {'SEG': 5}, {'SEG': 5}]


@pytest.mark.pulseq
def test_a_sparse_edit_exports_and_means_what_the_running_view_said(tmp_path):
  """End to end: label a range and one object, export, read it back.

  The numbers asserted are `to_running_labels`' own, so the file and the
  view of it that the panel shows before exporting cannot disagree.
  """
  from feelmri.PulseqAdapter import import_pulseq
  from feelmri.gui.model.labels import write_labelled_seq
  from feelmri.gui.model.session import Session
  from conftest import EXAMPLES_SEQ_DIR, skip_if_pypulseq_too_old
  path = EXAMPLES_SEQ_DIR / 'epi_pypulseq.seq'
  if not path.exists():
    pytest.skip('epi_pypulseq.seq not present')
  skip_if_pypulseq_too_old(path)

  session = Session()
  original = session.load_sequence(path)
  store = session.labels
  store.set_blocks(range(0, 10), 'SLC', 1)
  store.set_object(1, 'gradient', 0, 'SEG', 5)

  running = store.to_running_labels()
  out = write_labelled_seq(path, tmp_path / 'sparse.seq', running)
  back = import_pulseq(out)

  assert [dict(d) for d in back.block_labels] == running
  # Sticky, and the test says so rather than asserting the range naively.
  assert back.filter_blocks(SLC=1) == list(range(store.n_blocks))
  assert back.filter_blocks(SEG=5) == list(range(1, store.n_blocks))
  # The file's own SET convention is untouched by either addition.
  for value in sorted({d['SET'] for d in original.block_labels if 'SET' in d}):
    assert back.filter_blocks(SET=value) == original.filter_blocks(SET=value)
