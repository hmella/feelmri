"""Labels on blocks and on individual MR objects, with a .seq sidecar.

What the library has today is read-only and per block. `import_pulseq` folds
the file's LABELSET and LABELINC extensions into a running dict per block
(`PulseqImport.block_labels`), and `filter_blocks` queries it. Nothing writes a
label back, and no label can attach to a single RF pulse, gradient or ADC.

This adds the missing half. Two layers:

  - **block labels**, seeded from `block_labels` so an imported file arrives
    with its own convention already in place;
  - **object labels**, keyed by `(block, kind, ordinal)`, which the library has
    no equivalent of.

`to_block_labels()` merges the second layer down into the first, so
`PulseqImport.filter_blocks(**labels)` keeps working unchanged on the result.
That is deliberate: the selection path the examples already walk stays the
selection path, and nothing downstream has to learn about object labels.

Persistence is a YAML sidecar rather than a rewritten `.seq`. A sidecar cannot
corrupt a file the Pulseq golden tests pin, and it carries a **checksum of the
sequence it describes**, so a sidecar that has drifted from its file is
refused by name instead of silently labelling the wrong blocks.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

LabelValue = Union[int, str]
ObjectKey = Tuple[int, str, int]          # (block, kind, ordinal)

#: Kinds an object label may attach to, matching `SequenceModel.objects()`.
OBJECT_KINDS = ('rf', 'gradient', 'adc')

_SIDECAR_VERSION = 1


def sequence_checksum(path) -> str:
  """SHA-256 of a file, used to tie a sidecar to the sequence it describes."""
  h = hashlib.sha256()
  with open(path, 'rb') as fh:
    for chunk in iter(lambda: fh.read(65536), b''):
      h.update(chunk)
  return h.hexdigest()


class StaleSidecarError(RuntimeError):
  """The sidecar was written against a different sequence file."""


@dataclass
class LabelStore:
  """Block and object labels for one sequence.

  `n_blocks` is fixed at construction and every write is range-checked against
  it, because a label on a block that does not exist is the failure mode this
  whole class is exposed to: indices are the only thing tying a label to a
  sequence.
  """

  n_blocks: int
  block_labels: List[Dict[str, LabelValue]] = field(default_factory=list)
  object_labels: Dict[ObjectKey, Dict[str, LabelValue]] = field(default_factory=dict)
  source: Optional[str] = None
  checksum: Optional[str] = None

  def __post_init__(self):
    if self.n_blocks < 0:
      raise ValueError(f'LabelStore: n_blocks must be >= 0, got {self.n_blocks}')
    if not self.block_labels:
      self.block_labels = [{} for _ in range(self.n_blocks)]
    elif len(self.block_labels) != self.n_blocks:
      raise ValueError(
        f'LabelStore: got {len(self.block_labels)} label dicts for '
        f'{self.n_blocks} blocks')

  # -- construction ---------------------------------------------------------

  @classmethod
  def from_import(cls, imp, path=None) -> 'LabelStore':
    """Seed from a `PulseqImport`, keeping its running label state.

    The file's own convention arrives intact, so opening `epi_pypulseq.seq`
    shows the SET groups the writer put there rather than a blank slate.
    """
    blocks = [dict(d) for d in imp.block_labels]
    return cls(n_blocks=len(blocks), block_labels=blocks,
               source=None if path is None else str(path),
               checksum=None if path is None else sequence_checksum(path))

  # -- writing --------------------------------------------------------------

  def set_block(self, block: int, name: str, value: LabelValue) -> None:
    """Assign one label on one block."""
    self._check_block(block)
    self.block_labels[self._wrap(block)][str(name)] = value

  def set_blocks(self, blocks: Iterable[int], name: str,
                 value: LabelValue) -> None:
    """Assign the same label across a selection, which is the usual gesture."""
    for b in blocks:
      self.set_block(b, name, value)

  def clear_block(self, block: int, name: Optional[str] = None) -> None:
    """Drop one label, or all of them when `name` is None."""
    self._check_block(block)
    d = self.block_labels[self._wrap(block)]
    if name is None:
      d.clear()
    else:
      d.pop(str(name), None)

  def set_object(self, block: int, kind: str, ordinal: int,
                 name: str, value: LabelValue) -> None:
    """Assign a label to one MR object inside a block."""
    self._check_block(block)
    if kind not in OBJECT_KINDS:
      raise ValueError(
        f'set_object: kind must be one of {OBJECT_KINDS}, got {kind!r}')
    if ordinal < 0:
      raise ValueError(f'set_object: ordinal must be >= 0, got {ordinal}')
    key = (self._wrap(block), kind, int(ordinal))
    self.object_labels.setdefault(key, {})[str(name)] = value

  def clear_object(self, block: int, kind: str, ordinal: int,
                   name: Optional[str] = None) -> None:
    key = (self._wrap(block), kind, int(ordinal))
    if key not in self.object_labels:
      return
    if name is None:
      del self.object_labels[key]
    else:
      self.object_labels[key].pop(str(name), None)
      if not self.object_labels[key]:
        del self.object_labels[key]

  # -- reading --------------------------------------------------------------

  def to_block_labels(self) -> List[Dict[str, LabelValue]]:
    """Both layers merged to the per-block shape `filter_blocks` expects.

    An object label is promoted to its block. Where a block already carries the
    same label name, **the block's own value wins**: the block layer is what
    the file itself declared or what the user set explicitly on the block, and
    an object label is the finer annotation, so it must not silently override
    the coarser one that other tools read.
    """
    merged = [dict(d) for d in self.block_labels]
    for (block, _kind, _ordinal), labels in sorted(self.object_labels.items()):
      for name, value in labels.items():
        merged[block].setdefault(name, value)
    return merged

  def labels_on(self, block: int) -> Dict[str, LabelValue]:
    self._check_block(block)
    return dict(self.block_labels[self._wrap(block)])

  def labels_on_object(self, block: int, kind: str,
                       ordinal: int) -> Dict[str, LabelValue]:
    return dict(self.object_labels.get((self._wrap(block), kind, int(ordinal)), {}))

  def names(self) -> List[str]:
    """Every label name in use, sorted, for populating a picker."""
    seen = set()
    for d in self.block_labels:
      seen.update(d)
    for d in self.object_labels.values():
      seen.update(d)
    return sorted(seen)

  def select(self, **labels: LabelValue) -> List[int]:
    """Block indices whose merged labels match every keyword.

    Same AND semantics as `PulseqImport.filter_blocks`, and a block missing a
    requested name never matches, so the two agree on the merged state.
    """
    merged = self.to_block_labels()
    if not labels:
      return list(range(self.n_blocks))
    return [i for i, d in enumerate(merged)
            if all(name in d and d[name] == value
                   for name, value in labels.items())]

  # -- persistence ----------------------------------------------------------

  @staticmethod
  def sidecar_path(seq_path) -> Path:
    """`foo.seq` -> `foo.seq.labels.yaml`, beside the sequence."""
    return Path(str(seq_path) + '.labels.yaml')

  def save(self, path) -> str:
    """Write the sidecar.

    Object keys become `"block/kind/ordinal"` strings, since YAML has no tuple
    key. The checksum recorded at construction travels with them.
    """
    import yaml

    doc = {
      'version': _SIDECAR_VERSION,
      'source': self.source,
      'checksum': self.checksum,
      'n_blocks': self.n_blocks,
      'block_labels': [dict(d) for d in self.block_labels],
      'object_labels': {f'{b}/{k}/{o}': dict(v)
                        for (b, k, o), v in sorted(self.object_labels.items())},
    }
    with open(path, 'w') as fh:
      yaml.safe_dump(doc, fh, sort_keys=False)
    return str(path)

  @classmethod
  def load(cls, path, seq_path=None, *, check: bool = True) -> 'LabelStore':
    """Read a sidecar, refusing one that no longer matches its sequence.

    Labels are tied to their sequence by BLOCK INDEX and nothing else, so a
    sidecar written against a different file does not fail, it mislabels. When
    `seq_path` is given its checksum is compared and a mismatch raises
    `StaleSidecarError`. Pass `check=False` only to inspect a sidecar you
    already know is stale.
    """
    import yaml

    with open(path) as fh:
      doc = yaml.safe_load(fh) or {}

    version = doc.get('version')
    if version != _SIDECAR_VERSION:
      raise ValueError(
        f'{path}: sidecar version {version!r}, this build writes '
        f'{_SIDECAR_VERSION}')

    if seq_path is not None and check:
      actual = sequence_checksum(seq_path)
      recorded = doc.get('checksum')
      if recorded is not None and recorded != actual:
        raise StaleSidecarError(
          f'{path} was written for a sequence with checksum '
          f'{recorded[:12]}... but {seq_path} is {actual[:12]}.... The labels '
          f'are keyed by block index, so applying them would label the wrong '
          f'blocks. Re-create them, or pass check=False to inspect.')

    objects: Dict[ObjectKey, Dict[str, LabelValue]] = {}
    for key, value in (doc.get('object_labels') or {}).items():
      parts = str(key).split('/')
      if len(parts) != 3:
        raise ValueError(f'{path}: malformed object key {key!r}, '
                         f'expected "block/kind/ordinal"')
      block, kind, ordinal = int(parts[0]), parts[1], int(parts[2])
      if kind not in OBJECT_KINDS:
        raise ValueError(f'{path}: object key {key!r} has unknown kind {kind!r}')
      objects[(block, kind, ordinal)] = dict(value)

    return cls(n_blocks=int(doc.get('n_blocks', 0)),
               block_labels=[dict(d) for d in (doc.get('block_labels') or [])],
               object_labels=objects,
               source=doc.get('source'),
               checksum=doc.get('checksum'))

  # -- internals ------------------------------------------------------------

  def _check_block(self, block: int) -> None:
    if not -self.n_blocks <= int(block) < self.n_blocks:
      raise IndexError(
        f'LabelStore: block {block} is out of range for {self.n_blocks} blocks')

  def _wrap(self, block: int) -> int:
    return int(block) % self.n_blocks
