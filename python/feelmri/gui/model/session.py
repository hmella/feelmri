"""The document the panels share, and the notifications that keep them in step.

Every panel reads this and none of them talk to each other. A panel subscribes
to the signals it cares about and redraws when one fires, which is what lets
the view layer stay thin enough to be worth not unit-testing.

The observer is hand-rolled on purpose. It is forty lines, and taking a
dependency for it would sit badly next to a brief that asks for no heavy
libraries.

Nothing here imports a view, a plotting library or VTK, so a script can build a
`Session`, set a plan on it and read the derived geometry without a display.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .labels import LabelStore
from .mesh import element_centroids, load_mesh, surface_triangles
from .planning import FOVBox
from .sequence import SequenceModel


class Signal:
  """A minimal observer. Callbacks fire in the order they were connected."""

  def __init__(self, name: str = ''):
    self.name = name
    self._slots: List[Callable[..., None]] = []
    self._blocked = False

  def connect(self, fn: Callable[..., None]) -> Callable[..., None]:
    if fn not in self._slots:
      self._slots.append(fn)
    return fn

  def disconnect(self, fn: Callable[..., None]) -> None:
    if fn in self._slots:
      self._slots.remove(fn)

  def emit(self, *args, **kwargs) -> None:
    if self._blocked:
      return
    for fn in list(self._slots):
      fn(*args, **kwargs)

  @contextmanager
  def blocked(self):
    """Suppress emissions, so a batch of edits notifies once at the end."""
    previous, self._blocked = self._blocked, True
    try:
      yield
    finally:
      self._blocked = previous

  def __len__(self) -> int:
    return len(self._slots)


class Session:
  """What is loaded, what is planned, and what the panels are showing.

  The plan is the only thing with undo, because it is the only thing a user
  nudges repeatedly. Reloading a mesh or a sequence is a deliberate act and is
  not worth a history.
  """

  #: How many plan edits to remember.
  UNDO_DEPTH = 64

  def __init__(self):
    self.mesh_changed = Signal('mesh_changed')
    self.plan_changed = Signal('plan_changed')
    self.view_changed = Signal('view_changed')
    self.sequence_changed = Signal('sequence_changed')
    self.labels_changed = Signal('labels_changed')

    self.mesh_path: Optional[str] = None
    self.points: Optional[np.ndarray] = None
    self.cells: Optional[List[Tuple[str, np.ndarray]]] = None
    self.n_frames: int = 0
    self._reader = None
    self._surface: Optional[np.ndarray] = None
    self._centroids: Optional[np.ndarray] = None

    self.sequence: Optional[SequenceModel] = None
    self.sequence_path: Optional[str] = None
    self.labels: Optional[LabelStore] = None

    self._box: Optional[FOVBox] = None
    self._undo: List[Optional[FOVBox]] = []
    self._redo: List[Optional[FOVBox]] = []

    self._frame = 0
    self._field: Optional[str] = None
    self._warp_field: Optional[str] = None
    self._warp_scale = 1.0

  # -- the mesh -------------------------------------------------------------

  def load_mesh(self, path, scale_factor: float = 1.0) -> None:
    """Read a mesh and drop every derived cache.

    No `FEMPhantom` is constructed. That keeps the viewer clear of the
    partitioning a phantom runs at construction, and of the
    `local_to_global_nodes` property whose mere read binds the partition.

    `scale_factor` multiplies the coordinates, exactly as `FEMPhantom`'s own
    argument does, and for the same reason: **the shipped phantoms are not in
    one unit** -- metres, centimetres and millimetres across five files. The
    plan is in metres, so without it the field of view is drawn a hundred or a
    thousand times too small and the submesh comes out empty.
    `mesh.suggest_scale_factor` proposes one; it is never applied silently.
    """
    points, cells, n_frames, reader = load_mesh(path)
    scale_factor = float(scale_factor)
    if not np.isfinite(scale_factor) or scale_factor <= 0:
      raise ValueError(
        f'Session.load_mesh: scale_factor must be positive, got {scale_factor}')
    if scale_factor != 1.0:
      points = points * scale_factor
    self.scale_factor = scale_factor
    self.mesh_path = str(path)
    self.points, self.cells, self.n_frames, self._reader = (
      points, cells, n_frames, reader)
    self._surface = None
    self._centroids = None
    self._frame = 0
    self.mesh_changed.emit(self)

  @property
  def has_mesh(self) -> bool:
    return self.points is not None

  @property
  def surface(self) -> np.ndarray:
    """Boundary triangles, extracted once and cached."""
    self._require_mesh('surface')
    if self._surface is None:
      self._surface = surface_triangles(self.points, self.cells)
    return self._surface

  @property
  def centroids(self) -> np.ndarray:
    """Element centroids, extracted once and cached.

    Cached because the plan panel recomputes the submesh on every drag, and
    the centroids are the expensive half of that on a large mesh.
    """
    self._require_mesh('centroids')
    if self._centroids is None:
      self._centroids = element_centroids(self.points, self.cells)
    return self._centroids

  def read_frame(self, index: int) -> Tuple[float, Dict, Dict]:
    """One frame's point and cell data, or empty dicts for a static mesh."""
    from .mesh import read_frame
    return read_frame(self._reader, index)

  # -- the plan -------------------------------------------------------------

  @property
  def box(self) -> Optional[FOVBox]:
    return self._box

  @box.setter
  def box(self, value: Optional[FOVBox]) -> None:
    if value is not None and not isinstance(value, FOVBox):
      raise TypeError(f'Session.box must be a FOVBox or None, got '
                      f'{type(value).__name__}')
    if self._same_box(value, self._box):
      return                                    # no notification for a no-op
    self._undo.append(copy.deepcopy(self._box))
    del self._undo[:-self.UNDO_DEPTH]
    self._redo.clear()
    self._box = value
    self.plan_changed.emit(self)

  @staticmethod
  def _same_box(a: Optional[FOVBox], b: Optional[FOVBox]) -> bool:
    if a is None or b is None:
      return a is b
    return (np.array_equal(a.fov, b.fov) and np.array_equal(a.loc, b.loc)
            and np.array_equal(a.angles, b.angles))

  @property
  def can_undo(self) -> bool:
    return bool(self._undo)

  @property
  def can_redo(self) -> bool:
    return bool(self._redo)

  def undo(self) -> bool:
    """Step the plan back. Returns whether anything moved."""
    if not self._undo:
      return False
    self._redo.append(copy.deepcopy(self._box))
    self._box = self._undo.pop()
    self.plan_changed.emit(self)
    return True

  def redo(self) -> bool:
    if not self._redo:
      return False
    self._undo.append(copy.deepcopy(self._box))
    self._box = self._redo.pop()
    self.plan_changed.emit(self)
    return True

  def submesh_markers(self, axis: int = 2) -> np.ndarray:
    """Elements inside the planned slab, as a mask over GLOBAL ELEMENTS.

    This is what `FEMPhantom.create_submesh` takes. The centroids are moved
    into the imaging frame first, matching the examples, which compute their
    markers after `orient`.
    """
    self._require_mesh('submesh_markers')
    if self._box is None:
      raise RuntimeError('Session.submesh_markers: no plan has been set')
    from .mesh import slab_markers
    return slab_markers(self._box.to_imaging(self.centroids),
                        0.5 * self._box.fov[axis], axis=axis)

  # -- the sequence and its labels -----------------------------------------

  def set_sequence(self, sequence, path=None,
                   labels: Optional[LabelStore] = None) -> None:
    """Attach a sequence, and a label store to go with it."""
    self.sequence = SequenceModel(sequence)
    self.sequence_path = None if path is None else str(path)
    self.labels = labels if labels is not None else LabelStore(
      n_blocks=len(sequence.blocks))
    self.sequence_changed.emit(self)
    self.labels_changed.emit(self)

  def load_sequence(self, path, scanner=None) -> None:
    """Read a Pulseq `.seq` and attach it WITH the labels it already carries.

    Seeding the store from the import is the whole point: `set_sequence` takes
    a `Sequence`, which no longer knows its `LABELSET` state, so a caller that
    imports and then calls it directly gets a blank slate and silently loses
    the file's own convention -- which is what a user opens a labelled
    sequence to see. Doing it here rather than in the shell keeps that out of
    view code, where it cannot be tested.
    """
    from ...PulseqAdapter import import_pulseq   # pypulseq lives behind this
    imported = import_pulseq(str(path), scanner=scanner) if scanner is not None \
      else import_pulseq(str(path))
    self.set_sequence(imported.feelmri_seq, path=path,
                      labels=LabelStore.from_import(imported, path))
    return imported

  def notify_labels(self) -> None:
    """Announce a label edit. The store is mutable, so this is explicit."""
    self.labels_changed.emit(self)

  # -- what the panels are showing -----------------------------------------

  @property
  def frame(self) -> int:
    return self._frame

  @frame.setter
  def frame(self, value: int) -> None:
    value = int(value)
    if self.n_frames and not 0 <= value < self.n_frames:
      raise IndexError(
        f'Session.frame: {value} is out of range for {self.n_frames} frames')
    if value != self._frame:
      self._frame = value
      self.view_changed.emit(self)

  @property
  def field(self) -> Optional[str]:
    return self._field

  @field.setter
  def field(self, value: Optional[str]) -> None:
    if value != self._field:
      self._field = value
      self.view_changed.emit(self)

  @property
  def warp_field(self) -> Optional[str]:
    return self._warp_field

  @warp_field.setter
  def warp_field(self, value: Optional[str]) -> None:
    if value != self._warp_field:
      self._warp_field = value
      self.view_changed.emit(self)

  @property
  def warp_scale(self) -> float:
    return self._warp_scale

  @warp_scale.setter
  def warp_scale(self, value: float) -> None:
    value = float(value)
    if not np.isfinite(value):
      raise ValueError(f'Session.warp_scale must be finite, got {value}')
    if value != self._warp_scale:
      self._warp_scale = value
      self.view_changed.emit(self)

  # -- internals ------------------------------------------------------------

  def _require_mesh(self, who: str) -> None:
    if self.points is None:
      raise RuntimeError(f'Session.{who}: no mesh is loaded')

  def summary(self) -> Dict[str, Any]:
    """A flat description, for a status bar or a bug report."""
    return {
      'mesh': self.mesh_path,
      'nodes': None if self.points is None else int(len(self.points)),
      'elements': None if self.cells is None
                  else int(sum(len(c) for _, c in self.cells)),
      'frames': self.n_frames,
      'frame': self._frame,
      'field': self._field,
      'sequence': self.sequence_path,
      'blocks': None if self.sequence is None else len(self.sequence.spans),
      'planned': self._box is not None,
    }
