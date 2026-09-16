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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .fields import (MAX_GLYPHS, auto_glyph_factor, field_groups,
                     field_labels, glyph_sample, resolve, resolve_vector,
                     vector_names)
from .labels import LabelStore
from .mesh import element_centroids, load_mesh, surface_triangles
from .planning import FOVBox
from .sequence import SequenceModel

#: How solid the planned volume is drawn. Non-zero by default: the plan was a
#: wireframe only, so where it cut the phantom was invisible, and a translucent
#: solid is what shows the intersection. Low enough that it tints rather than
#: hides what is behind it.
DEFAULT_PLAN_OPACITY = 0.25

#: The drawable layers, which is what a pipeline browser lists and toggles.
#: Names rather than actors: the session holds no VTK, and each viewport
#: backend maps a name onto whatever it drew for it.
LAYERS = ('phantom', 'plan', 'arrows', 'glyphs')

#: How the phantom surface is drawn. ParaView's representation menu, less the
#: entries that need a volume renderer or a filter this viewer does not have.
REPRESENTATIONS = ('Surface', 'Surface With Edges', 'Wireframe', 'Points')


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
    self.result_changed = Signal('result_changed')
    self.view_changed = Signal('view_changed')
    self.sequence_changed = Signal('sequence_changed')
    self.labels_changed = Signal('labels_changed')

    self.mesh_path: Optional[str] = None
    self.points: Optional[np.ndarray] = None
    self.cells: Optional[List[Tuple[str, np.ndarray]]] = None
    self.n_frames: int = 0
    self._reader = None
    self._surface: Optional[np.ndarray] = None
    self._surface_source: Optional[np.ndarray] = None
    self._centroids: Optional[np.ndarray] = None

    self.sequence: Optional[SequenceModel] = None
    self.sequence_path: Optional[str] = None
    self.result: Optional[dict] = None
    self.result_path: Optional[str] = None
    self.labels: Optional[LabelStore] = None

    self._box: Optional[FOVBox] = None
    self._undo: List[Optional[FOVBox]] = []
    self._redo: List[Optional[FOVBox]] = []

    self._frame = 0
    self._field: Optional[str] = None
    self._warp_field: Optional[str] = None
    self._warp_scale = 1.0
    self._glyph_field: Optional[str] = None
    self._glyph_scale = 1.0
    self._glyph_count = MAX_GLYPHS
    self._opacity = 1.0
    self._plan_opacity = DEFAULT_PLAN_OPACITY
    self._representation = REPRESENTATIONS[0]
    self._visible = {name: True for name in LAYERS}

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
    self._surface_source = None
    self._centroids = None
    self._frame = 0
    self._forget_result()
    self.mesh_changed.emit(self)

  @property
  def has_mesh(self) -> bool:
    return self.points is not None

  @property
  def surface(self) -> np.ndarray:
    """Boundary triangles, extracted once and cached."""
    self._extract_surface()
    return self._surface

  @property
  def surface_source(self) -> np.ndarray:
    """Which ELEMENT each surface triangle came from.

    A cell field carries one value per element and the surface carries
    triangles, so this is the only thing that lets one be drawn on the other.
    """
    self._extract_surface()
    return self._surface_source

  def _extract_surface(self) -> None:
    self._require_mesh('surface')
    if self._surface is None:
      self._surface, self._surface_source = surface_triangles(
        self.points, self.cells, with_source=True)

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
    self._forget_result()
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

  def _forget_result(self) -> None:
    """Drop a loaded result when its inputs change.

    A result belongs to the phantom and sequence that produced it, and one
    left on screen beside a different phantom is a plausible wrong answer --
    the same failure the label sidecar's checksum exists to prevent. Nothing
    is announced: there is no result to show, which panels read as empty.
    """
    self.result = None
    self.result_path = None

  def load_result(self, path) -> dict:
    """Read a finished run's `kspace.npz` back in.

    The run is a subprocess writing a file, so this is the only way a result
    returns: nothing is shared with it in memory. Kept in the model rather
    than in the run panel because a result outlives the run that made it --
    an old one can be opened without launching anything.
    """
    import numpy as _np

    path = Path(path)
    if path.is_dir():
      path = path / 'kspace.npz'
    with _np.load(path) as handle:
      result = {key: handle[key] for key in handle.files}
    if 'kspace' not in result:
      raise ValueError(
        f'{path}: no "kspace" array -- this is not a feelmri run result '
        f'(it holds {sorted(result) or "nothing"})')
    self.result = result
    self.result_path = str(path)
    self.result_changed.emit(self)
    return result

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
    self._set_number('warp_scale', value)

  @property
  def glyph_field(self) -> Optional[str]:
    """Vector field drawn as arrows, or None. Independent of `warp_field`:
    warping moves the mesh, glyphs annotate it, and a user often wants one
    field doing each."""
    return self._glyph_field

  @glyph_field.setter
  def glyph_field(self, value: Optional[str]) -> None:
    if value != self._glyph_field:
      self._glyph_field = value
      self.view_changed.emit(self)

  @property
  def glyph_scale(self) -> float:
    """A unitless multiplier on the automatic arrow length.

    Unitless on purpose: the automatic factor already puts the longest arrow
    at a fixed fraction of the mesh, so 1.0 shows something whether the field
    is a displacement in metres or a velocity in metres per second, which
    differ here by two orders of magnitude.
    """
    return self._glyph_scale

  @glyph_scale.setter
  def glyph_scale(self, value: float) -> None:
    self._set_number('glyph_scale', value)

  @property
  def glyph_count(self) -> int:
    """How many arrows to draw at most, before the field is subsampled."""
    return self._glyph_count

  @glyph_count.setter
  def glyph_count(self, value: int) -> None:
    value = int(value)
    if value < 1:
      raise ValueError(f'Session.glyph_count must be >= 1, got {value}')
    if value != self._glyph_count:
      self._glyph_count = value
      self.view_changed.emit(self)

  @property
  def opacity(self) -> float:
    """How solid the phantom surface is drawn, 0 to 1."""
    return self._opacity

  @opacity.setter
  def opacity(self, value: float) -> None:
    self._set_number('opacity', value, low=0.0, high=1.0)

  @property
  def plan_opacity(self) -> float:
    """How solid the planned volume is drawn, 0 to 1. 0 leaves the wireframe
    box alone, which is what the viewer did before."""
    return self._plan_opacity

  @plan_opacity.setter
  def plan_opacity(self, value: float) -> None:
    self._set_number('plan_opacity', value, low=0.0, high=1.0,
                     signal=self.plan_changed)

  @property
  def representation(self) -> str:
    """How the phantom surface is drawn, one of `REPRESENTATIONS`."""
    return self._representation

  @representation.setter
  def representation(self, value: str) -> None:
    value = str(value)
    if value not in REPRESENTATIONS:
      raise ValueError(
        f'Session.representation must be one of {REPRESENTATIONS}, '
        f'got {value!r}')
    if value != self._representation:
      self._representation = value
      self.view_changed.emit(self)

  def is_visible(self, layer: str) -> bool:
    """Whether a drawable layer is shown."""
    self._check_layer(layer)
    return self._visible[layer]

  def set_visible(self, layer: str, shown: bool) -> None:
    """Show or hide one layer.

    **Visibility is not opacity.** Opacity 0 still costs the render and still
    depth-sorts against everything behind it; hiding removes the actor. More
    to the point they mean different things to a user -- "I do not want to see
    the plan right now" against "I want to see through it" -- and ParaView
    gives them separate controls for exactly that reason.

    The plan layers announce on `plan_changed`, which is what redraws the
    overlay; the phantom and its arrows on `view_changed`, which rebuilds the
    mesh. Sending all four to one signal would rebuild a 100 000-triangle
    surface to hide a box.
    """
    self._check_layer(layer)
    shown = bool(shown)
    if shown == self._visible[layer]:
      return
    self._visible[layer] = shown
    signal = self.plan_changed if layer in ('plan', 'arrows') \
      else self.view_changed
    signal.emit(self)

  def _check_layer(self, layer: str) -> None:
    if layer not in self._visible:
      raise ValueError(
        f'Session: unknown layer {layer!r}, expected one of {LAYERS}')

  def step_frame(self, delta: int = 1) -> int:
    """Move the frame on, wrapping at the ends. Returns the new frame.

    Wrapping rather than clamping because these are CINES -- the last frame of
    a cardiac cycle is followed by the first, and a play button that stops
    dead at the end of a loop is showing the data wrongly.
    """
    if self.n_frames <= 1:
      return self._frame
    self.frame = (self._frame + int(delta)) % self.n_frames
    return self._frame

  def _set_number(self, name: str, value, low=None, high=None,
                  signal: Optional[Signal] = None) -> None:
    """Assign a validated float and notify, or do nothing if it has not moved."""
    value = float(value)
    if not np.isfinite(value):
      raise ValueError(f'Session.{name} must be finite, got {value}')
    if low is not None and not low <= value <= high:
      raise ValueError(
        f'Session.{name} must be between {low} and {high}, got {value}')
    attribute = f'_{name}'
    if value != getattr(self, attribute):
      setattr(self, attribute, value)
      (signal or self.view_changed).emit(self)

  # -- what the display resolves to ----------------------------------------
  #
  # Both viewport backends -- the native window and the blitting canvas -- ask
  # these rather than reading the frame themselves. They used to carry a copy
  # of the resolution each, which is the shape of duplication this project has
  # already been bitten by: a copy cannot fail when the original is wrong.

  def field_choices(self, frame: Optional[int] = None):
    """`(colour labels, vector names)` for a frame, ready for two comboboxes."""
    if not self.has_mesh:
      return [], []
    _, point_data, cell_data = self.read_frame(
      self._frame if frame is None else frame)
    return field_labels(point_data, cell_data), vector_names(point_data)

  def field_group_choices(self, frame: Optional[int] = None):
    """`field_groups` for a frame -- a field list and its component lists."""
    if not self.has_mesh:
      return []
    _, point_data, cell_data = self.read_frame(
      self._frame if frame is None else frame)
    return field_groups(point_data, cell_data)

  def information(self) -> List[Tuple[str, str]]:
    """`(label, value)` rows describing what is loaded.

    ParaView's Information tab, and it earns its place here for one specific
    reason: **the shipped phantoms are in three different units**, and the
    extent in metres beside the applied scale is what makes a wrong one
    obvious before a submesh silently reads zero.
    """
    if not self.has_mesh:
      return [('file', 'nothing loaded')]
    lo, hi = self.points.min(axis=0), self.points.max(axis=0)
    rows = [
      ('file', str(self.mesh_path).rsplit('/', 1)[-1]),
      ('nodes', f'{len(self.points):,}'),
      ('elements', f'{sum(len(c) for _, c in self.cells):,}'),
      ('cell types', ', '.join(t for t, _ in self.cells)),
      ('surface', f'{len(self.surface):,} triangles'),
      ('frames', str(self.n_frames)),
      ('scale applied', f'{getattr(self, "scale_factor", 1.0):g}'),
      ('extent (m)', ' x '.join(f'{v:.4g}' for v in (hi - lo))),
      ('centre (m)', ' '.join(f'{v:.4g}' for v in 0.5 * (lo + hi))),
    ]
    resolved = self.colour_values()
    if resolved is not None:
      values = resolved[0]
      rows.append((f'range of {self._field}',
                   f'{float(np.min(values)):.4g} to {float(np.max(values)):.4g}'))
    if self._box is not None:
      markers = self.submesh_markers()
      rows.append(('in the slab',
                   f'{int(markers.sum()):,} of {markers.size:,} elements'))
    return rows

  def colour_values(self):
    """`(values, association)` for the chosen field, or None.

    A CELL field is returned per element; it is the caller's job to map it
    onto the surface with `surface_source`, since only the caller knows what
    geometry it is drawing.
    """
    if not self._field:
      return None
    _, point_data, cell_data = self.read_frame(self._frame)
    return resolve(self._field, point_data, cell_data)

  def surface_colour_values(self):
    """The chosen field as one value per surface TRIANGLE or per node.

    Returns `(values, association)`, where a point field comes back untouched
    and a cell field has been indexed through `surface_source`.
    """
    resolved = self.colour_values()
    if resolved is None:
      return None
    values, association = resolved
    # A field that does not describe THIS mesh is reported absent rather than
    # drawn short: a truncated colour array is a picture of the right shape
    # carrying the wrong numbers, which is the failure mode with no symptom.
    if association != 'cell':
      return (values, association) if values.shape[0] >= len(self.points) \
        else None
    source = self.surface_source
    if source.size and values.shape[0] <= int(source.max()):
      return None
    return values[source], association

  def warp_vectors(self) -> Optional[np.ndarray]:
    """The displacement applied to the mesh, unscaled, or None."""
    _, point_data, _ = self.read_frame(self._frame)
    return resolve_vector(self._warp_field, point_data)

  def glyph_arrows(self):
    """`(points, vectors, factor)` for the arrow overlay, or None.

    `points` are the warped positions, so arrows sit on the mesh as drawn
    rather than on its reference configuration. `factor` already carries the
    automatic scale and the user's multiplier, so the caller multiplies by
    nothing.
    """
    if not self._glyph_field or not self.has_mesh:
      return None
    _, point_data, _ = self.read_frame(self._frame)
    vectors = resolve_vector(self._glyph_field, point_data)
    if vectors is None or len(vectors) != len(self.points):
      return None
    points = self.points
    warp = self.warp_vectors()
    if warp is not None and warp.shape == points.shape:
      points = points + self._warp_scale * warp
    points, vectors = glyph_sample(points, vectors, self._glyph_count)
    extent = float(np.max(self.points.max(axis=0) - self.points.min(axis=0)))
    return points, vectors, auto_glyph_factor(vectors, extent) * self._glyph_scale

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
