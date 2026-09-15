"""What can be coloured by, and how a chosen name resolves to numbers.

**The viewer offered almost none of the data it had loaded.** The shell
enumerated point fields with `ndim == 1` for colouring and `shape == (N, 3)`
for warping, and measured against the five shipped phantoms that leaves the
`Colour by` list EMPTY on every one of them:

| phantom | stored | offered |
|---|---|---|
| `heart_P1_hex`, `heart_P2_tetra` | `displacement (N,3)` | nothing |
| `aorta_P1_tetra` | `velocity (N,3)`, `pressure (N,1)` | nothing |
| `water_fat_P1_prism` | `point_markers (N,1)`, `cell_markers` | nothing |

Two independent reasons, and the second is the less obvious one: a vector was
never colourable at all, and `pressure` and `point_markers` ARE scalars but
are stored as `(N, 1)`, so `ndim == 1` rejects them too. `cell_markers` is cell
data, which nothing read.

So this module enumerates a data dictionary the way ParaView does -- a scalar
by name, a vector expanded into its magnitude and its components -- and
resolves a chosen label back to one value per node or per element. It holds no
figure and no VTK, so the rule the two viewport backends share is testable and
written once rather than copied into each.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Component names for a 3-vector, in ParaView's spelling. Anything wider is
#: numbered instead, since X/Y/Z would be a guess about what the axes mean.
AXIS_COMPONENTS = ('X', 'Y', 'Z')

#: The label suffix marking a field that lives on ELEMENTS rather than nodes.
#: Point and cell fields can share a name and mean different things, so the
#: association has to survive in the label the session stores.
CELL_SUFFIX = ' [cell]'

#: `component` value meaning "the vector's length", which is what a viewer
#: shows by default and the only entry that is independent of the frame.
MAGNITUDE = -1


@dataclass(frozen=True)
class FieldRef:
  """A choice of field: its name, where it lives, and which component.

  `component` is None for a field that is already scalar, `MAGNITUDE` for the
  vector norm, and otherwise a column index.
  """

  name: str
  association: str = 'point'          # 'point' or 'cell'
  component: Optional[int] = None

  def __post_init__(self):
    if self.association not in ('point', 'cell'):
      raise ValueError(
        f"FieldRef: association must be 'point' or 'cell', got "
        f"{self.association!r}")

  @property
  def label(self) -> str:
    """The string a combobox shows, and the one `parse` reads back."""
    text = self.name
    if self.component is not None:
      text = f'{text} ({component_name(self.component)})'
    return text + (CELL_SUFFIX if self.association == 'cell' else '')

  @classmethod
  def parse(cls, label: str,
            point_names: Sequence[str] = (),
            cell_names: Sequence[str] = ()) -> 'FieldRef':
    """Read a label back into a reference.

    **An exact name wins over the suffix grammar.** A field genuinely called
    `pressure (X)` is legal in a data file, and taking the suffix off it would
    silently show a different array; so the known names are checked first when
    they are supplied, and the suffixes are only stripped from what is left.
    """
    if label in point_names:
      return cls(label, 'point')
    if label in cell_names:
      return cls(label, 'cell')

    association = 'point'
    if label.endswith(CELL_SUFFIX):
      association, label = 'cell', label[:-len(CELL_SUFFIX)]
      if label in cell_names:
        return cls(label, 'cell')

    component = None
    if label.endswith(')') and ' (' in label:
      head, _, tail = label.rpartition(' (')
      component = _component_index(tail[:-1])
      if component is not None:
        label = head
    return cls(label, association, component)


def component_name(component: int) -> str:
  """`MAGNITUDE` to `'Magnitude'`, 0 to `'X'`, 7 to `'7'`."""
  if component == MAGNITUDE:
    return 'Magnitude'
  if 0 <= component < len(AXIS_COMPONENTS):
    return AXIS_COMPONENTS[component]
  return str(component)


def _component_index(name: str) -> Optional[int]:
  """The inverse of `component_name`, or None if it names no component."""
  if name == 'Magnitude':
    return MAGNITUDE
  if name in AXIS_COMPONENTS:
    return AXIS_COMPONENTS.index(name)
  return int(name) if name.isdigit() else None


# -- shapes -----------------------------------------------------------------

def width(values) -> int:
  """Number of components per entry: 1 for a scalar, 3 for a 3-vector.

  `(N,)` and `(N, 1)` are both scalars -- the second is how meshio hands back
  an XDMF scalar attribute, and treating it as a vector is what hid `pressure`
  and `point_markers` from the viewer.
  """
  array = np.asarray(values)
  if array.ndim == 1:
    return 1
  if array.ndim == 2:
    return int(array.shape[1])
  raise ValueError(
    f'fields.width: expected a 1-D or 2-D array, got shape {array.shape}')


def is_vector(values) -> bool:
  """Whether a field can be drawn as an arrow, i.e. has three components."""
  try:
    return width(values) == 3
  except ValueError:
    return False


def as_columns(values) -> np.ndarray:
  """The field as `(N, components)`, whatever shape it arrived in."""
  array = np.asarray(values)
  return array.reshape(len(array), -1) if array.ndim == 1 else array


# -- enumeration ------------------------------------------------------------

def field_refs(point_data: Optional[Dict] = None,
               cell_data: Optional[Dict] = None) -> List[FieldRef]:
  """Every selectable field, scalars by name and vectors by component.

  Order is: point fields then cell fields, each alphabetical, and within a
  vector the magnitude first -- which is the entry a user almost always wants
  and the only one whose meaning does not depend on the frame of reference.
  """
  refs: List[FieldRef] = []
  for association, data in (('point', point_data), ('cell', cell_data)):
    for name in sorted(data or {}):
      values = _concat(data[name])
      try:
        n = width(values)
      except ValueError:
        continue                    # a tensor or something stranger; skip it
      if n <= 1:
        refs.append(FieldRef(name, association))
      else:
        refs.append(FieldRef(name, association, MAGNITUDE))
        refs.extend(FieldRef(name, association, i) for i in range(n))
  return refs


def field_labels(point_data: Optional[Dict] = None,
                 cell_data: Optional[Dict] = None) -> List[str]:
  """`field_refs` as the strings a combobox shows."""
  return [ref.label for ref in field_refs(point_data, cell_data)]


def vector_names(data: Optional[Dict] = None) -> List[str]:
  """Names of the three-component fields, which are what warp and glyph take."""
  return sorted(name for name, values in (data or {}).items()
                if is_vector(_concat(values)))


# -- resolution -------------------------------------------------------------

def resolve(label, point_data: Optional[Dict] = None,
            cell_data: Optional[Dict] = None) -> Optional[Tuple[np.ndarray, str]]:
  """`(values, association)` for a chosen label, or None if it is not there.

  None rather than an exception: a frame of a time series need not carry every
  field, and a viewer that raises on the frame slider is worse than one that
  draws the mesh plain.
  """
  if not label:
    return None
  ref = label if isinstance(label, FieldRef) else FieldRef.parse(
    str(label), tuple(point_data or ()), tuple(cell_data or ()))
  data = point_data if ref.association == 'point' else cell_data
  if not data or ref.name not in data:
    return None

  columns = as_columns(_concat(data[ref.name]))
  if ref.component is None or columns.shape[1] == 1:
    return columns[:, 0], ref.association
  if ref.component == MAGNITUDE:
    return np.linalg.norm(columns, axis=1), ref.association
  if 0 <= ref.component < columns.shape[1]:
    return columns[:, ref.component], ref.association
  return None                     # the field narrowed between frames


def resolve_vector(name, data: Optional[Dict] = None) -> Optional[np.ndarray]:
  """An `(N, 3)` field by name, for warping or glyphing, or None."""
  if not name or not data or name not in data:
    return None
  values = _concat(data[name])
  return np.asarray(values, dtype=float) if is_vector(values) else None


def _concat(values) -> np.ndarray:
  """meshio hands cell data back as one array PER CELL BLOCK.

  Concatenating them in block order gives one value per global element, which
  is the same ordering `element_centroids` and `surface_triangles` index by,
  so a cell field lines up with the surface without a second convention.
  """
  if isinstance(values, (list, tuple)):
    if not values:
      return np.empty(0)
    return np.concatenate([np.asarray(v) for v in values], axis=0)
  return np.asarray(values)


# -- glyphs -----------------------------------------------------------------

#: How many arrows a glyph overlay draws before it starts subsampling. One
#: arrow per node is not a choice on these meshes: `heart_P2_tetra` has 191 576
#: of them, and the picture is solid colour long before the frame rate matters.
MAX_GLYPHS = 3000

#: The longest arrow, as a fraction of the mesh diagonal, at scale 1. Sets the
#: automatic factor so the default shows something on a field of any
#: magnitude -- a velocity in m/s and a displacement in m differ by orders of
#: magnitude, and a fixed factor draws either a hairline or a thicket.
GLYPH_FRACTION = 0.06


def glyph_stride(n_points: int, max_glyphs: int = MAX_GLYPHS) -> int:
  """Take every `stride`-th point, so at most `max_glyphs` arrows are drawn.

  Deterministic rather than random, which is what ParaView's "Every Nth Point"
  masking does: the same mesh gives the same arrows on every redraw, so a
  frame slider does not make the field shimmer.
  """
  if max_glyphs < 1:
    raise ValueError(f'glyph_stride: max_glyphs must be >= 1, got {max_glyphs}')
  return max(1, int(np.ceil(n_points / float(max_glyphs))))


def glyph_sample(points: np.ndarray, vectors: np.ndarray,
                 max_glyphs: int = MAX_GLYPHS
                 ) -> Tuple[np.ndarray, np.ndarray]:
  """Thin a point cloud and its vectors down to a drawable number of arrows."""
  points = np.asarray(points, dtype=float)
  vectors = np.asarray(vectors, dtype=float)
  if len(points) != len(vectors):
    raise ValueError(
      f'glyph_sample: {len(points)} points against {len(vectors)} vectors')
  stride = glyph_stride(len(points), max_glyphs)
  return points[::stride], vectors[::stride]


def auto_glyph_factor(vectors: np.ndarray, extent: float,
                      fraction: float = GLYPH_FRACTION) -> float:
  """A scale factor putting the LONGEST arrow at `fraction` of `extent`.

  Returned rather than applied, and multiplied by the user's own unitless
  scale at the call site, so the slider means the same thing whatever field is
  chosen. A degenerate field -- all zeros, or a mesh with no extent -- gives
  1.0, which draws nothing rather than raising inside a redraw.
  """
  vectors = np.asarray(vectors, dtype=float)
  if vectors.size == 0 or not np.isfinite(extent) or extent <= 0:
    return 1.0
  longest = float(np.max(np.linalg.norm(vectors, axis=1)))
  if not np.isfinite(longest) or longest <= 0:
    return 1.0
  return fraction * float(extent) / longest
