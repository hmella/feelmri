"""Mesh geometry for the viewer, with no display and no VTK.

Everything here works on plain numpy arrays so it can be tested headless and
reused from a script. The VTK-backed viewer consumes the same functions; it
does not carry a second copy of the geometry.

Two things are deliberately duplicated from the library rather than imported.
The loader mirrors `FEMPhantom._prepare_reader`, because the shipped phantoms
need both paths: `water_fat_P1_prism.xdmf` is a time series that plain
`meshio.read` refuses, and `abdomen_P1_tetra.xdmf` is a static grid that
`meshio.xdmf.TimeSeriesReader` refuses. And `surface_triangles` re-derives the
boundary in numpy so the no-VTK fallback still has something to draw.

This module itself imports only numpy and meshio, and it constructs no
`FEMPhantom`, so opening a mesh runs no partitioning and needs no compiled
extension. Importing it through the `feelmri` package still pulls in whatever
`feelmri/__init__` pulls in, mpi4py included; import it by path if that matters.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Corner-node faces per meshio cell type. Higher-order types reuse their
# corner block: the extra nodes sit on edges and add no new faces, so the
# boundary is the same set. Winding is not normalised here; `surface_triangles`
# sorts indices to pair a face with its neighbour, and the renderer computes
# its own normals.
_FACES: Dict[str, Tuple[Tuple[int, ...], ...]] = {
  'tetra':       ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
  'tetra10':     ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
  'hexahedron':  ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                  (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
  'hexahedron20': ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                   (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
  'wedge':       ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
  'pyramid':     ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
  'triangle':    ((0, 1, 2),),
  'triangle6':   ((0, 1, 2),),
  'quad':        ((0, 1, 2, 3),),
}

# Cell types that are already a surface: every face is a boundary face, so the
# shared-face test below would discard all of them.
_SURFACE_TYPES = frozenset({'triangle', 'triangle6', 'quad', 'quad8', 'line'})


def supported_cell_types() -> Tuple[str, ...]:
  """Cell types `surface_triangles` can handle, sorted."""
  return tuple(sorted(_FACES))


def load_mesh(path):
  """Read a mesh, returning `(points, cells, n_frames, reader)`.

  `cells` is a list of `(cell_type, connectivity)` pairs. `reader` is the open
  `TimeSeriesReader` when the file is a time series and `None` otherwise; hold
  on to it to read frames later.

  The two-path fallback is not defensive coding, it is required: the shipped
  phantoms include one file of each kind and neither reader accepts both.
  """
  import meshio

  try:
    reader = meshio.xdmf.TimeSeriesReader(path)
    points, cells = reader.read_points_cells()
    return np.asarray(points), _as_pairs(cells), int(reader.num_steps), reader
  except Exception:
    mesh = meshio.read(path)
    return np.asarray(mesh.points), _as_pairs(mesh.cells), 1, None


def _as_pairs(cells) -> List[Tuple[str, np.ndarray]]:
  """Normalise meshio's cell container to `[(type, connectivity), ...]`."""
  out = []
  for block in cells:
    if isinstance(block, tuple):
      out.append((block[0], np.asarray(block[1])))
    else:
      out.append((block.type, np.asarray(block.data)))
  return out


def read_frame(reader, index: int) -> Tuple[float, Dict, Dict]:
  """One frame of a time series: `(time, point_data, cell_data)`.

  Returns empty dicts when `reader` is None, so a static mesh needs no branch
  at the call site.
  """
  if reader is None:
    return 0.0, {}, {}
  t, point_data, cell_data = reader.read_data(index)
  return float(t), point_data, cell_data


def element_centroids(points: np.ndarray,
                      cells: Sequence[Tuple[str, np.ndarray]]) -> np.ndarray:
  """Centroid of every element, concatenated in cell-block order.

  The mean is over the CORNER nodes only for higher-order types: averaging the
  mid-edge nodes as well would bias the centroid towards curved faces.
  """
  chunks = []
  for cell_type, conn in cells:
    n_corner = _n_corner(cell_type, conn.shape[1])
    chunks.append(points[conn[:, :n_corner]].mean(axis=1))
  if not chunks:
    return np.empty((0, 3), dtype=points.dtype)
  return np.concatenate(chunks, axis=0)


def _n_corner(cell_type: str, n_nodes: int) -> int:
  """Number of corner nodes, which is what the face table indexes."""
  faces = _FACES.get(cell_type)
  if faces is None:
    return n_nodes
  return max(max(f) for f in faces) + 1


def surface_triangles(points: np.ndarray,
                      cells: Sequence[Tuple[str, np.ndarray]],
                      with_source: bool = False):
  """Boundary of the mesh as an `(n_tri, 3)` index array.

  A face interior to the mesh is shared by exactly two elements; a boundary
  face by one. Sorting each face's node indices makes the two spellings of a
  shared face identical, so counting occurrences separates them. Quads are
  split into two triangles AFTER the test, so the split cannot affect it.

  This is the same reduction the VTK path performs, and it is what makes large
  meshes tractable: `abdomen_P1_tetra` goes from 4 364 561 cells to roughly a
  hundred thousand triangles.

  `with_source=True` also returns, per triangle, the index of the ELEMENT it
  came from. That index counts elements across cell blocks in the order they
  are given -- the same numbering `element_centroids` concatenates in, and the
  same one `meshio`'s per-block cell data concatenates to -- so a cell field
  reaches the drawn surface as `values[source]` with no second convention.
  Without it a cell field cannot be displayed at all, which is what kept
  `water_fat_P1_prism`'s `cell_markers` off the screen.
  """
  polys: List[np.ndarray] = []
  sources: List[np.ndarray] = []
  base = 0
  for cell_type, conn in cells:
    elements = np.arange(base, base + conn.shape[0], dtype=np.int64)
    base += conn.shape[0]
    if cell_type in _SURFACE_TYPES:
      # Already a surface: every face is a boundary face.
      _collect(_fan(conn[:, :_n_corner(cell_type, conn.shape[1])], elements),
               polys, sources)
      continue
    faces = _FACES.get(cell_type)
    if faces is None:
      raise ValueError(
        f"surface_triangles: cell type {cell_type!r} is not supported; "
        f"supported types are {', '.join(supported_cell_types())}")
    by_size: Dict[int, List[np.ndarray]] = {}
    for f in faces:
      by_size.setdefault(len(f), []).append(conn[:, list(f)])
    for size, group in by_size.items():
      stacked = np.concatenate(group, axis=0)
      # `concatenate` lays the group out face-table-major, so the element each
      # row belongs to repeats once per face of that size.
      owners = np.tile(elements, len(group))
      keep = _boundary_mask(stacked)
      _collect(_fan(stacked[keep], owners[keep]), polys, sources)

  if not polys:
    empty = (np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.int64))
    return empty if with_source else empty[0]
  tris = np.concatenate(polys, axis=0)
  return (tris, np.concatenate(sources)) if with_source else tris


def _collect(chunks, polys: List[np.ndarray],
             sources: List[np.ndarray]) -> None:
  """Append `(triangles, owners)` pairs onto the two parallel lists."""
  for tris, owners in chunks:
    polys.append(tris)
    sources.append(owners)


def _boundary_mask(faces: np.ndarray) -> np.ndarray:
  """Which faces occur exactly once, i.e. lie on the boundary.

  A mask rather than the filtered faces, so the element each face belongs to
  can be filtered the same way.
  """
  keys = np.sort(faces, axis=1)
  _, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                 return_counts=True)
  return counts[inverse.ravel()] == 1


def _fan(faces: np.ndarray, elements: np.ndarray
         ) -> List[Tuple[np.ndarray, np.ndarray]]:
  """Triangulate a block of equal-sided faces by a fan from vertex 0.

  Every triangle of a face inherits that face's element, so the source array
  stays parallel to the triangles through the split.
  """
  if faces.size == 0:
    return []
  n = faces.shape[1]
  if n == 3:
    return [(faces.astype(np.int64), elements)]
  return [(np.stack([faces[:, 0], faces[:, i], faces[:, i + 1]], axis=1)
           .astype(np.int64), elements) for i in range(1, n - 1)]


def slab_markers(centroids: np.ndarray,
                 half_thickness: float,
                 axis: int = 2,
                 offset: float = 0.0) -> np.ndarray:
  """Boolean mask of the elements whose centroid lies within a slab.

  `FEMPhantom.create_submesh` takes a mask over GLOBAL ELEMENTS, not nodes,
  which is what this returns.

  This is the idiom nine examples spell out by hand, for instance
  `examples/phase_contrast.py`:

      mp = phantom.global_nodes[phantom.global_elements].mean(axis=1)
      markers = np.abs(mp[:, 2]) <= 0.5 * planning.FOV[2].m_as('m')

  Note the examples compute it AFTER `orient`, so `axis=2` is the slice axis of
  the imaging frame rather than a patient axis.
  """
  if centroids.ndim != 2 or centroids.shape[1] != 3:
    raise ValueError(
      f"slab_markers: centroids must be (n_elements, 3), got {centroids.shape}")
  if not 0 <= axis <= 2:
    raise ValueError(f"slab_markers: axis must be 0, 1 or 2, got {axis}")
  if not np.isfinite(half_thickness) or half_thickness < 0:
    raise ValueError(
      f"slab_markers: half_thickness must be finite and non-negative, "
      f"got {half_thickness}")
  return np.abs(centroids[:, axis] - offset) <= half_thickness


def box_markers(centroids: np.ndarray,
                centre: Sequence[float],
                lengths: Sequence[float],
                rotation: Optional[np.ndarray] = None) -> np.ndarray:
  """Boolean mask of the elements inside an oriented box.

  `rotation` follows the same convention as `FEMPhantom.orient`: its COLUMNS
  are the box axes expressed in the coordinates `centroids` is given in, so a
  point is tested as `R^T (x - centre)` against `lengths / 2`. Passing None
  means an axis-aligned box.
  """
  centroids = np.asarray(centroids, dtype=np.float64)
  if centroids.ndim != 2 or centroids.shape[1] != 3:
    raise ValueError(
      f"box_markers: centroids must be (n_elements, 3), got {centroids.shape}")
  centre = np.asarray(centre, dtype=np.float64).reshape(3)
  half = 0.5 * np.asarray(lengths, dtype=np.float64).reshape(3)
  if not np.all(np.isfinite(half)) or np.any(half < 0):
    raise ValueError(f"box_markers: lengths must be finite and non-negative, "
                     f"got {tuple(np.asarray(lengths).ravel())}")

  local = centroids - centre
  if rotation is not None:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
      raise ValueError(
        f"box_markers: rotation must be 3x3, got {rotation.shape}")
    local = local @ rotation      # rows: (x - c) @ R == R^T (x - c)
  return np.all(np.abs(local) <= half, axis=1)


def warp(points: np.ndarray,
         displacement: np.ndarray,
         scale: float = 1.0) -> np.ndarray:
  """`points + scale * displacement`, shape-checked.

  The displacement is what `POD.__call__(t)` returns, one 3-vector per node.
  """
  points = np.asarray(points)
  displacement = np.asarray(displacement)
  if displacement.shape != points.shape:
    raise ValueError(
      f"warp: displacement is {displacement.shape} but the mesh has "
      f"{points.shape[0]} nodes, so it must be {points.shape}")
  return points + scale * displacement


#: Plausible extent of an MRI FIELD OF VIEW, in metres -- not of a body.
#:
#: The upper bound is the load-bearing one and it is deliberately tight. With
#: 3.0 m, `abdomen_P1_tetra` (extent 158.8) reads as centimetres, giving a
#: plausible-looking 1.588 m whole body -- and the example that uses it passes
#: 0.001, i.e. 15.9 cm. At 0.6 m the centimetre reading is rejected and the
#: millimetre one is chosen, which is what the example means. A genuinely
#: body-sized mesh now matches nothing and is reported as such, which is the
#: right answer for a viewer that must not guess.
PLAUSIBLE_EXTENT_M = (0.02, 0.6)

#: The scales the shipped phantoms actually need, and what each means.
KNOWN_SCALES = ((1.0, 'metres'), (1e-2, 'centimetres'), (1e-3, 'millimetres'))


def suggest_scale_factor(points: np.ndarray) -> Tuple[float, str]:
  """`(factor, why)` -- the scale that puts this mesh at a human size.

  **The shipped phantoms are not in one unit.** `heart_P1_hex` and
  `heart_P2_tetra` are metres, `aorta_P1_tetra` and `water_fat_P1_prism` are
  centimetres, `abdomen_P1_tetra` is millimetres -- three scales among five
  files, each spelled out as a `scale_factor` in the example that uses it. A
  viewer that assumes metres draws a field of view a hundred times too small
  against the phantom and reports an empty submesh, which is what happened.

  This SUGGESTS and never applies: guessing units silently is how a plausible
  wrong answer gets published. The caller shows the number and the reason.
  """
  extent = float(np.max(points.max(axis=0) - points.min(axis=0)))
  low, high = PLAUSIBLE_EXTENT_M
  if extent <= 0:
    return 1.0, 'the mesh has no extent'
  for factor, unit in KNOWN_SCALES:
    if low <= extent * factor <= high:
      scaled = extent * factor
      return factor, (f'extent {extent:.4g} reads as {unit} '
                      f'({scaled:.4g} m across)')
  return 1.0, (f'extent {extent:.4g} matches no usual unit; '
               f'set the scale by hand')


def bounds(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """`(min, max)` corner of the axis-aligned bounding box."""
  return points.min(axis=0), points.max(axis=0)
