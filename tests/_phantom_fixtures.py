"""Shared analytical-phantom mesh builders for the FEelMRI test suite.

All helpers write a `.vtu` file via meshio so they can be ingested by
:class:`feelmri.Phantom.FEMPhantom`. They are deliberately tiny
(2-50 nodes) so a full test that constructs a phantom and runs the
C++ kernel stays well under one second.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def make_minimal_tet_mesh(path: Path, scale: float = 1e-2) -> Path:
  """2-tetrahedron, 5-node mesh used by the Pulseq dual-path tests and
  by the closed-form Bloch / signal-assembly tests."""
  import meshio
  points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
  ], dtype=np.float64) * scale
  cells = [('tetra', np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64))]
  meshio.write_points_cells(str(path), points, cells)
  return path


def make_1d_rod_mesh(path: Path, length: float = 1e-2,
                     n_segments: int = 8,
                     transverse_width: float = 1e-4) -> Tuple[Path, float]:
  """A pseudo-1D rod along x, triangulated into `2*n_segments`
  tetrahedra. Each prismatic slab spans `length / n_segments` along
  x and `transverse_width` in y and z, so the integrated volume is
  ``length * transverse_width**2`` and the mass-along-x integral is
  approximately ``length * transverse_width**2`` (the integrand is
  constant if Mxy is constant per node).

  Returns the file path and the total integrated volume."""
  import meshio
  n = int(n_segments)
  xs = np.linspace(-0.5 * length, 0.5 * length, n + 1, dtype=np.float64)
  half_w = 0.5 * transverse_width
  # Each x-slab has 4 corner nodes at +-half_w in y and z.
  # The slab is decomposed into two tetrahedra sharing the diagonal.
  pts = np.zeros((4 * (n + 1), 3), dtype=np.float64)
  for i, x in enumerate(xs):
    base = 4 * i
    pts[base + 0] = [x, -half_w, -half_w]
    pts[base + 1] = [x, +half_w, -half_w]
    pts[base + 2] = [x, +half_w, +half_w]
    pts[base + 3] = [x, -half_w, +half_w]
  cells = []
  for i in range(n):
    a = 4 * i
    b = 4 * (i + 1)
    # Three tetrahedra per prismatic slab (the standard 3-tet split).
    cells.append([a + 0, a + 1, a + 2, b + 0])
    cells.append([a + 2, a + 3, a + 0, b + 0])
    cells.append([b + 0, b + 1, b + 2, a + 2])
  cells_arr = np.asarray(cells, dtype=np.int64)
  meshio.write_points_cells(str(path), pts, [('tetra', cells_arr)])
  volume = float(length * transverse_width * transverse_width)
  return path, volume


def make_2d_disk_mesh(path: Path, radius: float = 5e-3,
                      n_radial: int = 4, n_angular: int = 12,
                      thickness: float = 1e-4) -> Tuple[Path, float]:
  """Disk of radius ``radius`` in the xy plane, extruded into a thin
  slab of thickness ``thickness``. Triangulated radially (`n_radial`
  rings) and angularly (`n_angular` sectors), with each prismatic
  cell split into three tetrahedra.

  Returns the file path and the disk's nominal volume
  ``pi * radius^2 * thickness``."""
  import meshio
  rs = np.linspace(0.0, radius, int(n_radial) + 1, dtype=np.float64)
  thetas = np.linspace(0.0, 2.0 * np.pi, int(n_angular) + 1)[:-1]

  bottom_nodes = []
  top_nodes = []
  # Centre nodes shared at bottom and top.
  bottom_nodes.append([0.0, 0.0, -0.5 * thickness])
  top_nodes.append([0.0, 0.0, +0.5 * thickness])
  for r in rs[1:]:
    for th in thetas:
      bottom_nodes.append([r * np.cos(th), r * np.sin(th), -0.5 * thickness])
      top_nodes.append([r * np.cos(th),    r * np.sin(th), +0.5 * thickness])
  pts = np.array(bottom_nodes + top_nodes, dtype=np.float64)
  half = len(bottom_nodes)

  def bot(ring, angular):
    if ring == 0:
      return 0
    return 1 + (ring - 1) * len(thetas) + (angular % len(thetas))

  def top(ring, angular):
    return half + bot(ring, angular)

  cells = []
  for ring in range(int(n_radial)):
    for a in range(len(thetas)):
      v0b = bot(ring,     a)
      v1b = bot(ring + 1, a)
      v2b = bot(ring + 1, a + 1)
      v3b = bot(ring,     a + 1)
      v0t = top(ring,     a)
      v1t = top(ring + 1, a)
      v2t = top(ring + 1, a + 1)
      v3t = top(ring,     a + 1)
      # Standard prism->3-tet split of the (v0..v3) wedge.
      cells.append([v0b, v1b, v2b, v0t])
      cells.append([v0b, v2b, v3b, v0t])
      cells.append([v0t, v1t, v2t, v2b])
  cells_arr = np.asarray(cells, dtype=np.int64)
  meshio.write_points_cells(str(path), pts, [('tetra', cells_arr)])
  volume = float(np.pi * radius * radius * thickness)
  return path, volume


def make_cube_mesh(path: Path, cell_type: str, n: int = 2,
                   scale: float = 2e-3) -> Tuple[Path, float]:
  """Structured mesh of a cube, in any of the 3-D cell types the assembler
  supports (``tetra``, ``tetra10``, ``hexahedron``, ``wedge``).

  All four discretise the *same* domain with straight edges, so any
  volume integral over them must agree. That makes them a direct check
  on the meshio -> Basix DOF ordering, which differs per cell type.

  Returns the file path and the exact cube volume ``(n * scale) ** 3``.
  """
  import meshio
  c = np.arange(n + 1, dtype=np.float64) * scale
  pts = [[x, y, z] for z in c for y in c for x in c]

  def idx(i, j, k):
    return i + (n + 1) * (j + (n + 1) * k)

  # VTK hexahedron: bottom face walked cyclically, then the top face.
  hexes = [[idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
            idx(i, j, k + 1), idx(i + 1, j, k + 1),
            idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)]
           for k in range(n) for j in range(n) for i in range(n)]
  volume = float((n * scale) ** 3)

  if cell_type == 'hexahedron':
    cells = hexes
  elif cell_type == 'wedge':
    # Cut each cube along the bottom-face diagonal into two prisms.
    cells = []
    for h in hexes:
      cells.append([h[0], h[1], h[2], h[4], h[5], h[6]])
      cells.append([h[0], h[2], h[3], h[4], h[6], h[7]])
  else:
    # Six-tetrahedron decomposition of each cube.
    tets = []
    for h in hexes:
      for a, b, cc, d in ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
                          (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)):
        tets.append([h[a], h[b], h[cc], h[d]])
    if cell_type == 'tetra':
      cells = tets
    elif cell_type == 'tetra10':
      # Append edge midpoints in VTK edge order (0,1)(1,2)(0,2)(0,3)(1,3)(2,3).
      midpoints = {}

      def mid(u, v):
        key = (min(u, v), max(u, v))
        if key not in midpoints:
          midpoints[key] = len(pts)
          pts.append(list(0.5 * (np.asarray(pts[u]) + np.asarray(pts[v]))))
        return midpoints[key]

      cells = [list(t) + [mid(t[0], t[1]), mid(t[1], t[2]), mid(t[0], t[2]),
                          mid(t[0], t[3]), mid(t[1], t[3]), mid(t[2], t[3])]
               for t in tets]
    else:
      raise ValueError(f'unsupported cell type: {cell_type}')

  meshio.write_points_cells(str(path), np.asarray(pts, dtype=np.float64),
                            [(cell_type, np.asarray(cells, dtype=np.int64))])
  return path, volume
