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
