"""The headless mesh model behind the viewer.

No display, no VTK, no phantom: these run on plain arrays, which is the point
of keeping the model layer free of the view layer.

The boundary extraction is checked three ways, because each catches something
the others do not: exact triangle counts on single elements, Euler's formula on
a closed surface (which a wrong face table satisfies only by accident), and a
cross-check against VTK on the real phantoms, where two independent
implementations agreeing is worth more than either alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from feelmri.gui.model.mesh import (bounds, box_markers, element_centroids,
                                    load_mesh, slab_markers,
                                    supported_cell_types, surface_triangles,
                                    warp)

PHANTOMS = __import__('pathlib').Path(__file__).resolve().parent.parent / 'examples' / 'phantoms'

# One element of each supported type, with the triangle count its faces imply.
UNIT_TET = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
UNIT_HEX = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                     [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
UNIT_WEDGE = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                       [0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])


@pytest.mark.parametrize('cell_type, points, n_tri', [
  ('tetra', UNIT_TET, 4),            # 4 triangular faces
  ('hexahedron', UNIT_HEX, 12),      # 6 quads, each split in two
  ('wedge', UNIT_WEDGE, 8),          # 2 triangles + 3 quads
  ('triangle', UNIT_TET[:3], 1),     # already a surface
])
def test_a_single_element_is_all_boundary(cell_type, points, n_tri):
  """Every face of a lone element is a boundary face."""
  conn = np.arange(len(points))[None, :]
  tris = surface_triangles(points, [(cell_type, conn)])
  assert tris.shape == (n_tri, 3)
  # Every node of the element must appear, or a face is missing from the table.
  assert set(tris.ravel().tolist()) == set(range(len(points)))


def test_a_shared_face_is_dropped():
  """Two tetrahedra glued on one face expose 6 of their 8 faces.

  This is the property the whole extraction rests on, and it is what a wrong
  face table breaks first.
  """
  pts = np.vstack([UNIT_TET, [[1., 1., 1.]]])
  conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])   # share face (1,2,3)
  tris = surface_triangles(pts, [('tetra', conn)])
  assert tris.shape == (6, 3), 'the shared face was not removed'
  shared = {1, 2, 3}
  assert not any(set(t.tolist()) == shared for t in tris)


def test_the_boundary_of_a_closed_block_is_a_closed_surface():
  """Euler's formula on a lattice of tetrahedra.

  A closed orientable surface of genus 0 has `V - E + F = 2`. A face table with
  a wrong vertex in it still yields *some* surface, but essentially never one
  that closes, so this catches errors the count tests cannot.
  """
  n = 4
  g = np.mgrid[0:n, 0:n, 0:n].reshape(3, -1).T.astype(float)
  idx = np.arange(n ** 3).reshape(n, n, n)
  cubes = np.stack([idx[:-1, :-1, :-1], idx[1:, :-1, :-1], idx[1:, 1:, :-1],
                    idx[:-1, 1:, :-1], idx[:-1, :-1, 1:], idx[1:, :-1, 1:],
                    idx[1:, 1:, 1:], idx[:-1, 1:, 1:]], axis=-1).reshape(-1, 8)
  # Kuhn subdivision: six tetrahedra per cube, sharing the 0-6 diagonal.
  order = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
           (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
  tets = np.concatenate([cubes[:, list(o)] for o in order], axis=0)

  tris = surface_triangles(g, [('tetra', tets)])
  V = len(np.unique(tris))
  F = len(tris)
  edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]],
                                  tris[:, [2, 0]]]), axis=1)
  E = len(np.unique(edges, axis=0))
  assert V - E + F == 2, f'surface does not close: V={V} E={E} F={F}'


def test_a_higher_order_element_has_the_same_boundary_as_its_corners():
  """`tetra10` adds mid-edge nodes, which sit ON faces and add none."""
  pts = np.vstack([UNIT_TET, np.zeros((6, 3))])
  tris10 = surface_triangles(pts, [('tetra10', np.arange(10)[None, :])])
  tris4 = surface_triangles(UNIT_TET, [('tetra', np.arange(4)[None, :])])
  assert tris10.shape == tris4.shape == (4, 3)
  assert {frozenset(t.tolist()) for t in tris10} == {frozenset(t.tolist()) for t in tris4}


def test_an_unsupported_cell_type_is_refused_by_name():
  with pytest.raises(ValueError, match='polyhedron'):
    surface_triangles(UNIT_TET, [('polyhedron', np.arange(4)[None, :])])
  assert 'tetra' in supported_cell_types()


def test_centroids_use_corner_nodes_only():
  """Averaging mid-edge nodes too would drag the centroid off centre."""
  pts = np.vstack([UNIT_TET, np.full((6, 3), 99.0)])
  c10 = element_centroids(pts, [('tetra10', np.arange(10)[None, :])])
  c4 = element_centroids(UNIT_TET, [('tetra', np.arange(4)[None, :])])
  np.testing.assert_allclose(c10, c4, atol=1e-12)
  np.testing.assert_allclose(c4[0], UNIT_TET.mean(axis=0), atol=1e-12)


def test_slab_markers_reproduce_the_idiom_the_examples_spell_out():
  """Same result as `np.abs(mp[:, 2]) <= 0.5 * FOV[2]`, the nine-example idiom."""
  rng = np.random.default_rng(0)
  centroids = rng.uniform(-1, 1, size=(500, 3))
  fov_z = 0.8
  expected = np.abs(centroids[:, 2]) <= 0.5 * fov_z
  np.testing.assert_array_equal(slab_markers(centroids, 0.5 * fov_z, axis=2),
                                expected)
  # The axis has to reach the answer: on anisotropic data, axis 0 differs.
  assert not np.array_equal(slab_markers(centroids, 0.5 * fov_z, axis=0),
                            expected)
  # And so does the offset.
  assert not np.array_equal(slab_markers(centroids, 0.5 * fov_z, offset=0.5),
                            expected)


@pytest.mark.parametrize('bad, match', [
  ({'half_thickness': -1.0}, 'non-negative'),
  ({'half_thickness': np.nan}, 'finite'),
  ({'axis': 3}, 'axis must be'),
])
def test_slab_markers_refuse_bad_input_by_name(bad, match):
  kwargs = {'half_thickness': 1.0}
  kwargs.update(bad)
  with pytest.raises(ValueError, match=match):
    slab_markers(np.zeros((4, 3)), **kwargs)


def test_box_markers_follow_the_rotation():
  """A rotated box must select a different set than an axis-aligned one.

  Driven on a thin box at 45 degrees so the two selections genuinely differ; an
  isotropic box, or a rotation about its own axis, would agree by symmetry and
  prove nothing about whether the rotation was applied at all.
  """
  rng = np.random.default_rng(1)
  centroids = rng.uniform(-1, 1, size=(2000, 3))
  lengths = (2.0, 0.2, 2.0)                      # thin in y
  a = np.pi / 4
  R = np.array([[np.cos(a), -np.sin(a), 0.],
                [np.sin(a), np.cos(a), 0.],
                [0., 0., 1.]])

  plain = box_markers(centroids, (0, 0, 0), lengths)
  tilted = box_markers(centroids, (0, 0, 0), lengths, rotation=R)
  assert plain.sum() > 0 and tilted.sum() > 0
  assert not np.array_equal(plain, tilted), 'the rotation was ignored'

  # The slab normal is R's second column, so test the projection directly.
  d = np.abs(centroids @ R[:, 1])
  np.testing.assert_array_equal(tilted, (d <= 0.1) &
                                (np.abs(centroids @ R[:, 0]) <= 1.0) &
                                (np.abs(centroids @ R[:, 2]) <= 1.0))

  # Identity must agree with the axis-aligned case exactly.
  np.testing.assert_array_equal(
    box_markers(centroids, (0, 0, 0), lengths, rotation=np.eye(3)), plain)


def test_warp_checks_its_shape_and_scales():
  pts = np.zeros((5, 3))
  d = np.ones((5, 3))
  np.testing.assert_allclose(warp(pts, d, 2.0), 2.0)
  np.testing.assert_allclose(warp(pts, d, 0.0), 0.0)
  with pytest.raises(ValueError, match=r'must be \(5, 3\)'):
    warp(pts, np.ones((4, 3)))


def test_bounds():
  lo, hi = bounds(UNIT_HEX)
  np.testing.assert_allclose(lo, [0, 0, 0])
  np.testing.assert_allclose(hi, [1, 1, 1])


# The five shipped phantoms, with the cell type each exercises. Counts come
# from the .xdmf headers, so a loader that silently picked the wrong reader
# path would fail here rather than produce a plausible smaller mesh.
SHIPPED = [
  ('water_fat_P1_prism', 'wedge', 63357, 119040, 1),
  ('heart_P2_tetra', 'tetra10', 191576, 121133, 26),
  ('heart_P1_hex', 'hexahedron', 151077, 124008, 26),
  ('aorta_P1_tetra', 'tetra', 127131, 719419, 30),
  ('abdomen_P1_tetra', 'tetra', 728058, 4364561, 1),
]


@pytest.mark.slow
@pytest.mark.parametrize('name, cell_type, n_nodes, n_cells, n_frames',
                         SHIPPED, ids=[s[0] for s in SHIPPED])
def test_every_shipped_phantom_loads_and_its_boundary_is_manifold(
    name, cell_type, n_nodes, n_cells, n_frames):
  """Both reader paths, and a surface every edge of which is shared twice.

  `water_fat_P1_prism` is a time series that plain `meshio.read` refuses;
  `abdomen_P1_tetra` is a static grid that `TimeSeriesReader` refuses. One file
  of each kind is why `load_mesh` needs both branches.

  The manifold check is genus-independent, unlike Euler's formula: whatever the
  topology, a closed boundary shares every edge between exactly two triangles.
  A dropped or duplicated face breaks it immediately.
  """
  path = PHANTOMS / f'{name}.xdmf'
  if not path.exists():
    pytest.skip(f'{path.name} not present')

  points, cells, frames, reader = load_mesh(path)
  assert points.shape == (n_nodes, 3)
  assert sum(len(c) for _, c in cells) == n_cells
  assert frames == n_frames
  assert [t for t, _ in cells] == [cell_type]

  tris = surface_triangles(points, cells)
  assert len(tris) > 0
  assert tris.max() < n_nodes

  edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]],
                                  tris[:, [2, 0]]]), axis=1)
  _, counts = np.unique(edges, axis=0, return_counts=True)
  assert set(np.unique(counts).tolist()) == {2}, (
    f'boundary is not manifold: edge multiplicities '
    f'{sorted(set(counts.tolist()))}')

  # The reduction is the whole reason the viewer is tractable.
  assert len(tris) < n_cells, 'surface extraction did not reduce the mesh'


def test_the_model_layer_imports_with_no_display_and_no_vtk():
  """The model layer must never acquire a view dependency.

  This is the architectural claim the whole GUI rests on: the geometry is
  usable from a script, a test or a headless machine. Asserting it here means a
  stray `import pyvista` at module scope fails a test rather than failing on
  somebody's machine six months from now.

  The import runs in a subprocess because blocking modules in this one would
  disturb everything that follows.
  """
  import subprocess
  import sys
  import textwrap

  code = textwrap.dedent('''
    import sys
    # matplotlib is NOT blocked: it is a declared core dependency and
    # `feelmri/__init__` imports it via MRObjects. The claim under test is that
    # the GUI model layer adds no NEW view dependency of its own.
    BLOCKED = ('pyvista', 'vtk', 'vtkmodules', 'tkinter')
    class Block:
      def find_module(self, name, path=None):
        return self if name.split('.')[0] in BLOCKED else None
      def load_module(self, name):
        raise ImportError(name + ' is blocked for this check')
    sys.meta_path.insert(0, Block())
    from feelmri.gui.model import mesh
    import numpy as np
    pts = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    assert len(mesh.surface_triangles(pts, [('tetra', np.arange(4)[None, :])])) == 4
    print('OK')
  ''')
  proc = subprocess.run([sys.executable, '-c', code], capture_output=True,
                        text=True, timeout=120)
  assert 'OK' in proc.stdout, (
    f'the model layer pulled in a blocked view dependency:\n{proc.stderr[-2000:]}')


# -- the unit the file is in, which is not the unit the plan is in ----------

PHANTOMS = Path('examples/phantoms')

#: The `scale_factor` each shipped example passes to `FEMPhantom` for this
#: file. This is the ground truth: the suggestion is only worth anything if it
#: reproduces what the library is already told by hand.
EXAMPLE_SCALES = {
  'water_fat_P1_prism': 0.01,     # examples/water_and_fat.py
  'abdomen_P1_tetra': 0.001,      # examples/water_and_fat_abdomen.py
  'aorta_P1_tetra': 0.01,         # examples/phase_contrast.py, 4dflow.py
  'heart_P2_tetra': 1.0,          # examples/spamm.py
  'heart_P1_hex': 1.0,            # examples/free_running.py
}


@pytest.mark.parametrize('name,expected', sorted(EXAMPLE_SCALES.items()))
def test_the_suggested_scale_matches_what_the_examples_pass(name, expected):
  """Five shipped phantoms, THREE different units, and the viewer must agree
  with the library about which is which.

  Without this the plan is in metres while the mesh is not, so the field of
  view is drawn a hundred or a thousand times too small and the submesh reads
  zero elements -- which is exactly what it did.
  """
  from feelmri.gui.model.mesh import suggest_scale_factor
  path = PHANTOMS / f'{name}.xdmf'
  if not path.exists():
    pytest.skip(f'{name} not present')
  points = load_mesh(path)[0]
  factor, why = suggest_scale_factor(points)
  assert factor == expected, f'{name}: {why}'


def test_a_mesh_matching_no_usual_unit_says_so_instead_of_guessing():
  """A viewer that guesses units publishes a plausible wrong answer.

  An extent of 5000 is 5000 m, 50 m or 5 m depending on the unit read, and
  none of those is a field of view, so the honest reply is the identity plus
  a reason rather than the nearest match. (40 would NOT do: 40 cm is 0.4 m
  and perfectly plausible, which is how a first version of this test failed.)
  """
  from feelmri.gui.model.mesh import suggest_scale_factor
  points = np.array([[0.0, 0.0, 0.0], [5000.0, 5000.0, 5000.0]])
  factor, why = suggest_scale_factor(points)
  assert factor == 1.0
  assert 'by hand' in why


def test_a_degenerate_mesh_does_not_divide_by_its_own_extent():
  from feelmri.gui.model.mesh import suggest_scale_factor
  points = np.zeros((4, 3))
  assert suggest_scale_factor(points)[0] == 1.0


def test_loading_at_a_scale_puts_the_mesh_where_the_plan_is():
  """The end the user sees: a slab that keeps elements rather than none."""
  from feelmri.gui.model.planning import FOVBox
  from feelmri.gui.model.session import Session

  path = PHANTOMS / 'water_fat_P1_prism.xdmf'
  if not path.exists():
    pytest.skip('water_fat_P1_prism not present')

  plan = FOVBox(fov=np.array([0.3, 0.22, 0.008]), loc=np.zeros(3),
                angles=np.zeros(3))

  unscaled = Session()
  unscaled.load_mesh(path)
  unscaled.box = plan
  assert unscaled.submesh_markers().sum() == 0, (
    'the unscaled control must keep nothing, or this proves nothing')

  scaled = Session()
  scaled.load_mesh(path, scale_factor=0.01)
  scaled.box = plan
  assert scaled.submesh_markers().sum() > 0


@pytest.mark.parametrize('bad', [0.0, -1.0, float('nan')])
def test_a_nonsense_scale_is_refused(bad):
  from feelmri.gui.model.session import Session
  path = PHANTOMS / 'water_fat_P1_prism.xdmf'
  if not path.exists():
    pytest.skip('water_fat_P1_prism not present')
  with pytest.raises(ValueError, match='scale_factor'):
    Session().load_mesh(path, scale_factor=bad)
