"""Which stored fields the viewer offers, and what a chosen one resolves to.

The defect these exist for is not a crash. **The `Colour by` list was empty on
every shipped phantom**, because the shell filtered point data to `ndim == 1`:
that excludes every vector by design, and it also excludes `pressure` and
`point_markers`, which are genuine scalars stored as `(N, 1)`. `cell_markers`
is cell data, which was never read at all. So the first test below is a direct
regression on what the five phantoms offer.

Everything here is headless: plain dictionaries of numpy arrays, plus the
shipped meshes for the cell-data path, which needs a real element-to-triangle
map to mean anything.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from feelmri.gui.model import fields as F
from feelmri.gui.model.mesh import load_mesh, read_frame, surface_triangles
from feelmri.gui.model.session import Session

PHANTOMS = Path(__file__).resolve().parent.parent / 'examples' / 'phantoms'

#: A vector whose three components and magnitude are all different, so a test
#: that reads the wrong column cannot pass by coincidence. The 3-4-5 triple
#: makes the magnitude exact in floating point.
VECTOR = np.array([[3.0, -4.0, 12.0],
                   [6.0, 8.0, 0.0],
                   [-1.0, 2.0, -2.0]])


def phantom(name):
  path = PHANTOMS / f'{name}.xdmf'
  if not path.exists():
    pytest.skip(f'{path.name} not present')
  return path


# -- what is offered --------------------------------------------------------

@pytest.mark.parametrize('name, expected', [
  ('heart_P2_tetra', ['displacement (Magnitude)', 'displacement (X)',
                      'displacement (Y)', 'displacement (Z)']),
  ('aorta_P1_tetra', ['pressure', 'velocity (Magnitude)', 'velocity (X)',
                      'velocity (Y)', 'velocity (Z)']),
  ('water_fat_P1_prism', ['point_markers', 'cell_markers [cell]']),
])
def test_every_stored_field_is_offered(name, expected):
  """The regression. Measured before this: all three lists were EMPTY.

  Both reasons are represented -- `displacement` and `velocity` are vectors,
  which were excluded by design; `pressure` and `point_markers` are scalars
  shaped `(N, 1)`, which `ndim == 1` excluded by accident; and `cell_markers`
  lives on elements, which nothing read.
  """
  _, cells, _, reader = load_mesh(phantom(name))
  _, point_data, cell_data = read_frame(reader, 0)
  assert F.field_labels(point_data, cell_data) == expected


def test_a_vector_is_offered_as_magnitude_first_then_components():
  """Magnitude leads because it is the entry whose meaning does not depend on
  which way the patient is lying."""
  labels = F.field_labels({'v': VECTOR})
  assert labels == ['v (Magnitude)', 'v (X)', 'v (Y)', 'v (Z)']


def test_a_scalar_is_offered_under_its_bare_name():
  """No component suffix, in either stored shape."""
  assert F.field_labels({'p': np.zeros(4)}) == ['p']
  assert F.field_labels({'p': np.zeros((4, 1))}) == ['p']


def test_point_and_cell_fields_of_the_same_name_stay_distinct():
  """They are different arrays of different lengths, and a label that could
  not tell them apart would draw one while naming the other."""
  labels = F.field_labels({'marker': np.zeros(4)}, {'marker': np.zeros(9)})
  assert labels == ['marker', 'marker [cell]']
  point, _ = F.resolve('marker', {'marker': np.zeros(4)},
                       {'marker': np.zeros(9)})
  cell, association = F.resolve('marker [cell]', {'marker': np.zeros(4)},
                                {'marker': np.zeros(9)})
  assert len(point) == 4
  assert len(cell) == 9 and association == 'cell'


# -- reading a label back ---------------------------------------------------

@pytest.mark.parametrize('ref', [
  F.FieldRef('p'),
  F.FieldRef('v', 'point', F.MAGNITUDE),
  F.FieldRef('v', 'point', 2),
  F.FieldRef('m', 'cell'),
  F.FieldRef('s', 'cell', F.MAGNITUDE),
  F.FieldRef('t', 'point', 5),
])
def test_a_label_round_trips(ref):
  assert F.FieldRef.parse(ref.label) == ref


def test_an_exact_name_beats_the_suffix_grammar():
  """A field may legitimately be CALLED `pressure (X)`.

  Stripping the suffix from it would silently show a different array, which is
  the quiet kind of wrong this whole module exists to avoid. The known names
  are consulted first.
  """
  data = {'pressure (X)': np.array([1.0, 2.0]), 'pressure': VECTOR[:2]}
  ref = F.FieldRef.parse('pressure (X)', tuple(data))
  assert ref == F.FieldRef('pressure (X)', 'point', None)
  values, _ = F.resolve('pressure (X)', data)
  assert np.array_equal(values, [1.0, 2.0]), 'the component was taken instead'


def test_a_field_that_is_not_in_this_frame_reads_as_absent():
  """None, not an exception: a frame of a time series need not carry every
  field, and raising from the frame slider is worse than drawing plain."""
  assert F.resolve('missing', {'p': np.zeros(3)}) is None
  assert F.resolve('', {'p': np.zeros(3)}) is None
  assert F.resolve('v (Z)', {'v': np.zeros((3, 2))}) is None


# -- what a label resolves to ----------------------------------------------

@pytest.mark.parametrize('label, expected', [
  ('v (X)', VECTOR[:, 0]),
  ('v (Y)', VECTOR[:, 1]),
  ('v (Z)', VECTOR[:, 2]),
  ('v (Magnitude)', np.array([13.0, 10.0, 3.0])),
])
def test_each_component_resolves_to_its_own_column(label, expected):
  """Every column and the magnitude are different numbers here, so reading the
  wrong one fails rather than agreeing by symmetry."""
  values, association = F.resolve(label, {'v': VECTOR})
  assert association == 'point'
  assert np.allclose(values, expected)


def test_the_magnitude_is_the_norm_and_never_negative():
  values, _ = F.resolve('v (Magnitude)', {'v': VECTOR})
  assert np.allclose(values, np.linalg.norm(VECTOR, axis=1))
  assert (values >= 0).all()


def test_an_n_by_one_scalar_comes_back_flat():
  """`(N, 1)` is how meshio hands back an XDMF scalar attribute; a viewer that
  gets a column vector instead paints one value per node wrongly or not at
  all."""
  values, _ = F.resolve('p', {'p': np.array([[1.0], [2.0], [3.0]])})
  assert values.shape == (3,)
  assert np.array_equal(values, [1.0, 2.0, 3.0])


def test_cell_data_arrives_as_one_array_per_block_and_is_concatenated():
  """meshio splits cell data by cell block. The concatenation has to be in
  block order, because that is the order `surface_triangles` numbers elements
  in and the two must agree."""
  values, association = F.resolve(
    'm [cell]', {}, {'m': [np.array([1.0, 2.0]), np.array([3.0])]})
  assert association == 'cell'
  assert np.array_equal(values, [1.0, 2.0, 3.0])


def test_only_three_component_fields_can_be_warped_or_glyphed():
  data = {'v': VECTOR, 'p': np.zeros((3, 1)), 'flat': np.zeros(3),
          'wide': np.zeros((3, 6))}
  assert F.vector_names(data) == ['v']
  assert F.resolve_vector('v', data).shape == (3, 3)
  assert F.resolve_vector('p', data) is None
  assert F.resolve_vector('missing', data) is None


# -- the triangle-to-element map, which is what makes cell data drawable ----

def test_tracking_the_source_element_does_not_change_the_triangles():
  """The map is added alongside, so a mesh drawn before and after this is the
  same mesh."""
  points, cells, _, _ = load_mesh(phantom('water_fat_P1_prism'))
  plain = surface_triangles(points, cells)
  tris, source = surface_triangles(points, cells, with_source=True)
  assert np.array_equal(plain, tris)
  assert source.shape == (len(tris),)


@pytest.mark.parametrize('name', ['water_fat_P1_prism', 'heart_P1_hex',
                                  'aorta_P1_tetra'])
def test_every_triangle_is_a_face_of_the_element_it_claims(name):
  """The map is only worth anything if it is RIGHT, and a plausible-looking
  index array is exactly what a wrong one looks like.

  Three cell types, because the face tables differ: a wedge mixes triangles
  and quads, a hexahedron is six quads each split in two, and a tetrahedron is
  four triangles. The quad split is where a source array most easily slips.
  """
  points, cells, _, _ = load_mesh(phantom(name))
  assert len(cells) == 1, 'this check assumes a single cell block'
  conn = cells[0][1]
  tris, source = surface_triangles(points, cells, with_source=True)

  rng = np.random.default_rng(0)
  sample = rng.choice(len(tris), size=min(2000, len(tris)), replace=False)
  for i in sample:
    assert set(tris[i].tolist()) <= set(conn[source[i]].tolist()), (
      f'triangle {i} is not a face of element {source[i]}')


def test_a_cell_field_reaches_the_drawn_surface():
  """`cell_markers` is the water/fat phantom's tissue map, and it could not be
  displayed at all before: the surface knew nothing about which element each
  triangle came from."""
  session = Session()
  session.load_mesh(phantom('water_fat_P1_prism'), scale_factor=0.01)
  session.field = 'cell_markers [cell]'

  values, association = session.surface_colour_values()
  assert association == 'cell'
  assert values.shape == (len(session.surface),)

  # Every tissue label survives onto the surface. A map that collapsed to one
  # value would still be an array of the right length.
  _, _, cell_data = session.read_frame(0)
  whole = F.resolve('cell_markers [cell]', {}, cell_data)[0]
  assert len(np.unique(values)) > 1
  assert set(np.unique(values)) <= set(np.unique(whole))
  assert np.array_equal(values, whole[session.surface_source])


def test_a_point_field_is_passed_through_untouched():
  """One value per NODE, not per triangle: the surface indexes the mesh's own
  points, so the array must not be remapped."""
  session = Session()
  session.load_mesh(phantom('water_fat_P1_prism'), scale_factor=0.01)
  session.field = 'point_markers'
  values, association = session.surface_colour_values()
  assert association == 'point'
  assert values.shape == (len(session.points),)


# -- glyphs -----------------------------------------------------------------

def test_the_arrow_count_is_capped_however_large_the_mesh():
  """One arrow per node is not a choice: `heart_P2_tetra` has 191 576, and the
  picture is solid colour long before the frame rate matters."""
  points = np.zeros((191576, 3))
  vectors = np.ones((191576, 3))
  sampled, _ = F.glyph_sample(points, vectors, max_glyphs=3000)
  assert len(sampled) <= 3000


def test_sampling_is_deterministic_and_keeps_points_with_their_vectors():
  """Deterministic, or the arrows shimmer as the frame slider moves. Paired,
  or every arrow is drawn at someone else's position.
  """
  points = np.arange(300, dtype=float).reshape(100, 3)
  vectors = points * 2.0
  a_points, a_vectors = F.glyph_sample(points, vectors, max_glyphs=10)
  b_points, b_vectors = F.glyph_sample(points, vectors, max_glyphs=10)
  assert np.array_equal(a_points, b_points)
  assert np.allclose(a_vectors, 2.0 * a_points), 'a vector lost its point'
  assert np.array_equal(a_vectors, b_vectors)


def test_a_mismatched_pair_is_refused():
  with pytest.raises(ValueError, match='points against'):
    F.glyph_sample(np.zeros((5, 3)), np.zeros((4, 3)))


def test_the_automatic_scale_puts_the_longest_arrow_at_a_fixed_fraction():
  """The whole reason the factor is automatic: the shipped fields differ by
  orders of magnitude -- a displacement in metres against a velocity in
  centimetres per second -- and one fixed factor draws either a hairline or a
  thicket."""
  for magnitude in (1e-4, 1.0, 250.0):
    vectors = np.array([[magnitude, 0.0, 0.0], [0.0, 0.5 * magnitude, 0.0]])
    factor = F.auto_glyph_factor(vectors, extent=0.2, fraction=0.06)
    longest = factor * float(np.max(np.linalg.norm(vectors, axis=1)))
    assert longest == pytest.approx(0.06 * 0.2)


@pytest.mark.parametrize('vectors, extent', [
  (np.zeros((4, 3)), 0.2),            # a field that is identically zero
  (np.ones((4, 3)), 0.0),             # a mesh with no extent
  (np.empty((0, 3)), 0.2),            # nothing to draw
])
def test_a_degenerate_glyph_field_gives_a_harmless_factor(vectors, extent):
  """It runs inside a redraw, so it returns a number rather than raising."""
  assert F.auto_glyph_factor(vectors, extent) == 1.0


def test_the_glyph_arrows_follow_the_warp():
  """Arrows belong on the mesh as DRAWN. Left on the reference configuration
  they float away from it exactly when the motion is most interesting."""
  session = Session()
  session.load_mesh(phantom('heart_P1_hex'))
  session.frame = 12
  session.glyph_field = 'displacement'
  at_rest = session.glyph_arrows()[0]

  session.warp_field = 'displacement'
  session.warp_scale = 1.0
  warped = session.glyph_arrows()[0]
  assert not np.allclose(at_rest, warped), 'the arrows ignored the warp'


# -- the display state ------------------------------------------------------

@pytest.mark.parametrize('attribute', ['opacity', 'plan_opacity'])
@pytest.mark.parametrize('bad', [-0.1, 1.5, float('nan')])
def test_an_opacity_outside_zero_to_one_is_refused(attribute, bad):
  with pytest.raises(ValueError, match=attribute):
    setattr(Session(), attribute, bad)


def test_the_plan_opacity_notifies_the_plan_rather_than_the_view():
  """It changes the OVERLAY, and the overlay is what `plan_changed` redraws.
  Sending it to `view_changed` would rebuild the whole mesh for a tint."""
  session = Session()
  seen = []
  session.plan_changed.connect(lambda *_: seen.append('plan'))
  session.view_changed.connect(lambda *_: seen.append('view'))
  session.plan_opacity = 0.5
  assert seen == ['plan']


def test_a_glyph_count_below_one_is_refused():
  with pytest.raises(ValueError, match='glyph_count'):
    Session().glyph_count = 0


def test_setting_a_display_value_to_what_it_already_is_notifies_nobody():
  """A redraw costs a full rebuild of the surface, and a slider that lands on
  the value it started at should cost nothing."""
  session = Session()
  seen = []
  session.view_changed.connect(lambda *_: seen.append(1))
  session.opacity = 1.0            # the default
  session.glyph_scale = 1.0
  assert seen == []
  session.opacity = 0.4
  assert len(seen) == 1


@pytest.mark.parametrize('field, short', [
  ('point_markers', 'point'),
  ('cell_markers [cell]', 'cell'),
])
def test_a_field_that_does_not_describe_this_mesh_reads_as_absent(field, short):
  """Drawn short, a colour array is a picture of the right shape carrying the
  wrong numbers -- the failure mode with no symptom. Reported absent instead,
  which draws the mesh plain."""
  session = Session()
  session.load_mesh(phantom('water_fat_P1_prism'), scale_factor=0.01)
  session.field = field

  assert session.surface_colour_values() is not None, 'the fixture is wrong'
  frame = session.read_frame
  def truncated(index):
    _, point_data, cell_data = frame(index)
    name = field.split(' [')[0]
    data = point_data if short == 'point' else cell_data
    data[name] = F._concat(data[name])[:-5]
    return 0.0, point_data, cell_data
  session.read_frame = truncated
  assert session.surface_colour_values() is None
