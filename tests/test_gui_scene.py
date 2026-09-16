"""The scene state a ParaView-shaped Plan view drives: layers, representation,
camera axes, and the field/component split.

All headless. The widgets that drive these need a display and are verified by
screenshot; what is tested here is the state they write and the rules that
state has to obey.
"""
from __future__ import annotations

import numpy as np
import pytest

from feelmri.gui.model import fields as F
from feelmri.gui.model.camera import AXIS_VIEWS, STANDARD_VIEWS, Camera
from feelmri.gui.model.session import LAYERS, REPRESENTATIONS, Session

VECTOR = np.array([[3.0, -4.0, 12.0], [6.0, 8.0, 0.0]])


# -- layer visibility -------------------------------------------------------

def test_every_layer_starts_shown():
  session = Session()
  assert all(session.is_visible(layer) for layer in LAYERS)


@pytest.mark.parametrize('layer, signal', [
  ('phantom', 'view_changed'),
  ('glyphs', 'view_changed'),
  ('plan', 'plan_changed'),
  ('arrows', 'plan_changed'),
])
def test_a_layer_announces_on_the_signal_that_redraws_it(layer, signal):
  """The plan layers go to `plan_changed`, which redraws the OVERLAY; the
  phantom and its glyphs to `view_changed`, which rebuilds the mesh. Sending
  all four to one signal would rebuild a hundred thousand triangles to hide a
  box."""
  session = Session()
  seen = []
  session.plan_changed.connect(lambda *_: seen.append('plan_changed'))
  session.view_changed.connect(lambda *_: seen.append('view_changed'))
  session.set_visible(layer, False)
  assert seen == [signal]


def test_hiding_a_layer_that_is_already_hidden_notifies_nobody():
  session = Session()
  session.set_visible('plan', False)
  seen = []
  session.plan_changed.connect(lambda *_: seen.append(1))
  session.set_visible('plan', False)
  assert seen == []


def test_an_unknown_layer_is_refused_by_name():
  """A typo would otherwise create a layer nothing draws, and the browser
  would show an eye for something that is not in the scene."""
  session = Session()
  with pytest.raises(ValueError, match='unknown layer'):
    session.set_visible('surface', False)
  with pytest.raises(ValueError, match='unknown layer'):
    session.is_visible('surface')


def test_the_pipeline_browser_lists_exactly_the_session_layers():
  """Two lists that must agree, in two files. The browser asserts it at
  import; this is the same check where a test runner will see it."""
  from feelmri.gui.view.pipeline import LAYER_NAMES
  assert tuple(LAYER_NAMES) == LAYERS


# -- representation ---------------------------------------------------------

def test_the_representation_round_trips():
  session = Session()
  for name in REPRESENTATIONS:
    session.representation = name
    assert session.representation == name


def test_an_unknown_representation_is_refused():
  with pytest.raises(ValueError, match='representation'):
    Session().representation = 'Volume'


def test_every_representation_maps_onto_a_pyvista_style():
  """The viewport looks each one up rather than passing it through, because
  `Surface With Edges` is a flag on the surface style and not a fourth style.
  A new entry with no mapping would raise inside a redraw."""
  styles = {'Surface': 'surface', 'Surface With Edges': 'surface',
            'Wireframe': 'wireframe', 'Points': 'points'}
  assert set(styles) == set(REPRESENTATIONS)


# -- the camera axes --------------------------------------------------------

def test_the_axis_views_are_all_real_views():
  assert set(AXIS_VIEWS) <= set(STANDARD_VIEWS)
  assert len(AXIS_VIEWS) == 6


@pytest.mark.parametrize('anatomical, axis', [
  ('sagittal', '-X'), ('coronal', '-Y'), ('axial', '+Z'),
])
def test_the_anatomical_names_are_the_axis_names(anatomical, axis):
  """Documented as equal, so it is asserted rather than left to be re-derived
  by the next reader of the table."""
  assert STANDARD_VIEWS[anatomical] == STANDARD_VIEWS[axis]


@pytest.mark.parametrize('name', AXIS_VIEWS)
def test_an_axis_view_looks_along_its_own_axis(name):
  """`+X` puts the camera on the +X side, so it looks back along -X. Getting
  the sign backwards is invisible on a symmetric phantom, which is why this
  checks the camera POSITION rather than only the direction."""
  camera = Camera((0.0, 0.0, 0.4), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
  direction, up = STANDARD_VIEWS[name]
  view = camera.look_along(direction, up)

  axis = 'XYZ'.index(name[1])
  sign = 1.0 if name[0] == '+' else -1.0
  offset = view.position - view.focal_point
  assert np.sign(offset[axis]) == sign, 'the camera is on the wrong side'
  assert abs(offset[axis]) == pytest.approx(np.linalg.norm(offset), rel=1e-9)


# -- the field / component split -------------------------------------------

def test_a_vector_becomes_one_group_with_four_components():
  """One flat list grows as the product; two boxes do not."""
  groups = dict(F.field_groups({'v': VECTOR}))
  assert list(groups) == ['v']
  assert [c for c, _ in groups['v']] == ['Magnitude', 'X', 'Y', 'Z']
  assert [label for _, label in groups['v']] == [
    'v (Magnitude)', 'v (X)', 'v (Y)', 'v (Z)']


def test_a_scalar_becomes_one_group_with_one_nameless_component():
  """`''` rather than an empty list, so a caller drives the second box off the
  list with no special case."""
  groups = dict(F.field_groups({'p': np.zeros((4, 1))}))
  assert groups['p'] == [('', 'p')]


def test_a_cell_field_keeps_its_marker_in_the_group_name():
  """The group is what the first combobox shows, and a point and a cell field
  may share a name."""
  groups = dict(F.field_groups({'m': np.zeros(4)}, {'m': np.zeros(9)}))
  assert set(groups) == {'m', 'm [cell]'}


def test_every_group_label_resolves_back_to_a_real_field():
  """The two boxes compose a label, and that label has to be one `resolve`
  knows -- otherwise picking a component silently draws nothing."""
  point = {'v': VECTOR, 'p': np.zeros((2, 1))}
  cell = {'m': np.zeros(5)}
  for _group, components in F.field_groups(point, cell):
    for _component, label in components:
      assert F.resolve(label, point, cell) is not None, label


# -- frame stepping ---------------------------------------------------------

def test_stepping_wraps_at_both_ends():
  """These are cines: the last frame of a cardiac cycle is followed by the
  first, and a play button that stops dead at the end shows the data wrongly.
  """
  session = Session()
  session.n_frames = 4
  session.frame = 3
  assert session.step_frame(1) == 0
  assert session.step_frame(-1) == 3


def test_stepping_a_static_mesh_does_nothing():
  session = Session()
  session.n_frames = 1
  assert session.step_frame(1) == 0
  session.n_frames = 0
  assert session.step_frame(1) == 0
