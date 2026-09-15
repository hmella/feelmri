"""Writing ParaView state files that `PVSMParser` reads back.

The GUI is an alternative to the ParaView planning workflow, not a
replacement, so a plan has to survive the trip in both directions. The reader
already exists and is unchanged; these tests drive the writer against it.

Round-tripping through the real reader is the acceptance check, because the
reader is the contract. A writer tested against its own assumptions would agree
with itself and tell us nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from feelmri.Parameters import PVSMParser
from feelmri.gui.model.planning import FOVBox, euler_to_mps, is_rotation
from feelmri.gui.model.pvsm import patch_pvsm, write_pvsm

PLANNING = __import__('pathlib').Path(__file__).resolve().parent.parent / 'examples' / 'planning'
PVSM_FILES = ['4dflow', 'abdomen', 'beating_heart', 'phase_contrast', 'water_and_fat']


def test_a_written_state_reads_back_with_the_same_numbers(tmp_path):
  """The acceptance check: write, then read with the real parser."""
  fov = [0.30, 0.22, 0.008]
  loc = [0.011, -0.023, 0.047]
  rot = [12.5, -33.0, 97.25]          # degrees

  out = tmp_path / 'plan.pvsm'
  write_pvsm(out, fov, loc, rot)
  assert out.exists() and out.stat().st_size > 0

  parser = PVSMParser(str(out))
  np.testing.assert_allclose(parser.FOV.m_as('m'), fov, rtol=1e-12)
  np.testing.assert_allclose(parser.LOC.m_as('m'), loc, rtol=1e-12)
  np.testing.assert_allclose(parser.Rotation.m_as('deg'), rot, rtol=1e-12)
  assert is_rotation(parser.MPS)


def test_the_orientation_survives_the_trip(tmp_path):
  """A box built from a matrix, written, and read, gives that matrix back.

  This is the whole chain the GUI depends on: an interactive rotation becomes
  three Z-X-Y angles, becomes degrees in a file, becomes a matrix again.
  """
  rng = np.random.default_rng(11)
  worst = 0.0
  for i in range(60):
    tx = rng.uniform(-1.4, 1.4)
    ty, tz = rng.uniform(-np.pi, np.pi, size=2)
    mps = euler_to_mps(tx, ty, tz)
    box = FOVBox.from_mps([0.3, 0.2, 0.01], [0.01, 0.02, -0.03], mps)

    out = tmp_path / f'r{i}.pvsm'
    write_pvsm(out, box.fov, box.loc, np.degrees(box.angles))
    worst = max(worst, float(np.abs(PVSMParser(str(out)).MPS - mps).max()))
  assert worst < 1e-12, f'worst orientation round-trip error {worst:.3e}'


def test_the_writer_refuses_impossible_geometry(tmp_path):
  with pytest.raises(ValueError, match='non-negative'):
    write_pvsm(tmp_path / 'a.pvsm', [-1, 1, 1], [0, 0, 0], [0, 0, 0])
  with pytest.raises(ValueError, match='finite'):
    write_pvsm(tmp_path / 'b.pvsm', [1, 1, np.nan], [0, 0, 0], [0, 0, 0])
  with pytest.raises(ValueError, match='finite'):
    write_pvsm(tmp_path / 'c.pvsm', [1, 1, 1], [0, 0, 0], [np.inf, 0, 0])


def test_custom_proxy_names_round_trip(tmp_path):
  """The reader takes the names as arguments, so the writer must too."""
  out = tmp_path / 'named.pvsm'
  write_pvsm(out, [1, 2, 3], [4, 5, 6], [0, 0, 0],
             box_name='MyBox', transform_name='MyTransform')
  parser = PVSMParser(str(out), box_name='MyBox', transform_name='MyTransform')
  np.testing.assert_allclose(parser.FOV.m_as('m'), [1, 2, 3])
  # The default names must genuinely not be there, or the test is vacuous.
  with pytest.raises(KeyError):
    PVSMParser(str(out))


@pytest.mark.parametrize('name', PVSM_FILES)
def test_patching_a_shipped_state_changes_only_what_was_asked(tmp_path, name):
  """Patch a real state file and confirm the reader sees the new plan.

  The template carries a phantom reader, a camera and colour maps that a
  from-scratch file does not. Patching is how a GUI plan keeps them, so the
  check is that the six numbers move and the file stays a valid state.
  """
  src = PLANNING / f'{name}.pvsm'
  if not src.exists():
    pytest.skip(f'{src.name} not present')

  before = PVSMParser(str(src))
  new_fov = [7.5, 3.25, 11.0]
  new_loc = [1.5, -2.5, 3.5]
  new_rot = [10.0, 20.0, -30.0]

  out = tmp_path / f'{name}_patched.pvsm'
  patch_pvsm(src, out, new_fov, new_loc, new_rot)

  after = PVSMParser(str(out))
  np.testing.assert_allclose(after.FOV.m, new_fov, rtol=1e-12)
  np.testing.assert_allclose(after.LOC.m, new_loc, rtol=1e-12)
  np.testing.assert_allclose(after.Rotation.m, new_rot, rtol=1e-12)

  # It must actually be a change, or the test proves nothing.
  assert not np.allclose(before.FOV.m, new_fov)

  # The template's bulk survives: a from-scratch file is tiny by comparison.
  assert out.stat().st_size > 0.5 * src.stat().st_size, (
    'patching appears to have discarded most of the template')


@pytest.mark.parametrize('name', PVSM_FILES)
def test_patching_nothing_leaves_the_plan_alone(tmp_path, name):
  """All three values optional: omitting them must not perturb the geometry."""
  src = PLANNING / f'{name}.pvsm'
  if not src.exists():
    pytest.skip(f'{src.name} not present')
  out = tmp_path / f'{name}_same.pvsm'
  patch_pvsm(src, out)

  a, b = PVSMParser(str(src)), PVSMParser(str(out))
  np.testing.assert_allclose(b.FOV.m, a.FOV.m, rtol=1e-12)
  np.testing.assert_allclose(b.LOC.m, a.LOC.m, rtol=1e-12)
  np.testing.assert_allclose(b.Rotation.m, a.Rotation.m, rtol=1e-12)
  np.testing.assert_allclose(b.MPS, a.MPS, atol=1e-15)


def test_patch_refuses_a_template_without_the_named_proxies(tmp_path):
  """A file the reader would reject must be refused here, by name."""
  bogus = tmp_path / 'bogus.pvsm'
  bogus.write_text('<ParaView><ServerManagerState version="6.0.0"/></ParaView>')
  with pytest.raises(KeyError, match='Box1'):
    patch_pvsm(bogus, tmp_path / 'out.pvsm', [1, 1, 1])


# -- the parameter file that points at the state file -----------------------

def test_the_exported_yaml_is_readable_by_the_librarys_own_handler(tmp_path):
  """The `.pvsm` holds the plan; this is the file an example opens.

  Checked through `ParameterHandler` rather than by reading the text, because
  what matters is that the library accepts it and converts the units.
  """
  from feelmri.Parameters import ParameterHandler
  from feelmri.gui.model.pvsm import write_plan_yaml

  path = tmp_path / 'plan.yaml'
  write_plan_yaml(path, 'plan.pvsm')

  handler = ParameterHandler(str(path))
  assert handler.Formatting.planning == 'plan.pvsm'
  assert handler.Formatting.units == 'm'
  assert handler.Imaging.FlipAngle.m_as('deg') == 15
  assert handler.Hardware.G_max.m_as('mT/m') == 33.0
  assert handler.Phantom.T1.m_as('ms') == 800


def test_the_yaml_matches_the_shape_the_shipped_examples_read(tmp_path):
  """Same four sections and the same keys as an example's own parameter file.

  A file the GUI writes has to be a drop-in for one a user already has, or
  `parameters.Imaging.RES` fails at the call site rather than here.
  """
  import yaml

  from feelmri.gui.model.pvsm import write_plan_yaml

  shipped = Path('examples/parameters/trajectories.yaml')
  if not shipped.exists():
    pytest.skip('trajectories.yaml not present')

  path = tmp_path / 'plan.yaml'
  write_plan_yaml(path, 'plan.pvsm')
  written = yaml.safe_load(path.read_text())
  reference = yaml.safe_load(shipped.read_text())

  assert set(written) == set(reference)
  for section in ('Imaging', 'Hardware', 'Formatting', 'Phantom'):
    assert set(written[section]) == set(reference[section]), section


def test_the_planning_path_is_written_verbatim(tmp_path):
  """It is resolved against the SCRIPT directory, not against the yaml, so
  the caller decides the spelling and this must not rewrite it."""
  import yaml

  from feelmri.gui.model.pvsm import write_plan_yaml

  path = tmp_path / 'plan.yaml'
  write_plan_yaml(path, 'planning/beating_heart.pvsm')
  assert yaml.safe_load(path.read_text())['Formatting']['planning'] == \
         'planning/beating_heart.pvsm'


def test_the_exported_pair_round_trips_through_the_real_parsers(tmp_path):
  """Export both, then read the plan back the way an example does.

  `ParameterHandler` for the parameters, `PVSMParser` for the geometry it
  names -- the two halves the GUI writes have to agree with each other.
  """
  import numpy as np

  from feelmri.Parameters import ParameterHandler, PVSMParser
  from feelmri.gui.model.pvsm import write_plan_yaml, write_pvsm

  fov = [0.30, 0.22, 0.008]
  loc = [0.021, -0.034, 0.047]
  rotation = [0.0, 0.0, 25.0]
  write_pvsm(tmp_path / 'plan.pvsm', fov, loc, rotation)
  write_plan_yaml(tmp_path / 'plan.yaml', 'plan.pvsm')

  handler = ParameterHandler(str(tmp_path / 'plan.yaml'))
  planning = PVSMParser(tmp_path / handler.Formatting.planning)

  np.testing.assert_allclose(planning.FOV.m_as('m'), fov, atol=1e-9)
  np.testing.assert_allclose(planning.LOC.m_as('m'), loc, atol=1e-9)
