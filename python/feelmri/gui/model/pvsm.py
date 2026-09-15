"""Writing ParaView state files, so a plan made here opens there.

`PVSMParser` (`Parameters.py`) reads a `.pvsm` and yields `FOV`, `LOC`,
`Rotation` and `MPS`. This is the other direction. The GUI is meant to be an
alternative to the ParaView planning workflow rather than a replacement, so a
plan has to travel both ways.

Two routes, because they trade different things:

`write_pvsm` builds a minimal state from scratch. It carries only the Box and
the Transform, which is everything `PVSMParser` reads and enough for ParaView
to open and show the field of view, but it has no phantom in it.

`patch_pvsm` copies an existing state and rewrites only the six numbers. That
keeps the reader, the camera, the colour maps and every representation the
original author set up, so it is the better route whenever a template exists.
The five files under `examples/planning/` are ready-made templates.

What the reader requires, traced from `PVSMParser`:

  - a `CubeSource` proxy in group `sources` carrying `XLength`, `YLength` and
    `ZLength`, registered in some `ProxyCollection` under the box name;
  - a `TransformFilter` proxy in group `filters`, registered under the
    transform name, whose `Transform` property points at another proxy;
  - a collection named `pq_helper_proxies.<filter id>` holding an `Item` for
    that proxy whose `logname` ENDS in the proxy's type after splitting on
    `/`, since the type is what the reader then searches for in group
    `extended_sources`;
  - that proxy carrying `Position` and `Rotation`, three `Element`s each.

Angles are DEGREES, matching `PVSMParser`'s `angle_units='deg'` default.
Lengths are in whatever unit the reader will be told to assume, so the caller
and the reader have to agree; several shipped files are in centimetres.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional, Sequence

import numpy as np

#: Version stamped on a from-scratch file. Matches the shipped states.
DEFAULT_VERSION = '6.0.0'

# Arbitrary but distinct ids for a generated file. ParaView renumbers on load,
# and nothing here refers to a proxy it does not also define.
_BOX_ID = '1001'
_FILTER_ID = '1002'
_TRANSFORM_ID = '1003'


def _vector_property(parent, name: str, values: Sequence[float], owner: str):
  """A three-element vector property, the shape the reader expects."""
  prop = ET.SubElement(parent, 'Property',
                       {'name': name, 'id': f'{owner}.{name}',
                        'number_of_elements': '3'})
  for i, v in enumerate(values):
    ET.SubElement(prop, 'Element', {'index': str(i), 'value': repr(float(v))})
  return prop


def _scalar_property(parent, name: str, value: float, owner: str):
  prop = ET.SubElement(parent, 'Property',
                       {'name': name, 'id': f'{owner}.{name}',
                        'number_of_elements': '1'})
  ET.SubElement(prop, 'Element', {'index': '0', 'value': repr(float(value))})
  return prop


def write_pvsm(path,
               fov: Sequence[float],
               loc: Sequence[float],
               rotation_deg: Sequence[float],
               *,
               box_name: str = 'Box1',
               transform_name: str = 'Transform1',
               version: str = DEFAULT_VERSION) -> str:
  """Write a minimal ParaView state describing one field of view.

  `fov` is the box extent, `loc` the centre and `rotation_deg` the three Euler
  angles in degrees, in the `Rz(tz) Rx(tx) Ry(ty)` order `PVSMParser`
  reassembles them with.

  The file contains no phantom. Use `patch_pvsm` when you have a template and
  want the mesh, camera and colour maps preserved.
  """
  fov = np.asarray(fov, dtype=np.float64).reshape(3)
  loc = np.asarray(loc, dtype=np.float64).reshape(3)
  rot = np.asarray(rotation_deg, dtype=np.float64).reshape(3)
  if not (np.all(np.isfinite(fov)) and np.all(np.isfinite(loc))
          and np.all(np.isfinite(rot))):
    raise ValueError('write_pvsm: fov, loc and rotation must all be finite')
  if np.any(fov < 0):
    raise ValueError(f'write_pvsm: fov must be non-negative, got {tuple(fov)}')

  root = ET.Element('ParaView')
  state = ET.SubElement(root, 'ServerManagerState', {'version': version})

  box = ET.SubElement(state, 'Proxy', {'group': 'sources', 'type': 'CubeSource',
                                       'id': _BOX_ID, 'servers': '1'})
  for axis, value in zip('XYZ', fov):
    _scalar_property(box, f'{axis}Length', value, _BOX_ID)

  transform = ET.SubElement(state, 'Proxy',
                            {'group': 'extended_sources', 'type': 'Transform3',
                             'id': _TRANSFORM_ID, 'servers': '1'})
  _vector_property(transform, 'Position', loc, _TRANSFORM_ID)
  _vector_property(transform, 'Rotation', rot, _TRANSFORM_ID)
  _vector_property(transform, 'Scale', (1.0, 1.0, 1.0), _TRANSFORM_ID)

  filt = ET.SubElement(state, 'Proxy',
                       {'group': 'filters', 'type': 'TransformFilter',
                        'id': _FILTER_ID, 'servers': '1'})
  inp = ET.SubElement(filt, 'Property', {'name': 'Input',
                                         'id': f'{_FILTER_ID}.Input',
                                         'number_of_elements': '1'})
  ET.SubElement(inp, 'Proxy', {'value': _BOX_ID, 'output_port': '0'})
  tprop = ET.SubElement(filt, 'Property', {'name': 'Transform',
                                           'id': f'{_FILTER_ID}.Transform',
                                           'number_of_elements': '1'})
  ET.SubElement(tprop, 'Proxy', {'value': _TRANSFORM_ID})

  sources = ET.SubElement(state, 'ProxyCollection', {'name': 'sources'})
  ET.SubElement(sources, 'Item', {'id': _BOX_ID, 'name': box_name})
  ET.SubElement(sources, 'Item', {'id': _FILTER_ID, 'name': transform_name})

  # The reader takes the proxy TYPE from this logname's last '/' segment, so
  # the tail has to be the type it will then look for in extended_sources.
  helper = ET.SubElement(state, 'ProxyCollection',
                         {'name': f'pq_helper_proxies.{_FILTER_ID}'})
  ET.SubElement(helper, 'Item',
                {'id': _TRANSFORM_ID, 'name': 'Transform',
                 'logname': f'{transform_name}/Transform/Transform3'})

  tree = ET.ElementTree(root)
  ET.indent(tree, space='  ')
  tree.write(str(path), encoding='utf-8', xml_declaration=False)
  return str(path)


def patch_pvsm(template,
               path,
               fov: Optional[Sequence[float]] = None,
               loc: Optional[Sequence[float]] = None,
               rotation_deg: Optional[Sequence[float]] = None,
               *,
               box_name: str = 'Box1',
               transform_name: str = 'Transform1') -> str:
  """Copy `template` and rewrite only the box extent and the transform.

  Everything else survives: the phantom reader, the camera, the colour maps,
  every representation. Any of the three values may be left as None to keep the
  template's own.

  The proxies are located exactly the way `PVSMParser` locates them, so a
  template this function accepts is a template the reader accepts.
  """
  tree = ET.parse(str(template))
  root = tree.getroot()

  def proxy_id(name: str) -> str:
    for pc in root.findall('.//ProxyCollection'):
      for item in pc.findall('Item'):
        if item.get('name') == name:
          return item.get('id')
    raise KeyError(f"patch_pvsm: no proxy named {name!r} in {template}")

  if fov is not None:
    fov = np.asarray(fov, dtype=np.float64).reshape(3)
    if np.any(fov < 0) or not np.all(np.isfinite(fov)):
      raise ValueError(f'patch_pvsm: fov must be finite and non-negative, '
                       f'got {tuple(fov)}')
    box = root.find(".//Proxy[@group='sources'][@type='CubeSource'][@id='%s']"
                    % proxy_id(box_name))
    if box is None:
      raise KeyError(f"patch_pvsm: {box_name!r} is not a CubeSource")
    for axis, value in zip('XYZ', fov):
      el = box.find("./Property[@name='%sLength']/Element" % axis)
      if el is None:
        raise KeyError(f'patch_pvsm: CubeSource has no {axis}Length')
      el.set('value', repr(float(value)))

  if loc is None and rotation_deg is None:
    tree.write(str(path), encoding='utf-8', xml_declaration=False)
    return str(path)

  filter_id = proxy_id(transform_name)
  filt = root.find(".//Proxy[@group='filters'][@type='TransformFilter'][@id='%s']"
                   % filter_id)
  if filt is None:
    raise KeyError(f"patch_pvsm: {transform_name!r} is not a TransformFilter")
  ref = filt.find("./Property[@name='Transform']/Proxy")
  if ref is None or 'value' not in ref.attrib:
    raise KeyError('patch_pvsm: TransformFilter has no Transform proxy')
  transform_id = ref.get('value')

  helper = root.find(".//ProxyCollection[@name='pq_helper_proxies.%s']" % filter_id)
  if helper is None:
    raise KeyError(f'patch_pvsm: no helper collection for filter {filter_id}')
  item = helper.find("Item[@id='%s']" % transform_id)
  if item is None:
    raise KeyError(f'patch_pvsm: helper collection has no item {transform_id}')
  type_name = item.get('logname', '').split('/')[-1]
  tr = root.find(".//Proxy[@group='extended_sources'][@type='%s'][@id='%s']"
                 % (type_name, transform_id))
  if tr is None:
    raise KeyError(f'patch_pvsm: no {type_name} proxy with id {transform_id}')

  for name, values in (('Position', loc), ('Rotation', rotation_deg)):
    if values is None:
      continue
    values = np.asarray(values, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(values)):
      raise ValueError(f'patch_pvsm: {name} must be finite, got {tuple(values)}')
    elems = tr.findall("./Property[@name='%s']/Element" % name)
    if len(elems) != 3:
      raise KeyError(f'patch_pvsm: {name} does not have three elements')
    for el, v in zip(elems, values):
      el.set('value', repr(float(v)))

  tree.write(str(path), encoding='utf-8', xml_declaration=False)
  return str(path)
