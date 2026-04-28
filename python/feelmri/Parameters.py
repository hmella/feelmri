"""
Parameter loading and parsing utilities for feelmri configuration files.

Provides :class:`ParameterHandler` for YAML-based parameter files with
physical units (via pint), and :class:`PVSMParser` for extracting geometry
from ParaView ``.pvsm`` state files.
"""
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from pint import Quantity as Q_

from feelmri.Math import Rx, Ry, Rz


class PVSMParser:
    """Parse a ParaView ``.pvsm`` state file to extract scan geometry.

    Reads the bounding box dimensions from a CubeSource and the position /
    rotation from a TransformFilter chain, then assembles the
    machine-to-patient-space rotation matrix.

    Parameters
    ----------
    filepath : str
        Path to the ``.pvsm`` file.
    box_name : str, optional
        ``name`` attribute of the Box source proxy. Default is ``'Box1'``.
    transform_name : str, optional
        ``name`` attribute of the Transform filter proxy.
        Default is ``'Transform1'``.
    length_units : str, optional
        Physical unit string for length values (pint-compatible).
        Default is ``'m'``.
    angle_units : str, optional
        Physical unit string for angle values (pint-compatible).
        Default is ``'deg'``.
    """

    def __init__(self, filepath: str,
                 box_name: str = "Box1",
                 transform_name: str = "Transform1",
                 length_units: str = "m",
                 angle_units: str = "deg"):
        self.tree = ET.parse(filepath)
        self.root = self.tree.getroot()
        self.box_name = box_name
        self.transform_name = transform_name
        self.length_units = length_units
        self.angle_units = angle_units
        self.Rotation = Q_(np.array(self.get_transform()['Rotation']), angle_units)
        self.FOV = Q_(np.array(list(self.get_box_lengths().values())), length_units)
        self.LOC = Q_(np.array(self.get_transform()['Position']), length_units)
        tx, ty, tz = self.Rotation.m_as('rad')
        self.MPS = Rz(tz) @ Rx(tx) @ Ry(ty)

    def _get_proxy_id(self, proxy_name: str) -> str:
        """Return the proxy ID for a given proxy name from the ProxyCollection.

        Parameters
        ----------
        proxy_name : str
            The ``name`` attribute of the proxy item to look up.

        Returns
        -------
        str
            The corresponding proxy ``id`` attribute.

        Raises
        ------
        KeyError
            If no proxy with the given name is found.
        """
        for pc in self.root.findall('.//ProxyCollection'):
            for item in pc.findall('Item'):
                if item.get('name') == proxy_name:
                    return item.get('id')
        raise KeyError(f"No proxy named '{proxy_name}' found.")

    def _get_transform_type(self, filter_id: str, transform_id: str) -> str:
        """Extract the transform type name (e.g. ``'Transform3'``) from the log name.

        Parameters
        ----------
        filter_id : str
            Proxy collection name suffix of the TransformFilter.
        transform_id : str
            Proxy ID of the associated Transform3 proxy.

        Returns
        -------
        str
            Type name parsed from the logname attribute.

        Raises
        ------
        KeyError
            If the proxy collection, item, or logname cannot be resolved.
        """
        pc_name = f"pq_helper_proxies.{filter_id}"
        pc = self.root.find(f".//ProxyCollection[@name='{pc_name}']")
        if pc is None:
            raise KeyError(f"ProxyCollection with name '{pc_name}' not found.")
        item = pc.find(f"Item[@id='{transform_id}']")
        if item is None:
            raise KeyError(f"Item with id={transform_id} not found in {pc_name}.")
        logname = item.get('logname', '')
        type_name = logname.split('/')[-1]
        if not type_name:
            raise KeyError(f"Could not parse type from logname '{logname}'.")
        return type_name

    def get_box_lengths(self) -> dict:
        """Extract XLength, YLength, ZLength from the CubeSource proxy.

        Returns
        -------
        dict
            Keys ``'XLength'``, ``'YLength'``, ``'ZLength'`` with float values.

        Raises
        ------
        KeyError
            If the CubeSource proxy or any length property is not found.
        """
        proxy_id = self._get_proxy_id(self.box_name)
        proxy = self.root.find(
            f".//Proxy[@group='sources'][@type='CubeSource'][@id='{proxy_id}']"
        )
        if proxy is None:
            raise KeyError(f"CubeSource with id={proxy_id} not found.")

        def _read_scalar(prop_name):
            elem = proxy.find(f"./Property[@name='{prop_name}']/Element")
            if elem is None:
                raise KeyError(f"Property '{prop_name}' missing on CubeSource id={proxy_id}.")
            return float(elem.get('value'))

        return {
            'XLength': _read_scalar('XLength'),
            'YLength': _read_scalar('YLength'),
            'ZLength': _read_scalar('ZLength'),
        }

    def get_transform(self) -> dict:
        """Extract Position and Rotation from a TransformFilter chain.

        Returns
        -------
        dict
            Dictionary with keys:

            * ``'Position'`` — ``(x, y, z)`` tuple of floats.
            * ``'Rotation'`` — ``(theta_x, theta_y, theta_z)`` tuple of floats.

        Raises
        ------
        KeyError
            If any expected proxy, property, or element is missing.
        """
        filter_id = self._get_proxy_id(self.transform_name)
        filter_proxy = self.root.find(
            f".//Proxy[@group='filters'][@type='TransformFilter'][@id='{filter_id}']"
        )
        if filter_proxy is None:
            raise KeyError(f"TransformFilter with id={filter_id} not found.")

        tr_prop = filter_proxy.find("./Property[@name='Transform']/Proxy")
        if tr_prop is None or 'value' not in tr_prop.attrib:
            raise KeyError(
                f"Transform property not found on TransformFilter id={filter_id}.")
        transform_id = tr_prop.get('value')

        transform3_type = self._get_transform_type(filter_id, transform_id)

        tr3 = self.root.find(
            f".//Proxy[@group='extended_sources'][@type='{transform3_type}'][@id='{transform_id}']"
        )
        if tr3 is None:
            raise KeyError(f"{transform3_type} with id={transform_id} not found.")

        def _read_vector(prop_name):
            elems = tr3.findall(f"./Property[@name='{prop_name}']/Element")
            if len(elems) != 3:
                raise KeyError(f"Property '{prop_name}' does not have 3 elements.")
            return tuple(float(e.get('value')) for e in elems)

        return {
            'Position': _read_vector('Position'),
            'Rotation': _read_vector('Rotation'),
        }


class DotDict(dict):
    """Dictionary subclass that also supports attribute-style access."""

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]


class ParameterHandler:
    """Load a YAML parameter file and expose parameters as a dot-accessible object.

    Values annotated with ``value`` / ``unit`` keys in the YAML are
    automatically converted to :class:`pint.Quantity` objects. Nested
    dictionaries become nested :class:`DotDict` instances.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file.

    Examples
    --------
    Given a YAML file::

        TR:
          value: 5.0
          unit: ms

    Usage::

        params = ParameterHandler('config.yaml')
        print(params.TR)  # 5.0 ms (pint Quantity)
    """

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.params = self._load_and_parse()

    def _load_and_parse(self):
        with open(self.yaml_path, 'r') as f:
            raw = yaml.safe_load(f)
        return self._parse_dict(raw)

    def _parse_dict(self, d):
        parsed = DotDict()
        for k, v in d.items():
            if isinstance(v, dict):
                if 'value' in v and 'unit' in v:
                    parsed[k] = Q_(v['value'], v['unit'])
                else:
                    parsed[k] = self._parse_dict(v)
            elif isinstance(v, list):
                parsed[k] = [self._parse_dict(i) if isinstance(i, dict) else i for i in v]
            else:
                parsed[k] = v
        return parsed

    def __getattr__(self, attr):
        return getattr(self.params, attr)
