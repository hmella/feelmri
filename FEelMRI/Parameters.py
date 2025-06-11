import numpy as np
import yaml
from pint import Quantity


class DotDict(dict):
    """Dictionary with dot notation access."""
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
    def __init__(self, yaml_path):
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
                    parsed[k] = Quantity(v['value'], v['unit'])
                else:
                    parsed[k] = self._parse_dict(v)
            elif isinstance(v, list):
                parsed[k] = [self._parse_dict(i) if isinstance(i, dict) else i for i in v]
            else:
                parsed[k] = v
        return parsed

    def __getattr__(self, attr):
        return getattr(self.params, attr)