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


class ParameterHandler2:
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

class ParameterHandler:
  def __init__(self, file_path):
    self.file_path = file_path
    self.parameters = self.load_parameters()
    self.set_default_properties()

  def load_parameters(self):
    with open(self.file_path, 'r') as file:
      return yaml.safe_load(file)

  def get_parameter(self, section, key):
    return self.parameters.get(section, {}).get(key)

  def set_parameter(self, section, key, value):
    if section in self.parameters:
      self.parameters[section][key] = value
    else:
      self.parameters[section] = {key: value}
    self.save_parameters()

  def save_parameters(self):
    with open(self.file_path, 'w') as file:
      yaml.safe_dump(self.parameters, file)

  def set_default_properties(self):
    for section, params in self.parameters.items():
      for key, value in params.items():
        if isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
          value = np.array(value)
        setattr(self, key, value)
