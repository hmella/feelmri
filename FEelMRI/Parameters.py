import yaml
import numpy as np

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
