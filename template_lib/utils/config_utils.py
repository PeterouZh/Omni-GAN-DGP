from collections import Mapping
import json
from collections import OrderedDict
from pathlib import Path
import yaml


class YamlConfigParser(object):
  def __init__(self, fname, saved_fname):
    config = self.read_yaml(fname)
    self.write_yaml(config, saved_fname)
    self.config_dotdict = DotDict(config)

  @staticmethod
  def read_yaml(fname):
    if isinstance(fname, str):
      fname = Path(fname)
    with fname.open('rt') as handle:
      yaml_dict = yaml.load(handle)
      return yaml_dict

  @staticmethod
  def write_yaml(content, fname):
    if isinstance(fname, str):
      fname = Path(fname)
    with fname.open('wt') as handle:
      yaml.dump(content, handle, indent=2, sort_keys=False)


class JsonConfigParser(object):
  def __init__(self, fname, saved_fname):
    # self.dump_json_to_yaml(fname)
    config = self.read_json(fname)
    self.write_json(config, saved_fname)
    self.__config = config
    self.config_dotdict = DotDict(self.config)

  @staticmethod
  def read_json(fname):
    if isinstance(fname, str):
      fname = Path(fname)
    with fname.open('rt') as handle:
      return json.load(handle, object_hook=OrderedDict)

  @staticmethod
  def write_json(content, fname):
    if isinstance(fname, str):
      fname = Path(fname)
    with fname.open('wt') as handle:
      json.dump(content, handle, indent=2, sort_keys=False)

  @staticmethod
  def dump_json_to_yaml(fname):
    if isinstance(fname, str):
      fname = Path(fname)
    json_dict = yaml.load(json.dumps(json.load(fname.open('rt'))))
    yaml.dump(json_dict,
              open(fname.parent.joinpath('config.yaml'), 'w'),
              Dumper=_CustomDumper)
    pass

  @property
  def config(self):
    return self.__config


class DotDict(OrderedDict):
    '''
    Quick and dirty implementation of a dot-able dict, which allows access and
    assignment via object properties rather than dict indexing.
    '''

    def __init__(self, *args, **kwargs):
        # we could just call super(DotDict, self).__init__(*args, **kwargs)
        # but that won't get us nested dotdict objects
        od = OrderedDict(*args, **kwargs)
        for key, val in od.items():
            if isinstance(val, (dict, Mapping, OrderedDict)):
                val = DotDict(val)
            self[key] = val

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as ex:
            raise AttributeError(f"No attribute called: {name}") from ex

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as ex:
            raise AttributeError(f"No attribute called: {k}") from ex

    __setattr__ = OrderedDict.__setitem__


class _CustomDumper(yaml.Dumper):
  # Super neat hack to preserve the mapping key order. See
  # https://stackoverflow.com/a/52621703/1497385
  def represent_dict_preserve_order(self, data):
    return self.represent_dict(data.items())
_CustomDumper.add_representer(dict,
                              _CustomDumper.represent_dict_preserve_order)