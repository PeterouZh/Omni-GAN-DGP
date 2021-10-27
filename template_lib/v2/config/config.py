import copy
import json
from easydict import EasyDict
import yaml
from pathlib import Path


class YamlConfigParser(object):
  def __init__(self, fname, saved_fname):
    config = self.read_yaml(fname)
    self.write_yaml(config, saved_fname)
    self.cfg = EasyDict(config)
    pass

  @staticmethod
  def read_yaml(fname):
    if isinstance(fname, str):
      fname = Path(fname)
    with fname.open('rt') as handle:
      yaml_dict = yaml.safe_load(handle)
      return yaml_dict

  @staticmethod
  def write_yaml(content, fname):
    if isinstance(fname, str):
      fname = Path(fname)
    with fname.open('wt') as handle:
      yaml.dump(content, handle, indent=2, sort_keys=False)


def _get_config_from_file(config_file, saved_path):

  try:
    config_parser = YamlConfigParser(fname=config_file, saved_fname=saved_path)
    config = config_parser.cfg
    return config
  except ValueError:
    import traceback
    print(traceback.format_exc())
    exit(-1)


def config_inherit_from_base(config, configs, arg_base=[], overwrite_opts=False):
  base = getattr(config, 'base', [])
  if not isinstance(arg_base, list):
    arg_base = [arg_base]
  base += arg_base
  if not base:
    return EasyDict(config)

  super_config = EasyDict()
  for b in base:
    b_config = getattr(configs, b)
    b_config = config_inherit_from_base(b_config, configs, overwrite_opts=overwrite_opts)
    super_config = update_config(super_config, b_config, overwrite_opts=overwrite_opts)
  # update super_config by config
  super_config = update_config(super_config, config, overwrite_opts=overwrite_opts)
  return super_config


def setup_config(config_file, saved_config_file, args, overwrite_opts=False):
  # Parse config file
  config = _get_config_from_file(config_file, saved_path=saved_config_file)

  config_command = getattr(config, args.tl_command, None)
  if config_command:
    # inherit from base
    config_command = config_inherit_from_base(
      config=config_command, configs=config, overwrite_opts=overwrite_opts)
    config_command = convert_easydict_to_dict(config_command)

    saved_config = copy.deepcopy(config_command)
    if 'base' in saved_config:
      saved_config.pop('base')
    saved_config_command = {args.tl_command: saved_config}
    YamlConfigParser.write_yaml(
      saved_config_command, fname=args.tl_saved_config_command_file)
    # update command config
    config[args.tl_command] = EasyDict(config_command)
  return config, config_command


def convert_easydict_to_dict(config):
  ret_config = dict()
  for k in config:
    if isinstance(config[k], EasyDict):
      # config[k] = dict(config[k])
      ret_config.update({k: convert_easydict_to_dict(config[k])})
    else:
      val = {k: config[k]}
      if isinstance(config[k], list):
        if isinstance(config[k][0], dict):
          temp = list(map(dict, config[k]))
          val = {k: temp}
          pass
      ret_config.update(val)
  return ret_config


def update_config(super_config, config, overwrite_opts=True):
  """

  :param super_config:
  :param config:
  :param overwrite_opts: overwrite opts directly or overwrite its elements only
  :return:
  """
  super_config = EasyDict(super_config)
  ret_config = copy.deepcopy(super_config)
  for k in config:
    if k == 'opts' and not overwrite_opts and hasattr(super_config, k):
      ret_config[k] = update_opts(super_config[k], config[k])
    # merge dict element-wise
    elif isinstance(config[k], dict) and hasattr(super_config, k):
      if getattr(config[k], 'overwrite', False):
        sub_config = copy.deepcopy(config[k])
        sub_config.pop('overwrite')
      else:
        sub_config = update_config(super_config[k], config[k])
      setattr(ret_config, k, sub_config)
    else:
      setattr(ret_config, k, config[k])
  return ret_config


def update_opts(super_opts, opts):
  assert isinstance(super_opts, list) and len(super_opts) % 2 ==0
  assert isinstance(opts, list) and len(opts) % 2 == 0
  for k_idx in range(0, len(opts), 2):
    v_idx = k_idx + 1
    if opts[k_idx] in super_opts:
      s_k_idx = super_opts.index(opts[k_idx])
      s_v_idx = s_k_idx + 1
      super_opts[s_v_idx] = opts[v_idx]
    else:
      super_opts += [opts[k_idx], opts[v_idx]]
  return super_opts


global_cfg = EasyDict()

def set_global_cfg(cfg):
  global global_cfg
  global_cfg.clear()
  global_cfg.update(cfg)
  pass