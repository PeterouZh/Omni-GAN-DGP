from easydict import EasyDict
import logging
import yaml
import os

from template_lib.v2.config import get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, \
  start_cmd_run, update_config, get_dict_str, convert_easydict_to_dict


def update_nni_config_file(nni_config_file, update_nni_cfg_str):
  update_nni_cfg = yaml.safe_load(update_nni_cfg_str)
  # os.makedirs(update_nni_cfg['logDir'], exist_ok=True)

  with open(nni_config_file, 'r') as f:
    nni_cfg = yaml.safe_load(f)

  nni_cfg = update_config(nni_cfg, update_nni_cfg)
  nni_cfg = convert_easydict_to_dict(nni_cfg)
  logging.getLogger('tl').info('\nnni config:\n ' + get_dict_str(nni_cfg))

  updated_config_file = nni_config_file.split('.')[-2] + '_updated.' + nni_config_file.split('.')[-1]
  with open(updated_config_file, 'w') as f:
    yaml.dump(nni_cfg, f, indent=2, sort_keys=False)
  return updated_config_file


def nni_ss2cfg(data_dict, delimiter='.'):
  ret_dict = EasyDict()
  for k, v in data_dict.items():
    elem_list = k.split(delimiter)
    cur_dict = ret_dict
    for idx, elem in enumerate(elem_list):
      if idx == len(elem_list) - 1:
        setattr(cur_dict, elem, v)
      else:
        if elem not in cur_dict:
          setattr(cur_dict, elem, EasyDict())
        cur_dict = getattr(cur_dict, elem)
  return ret_dict


# def cfgnode_merge_tunner_params(cfg, search_space: dict):
#   ss_opt = []
#   for k, v in search_space.items():
#     ss_opt.extend([k, v])
#   cfg.merge_from_list(ss_opt, new_allowed=True)

def cfgnode_merge_tunner_params(cfg, search_space: dict):

  # search_space = {'Generator.weight_decay': 0,
  #                 'Discriminator.weight_decay': 0}

  for k, v in search_space.items():
    try:
      cfg.merge_from_list([k, v], new_allowed=True)
    except ValueError:
      import traceback
      logging.getLogger('tl').info(traceback.format_exc())

      original_type = type(eval(f'cfg.{k}'))
      v = original_type(v)
      cfg.merge_from_list([k, v], new_allowed=True)
      pass



