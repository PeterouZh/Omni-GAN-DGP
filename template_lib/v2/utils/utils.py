import os
import re
import logging
import sys
import importlib
import json
import pprint
import collections


def get_dict_str(dict_obj, use_pprint=True):
  message = ''
  message += '----------------- start ---------------\n'
  if use_pprint:
    message += pprint.pformat(collections.OrderedDict(dict_obj))
  else:
    message += json.dumps(dict_obj, indent=2)
  message += '\n----------------- End -------------------'
  return message


def reload_module(module):
  if module not in sys.modules:
    imported_module = importlib.import_module(module)
  else:
    importlib.reload(sys.modules[module])


def register_modules(register_modules):
  for module in register_modules:
    importlib.import_module(module)
    # reload_module(module=module)
    logging.getLogger('tl').info(f"  Register {module}")
  pass


def get_prefix_abb(prefix):
  # prefix_split = prefix.split('_')
  prefix_split = re.split('_|/', prefix)
  if len(prefix_split) == 1:
    prefix_abb = prefix
  else:
    prefix_abb = ''.join([k[0] for k in prefix_split])
  return prefix_abb


def get_git_hash(logger=None):
  if logger is not None:
    print = logger.info
  cwd = os.getcwd()
  # os.chdir(os.path.join(cwd, '..'))
  try:
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print('git hash: \n%s'%sha)
    print('git checkout sha')
    print('git submodule update --recursive')
  except:
    sha = 0
    import traceback
    print(traceback.format_exc())
  os.chdir(cwd)
  return sha


