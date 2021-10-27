import argparse
import copy
import os, sys
import shutil
import time
import logging
import collections
import json, yaml
from easydict import EasyDict
import pprint
from datetime import datetime

from .dirs import create_dirs
from . import logging_utils
from . import config_utils
from . import torch_utils
from . import shutil_utils
from . import modelarts_utils
from . import tensorboardX_utils
from . import is_debugging, args_parser, get_git_hash


def get_config_from_file(config_file, saved_path):

  try:
    if config_file.endswith('.json'):
      with open(config_file, 'r') as f:
        config_dict = json.load(f)
        config = EasyDict(config_dict)
    elif config_file.endswith('.yaml'):
      config_parser = config_utils.YamlConfigParser(fname=config_file,
                                                    saved_fname=saved_path)
      config = config_parser.config_dotdict

    return config
  except ValueError:
    print("INVALID JSON file format.. Please provide a good json file")
    exit(-1)


def setup_dirs_and_files(args, **kwargs):
  # create some important directories to be used for that experiment.
  args.abs_outdir = os.path.realpath(args.outdir)
  args.ckptdir = os.path.join(args.outdir, "models/")
  args.tbdir = os.path.join(args.outdir, "tb/")
  args.textlogdir = os.path.join(args.outdir, 'textlog/')
  args.imgdir = os.path.join(args.outdir, 'saved_imgs/')
  create_dirs([args.ckptdir, args.tbdir, args.textlogdir, args.imgdir])
  args.logfile = os.path.join(args.outdir, "log.txt")
  args.config_command_file = os.path.join(args.outdir, "config_command.yaml")

  # append log dir name in configfile
  if ('add_number_to_configfile' in kwargs) and kwargs['add_number_to_configfile']:
    args.configfile = os.path.join(args.outdir, "c_%s.yaml"%kwargs['log_number'])
  else:
    args.configfile = os.path.join(args.outdir, "config.yaml")
  pass


def setup_outdir(args, resume_root, resume, **kwargs):
  TIME_STR = bool(int(os.getenv('TIME_STR', 0)))
  time_str = datetime.now().strftime("%Y%m%d-%H_%M_%S_%f")[:-3]
  args.outdir = args.outdir if not TIME_STR else (args.outdir + '_' + time_str)
  args.time_str_suffix = time_str
  if 'log_number' in kwargs:
    args.outdir += '_%s'%kwargs['log_number']

  if resume_root and resume:
    args.outdir = resume_root
    print('Using config.yaml in resume_root: %s'%resume_root)
    args.config = os.path.join(args.outdir, "config.yaml")
  else:
    shutil.rmtree(args.outdir, ignore_errors=True)
    os.makedirs(args.outdir, exist_ok=True)
  #   try:
  #     print('Start copying code to outdir.')
  #     shutil.copytree('.', os.path.join(args.outdir, 'code'),
  #                     ignore=shutil_utils.ignoreAbsPath(['results', ]))
  #     shutil.copytree(
  #       '../submodule/template_lib',
  #       os.path.join(args.outdir, 'submodule/template_lib'),
  #       ignore=shutil_utils.ignoreNamePath(['results', 'submodule']))
  #     print('End copying code to outdir.')
  #   except:
  #     print("Error! Copying code to results.")
  return


def setup_logger_and_redirect_stdout(logfile, myargs):
  # sys.stdout is changed
  if isinstance(sys.stdout, logging_utils.StreamToLogger):
    sys.stdout = myargs.stdout
    sys.stderr = myargs.stderr
  # setup logging in the project
  logging_utils.get_logger(
    filename=logfile, logger_names=['template_lib', 'tl'], stream=True)
  logger = logging.getLogger('tl')
  myargs.logger = logger
  myargs.stdout = sys.stdout
  myargs.stderr = sys.stderr
  logging_utils.redirect_print_to_logger(logger=logger)
  return


def setup_config(config_file, saved_config_file, overwrite_opts, args, myargs):
  # Parse config file
  config = get_config_from_file(config_file, saved_path=saved_config_file)
  myargs.config = config

  config_command = getattr(config, args.command, None)
  if config_command:
    # inherit from base
    config_command = config_inherit_from_base(
      config=config_command, configs=config, overwrite_opts=overwrite_opts)
    config_command = convert_easydict_to_dict(config_command)

    saved_config = copy.deepcopy(config_command)
    if 'base' in saved_config:
      saved_config.pop('base')
    saved_config_command = {args.command: saved_config}
    config_utils.YamlConfigParser.write_yaml(
      saved_config_command, fname=args.config_command_file)
    # update command config
    myargs.config[args.command] = \
      config_utils.DotDict(config_command)

  return


def setup_tensorboardX(tbdir, args, config, myargs, start_tb=True):
  # tensorboard
  tbtool = tensorboardX_utils.TensorBoardTool(tbdir=tbdir)
  writer = tbtool.writer
  myargs.writer = writer
  if start_tb:
    tbtool.run()
  tbtool.add_text_md_args(args=args, name='args')
  tbtool.add_text_str_args(args=config, name='config')
  if hasattr(args, 'command'):
    command_config = getattr(config, args.command, 'None')
    tbtool.add_text_str_args(args=command_config, name='command')
    logger = logging.getLogger(__name__)
    logger.info('command config: \n{}'.format(pprint.pformat(command_config)))
  return


def setup_checkpoint(ckptdir, myargs):
  checkpoint = torch_utils.CheckpointTool(ckptdir=ckptdir)
  myargs.checkpoint = checkpoint
  myargs.checkpoint_dict = collections.OrderedDict()


def setup_args_and_myargs(args, myargs, start_tb=True, **kwargs):
  setup_outdir(args=args, resume_root=args.resume_root, resume=args.resume,
               **kwargs)
  setup_dirs_and_files(args=args, **kwargs)
  setup_logger_and_redirect_stdout(args.logfile, myargs)
  get_git_hash()

  myargs.textlogger = logging_utils.TextLogger(
    log_root=args.textlogdir, reinitialize=(not args.resume),
    logstyle='%10.6f')

  logger = logging.getLogger(__name__)
  logger.info("The outdir: \n\t{}".format(args.abs_outdir))
  logger.info("The args: \n{}".format(pprint.pformat(args)))

  setup_config(
    config_file=args.config, saved_config_file=args.configfile, overwrite_opts=args.overwrite_opts,
    args=args, myargs=myargs)
  setup_tensorboardX(tbdir=args.tbdir, args=args, config=myargs.config,
                     myargs=myargs, start_tb=start_tb)

  modelarts_utils.modelarts_setup(args, myargs)

  setup_checkpoint(ckptdir=args.ckptdir, myargs=myargs)

  args = EasyDict(args)
  myargs.config = EasyDict(myargs.config)
  return args, myargs


def parse_args_and_setup_myargs(argv_str=None, run_script=None, start_tb=False):
  """
  Usage:

  :return:
  """
  if 'TIME_STR' not in os.environ:
    os.environ['TIME_STR'] = '0' if is_debugging() else '1'
  if argv_str and not isinstance(argv_str, list):
    argv_str = argv_str.split()
  # parse args
  parser = args_parser.build_parser()
  unparsed_argv = []
  if argv_str:
    print('\npython \t%s \\\n  '%run_script + ' \\\n  '.join(argv_str))
    args, unparsed_argv = parser.parse_known_args(args=argv_str)
  else:
    args = parser.parse_args()
  args = config_utils.DotDict(vars(args))

  # setup args and myargs
  myargs = argparse.Namespace()
  args, myargs = setup_args_and_myargs(
    args=args, myargs=myargs, start_tb=start_tb)
  if argv_str:
    myargs.logger.info('\npython \t%s \\\n  '%run_script + ' \\\n  '.join(argv_str))
  return args, myargs, unparsed_argv


def config2args(config, args):
  logger = logging.getLogger(__name__)
  for k, v in config.items():
    if not k in args:
      logger.info('\n\t%s = %s \tnot in args' % (k, v))
    setattr(args, k, v)
  return args


def setup_myargs_for_multiple_processing(myargs):
  myargs.writer = None
  myargs.logger = None
  sys.stdout = myargs.stdout
  sys.stderr = myargs.stderr
  myargs.stdout = None
  myargs.stderr = None
  return myargs


# def update_config(super_config, config, overwrite_opts=True):
#   """
#
#   :param super_config:
#   :param config:
#   :param overwrite_opts: overwrite opts directly or overwrite its elements only
#   :return:
#   """
#   super_config = EasyDict(super_config)
#   ret_config = copy.deepcopy(super_config)
#   for k in config:
#     if k == 'opts' and not overwrite_opts and hasattr(super_config, k):
#       ret_config[k] = update_opts(super_config[k], config[k])
#     # merge dict element-wise
#     elif isinstance(config[k], dict) and hasattr(super_config, k):
#       if getattr(config[k], 'overwrite', False):
#         sub_config = copy.deepcopy(config[k])
#         sub_config.pop('overwrite')
#       else:
#         sub_config = update_config(super_config[k], config[k])
#       setattr(ret_config, k, sub_config)
#     else:
#       setattr(ret_config, k, config[k])
#   return ret_config


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


def config_inherit_from_base(config, configs, arg_base=[], overwrite_opts=False):
  base = getattr(config, 'base', [])
  if not isinstance(arg_base, list):
    arg_base = [arg_base]
  base += arg_base
  if not base:
    return EasyDict(config)

  super_config = EasyDict()
  for b in base:
    b_config = getattr(configs, b, {})
    b_config = config_inherit_from_base(b_config, configs, overwrite_opts=overwrite_opts)
    super_config = update_config(super_config, b_config, overwrite_opts=overwrite_opts)
  # update super_config by config
  super_config = update_config(super_config, config, overwrite_opts=overwrite_opts)
  return super_config


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