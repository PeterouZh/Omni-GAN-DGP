import sys
import argparse
import logging
import os
import time
import importlib

from easydict import EasyDict

from ..logger import get_logger, set_global_textlogger, TextLogger
from .config import TLCfgNode, set_global_cfg, global_cfg
from ..config.argparser import (get_command_and_outdir, _setup_outdir, get_dict_str, get_git_hash,
                                get_append_cmd_str, start_cmd_run, parser_set_defaults)


def build_parser(parser=None, append_local_rank=False):
  if not parser:
    parser = argparse.ArgumentParser()
  parser.add_argument('--tl_config_file', type=str, default='')
  parser.add_argument('--tl_command', type=str, default='')
  parser.add_argument('--tl_outdir', type=str, default='results/temp')
  parser.add_argument('--tl_opts', type=str, nargs='*', default=[])
  parser.add_argument('--tl_resume', action='store_true', default=False)
  parser.add_argument('--tl_resumedir', type=str, default='results/temp')
  parser.add_argument('--tl_debug', action='store_true', default=False)

  parser.add_argument('--tl_time_str', type=str, default='')
  if append_local_rank:
    parser.add_argument("--local_rank", type=int, default=0)
  return parser


def setup_config(config_file, args):
  """
  Load yaml and save command_cfg
  """
  cfg = TLCfgNode(new_allowed=True)
  cfg.merge_from_file(config_file)
  cfg.dump_to_file(args.tl_saved_config_file)

  command_cfg = TLCfgNode.load_yaml_with_command(config_file, command=args.tl_command)
  command_cfg.merge_from_list(args.tl_opts)

  if args.tl_resume:
    from deepdiff import DeepDiff

    resume_cfg_file = f"{os.path.dirname(args.tl_config_file_resume)}/config_resume.yaml"
    if not os.path.exists(resume_cfg_file):
      temp_cfg = TLCfgNode.load_yaml_file(args.tl_config_file_resume)
      temp_cfg.dump_to_file(resume_cfg_file)
    else:
      temp_cfg = TLCfgNode.load_yaml_file(resume_cfg_file)
    assert len(temp_cfg) == 1
    resume_cfg = list(temp_cfg.values())[0]

    logging.getLogger('tl').info(f"Updating resume_cfg: {args.tl_config_file_resume}")
    resume_cfg_clone = resume_cfg.clone()
    resume_cfg_clone.update(command_cfg)
    command_cfg =resume_cfg_clone

    ddiff = DeepDiff(resume_cfg, command_cfg)
    logging.getLogger('tl').info(f"diff between resume_cfg and cfg: \n{ddiff.pretty()}")


  command_cfg.dump_to_file_with_command(saved_file=args.tl_saved_config_command_file,
                                        command=args.tl_command)
  # saved_command_cfg = TLCfgNode(new_allowed=True)
  # setattr(saved_command_cfg, args.tl_command, command_cfg)
  # saved_command_cfg.dump_to_file(args.tl_saved_config_command_file)
  return cfg, command_cfg


def _register_modules(register_modules):
  for module in register_modules:
    if module not in sys.modules:
      imported_module = importlib.import_module(module)
    else:
      importlib.reload(sys.modules[module])
    logging.getLogger('tl').info(f"  Register {module}")
  pass

def register_modules():
  if "register_modules" in global_cfg:
    _register_modules(register_modules=global_cfg.register_modules)
  pass


def setup_outdir_and_yaml(argv_str=None, return_cfg=False, register_module=False):
  """
  Usage:

  :return:
  """
  argv_str = argv_str.split()
  parser = build_parser()
  args, unparsed_argv = parser.parse_known_args(args=argv_str)

  args = EasyDict(vars(args))
  _setup_outdir(args=args, resume=args.tl_resume)

  # get logger
  logger = get_logger(filename=args.tl_logfile, logger_names=['template_lib', 'tl'], stream=True)
  logger.info('\nargs:\n' + get_dict_str(args, use_pprint=False))

  # git
  # get_git_hash(logger)

  if args.tl_command.lower() == 'none':
    if return_cfg: return args, None
    else: return args

  # Load yaml
  _, command_cfg = setup_config(config_file=args.tl_config_file, args=args)
  logger.info(f"\nThe cfg: \n{command_cfg.dump()}")
  if return_cfg:
    global_cfg.merge_from_dict(command_cfg)
    # if register_module:
    #   register_modules()
    for k, v in vars(args).items():
      if k.startswith('tl_'):
        global_cfg.merge_from_dict({k: v})
    command_cfg.merge_from_dict(global_cfg)
    return args, command_cfg
  else:
    return args


def setup_logger_global_cfg_global_textlogger(args, tl_textdir, is_main_process=True):
  # log files
  tl_logfile = os.path.join(args.tl_outdir, "log.txt")
  if is_main_process:
    if len(logging.getLogger('tl').handlers) < 2:
      logger = get_logger(filename=tl_logfile)

  # textlogger
  if is_main_process:
    textlogger = TextLogger(log_root=tl_textdir)
    set_global_textlogger(textlogger=textlogger)

  # Load yaml file and update parser defaults
  if not args.tl_command.lower() == 'none':
    assert os.path.exists(args.tl_config_file)
    cfg = TLCfgNode.load_yaml_with_command(args.tl_config_file, args.tl_command)
    cfg.merge_from_list(args.tl_opts)

    cfg.tl_saved_config_file = f"{args.tl_outdir}/config_command.yaml"
    set_global_cfg(cfg)
    # logging.getLogger('tl').info("\nglobal_cfg: \n" + get_dict_str(global_cfg, use_pprint=False))
    logging.getLogger('tl').info("\nglobal_cfg: \n" + global_cfg.dump())
    time.sleep(0.1)
    if is_main_process:
      cfg.dump_to_file_with_command(saved_file=global_cfg.tl_saved_config_file, command=args.tl_command)
      # saved_command_cfg = TLCfgNode(new_allowed=True)
      # setattr(saved_command_cfg, args.tl_command, cfg)
      # saved_command_cfg.dump_to_file(global_cfg.tl_saved_config_file)
  else:
    cfg = TLCfgNode()
    cfg.merge_from_list(args.tl_opts, new_allowed=True)

    cfg.tl_saved_config_file = f"{args.tl_outdir}/config_command.yaml"
    set_global_cfg(cfg)
    logging.getLogger('tl').info("\nglobal_cfg: \n" + get_dict_str(global_cfg))
    if is_main_process:
      cfg.dump_to_file_with_command(saved_file=global_cfg.tl_saved_config_file, command=args.tl_command)
  return cfg, tl_logfile

def update_parser_defaults_from_yaml(parser, name='args', use_cfg_as_args=False,
                                     is_main_process=True, append_local_rank=False):
  parser = build_parser(parser, append_local_rank=append_local_rank)

  args, _ = parser.parse_known_args()
  tl_ckptdir = f'{args.tl_outdir}/ckptdir'
  tl_imgdir = f'{args.tl_outdir}/imgdir'
  tl_textdir = f'{args.tl_outdir}/textdir'

  os.makedirs(args.tl_outdir, exist_ok=True)
  os.makedirs(tl_ckptdir, exist_ok=True)
  os.makedirs(tl_imgdir, exist_ok=True)
  os.makedirs(tl_textdir, exist_ok=True)

  cfg, tl_logfile = setup_logger_global_cfg_global_textlogger(args, tl_textdir, is_main_process=is_main_process)

  if use_cfg_as_args:
    default_args = cfg
  else:
    default_args = cfg[name] if name in cfg else None

  parser_set_defaults(parser, cfg=default_args,
                      tl_imgdir=tl_imgdir, tl_ckptdir=tl_ckptdir, tl_textdir=tl_textdir,
                      tl_logfile=tl_logfile)
  logging.getLogger('tl').info('sys.argv: \n python \n' + ' \n'.join(sys.argv))
  args, _ = parser.parse_known_args()
  for k, v in vars(args).items():
    if k.startswith('tl_'):
      global_cfg.merge_from_dict({k: v})

  # if "register_modules" in global_cfg:
  #   for module in global_cfg.register_modules:
  #     importlib.import_module(module)
  #     logging.getLogger('tl').info(f"Register {module}...")

  return parser
