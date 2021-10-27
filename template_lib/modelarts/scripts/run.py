import pprint
import os
import sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))
os.chdir(proj_root)
sys.path.insert(0, proj_root)
def setup_package():
  packages = ['easydict', 'numpy', 'termcolor', '-I pyyaml', 'fvcore', 'matplotlib']
  command_template = f'{sys.executable} -m pip install %s'
  for pack in packages:
    command = command_template % pack
    print('=Installing %s'%pack)
    os.system(command)
  pass

setup_package()

from datetime import datetime
import logging
import multiprocessing
import shutil
from easydict import EasyDict
import time
import subprocess
import yaml
import sys
import unittest
import argparse
import random

from template_lib import utils
from template_lib.modelarts import modelarts_utils
from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, global_cfg, get_dict_str
from template_lib.v2.config_cfgnode.argparser import start_cmd_run

class Worker(multiprocessing.Process):
  def run(self):
    command = self._args[0]
    command = f"export PATH={os.path.dirname(sys.executable)}:$PATH && " + command
    print('%s'%command)
    # start_cmd_run(command)
    os.system(command)
    return


def modelarts_record_bash_command(args, myargs, command=None):
  try:
    import moxing as mox
    assert os.environ['DLS_TRAIN_URL']
    log_obs = os.environ['DLS_TRAIN_URL']
    command_file_obs = os.path.join(log_obs, 'commands.txt')
    command_file = os.path.join(args.outdir, 'commands.txt')
    with open(command_file, 'a') as f:
      if not command:
        f.write(args.outdir)
      else:
        f.write(command)
      f.write('\n')
    mox.file.copy(command_file, command_file_obs)

  except ModuleNotFoundError as e:
    myargs.logger.info("Don't use modelarts!")


def setup_log_obs_dir():
  """

  """
  try:
    import moxing as mox
    # Remove log_obs dir
    log_obs = f'{global_cfg.logdir}/{global_cfg.logsubdir}'
    if mox.file.exists(log_obs):
      mox.file.remove(log_obs, recursive=True)
    mox.file.make_dirs(log_obs)
  except:
    import traceback
    logger.info(traceback.format_exc())
    pass
  return

def main():
  logger = logging.getLogger('tl')

  modelarts_utils.setup_tl_outdir_obs(cfg=global_cfg)

  old_command = ''
  # Create bash_command.sh
  bash_file = os.path.join(global_cfg.tl_outdir, f'bash_{global_cfg.number}.sh')
  open(bash_file, 'w').close()
  config_file = f'{os.path.dirname(global_cfg.tl_saved_config_file)}/c_{global_cfg.number}.yaml'
  shutil.copy(global_cfg.tl_saved_config_file, config_file)
  global_cfg.tl_saved_config_file = config_file
  global_cfg.tl_saved_config_file_old = global_cfg.tl_saved_config_file + '.old'

  # copy outdir to outdir_obs, copy bash_file to outdir_obs
  modelarts_utils.modelarts_sync_results_dir(cfg=global_cfg, join=True)
  # disable moxing copy_parallel output
  # logger.disabled = True

  while True:
    try:
      try:
        import moxing as mox
        time.sleep(global_cfg.time_interval)
        # copy oudir_obs to outdir
        mox.file.copy_parallel(global_cfg.tl_outdir_obs, global_cfg.tl_outdir)
      except:
        if not os.path.exists(global_cfg.tl_saved_config_file):
          os.rename(global_cfg.tl_saved_config_file_old, global_cfg.tl_saved_config_file)
        if not os.path.exists(bash_file):
          open(bash_file, 'w').close()
        pass

      # parse command
      if not os.path.exists(bash_file) or not os.path.exists(global_cfg.tl_saved_config_file):
        continue
      shutil.copy(bash_file, os.curdir)
      try:
        with open(global_cfg.tl_saved_config_file, 'rt') as handle:
          config = yaml.load(handle)
          config = EasyDict(config)
        command = getattr(getattr(config, global_cfg.tl_command), 'command')
      except:
        logger.warning('Parse config.yaml error!')
        command = old_command

      # execute command
      if command != old_command:
        old_command = command
        if type(command) is list and command[0].startswith(('bash', )):
          p = Worker(name='Command worker', args=(command[0],))
          p.start()
        elif type(command) is list and len(command) == 1:
          if command[0] == 'exit':
            exit(0)
          command = list(map(str, command))
          # command = ' '.join(command)
          # print('===Execute: %s' % command)
          err_f = open(os.path.join(global_cfg.tl_outdir, 'err.txt'), 'w')
          try:
            cwd = os.getcwd()
            return_str = subprocess.check_output(
              command, encoding='utf-8', cwd=cwd, shell=True)
            print(return_str, file=err_f, flush=True)
          except subprocess.CalledProcessError as e:
            print("Oops!\n", e.output, "\noccured.",
                  file=err_f, flush=True)
            print(e.returncode, file=err_f, flush=True)
          err_f.close()
        elif type(command) is list and len(command) > 1:
          command = list(map(str, command))
          command = [command[0]]
          # command = ' '.join(command)
          print('===Execute: %s' % command)
          err_f = open(os.path.join(global_cfg.tl_outdir, 'err.txt'), 'w')
          try:
            cwd = os.getcwd()
            return_str = subprocess.check_output(
              command, encoding='utf-8', cwd=cwd, shell=True)
            print(return_str, file=err_f, flush=True)
          except subprocess.CalledProcessError as e:
            print("Oops!\n", e.output, "\noccured.",
                  file=err_f, flush=True)
            print(e.returncode, file=err_f, flush=True)
          err_f.close()
        logger.info('EE')

      # sync outdir to outdir_obs
      # del configfile in outdir
      os.rename(global_cfg.tl_saved_config_file, global_cfg.tl_saved_config_file_old)
      # del bash_file in outdir
      os.remove(bash_file)
      try:
        mox.file.copy_parallel(global_cfg.tl_outdir, global_cfg.tl_outdir_obs)
      except:
        pass

    except Exception as e:
      if str(e) == 'server is not set correctly':
        print(str(e))
      else:
        # modelarts_utils.modelarts_record_jobs(args, myargs, str_info='Exception!')
        import traceback
        logger.warning(traceback.format_exc())
      modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)

  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--number', type=int, default=1)
  tmp_args, _ = parser.parse_known_args()
  if bool(int(os.environ.get('TIME_STR', 1))):
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
  else:
    time_str = ''

  # setup sys.argv
  print(f"sys.argv in: ")
  pprint.pprint(sys.argv)
  argv = [sys.argv[0], ]
  for v in sys.argv[1:]:
    if '=' in v:
      name, value = v.split('=')
      if name == '--tl_opts':
        argv.append(name)
        argv.extend(value.split(' '))
      else:
        argv.extend([name, value])
    else:
      argv.append(v)
  sys.argv.clear()
  sys.argv.extend(argv)

  sys.argv[sys.argv.index('--tl_outdir') + 1] = f"{sys.argv[sys.argv.index('--tl_outdir') + 1]}-{time_str}_{tmp_args.number:02d}"
  shutil.rmtree(sys.argv[sys.argv.index('--tl_outdir') + 1], ignore_errors=True)

  print(f"sys.argv processed: ")
  pprint.pprint(sys.argv)
  parser = update_parser_defaults_from_yaml(parser=parser, use_cfg_as_args=True)
  logger = logging.getLogger('tl')

  args, _ = parser.parse_known_args()
  global_cfg.merge_from_dict(vars(args))
  print(get_dict_str(global_cfg))
  main()











