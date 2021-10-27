import shutil
from easydict import EasyDict
import multiprocessing
import time
import subprocess
import yaml
import os
import sys
import unittest
import argparse
import random

from template_lib import utils


class Worker(multiprocessing.Process):
  def run(self):
    command = self._args[0]
    print('%s'%command)
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

  :return:  kwargs['log_number']
  """
  kwargs = {}
  try:
    import moxing as mox
    # Remove log_obs dir
    log_obs = os.environ['DLS_TRAIN_URL']
    if mox.file.exists(log_obs):
      mox.file.remove(log_obs, recursive=True)
    mox.file.make_dirs(log_obs)
    # Get log dir number
    log_number = log_obs.strip('/').split('/')[-1]
    if not log_number.isdigit():
      print('DLS_TRAIN_URL does not end with digital, ignore.')
    else:
      kwargs['log_number'] = log_number
      kwargs['add_number_to_configfile'] = True
  except:
    pass
  return kwargs


class TestingUnit(unittest.TestCase):

  def test_bash(self, *tmp):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0
        export PORT=6001
        export TIME_STR=1
        export DLS_TRAIN_URL=/tmp/logs/1
        export RESULTS_OBS=s3://bucket-xx/ZhouPeng/results
        export PYTHONPATH=../..
        python -c "import test_bash; \
          test_bash.TestingUnit().test_bash($PORT)"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6001'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    if 'DLS_TRAIN_URL' not in os.environ:
      os.environ['DLS_TRAIN_URL'] = '/tmp/logs/1'
    if 'RESULTS_OBS' not in os.environ:
      os.environ['RESULTS_OBS'] = 's3://bucket-xx/ZhouPeng/results'

    from template_lib.utils import modelarts_utils

    print('DLS_TRAIN_URL: %s'%os.environ['DLS_TRAIN_URL'])
    print('RESULTS_OBS: %s'%os.environ['RESULTS_OBS'])
    # func name
    outdir = os.path.join('results', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ../configs/virtual_terminal.yaml 
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    kwargs = setup_log_obs_dir()
    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(
      args=args, myargs=myargs, start_tb=False, **kwargs)


    old_command = ''
    # Create bash_command.sh
    bash_file = os.path.join(args.outdir, 'bash_command.sh')
    open(bash_file, 'w').close()
    cwd = os.getcwd()
    args.configfile_old = args.configfile + '.old'
    # copy outdir to outdir_obs, copy bash_file to outdir_obs
    myargs.logger.info('outdir_obs: %s', args.outdir_obs)
    modelarts_utils.modelarts_sync_results(args, myargs, join=True)
    
    # disable moxing copy_parallel output
    import logging
    logger = logging.getLogger()
    logger.disabled = True
    
    while True:
      try:
        try:
          import moxing as mox
          time.sleep(0.8)
          # copy oudir_obs to outdir
          mox.file.copy_parallel(args.outdir_obs, args.outdir)
        except:
          if not os.path.exists(args.configfile):
            os.rename(args.configfile_old, args.configfile)
          if not os.path.exists(bash_file):
            open(bash_file, 'w').close()
          pass

        # parse command
        if not os.path.exists(bash_file) or not os.path.exists(args.configfile):
          continue
        shutil.copy(bash_file, cwd)
        try:
          with open(args.configfile, 'rt') as handle:
            config = yaml.load(handle)
            config = EasyDict(config)
          command = config.command
        except:
          logger = logging.getLogger(__name__)
          logger.warning('Parse config.yaml error!')
          command = old_command

        # execute command
        if command != old_command:
          old_command = command
          if type(command) is list and command[0].startswith('bash'):
            p = Worker(name='Command worker', args=(command[0], ))
            p.start()
          elif type(command) is list and len(command) == 1:
            command = list(map(str, command))
            # command = ' '.join(command)
            # print('===Execute: %s' % command)
            err_f = open(os.path.join(args.outdir, 'err.txt'), 'w')
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
            err_f = open(os.path.join(args.outdir, 'err.txt'), 'w')
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
          myargs.logger.info('EE')

        # sync outdir to outdir_obs
        # del configfile in outdir
        os.rename(args.configfile, args.configfile_old)
        # del bash_file in outdir
        os.remove(bash_file)
        # copy jobs.txt from log_obs to outdir
        try:
          log_obs = os.environ['DLS_TRAIN_URL']
          jobs_file_obs = os.path.join(log_obs, 'jobs.txt')
          jobs_file = os.path.join(args.outdir, 'jobs.txt')
          if mox.file.exists(jobs_file_obs):
            mox.file.copy(jobs_file_obs, jobs_file)
        except:
          pass

        try:
          mox.file.copy_parallel(args.outdir, args.outdir_obs)
        except:
          pass

      except Exception as e:
        if str(e) == 'server is not set correctly':
          print(str(e))
        else:
          modelarts_utils.modelarts_record_jobs(args, myargs, str_info='Exception!')
          import traceback
          logger = logging.getLogger(__name__)
          logger.warning(traceback.format_exc())
        modelarts_utils.modelarts_sync_results(args, myargs, join=True)

  def test_resnet(self, gpu='0,1,2,3,4,5,6,7', *tmp, **kwargs):
    """

    export PYTHONPATH=./
    python -c "from template_lib.examples import test_bash; \
      test_bash.TestingUnit().test_resnet(gpu='1')"

    """
    import torch
    # from multiprocessing import Queue
    from template_lib.utils.test_resnet import TorchResnetWorker, run
    from torch.multiprocessing import Queue
    import torch.multiprocessing as mp

    gpu = str(gpu)
    determine_bs = True
    q = Queue()

    try:
      # determine max bs
      torch.multiprocessing.spawn(run, args=(0, gpu, determine_bs, q),
                                  nprocs=1, join=True, daemon=False)
    except:
      import traceback
      traceback.print_exc()
      pass
    time.sleep(2)
    # bs = q.get()
    with open('results/max_bs.txt', 'r') as f:
      bs = int(f.read())

    determine_bs = False
    while True:
      # p = TorchResnetWorker(name='Command worker', args=(bs, gpu, determine_bs, q))
      try:
        # determine max bs
        torch.multiprocessing.spawn(run, args=(bs, gpu, determine_bs, q),
                                    nprocs=1, join=True, daemon=False)
      except KeyboardInterrupt:
        exit(0)
      except:
        pass

      bs -= 1
      if bs < 1:
        break
