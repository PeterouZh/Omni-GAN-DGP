import logging
import argparse
import sys
import unittest
import os, time
import multiprocessing
import shutil
import logging

# from . import config


class CopyObsProcessing(multiprocessing.Process):
  """
    worker = CopyObsProcessing(args=(s, d, copytree))
    worker.start()
    worker.join()
  """
  def run(self):
    logger = logging.getLogger()
    try:
      import moxing as mox
      s, d, copytree = self._args
      logger.info('====== Starting %s, Copying %s to\n %s' % (self.name, s, d))
      start_time = time.time()
      if copytree:
        logger = logging.getLogger()
        logger.disabled = True
        mox.file.copy_parallel(s, d)
        logger = logging.getLogger()
        logger.disabled = False
      else:
        mox.file.copy(s, d)
      elapsed_time = time.time() - start_time
      time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
      logger.info('End %s, elapsed time: %s'%(self.name, time_str))
    except:
      import traceback
      logger.info(traceback.format_exc())
      # if str(e) == 'server is not set correctly':
      #   print(str(e))
      # else:
      #   print('Exception %s' % (self.name))
    return


def copy_obs(s, d, copytree=False, join=False):
  worker = CopyObsProcessing(args=(s, d, copytree))
  worker.start()
  if join:
    worker.join()
  return


def modelarts_sync_results_dir(cfg, join=False, is_main_process=True):
  if not is_main_process:
    return
  logger = logging.getLogger('tl')
  try:
    logger.info(f'\n======Uploading results dir======')
    copy_obs(cfg.tl_outdir, cfg.tl_outdir_obs, copytree=True, join=join)
    logger.info(f'\n======End uploading results dir======')
  except:
    import traceback
    logger.info(traceback.format_exc())
    logger.info(f'\n======End uploading results dir======')
  return


def setup_tl_outdir_obs(cfg):
  """
  Setup tl_outdir_obs
  Backup code.zip to tl_outdir
  """
  from template_lib.utils import unzip_file

  logger = logging.getLogger('tl')
  try:
    import moxing as mox
    # modelarts_record_jobs(args, myargs)
    logger.info('root_obs: %s' % cfg.root_obs)
    proj_dir = os.path.basename(os.path.abspath(os.path.curdir))
    assert cfg.tl_outdir.startswith('results/')
    cfg.tl_outdir_obs = os.path.join(cfg.root_obs, 'results', proj_dir, cfg.tl_outdir[8:])
    logger.info(f"tl_outdir_obs: {cfg.tl_outdir_obs}")

    zip_code_file = cfg.get('zip_code_file', "code.zip")
    if os.path.exists(zip_code_file):
      unzip_file(zip_file=zip_code_file, dst_dir=f'{cfg.tl_outdir}/code_bak')
  except ModuleNotFoundError:
    import traceback
    traceback.print_exc()
  return


def modelarts_resume(args):
  try:
    import moxing as mox
    assert os.environ['RESULTS_OBS']
    args.results_obs = os.environ['RESULTS_OBS']

    exp_name = os.path.relpath(
      os.path.normpath(args.resume_root), './results')
    resume_root_obs = os.path.join(args.results_obs, exp_name)
    assert mox.file.exists(resume_root_obs)
    print('Copying %s \n to %s'%(resume_root_obs, args.resume_root))
    mox.file.copy_parallel(resume_root_obs, args.resume_root)
  except ModuleNotFoundError as e:
    print("Resume, don't use modelarts!")
  return


def modelarts_finetune(args, finetune_path):
  try:
    import moxing as mox
    args.finetune_path = finetune_path
    assert args.finetune_path.startswith('results/')
    args.finetune_path_obs = os.path.join(
      args.results_obs, args.finetune_path[8:])

    assert mox.file.exists(args.finetune_path_obs)
    print('Copying %s \n to %s'%(args.finetune_path_obs, args.finetune_path))
    mox.file.copy_parallel(args.finetune_path_obs, args.finetune_path)
  except:
    print("Finetune load failed!")
  return





def modelarts_record_jobs(args, myargs, end=False, str_info=''):
  try:
    import moxing as mox
    assert os.environ['DLS_TRAIN_URL']
    log_obs = os.environ['DLS_TRAIN_URL']
    jobs_file_obs = os.path.join(log_obs, 'jobs.txt')
    jobs_file = os.path.join(args.outdir, 'jobs.txt')
    if mox.file.exists(jobs_file_obs):
      mox.file.copy(jobs_file_obs, jobs_file)

    with open(jobs_file, 'a') as f:
      if not end:
        f.write(str_info + '\t' + args.outdir + ' ' + os.environ['PORT'])
      else:
        f.write('End: \t' + args.outdir + ' ' + os.environ['PORT'])
      f.write('\n')
    mox.file.copy(jobs_file, jobs_file_obs)

  except:
    import traceback
    logging.getLogger(__name__).info('\n\t%s', traceback.format_exc())


def modelarts_catch_exception(func):
  def inter_func(**kwargs):
    """

    :param kwargs: args, myargs
    :return:
    """
    try:
      func(**kwargs)
    except:
      args = getattr(kwargs['myargs'], 'args', kwargs['args'])
      modelarts_record_jobs(args=args, myargs=kwargs['myargs'],
                            str_info='Exception!')
      import traceback
      print(traceback.format_exc(), flush=True)
      modelarts_sync_results(args=args, myargs=kwargs['myargs'],
                             join=True)
  return inter_func


def start_process(func, args, myargs, loop=10):
  for idx in range(loop):
    start_tb = True if idx == 0 else False
    args, myargs = config.setup_args_and_myargs(
      args=args, myargs=myargs, start_tb=start_tb)
    func(args=args, myargs=myargs)
    if os.environ['TIME_STR'] == '0':
      return
    if idx > 0:
      shutil.rmtree(args.outdir, ignore_errors=True)


def modelarts_copy_data(datapath_obs, datapath, overwrite=False, download=True, unzip=False):
  print(f'=== {"Downloading" if download else "Uploading"} dataset ===')
  try:
    import moxing as mox
    assert datapath_obs.startswith('s3://')

    logger = logging.getLogger()
    datapath = os.path.expanduser(datapath)

    if download:
      if not mox.file.exists(datapath_obs):
        assert 0, datapath_obs

      if not overwrite and os.path.exists(datapath):
        print('Skip copying [%s] \n to [%s]' % (datapath_obs, datapath))
        return

      if mox.file.is_directory(datapath_obs):
        # disable info output
        logger.disabled = True
        # dir
        print('Downloading dir [%s] \n to [%s]' % (datapath_obs, os.path.abspath(datapath)))
        mox.file.copy_parallel(datapath_obs, datapath)
      else:
        # file
        print('Downloading file [%s] \n to [%s]' % (datapath_obs, os.path.abspath(datapath)))
        mox.file.copy(datapath_obs, datapath)
        if unzip:
          from template_lib.utils import unzip_file
          print('Unzipping file [%s] \n to [%s]' % (os.path.abspath(datapath), os.path.dirname(datapath)))
          unzip_file(zip_file=datapath, dst_dir=os.path.dirname(datapath))
      print('End downloading [%s] \n to [%s]' % (datapath_obs, os.path.abspath(datapath)))
    else:
      # print('=== Uploading dataset ===')
      if os.path.isdir(datapath):
        # disable info output
        logger.disabled = True
        # dir
        print('Uploading dir [%s] \n to [%s]' % (datapath, datapath_obs))
        mox.file.copy_parallel(datapath, datapath_obs)
      else:
        # file
        print('Uploading file [%s] \n to [%s]' % (datapath, datapath_obs))
        assert datapath_obs.endswith(os.path.basename(datapath))
        mox.file.copy(datapath, datapath_obs)
      print('End uploading [%s] \n to [%s]' % (datapath, datapath_obs))
  except ModuleNotFoundError:
    logger = logging.getLogger('tl')
    logger.info('\n\tIgnore datapath: %s' % datapath_obs)
    pass
  except:
    logger = logging.getLogger('tl')
    import traceback
    logger.info('\n%s', traceback.format_exc())
    logger.info('\n\tIgnore datapath: %s' % datapath_obs)
  finally:
    logger = logging.getLogger()
    logger.disabled = False


def prepare_dataset(modelarts_datasets, global_cfg=None, download=True):
  """
    modelarts_download:
      dataset_root:
        datapath_obs: 'root_obs/keras/coco'
        datapath: './datasets/coco'
        overwrite: false
        eval: true
        unzip: false
  :return:
  """
  for k, v in modelarts_datasets.items():
    if getattr(v, 'eval', False):
      v.datapath_obs = eval("f'{}'".format(v.datapath_obs))
      v.datapath = eval("f'{}'".format(v.datapath))
      v.pop('eval')
    if isinstance(v, dict):
      modelarts_copy_data(download=download, **v)


class TestingUnit(unittest.TestCase):

  def test_modelarts_copy_data(self):
    """
    Usage:
        exp_name=wgan-pytorch0
        export root_obs=s3://bucket-cv-competition/ZhouPeng
        mkdir -p /cache/.keras/ && rm -rf $HOME/.keras && ln -s /cache/.keras $HOME/.keras
        export RESULTS_OBS=$root_obs/results/$exp_name
        python /home/work/user-job-dir/code/copy_tool.py \
          -s $root_obs/code/$exp_name \
          -d /cache/code/$exp_name -t copytree
        ln -s /cache/code/$exp_name /cache/code/template_lib
        cd /cache/code/$exp_name/

        export CUDA_VISIBLE_DEVICES=0
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../
        python -c "from utils import modelarts_utils; \
          modelarts_utils.TestingUnit().test_modelarts_copy_data()"
    :return:
    """
    import template_lib.utils as utils
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    outdir = os.path.join('results', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/config.yaml 
            --command test_command
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

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(
      args=args, myargs=myargs, start_tb=False)

    modelarts_sync_results(args, myargs, join=True)

    datapath_obs = 's3://bucket-cv-competition/ZhouPeng/keras/cifar10'
    datapath = '~/.keras/cifar10'
    modelarts_copy_data(
      datapath_obs=datapath_obs, datapath=datapath, overwrite=True)

    datapath_obs = 's3://bucket-cv-competition/ZhouPeng/keras/cifar10/cifar10_inception_moments.npz'
    datapath = '~/.keras/cifar10_inception_moments.npz'
    modelarts_copy_data(
      datapath_obs=datapath_obs, datapath=datapath, overwrite=False)

    modelarts_sync_results(args, myargs, join=True, end=True)
    input('End %s' % outdir)
    return

  def test_modelarts_time(self):
    """
    Usage:
        exp_name=template_lib
        export root_obs=s3://bucket-1893/ZhouPeng
        mkdir -p /cache/.keras/ && rm -rf $HOME/.keras && ln -s /cache/.keras $HOME/.keras
        export RESULTS_OBS=$root_obs/results/$exp_name
        python /home/work/user-job-dir/code/copy_tool.py \
          -s $root_obs/code/$exp_name \
          -d /cache/code/$exp_name -t copytree
        cd /cache/code/$exp_name/

        export CUDA_VISIBLE_DEVICES=0
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../
        python -c "from utils import modelarts_utils; \
          modelarts_utils.TestingUnit().test_modelarts_time()"
    :return:
    """
    import template_lib.utils as utils
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    outdir = os.path.join('results', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/config.yaml 
            --command test_command
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

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(
      args=args, myargs=myargs, start_tb=False)

    modelarts_sync_results(args, myargs, join=True)

    modelarts_sync_results(args, myargs, join=True, end=True)
    input('End %s' % outdir)
    return

