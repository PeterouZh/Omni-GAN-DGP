import os
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils


class TestingUnit(unittest.TestCase):

  def test_func(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=./submodule
        python detectron2_exp/tests/run_detectron2.py \
          --config ./detectron2_exp/configs/detectron2.yaml \
          --command train_scratch_mask_rcnn_dense_R_50_FPN_3x_gn_2gpu \
          --outdir results/Detectron2/train_scratch_mask_rcnn_dense_R_50_FPN_3x_gn_2gpu

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') else self.__class__.__name__
    class_name = class_name.strip('_')
    outdir = f'results/{class_name}/{command}'

    argv_str = f"""
                --config domain_adaptive_faster_rcnn_pytorch_exp/configs/domain_faster_rcnn.yaml
                --command {command}
                --outdir {outdir}
                --overwrite_opts False
                """
    import run
    run(argv_str)

  def test_run(self):
    """

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'

    from datetime import datetime
    TIME_STR = bool(int(os.getenv('TIME_STR', 0)))
    time_str = datetime.now().strftime("%Y%m%d-%H_%M_%S_%f")[:-3]
    outdir = outdir if not TIME_STR else (outdir + '_' + time_str)
    print(outdir)

    import collections, shutil
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

  def test_run_with_config(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=./submodule
        python detectron2_exp/tests/run_detectron2.py \
          --config ./detectron2_exp/configs/detectron2.yaml \
          --command train_scratch_mask_rcnn_dense_R_50_FPN_3x_gn_2gpu \
          --outdir results/Detectron2/train_scratch_mask_rcnn_dense_R_50_FPN_3x_gn_2gpu

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'

    argv_str = f"""
                --config domain_adaptive_faster_rcnn_pytorch_exp/configs/domain_faster_rcnn.yaml
                --command {command}
                --outdir {outdir}
                """
    from template_lib.test.main import run
    args, myargs = run(argv_str)

    pass


def run(argv_str=None, return_args=False):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  from template_lib.utils import parser_set_default
  from template_lib.utils.modelarts_utils import prepare_dataset
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  if return_args:
    return args1, myargs

  if hasattr(myargs.config, 'datasets'):
    prepare_dataset(myargs.config.datasets, cfg=myargs.config)

  # main(myargs)
  return args1, myargs

if __name__ == '__main__':
  run()
  from template_lib.examples import test_bash
  test_bash.TestingUnit().test_resnet(gpu=os.environ['CUDA_VISIBLE_DEVICES'])