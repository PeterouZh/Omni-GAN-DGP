import os
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils


class TestingTFFIDISScore(unittest.TestCase):

  def test_case_calculate_fid_stat_CIFAR10(self):
    """
    export  LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32/lib64
    python -c "from template_lib.gans.tests.test_evaluate import TestingTFFIDISScore;\
      TestingTFFIDISScore().test_case_calculate_fid_stat_CIFAR10()"
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

    from template_lib.gans.evaluation.tf_FID_IS_score import TFFIDISScore
    TFFIDISScore.test_case_calculate_fid_stat_CIFAR10()
    pass

  def test_case_evaluate_FID_IS(self):
    """
    export  LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32/lib64
    python -c "from template_lib.gans.tests.test_evaluate import TestingTFFIDISScore;\
      TestingTFFIDISScore().test_case_evaluate_FID_IS()"
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

    from template_lib.gans.evaluation.tf_FID_IS_score import TFFIDISScore
    TFFIDISScore.test_case_evaluate_FID_IS()
    pass
