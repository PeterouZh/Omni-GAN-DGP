import os
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils


class TestingD2Template(unittest.TestCase):

  def test_train(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=5
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=./pt_darts_lib:.
        python 	pt_darts_lib/search.py \
          --config exp/configs/pt_darts.yaml \
          --command darts_search \
          --outdir results/PTDARTS/darts_search

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'

    argv_str = f"""
                --config template_lib/d2template/configs/template.yaml
                --command {command}
                --outdir {outdir}
                """
    from template_lib.d2template.scripts.run import run
    run(argv_str)

