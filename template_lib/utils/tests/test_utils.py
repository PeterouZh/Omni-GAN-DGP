import os
import subprocess
import sys
import unittest
import argparse

from template_lib import utils


class Testing_utils_func(unittest.TestCase):

  def test_make_zip(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    from template_lib.utils.utils_func import make_zip, unzip_file

    make_zip(source_dir='./template_lib', output_filename='code.zip')
    unzip_file(zip_file='code.zip', dst_dir='/tmp/code')

    pass

  def test_MaxToKeep(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    from template_lib.utils.utils_func import MaxToKeep

    max2keep = MaxToKeep(max_to_keep=2)
    for i in range(10):
      file_path = f"{outdir}/{i:03d}.txt"
      with open(file_path, 'w'):
        max2keep.step(file_path)
    pass

    max2keep = MaxToKeep(max_to_keep=2)
    for i in range(10):
      file_path = f"{outdir}/{i:03d}"
      os.makedirs(file_path, exist_ok=True)
      max2keep.step(file_path)
    pass