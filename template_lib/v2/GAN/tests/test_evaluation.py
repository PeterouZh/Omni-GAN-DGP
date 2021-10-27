import os
import sys
import unittest
import argparse

from template_lib import utils
from template_lib.v2.config import get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, \
  start_cmd_run


class TestingTFFIDISScore(unittest.TestCase):

  def test_calculate_fid_stat_CIFAR10(self, debug=True):
    """
    Usage:
        export LD_LIBRARY_PATH=$HOME/.keras/envs/cuda-10.0/lib64:$HOME/.keras/envs/cudnn-10.0-linux-x64-v7.6.5.32/lib64
        export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64

        export CUDA_VISIBLE_DEVICES=2
        export TIME_STR=0
        export PYTHONPATH=./
        python -c "from template_lib.v2.tests.test_GAN import TestingTFFIDISScore;\
          TestingTFFIDISScore().test_calculate_fid_stat_CIFAR10(debug=False)"

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
                --tl_config_file template_lib/v2/GAN/configs/TFFIDISScore.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.0/lib64'

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
        python template_lib/v2/GAN/scripts/calculate_fid_stat_CIFAR.py
        {get_append_cmd_str(args)}
        --tl_opts OUTPUT_DIR {args.tl_outdir}/detectron2
        """
    if debug:
      cmd_str += f"""
      
      """
    start_cmd_run(cmd_str)
    pass

  def test_calculate_fid_stat_CIFAR100(self, debug=True):
    """
    Usage:
        export LD_LIBRARY_PATH=$HOME/.keras/envs/cuda-10.0/lib64:$HOME/.keras/envs/cudnn-10.0-linux-x64-v7.6.5.32/lib64
        export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64

        export CUDA_VISIBLE_DEVICES=2
        export TIME_STR=0
        export PYTHONPATH=./
        python -c "from template_lib.v2.tests.test_GAN import TestingTFFIDISScore;\
          TestingTFFIDISScore().test_calculate_fid_stat_CIFAR100(debug=False)"

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
                --tl_config_file template_lib/v2/GAN/configs/TFFIDISScore.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.0/lib64'

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
        python template_lib/v2/GAN/scripts/calculate_fid_stat_CIFAR.py
        {get_append_cmd_str(args)}
        --tl_opts OUTPUT_DIR {args.tl_outdir}/detectron2
        """
    if debug:
      cmd_str += f"""

      """
    start_cmd_run(cmd_str)
    pass

  def test_case_calculate_fid_stat_CIFAR10_ddp(self):
    """
    Usage:
        export PYTHONWARNINGS=ignore
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export master_port=8887
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python -c "from template_lib.v2.tests.test_GAN import TestingTFFIDISScore;\
          TestingTFFIDISScore().test_case_calculate_fid_stat_CIFAR10_ddp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    nproc_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    master_port = os.environ.get('master_port', 8888)
    cmd_str = f"""
        python -m torch.distributed.launch --nproc_per_node={nproc_per_node} --master_port={master_port} 
          template_lib/v2/GAN/evaluation/tf_FID_IS_score.py 
            --run_func TFFIDISScore.test_case_calculate_fid_stat_CIFAR10
        """
    cmd_str += get_append_cmd_str(args)
    start_cmd_run(cmd_str)
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

    from template_lib.v2.GAN.evaluation.tf_FID_IS_score import TFFIDISScore
    TFFIDISScore.test_case_evaluate_FID_IS()
    pass

  def test_case_evaluate_FID_IS_ddp(self):
    """
    Usage:
        export PYTHONWARNINGS=ignore
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python -c "from template_lib.v2.tests.test_GAN import TestingTFFIDISScore;\
          TestingTFFIDISScore().test_case_evaluate_FID_IS_ddp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    nproc_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
        python -m torch.distributed.launch --nproc_per_node={nproc_per_node} --master_port=8888 
          template_lib/v2/GAN/evaluation/tf_FID_IS_score.py 
            --run_func TFFIDISScore.test_case_evaluate_FID_IS
        """
    cmd_str += get_append_cmd_str(args)
    start_cmd_run(cmd_str)
    pass


