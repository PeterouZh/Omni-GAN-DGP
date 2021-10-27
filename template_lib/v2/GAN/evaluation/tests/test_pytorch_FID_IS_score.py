import os
import sys
import unittest
import argparse

from template_lib import utils
from template_lib.v2.config_cfgnode.argparser import get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, \
  start_cmd_run


class TestingTorchFIDISScore(unittest.TestCase):

  def test_ddp_calculate_fid_stat_CIFAR10(self, debug=True):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=1
        export master_port=8887
        export TIME_STR=0
        export PYTHONPATH=./
        python -c "from template_lib.v2.GAN.evaluation.tests.test_pytorch_FID_IS_score import TestingTorchFIDISScore;\
          TestingTorchFIDISScore().test_ddp_calculate_fid_stat_CIFAR10(debug=False)" \
          --tl_opts build_dataloader.shuffle False save_fid_stat False

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file template_lib/v2/GAN/evaluation/configs/TorchFIDISScore.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    nproc_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    master_port = os.environ.get('master_port', 8887)
    cmd_str = f"""
        python -m torch.distributed.launch --nproc_per_node={nproc_per_node} --master_port={master_port} 
          template_lib/v2/GAN/evaluation/scripts/torch_calculate_fid_stat_CIFAR.py
          {get_append_cmd_str(args)}
        """

    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts 
                  """
    else:
      cmd_str += f"""
                  --tl_opts {tl_opts}
                  """
    start_cmd_run(cmd_str)
    pass

  def test_ddp_calculate_fid_stat_CIFAR100(self, debug=True):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=1,4
        export master_port=8887
        export TIME_STR=0
        export PYTHONPATH=./
        python -c "from template_lib.v2.GAN.evaluation.tests.test_pytorch_FID_IS_score import TestingTorchFIDISScore;\
          TestingTorchFIDISScore().test_ddp_calculate_fid_stat_CIFAR100(debug=False)" \
          --tl_opts build_dataloader.shuffle False save_fid_stat False

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file template_lib/v2/GAN/evaluation/configs/TorchFIDISScore.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    nproc_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    master_port = os.environ.get('master_port', 8887)
    cmd_str = f"""
        python -m torch.distributed.launch --nproc_per_node={nproc_per_node} --master_port={master_port} 
          template_lib/v2/GAN/evaluation/scripts/torch_calculate_fid_stat_CIFAR.py
          {get_append_cmd_str(args)}
        """

    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts 
                  """
    else:
      cmd_str += f"""
                  --tl_opts {tl_opts}
                  """
    start_cmd_run(cmd_str)
    pass

  def test_case_evaluate_FID_IS(self):
    """
    Usage:
        export PYTHONWARNINGS=ignore
        export CUDA_VISIBLE_DEVICES=0,7
        export master_port=8887
        export TIME_STR=1
        export PYTHONPATH=./
        python -c "from template_lib.v2.tests.test_GAN import TestingTorchFIDISScore;\
          TestingTorchFIDISScore().test_case_evaluate_FID_IS()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,7'
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
    master_port = os.environ.get('master_port', 8887)
    cmd_str = f"""
        python -m torch.distributed.launch --nproc_per_node={nproc_per_node} --master_port={master_port} 
          template_lib/v2/GAN/evaluation/pytorch_FID_IS_score.py 
            --run_func PyTorchFIDISScore.test_case_evaluate_FID_IS
        """
    cmd_str += get_append_cmd_str(args)
    start_cmd_run(cmd_str)
    pass
