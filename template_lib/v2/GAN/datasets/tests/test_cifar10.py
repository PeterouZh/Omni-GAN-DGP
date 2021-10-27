import os
import subprocess
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils
from template_lib.nni import update_nni_config_file


class Testing_cifar10(unittest.TestCase):

  def test_DDPDataLoaderCIFAR10(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file template_lib/v2/GAN/datasets/configs/cifar10.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
        python -m torch.distributed.launch --nproc_per_node={n_gpus} --master_port=8888 
        template_lib/v2/GAN/datasets/scripts/cifar10.py
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

