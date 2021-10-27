import os
import sys
import unittest
import argparse
from template_lib import utils


class TestingRun(unittest.TestCase):

  def test_run(self, *tmp):
    """
    Usage:
        export ANSI_COLORS_DISABLED=1
        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=1
        export PYTHONPATH=./
        python template_lib/modelarts/scripts/run.py \
          --tl_config_file template_lib/modelarts/tests/configs/run.yaml \
          --tl_command run \
          --tl_outdir results/Run/run \
          --number 1

        # default image
        /bucket-8280/ZhouPeng/codes/Omni-GAN-ImageNet/template_lib/modelarts/scripts/run.py
          number = 3
          tl_outdir = results/Run/run
          tl_config_file = template_lib/modelarts/tests/configs/run.yaml
          tl_opts = root_obs s3://bucket-7001/ZhouPeng/
          tl_command = run

        # self defined image
        bash /home/work/run_train.sh python /home/work/user-job-dir/Omni-GAN-ImageNet/template_lib/modelarts/scripts/run.py --tl_outdir=results/Run/run --tl_config_file=/home/work/user-job-dir/Omni-GAN-ImageNet/template_lib/modelarts/tests/configs/run.yaml --tl_command=run --tl_opts=root_obs s3://bucket-7001/ZhouPeng/ --number=2
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # if 'TIME_STR' not in os.environ:
    #   os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    os.environ['TIME_STR'] = '0'
    if 'RESULTS_OBS' not in os.environ:
      os.environ['RESULTS_OBS'] = 's3://bucket-xx/ZhouPeng/results'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                    --tl_config_file template_lib/modelarts/tests/configs/run.yaml
                    --tl_command {command}
                    --tl_outdir {outdir}
                    """
    args = setup_outdir_and_yaml(argv_str)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
            python template_lib/modelarts/scripts/run.py
            {get_append_cmd_str(args)}
            --number 1
            """
    start_cmd_run(cmd_str)

    pass




