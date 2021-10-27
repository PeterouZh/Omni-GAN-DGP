import os
import subprocess
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils
from template_lib.nni import update_nni_config_file


class Testing_omni_loss_OmniLoss(unittest.TestCase):

  def test_omni_GAN(self, debug=True):
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
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file template_lib/v2/GAN/loss/configs/omni_loss_OmniLoss.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
    import torch
    from template_lib.v2.GAN.loss import build_GAN_loss

    omni_loss = build_GAN_loss(cfg)

    b, nc = 32, 100
    out_dim = nc + 2

    pred = torch.rand(b, out_dim).cuda().requires_grad_()
    y = torch.randint(0, nc, (b,)).cuda()

    D_loss_real, logits_pos, logits_neg = omni_loss(pred=pred, positive=(y, out_dim - 2), return_logits=True)

    D_loss_fake = omni_loss(pred=pred, positive=(out_dim - 1, ))

    G_loss, logits_pos, logits_neg = omni_loss(pred=pred, positive=(y, out_dim - 2), return_logits=True)

    pass
