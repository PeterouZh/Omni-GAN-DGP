import os
import subprocess
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils
from template_lib.nni import update_nni_config_file


class Testing_CIFAR10(unittest.TestCase):

  def test_save_cifar10_images(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=deep-generative-prior-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=5
        export TIME_STR=1
        export PYTHONPATH=./:DGP_lib:BigGAN_Pytorch_lib
        python -c "from template_lib.proj.cifar.tests.test_cifar_dataset import Testing_CIFAR10;\
          Testing_CIFAR10().test_save_cifar10_images(debug=False)"


    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import tqdm
    import torchvision

    saved_dir = 'datasets/cifar10/cifar10_images'

    cifar10 = torchvision.datasets.CIFAR10(root='datasets/cifar10', train=True)
    for idx in tqdm.tqdm(range(len(cifar10))):
      img_pil, label_id = cifar10[idx]
      subdir = f"{saved_dir}/{cifar10.classes[label_id]}_{label_id:02d}"
      os.makedirs(subdir, exist_ok=True)
      img_pil.save(f"{subdir}/{idx:05d}.png")
    pass



