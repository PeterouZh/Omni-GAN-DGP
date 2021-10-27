import os
import subprocess
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils
from template_lib.nni import update_nni_config_file


class Testing_Image(unittest.TestCase):

  def test_imshow(self, debug=True):
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

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    import mmcv
    import numpy as np

    img_path = "template_lib/datasets/images/zebra_GT_target_origin.png"
    mmcv.imshow(img_path)

    # show image with bounding boxes
    img = np.random.rand(100, 100, 3)
    bboxes = np.array([[0, 0, 50, 50], [20, 20, 60, 60]])
    mmcv.imshow_bboxes(img, bboxes)

    pass

  def test_Image(self, debug=True):
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

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    import mmcv
    import numpy as np
    import matplotlib.pyplot as plt

    img_path = "template_lib/datasets/images/zebra_GT_target_origin.png"

    img = mmcv.imread(img_path)
    img_gray = mmcv.imread(img_path, flag='grayscale')
    img_ = mmcv.imread(img)  # nothing will happen, img_ = img
    mmcv.imwrite(img, f'{args.tl_outdir}/out.png')

    # mmcv.imshow(img)
    fig, axes = plt.subplots(2, 1)
    img = mmcv.bgr2rgb(img)
    axes[0].imshow(img)
    # plt.imshow(img)
    # plt.show()

    # ret = mmcv.imresize(img, (1000, 600), return_scale=True)
    # ret = mmcv.imrescale(img, (1000, 800))

    bboxes = np.array([10, 10, 100, 120])
    patch = mmcv.imcrop(img, bboxes)
    axes[1].imshow(patch)

    fig.show()
    pass

