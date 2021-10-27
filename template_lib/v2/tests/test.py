import os
import subprocess
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils
from template_lib.nni import update_nni_config_file


class Testing_v2(unittest.TestCase):

  def test_ddp(self):
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
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config import get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, \
      start_cmd_run

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/configs/styleganv2.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    nproc_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
        python -m torch.distributed.launch --nproc_per_node={nproc_per_node} --master_port=8888 
          exp/scripts/train.py 
          {get_append_cmd_str(args)}
        """
    start_cmd_run(cmd_str)
    pass

  def test_detectron2(self):
    """
    Usage:
        export ANSI_COLORS_DISABLED=1

        export CUDA_VISIBLE_DEVICES=2
        export TIME_STR=1
        export PYTHONPATH=.:./exp
        python -c "from exp.tests.test_nas_cgan import TestingPrepareData;\
          TestingPrepareData().test_calculate_fid_stat_CIFAR10()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config import get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, \
      start_cmd_run

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                    --tl_config_file exp/nas_cgan/config/prepare_data.yaml
                    --tl_command {command}
                    --tl_outdir {outdir}
                    """
    args = setup_outdir_and_yaml(argv_str)

    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
              python exp/scripts/train_net.py 
               {get_append_cmd_str(args)}
               --num-gpus {num_gpus}
              """
    start_cmd_run(cmd_str)
    pass

  def test_nni(self, use_nni=False):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./
        python -c "from exp.tests.test_nni import Testing_nni;\
          Testing_nni().test_mnist_pytorch(use_nni=False)"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config import get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, \
      start_cmd_run

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    if use_nni: outdir = outdir + '_nni'
    argv_str = f"""
                --tl_config_file exp/configs/nni.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    cmd_str = f"""
            python 
              exp/mnist_pytorch/mnist.py
              {get_append_cmd_str(args)}
            """

    # use_nni = True
    if use_nni:
      # update nni config file
      nni_config_file = "exp/mnist_pytorch/config.yml"
      python_command = ' '.join([s.strip(' ') for s in cmd_str.split('\n')]) + ' --tl_nni'
      update_nni_cfg_str = f"""
                              logDir: {os.path.abspath(args.tl_outdir)}/nni
                              trialConcurrency: {n_gpus}
                              trial:
                                command: {python_command}
                                codeDir: {os.path.abspath(os.path.curdir)}
                            """
      updated_config_file = update_nni_config_file(
        nni_config_file=nni_config_file, update_nni_cfg_str=update_nni_cfg_str)
      cmd_str = f"""
                 bash 
                 nnictl create --config {updated_config_file}
                 """
    start_cmd_run(cmd_str)
    pass

  def test_plot_lines_figure(self):
    """
    Usage:
        export LD_LIBRARY_PATH=~/anaconda3/envs/py36/lib/
        export TIME_STR=1
        export PYTHONPATH=./exp:./BigGAN_PyTorch_1_lib:./
        python -c "from exp.tests.test_BigGAN import TestingCIFAR10_BigGAN_v1;\
          TestingCIFAR10_BigGAN_v1().test_save_FID_cbn_index_012_figure()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config import get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, \
      start_cmd_run

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                    --tl_config_file exp/configs/BigGAN_v1.yaml
                    --tl_command {command}
                    --tl_outdir {outdir}
                    """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import matplotlib.pyplot as plt
    import numpy as np
    from template_lib.utils import colors_dict

    fig, ax = plt.subplots()

    ax.set_xticks(range(0, 600, 100))
    ax.tick_params(labelsize=cfg.fontsize.tick_fs)
    ax.set_xlabel(cfg.xlabel, fontsize=cfg.fontsize.xylabel_fs)
    ax.set_ylabel(cfg.ylabel, fontsize=cfg.fontsize.xylabel_fs)

    colors = list(colors_dict.values())
    # colors = [plt.cm.cool(i / float(num_plot - 1)) for i in range(num_plot)]

    ax.set(**cfg.properties)
    for idx, (_, data_dict) in enumerate(cfg.lines.items()):
      log_file = os.path.join(data_dict.result_dir, data_dict.sub_path)
      data = np.loadtxt(log_file, delimiter=':')

      if 'xlim' in cfg.properties:
        data_xlim = cfg.properties.xlim[-1]
        data = data[data[:, 0] <= data_xlim]

      if cfg.get_min_value:
        best_index = data[:, 1].argmin()
      else:
        best_index = data[:, 1].argmax()
      best_x = int(data[:, 0][best_index])
      best_y = data[:, 1][best_index]

      if cfg.add_auxi_label:
        data_dict.properties.label = f'x_{best_x}-y_{best_y:.3f}-' + getattr(data_dict.properties, 'label', '')
      ax.plot(data[:, 0], data[:, 1], color=colors[idx], **data_dict.properties)
      pass

    ax.legend(prop={'size': cfg.fontsize.legend_size})
    fig.show()
    saved_file = os.path.join(args.tl_outdir, cfg.saved_file)
    fig.savefig(saved_file, bbox_inches='tight', pad_inches=0.01)
    print(f'Save to {saved_file}')
    pass

# from template_lib.v2.config import update_parser_defaults_from_yaml
# update_parser_defaults_from_yaml(parser)


class Testing_v2_cfgnode(unittest.TestCase):

  def test_ddp(self, debug=True):
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
                --tl_config_file template_lib/v2/tests/configs/config.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
        python -m torch.distributed.launch --nproc_per_node={n_gpus} --master_port=8888 
        template_lib/v2/ddp/train.py
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
    # from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, global_cfg
    # from template_lib.modelarts import modelarts_utils
    # update_parser_defaults_from_yaml(parser)

    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass

  def test_save_early_collapse_on_cifar100(self):
    """
    Usage:
        export LD_LIBRARY_PATH=~/anaconda3/envs/py36/lib/
        export TIME_STR=1
        export PYTHONPATH=./exp:./BigGAN_PyTorch_1_lib:./
        python -c "from exp.tests.test_BigGAN import TestingCIFAR10_BigGAN_v1;\
          TestingCIFAR10_BigGAN_v1().test_save_FID_cbn_index_012_figure()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import get_command_and_outdir, setup_outdir_and_yaml, \
      get_append_cmd_str, start_cmd_run

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                    --tl_config_file exp/configs/Figures.yaml
                    --tl_command {command}
                    --tl_outdir {outdir}
                    """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    cfg_str = """
              xlabel: "Epoch"
              ylabel: "FID"
              fontsize:
                tick_fs: 14
                xylabel_fs: 20
                legend_size: 12
              clip_x:
                - 0
                - 24500096
            #  properties:
            #    xlim:
            #      - 0
            #      - 336
            #    ylim:
            #      - 0
            #      - 30
              get_min_value: true
              add_auxi_label: false
              lines:
                biggan_pd:
                  result_dir: "results/CIFAR10/train_cifar100_\
                  20201002-19_37_05_825"
                  sub_path: "textdir/evaltf.ma0.FID_tf.log"
                  properties:
                    label: "projection discriminator (BigGAN)"
                    marker: "."
              saved_file: "early_collapse_cifar100.pdf"
            """

    import matplotlib.pyplot as plt
    import numpy as np
    from template_lib.utils import colors_dict

    fig, ax = plt.subplots()
    # ax.set_xticks(range(0, 600, 100))
    ax.tick_params(labelsize=cfg.fontsize.tick_fs)
    ax.set_xlabel(cfg.xlabel, fontsize=cfg.fontsize.xylabel_fs)
    ax.set_ylabel(cfg.ylabel, fontsize=cfg.fontsize.xylabel_fs)

    colors = list(colors_dict.values())
    # colors = [plt.cm.cool(i / float(num_plot - 1)) for i in range(num_plot)]

    properties = cfg.get('properties', {})
    ax.set(**properties)
    for idx, (_, data_dict) in enumerate(cfg.lines.items()):
      log_file = os.path.join(data_dict.result_dir, data_dict.sub_path)
      data = np.loadtxt(log_file, delimiter=':')

      if 'clip_x' in cfg:
        data_xlim = cfg.clip_x[-1]
        data = data[data[:, 0] <= data_xlim]

      if cfg.get_min_value:
        best_index = data[:, 1].argmin()
      else:
        best_index = data[:, 1].argmax()
      best_x = int(data[:, 0][best_index])
      best_y = data[:, 1][best_index]

      if cfg.add_auxi_label:
        data_dict.properties.label = f'x_{best_x}-y_{best_y:.3f}-' + getattr(data_dict.properties, 'label', '')
      ax.plot(data[:, 0], data[:, 1], color=colors[idx], **data_dict.properties)
      pass

    ax.legend(prop={'size': cfg.fontsize.legend_size})
    fig.show()
    saved_file = os.path.join(args.tl_outdir, cfg.saved_file)
    fig.savefig(saved_file, bbox_inches='tight', pad_inches=0.01)
    print(f'Save to {saved_file}')
    pass

  def test_save_OmniGAN_ImageNet128_IS(self):
    """
    Usage:
        export TIME_STR=1
        export PYTHONPATH=./exp:./BigGAN_PyTorch_1_lib:./
        python -c "from exp.tests.test_BigGAN_v1 import Testing_Figures;\
          Testing_Figures().test_save_early_collapse_on_cifar100()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import get_command_and_outdir, setup_outdir_and_yaml, \
      get_append_cmd_str, start_cmd_run
    from template_lib.v2.matplot import set_font

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                    --tl_config_file exp/tests/configs/Figures.yaml
                    --tl_command {command}
                    --tl_outdir {outdir}
                    """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from template_lib.utils import colors_dict

    plt.style.use('seaborn-whitegrid')
    set_font()
    fig, ax = plt.subplots()
    # ax.set_xticks(range(0, 600, 100))
    ax.tick_params(labelsize=cfg.fontsize.tick_fs)
    ax.set_xlabel(cfg.xlabel, fontsize=cfg.fontsize.xylabel_fs)
    ax.set_ylabel(cfg.ylabel, fontsize=cfg.fontsize.xylabel_fs)

    properties = cfg.get('properties', {})
    ax.set(**properties)

    for idx, (_, data_dict) in enumerate(cfg.lines.items()):
      with open(data_dict.data_pkl, 'rb') as f:
        loaded_data = pickle.load(f)
      data = loaded_data[data_dict.dict_index][data_dict.data_index]
      if 'xlim' in data_dict:
        data_idx = np.logical_and(data[:, 0] < data_dict.xlim[1], data[:, 0] > data_dict.xlim[0])
        data = data[data_idx]

      if 'clip_x' in cfg:
        data_xlim = cfg.clip_x[-1]
        data = data[data[:, 0] <= data_xlim]

      if cfg.get_min_value:
        best_index = data[:, 1].argmin()
      else:
        best_index = data[:, 1].argmax()
      best_x = int(data[:, 0][best_index])
      best_y = data[:, 1][best_index]

      if cfg.get('add_auxi_label', False):
        data_dict.properties.label = f'x_{best_x}-y_{best_y:.3f}-' + getattr(data_dict.properties, 'label', '')
      ax.plot(data[:, 0], data[:, 1], color=colors_dict[data_dict.color], **data_dict.properties)
      pass

    ax.grid(b=True, which='major', color='#666666', linestyle='--', alpha=0.2)
    ax.legend(prop={'size': cfg.fontsize.legend_size}, ncol=1)
    fig.show()
    saved_file = os.path.join(args.tl_outdir, cfg.saved_file)
    fig.savefig(saved_file, bbox_inches='tight', pad_inches=0.0)
    print(f'Save to {saved_file}')
    pass

  def test__select_images(self, debug=True):
    """
    Usage:
        ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=2 root@localhost -p 2232

        export CUDA_VISIBLE_DEVICES=4
        export TIME_STR=1
        export PYTHONPATH=./:./ada_lib
        python -c "from exp.tests.test_ada_ultra import Testing_train_StyleUltraGAN_ADA_Transfer_MixedFaces;\
          Testing_train_StyleUltraGAN_ADA_Transfer_MixedFaces().test_projector_web(debug=False)" \
          --tl_opts port 8530 start_web True

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
    from pathlib import Path
    import shutil

    domain_dict = {}

    domain_dict['lso_sm'] = """
x0_lso: results/train_StyleUltraGAN_ADA_Transfer_MixedFaces/double_inversion_web-20210604_093503_613/exp/0024/29218_lso.jpg,
x1_lso: results/train_StyleUltraGAN_ADA_Transfer_MixedFaces/double_inversion_web-20210604_093503_613/exp/0014/29438_lso.jpg,
        """
    suffix_str = [''] * 2

    for domain, image_list in domain_dict.items():
      image_list = image_list.split(',')
      for item, suffix in zip(image_list, suffix_str):
        layer_name, image_path = item.strip().split(':')
        layer_name = layer_name.strip().replace(' ', '_')
        image_path = Path(image_path.strip())

        select_path = Path(f"{image_path.parent}/{image_path.stem}{suffix}.jpg")
        saved_path = Path(f"{outdir}/images/{domain}/{layer_name}_{select_path.name}")

        os.makedirs(saved_path.parent, exist_ok=True)
        shutil.copy(select_path, saved_path)

    print(outdir)
    pass