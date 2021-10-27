import collections
import numpy as np
import math
import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

from template_lib import utils


def SuperResolutionNet(upscale_factor, channel=3):
  return nn.Sequential(
    nn.Conv2d(channel, 64, kernel_size=5, padding=2),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 32, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, channel * upscale_factor ** 2, kernel_size=3, padding=1),
    Rearrange('b (c h2 w2) h w -> b c (h h2) (w w2)', h2=upscale_factor, w2=upscale_factor),
  )


def gram_matrix_new(y):
  b, ch, h, w = y.shape
  return torch.einsum('bchw,bdhw->bcd', [y, y]) / (h * w)


class Self_Attn(nn.Module):
  """ Self attention Layer"""

  def __init__(self, in_dim):
    super().__init__()
    self.query_conv = nn.Conv2d(in_dim, out_channels=in_dim // 8, kernel_size=1)
    self.key_conv = nn.Conv2d(in_dim, out_channels=in_dim // 8, kernel_size=1)
    self.value_conv = nn.Conv2d(in_dim, out_channels=in_dim, kernel_size=1)
    self.gamma = nn.Parameter(torch.zeros([1]))

  def forward(self, x):
    proj_query = rearrange(self.query_conv(x), 'b c h w -> b (h w) c')
    proj_key = rearrange(self.key_conv(x), 'b c h w -> b c (h w)')
    proj_value = rearrange(self.value_conv(x), 'b c h w -> b (h w) c')
    energy = torch.bmm(proj_query, proj_key)
    attention = F.softmax(energy, dim=2)
    out = torch.bmm(attention, proj_value)
    out = x + self.gamma * rearrange(out, 'b (h w) c -> b c h w', **parse_shape(x, 'b c h w'))
    return out, attention


class MultiHeadAttention(nn.Module):
  def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
    super().__init__()
    self.n_head = n_head

    self.w_qs = nn.Linear(d_model, n_head * d_k)
    self.w_ks = nn.Linear(d_model, n_head * d_k)
    self.w_vs = nn.Linear(d_model, n_head * d_v)

    nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
    nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
    nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    self.fc = nn.Linear(n_head * d_v, d_model)
    nn.init.xavier_normal_(self.fc.weight)
    self.dropout = nn.Dropout(p=dropout)
    self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, q, k, v, mask=None):
    residual = q
    q = rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
    k = rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
    v = rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
    attn = torch.einsum('hblk,hbtk->hblt', [q, k]) / np.sqrt(q.shape[-1])
    if mask is not None:
      attn = attn.masked_fill(mask[None], -np.inf)
    attn = torch.softmax(attn, dim=3)
    output = torch.einsum('hblt,hbtv->hblv', [attn, v])
    output = rearrange(output, 'head b l v -> b l (head v)')
    output = self.dropout(self.fc(output))
    output = self.layer_norm(output + residual)
    return output, attn


class SpacialTransformNet(nn.Module):
  def __init__(self):
    super(SpacialTransformNet, self).__init__()
    # Spatial transformer localization-network
    linear = nn.Linear(32, 3 * 2)
    # Initialize the weights/bias with identity transformation
    linear.weight.data.zero_()
    linear.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    self.compute_theta = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size=7),
      nn.MaxPool2d(2, stride=2),
      nn.ReLU(True),
      nn.Conv2d(8, 10, kernel_size=5),
      nn.MaxPool2d(2, stride=2),
      nn.ReLU(True),
      Rearrange('b c h w -> b (c h w)', h=3, w=3),
      nn.Linear(10 * 3 * 3, 32),
      nn.ReLU(True),
      linear,
      Rearrange('b (row col) -> b row col', row=2, col=3),
    )

  # Spatial transformer network forward function
  def stn(self, x):
    grid = F.affine_grid(self.compute_theta(x), x.size())
    out = F.grid_sample(x, grid)
    return out


class Testing_networks(unittest.TestCase):

  def test_ConvNet(self, debug=True):
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
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
    from template_lib.proj.pytorch.pytorch_hook import VerboseModel

    conv_net_new = nn.Sequential(collections.OrderedDict([
      ('conv1', nn.Conv2d(1, 10, kernel_size=5, padding=2)),
      ('pool1', nn.MaxPool2d(kernel_size=2)),
      ('relu1', nn.ReLU()),
      ('conv2', nn.Conv2d(10, 20, kernel_size=5, padding=2)),
      ('pool2', nn.MaxPool2d(kernel_size=2)),
      ('relu2', nn.ReLU()),
      ('dropout1', nn.Dropout2d()),
      ('rerange', Rearrange('b c h w -> b (c h w)')),
      ('linear1', nn.Linear(320, 50)),
      ('relu3', nn.ReLU()),
      ('dropout2', nn.Dropout()),
      ('linear', nn.Linear(50, 10)),
      ('logsoftmax', nn.LogSoftmax(dim=1))
    ]))

    net = VerboseModel(model=conv_net_new)

    x = torch.rand(1, 1, 16, 16)
    out = net(x)
    pass

  def test_SuperResolutionNet(self, debug=True):
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
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
    from template_lib.proj.pytorch.pytorch_hook import VerboseModel

    net = SuperResolutionNet(upscale_factor=4)

    net = VerboseModel(model=net)

    x = torch.rand(1, 3, 16, 16)
    out = net(x)
    pass

  def test_Self_Attn(self, debug=True):
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
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
    from template_lib.proj.pytorch.pytorch_hook import VerboseModel

    net = Self_Attn(in_dim=32)

    net = VerboseModel(model=net)

    x = torch.rand(1, 32, 16, 16)
    out = net(x)
    pass

  def test_SpacialTransformNet(self, debug=True):
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
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
    from template_lib.proj.pytorch.pytorch_hook import VerboseModel

    net = SpacialTransformNet()

    net = VerboseModel(model=net)
    assert 0
    x = torch.rand(1, 32, 16, 16)
    out = net(x)
    pass



