import yaml
import math
from easydict import EasyDict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from template_lib.d2.layers_v2.build import D2LAYERv2_REGISTRY, build_d2layer_v2
from template_lib.utils import get_attr_kwargs
from template_lib.v2.config import update_config




@D2LAYERv2_REGISTRY.register()
class Conv2d(nn.Conv2d):
  """
  """
  def __init__(self, cfg, **kwargs):

    # fmt: off
    in_channels                   = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    out_channels                  = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    kernel_size                   = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    stride                        = get_attr_kwargs(cfg, 'stride', default=1, **kwargs)
    padding                       = get_attr_kwargs(cfg, 'padding', default=0, **kwargs)
    dilation                      = get_attr_kwargs(cfg, 'dilation', default=1, **kwargs)
    groups                        = get_attr_kwargs(cfg, 'groups', default=1, **kwargs)
    bias                          = get_attr_kwargs(cfg, 'bias', default=True, **kwargs)
    padding_mode                  = get_attr_kwargs(cfg, 'padding_mode', default='zeros', **kwargs)
    # fmt: on

    super(Conv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias,
                                 padding_mode=padding_mode)

  def forward(self, input, **kargs):
    x = super(Conv2d, self).forward(input)
    return x


