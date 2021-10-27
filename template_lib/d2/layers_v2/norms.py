from easydict import EasyDict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.layers_v2.build import D2LAYERv2_REGISTRY, build_d2layer_v2
from template_lib.utils import get_attr_kwargs


@D2LAYERv2_REGISTRY.register()
class BatchNorm2d(nn.BatchNorm2d):
  def __init__(self, cfg, **kwargs):

    self.num_features                  = get_attr_kwargs(cfg, 'num_features', **kwargs)
    self.eps                           = get_attr_kwargs(cfg, 'eps', default=1e-5, **kwargs)
    self.momentum                      = get_attr_kwargs(cfg, 'momentum', default=0.1, **kwargs)
    self.affine                        = get_attr_kwargs(cfg, 'affine', default=True, **kwargs)
    self.track_running_stats           = get_attr_kwargs(cfg, 'track_running_stats', default=True, **kwargs)

    super(BatchNorm2d, self).__init__(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                      affine=self.affine, track_running_stats=self.track_running_stats)
    pass

  def forward(self, input, *args):

    x = super(BatchNorm2d, self).forward(input)
    return x
