import torch
import torch.nn as nn

import BigGAN_Pytorch_lib.layers as biggan_layers
from .build import LAYER_REGISTRY

from template_lib.utils import get_attr_kwargs
from template_lib.d2.layers_v2 import comm_layers



@LAYER_REGISTRY.register()
class SNLinear(biggan_layers.SNLinear):
  def __init__(self, cfg, **kwargs):

    # fmt: off
    in_features                   = get_attr_kwargs(cfg, 'in_features', **kwargs)
    out_features                  = get_attr_kwargs(cfg, 'out_features', **kwargs)
    bias                          = get_attr_kwargs(cfg, 'bias', default=True, **kwargs)
    num_svs                       = get_attr_kwargs(cfg, 'num_svs', default=1, **kwargs)
    num_itrs                      = get_attr_kwargs(cfg, 'num_itrs', default=1, **kwargs)
    eps                           = get_attr_kwargs(cfg, 'eps', default=1e-6, **kwargs)

    # fmt: on

    super(SNLinear, self).__init__(in_features, out_features, bias, num_svs, num_itrs, eps)
    pass


@LAYER_REGISTRY.register()
class Linear(comm_layers.Linear):
  pass

