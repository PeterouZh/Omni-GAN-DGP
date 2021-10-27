import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.layers.build import D2LAYER_REGISTRY
from template_lib.utils import get_attr_kwargs


@D2LAYER_REGISTRY.register()
class ReLU(nn.ReLU):

  def __init__(self, cfg, **kwargs):

    inplace             = get_attr_kwargs(cfg, 'inplace', default=False, **kwargs)

    super(ReLU, self).__init__(inplace=inplace)


@D2LAYER_REGISTRY.register()
class ELU(nn.ELU):

  def __init__(self, cfg, **kwargs):

    alpha               = get_attr_kwargs(cfg, 'alpha', default=1.0, **kwargs)
    inplace             = get_attr_kwargs(cfg, 'inplace', default=False, **kwargs)

    super(ELU, self).__init__(alpha=alpha, inplace=inplace)


@D2LAYER_REGISTRY.register()
class LeakyReLU(nn.LeakyReLU):

  def __init__(self, cfg, **kwargs):

    negative_slope      = get_attr_kwargs(cfg, 'negative_slope', default=0.01, **kwargs)
    inplace             = get_attr_kwargs(cfg, 'inplace', default=False, **kwargs)

    super(LeakyReLU, self).__init__(negative_slope=negative_slope, inplace=inplace)


@D2LAYER_REGISTRY.register()
class PReLU(nn.PReLU):

  def __init__(self, cfg, **kwargs):

    num_parameters          = get_attr_kwargs(cfg, 'num_parameters', default=1, **kwargs)
    init                    = get_attr_kwargs(cfg, 'init', default=0.25, **kwargs)

    super(PReLU, self).__init__(num_parameters=num_parameters, init=init)


@D2LAYER_REGISTRY.register()
class ReLU6(nn.ReLU6):

  def __init__(self, cfg, **kwargs):

    inplace                 = get_attr_kwargs(cfg, 'inplace', default=False, **kwargs)

    super(ReLU6, self).__init__(inplace=inplace)


@D2LAYER_REGISTRY.register()
class SELU(nn.SELU):

  def __init__(self, cfg, **kwargs):

    inplace                 = get_attr_kwargs(cfg, 'inplace', default=False, **kwargs)

    super(SELU, self).__init__(inplace=inplace)


@D2LAYER_REGISTRY.register()
class CELU(nn.CELU):

  def __init__(self, cfg, **kwargs):

    alpha                   = get_attr_kwargs(cfg, 'alpha', default=1.0, **kwargs)
    inplace                 = get_attr_kwargs(cfg, 'inplace', default=False, **kwargs)

    super(CELU, self).__init__(alpha=alpha, inplace=inplace)


@D2LAYER_REGISTRY.register()
class Mish(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()
    pass

  def forward(self, x):
    # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
    return x * (torch.tanh(F.softplus(x)))


@D2LAYER_REGISTRY.register()
class NoAct(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()

  def forward(self, x):
    return x