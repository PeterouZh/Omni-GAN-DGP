import math
import torch
from torch import optim
import torch.nn as nn
from template_lib.utils import get_attr_kwargs

from .build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class Adam(optim.Adam):
  """
  params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False
  """
  def __init__(self, cfg, params, **kwargs):

    # fmt: off
    lr                = get_attr_kwargs(cfg, 'lr', default=1e-3, **kwargs)
    betas             = get_attr_kwargs(cfg, 'betas', default=(0.9, 0.999), **kwargs)
    eps               = get_attr_kwargs(cfg, 'eps', default=1e-8, **kwargs)
    weight_decay      = get_attr_kwargs(cfg, 'weight_decay', default=0, **kwargs)
    amsgrad           = get_attr_kwargs(cfg, 'amsgrad', default=False, **kwargs)
    # fmt: on

    super(Adam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                               amsgrad=amsgrad)
    pass


@OPTIMIZER_REGISTRY.register()
class SGD(optim.SGD):
  """
  params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False
  """
  def __init__(self, cfg, **kwargs):
    # fmt: off
    params            = kwargs['params']
    lr                = get_attr_kwargs(cfg, 'lr', **kwargs)
    momentum          = get_attr_kwargs(cfg, 'momentum', default=0, **kwargs)
    dampening         = get_attr_kwargs(cfg, 'dampening', default=0, **kwargs)
    weight_decay      = get_attr_kwargs(cfg, 'weight_decay', default=0, **kwargs)
    nesterov          = get_attr_kwargs(cfg, 'nesterov', default=False, **kwargs)

    # fmt: on
    super(SGD, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                              weight_decay=weight_decay, nesterov=nesterov)
    pass


@OPTIMIZER_REGISTRY.register()
class NoneOptim(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(NoneOptim, self).__init__()
    pass
