import torch.nn as nn

from template_lib.utils import get_attr_kwargs

from .build import D2LAYERv2_REGISTRY


@D2LAYERv2_REGISTRY.register()
class Linear(nn.Linear):
  def __init__(self, cfg, **kwargs):

    # fmt: off
    in_features                   = get_attr_kwargs(cfg, 'in_features', **kwargs)
    out_features                  = get_attr_kwargs(cfg, 'out_features', **kwargs)
    bias                          = get_attr_kwargs(cfg, 'bias', default=True, **kwargs)

    # fmt: on

    super(Linear, self).__init__(in_features, out_features, bias)
    pass


@D2LAYERv2_REGISTRY.register()
class ReLU(nn.ReLU):
  def __init__(self, cfg, **kwargs):

    # fmt: off
    inplace                   = get_attr_kwargs(cfg, 'inplace', default=False, **kwargs)
    # fmt: on

    super(ReLU, self).__init__(inplace)
    pass


@D2LAYERv2_REGISTRY.register()
class LeakyReLU(nn.LeakyReLU):
  def __init__(self, cfg, **kwargs):
    # fmt: off
    negative_slope                   = get_attr_kwargs(cfg, 'negative_slope', default=1e-2, **kwargs)
    inplace                          = get_attr_kwargs(cfg, 'inplace', default=False, **kwargs)

    # fmt: on

    super(LeakyReLU, self).__init__(negative_slope, inplace)
    pass

