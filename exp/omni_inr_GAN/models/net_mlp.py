import logging

import torch.nn as nn

from template_lib.utils import get_dict_str, get_attr_kwargs

from . import MODEL_REGISTRY


@MODEL_REGISTRY.register(name=f"{__name__}.MLP")
class MLP(nn.Module):

  def __init__(self, cfg, which_linear=None, verbose=True, **kwargs):
    super().__init__()
    kwargs['tl_ret_kwargs'] = {}
    # fmt: off
    in_dim                = get_attr_kwargs(cfg, 'in_dim', **kwargs)
    out_dim               = get_attr_kwargs(cfg, 'out_dim', **kwargs)
    hidden_list           = get_attr_kwargs(cfg, 'hidden_list', **kwargs)
    # fmt: on
    if verbose:
      logging.getLogger('tl').info(f"  MLP kwargs: \n{get_dict_str(kwargs['tl_ret_kwargs'])}")

    if which_linear is None:
      which_linear = nn.Linear

    layers = []
    lastv = in_dim
    for hidden in hidden_list:
      layers.append(which_linear(lastv, hidden))
      layers.append(nn.ReLU())
      lastv = hidden
    layers.append(which_linear(lastv, out_dim))
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    shape = x.shape[:-1]
    x = self.layers(x.view(-1, x.shape[-1]))
    return x.view(*shape, -1)
