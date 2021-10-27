import torch
import torch.nn as nn

from .build import MODEL_REGISTRY
from exp.comm.layers.build import build_layer

from template_lib.utils import get_attr_kwargs


@MODEL_REGISTRY.register()
class FCHyperNet(nn.Module):
  def __init__(self, cfg, **kwargs):
    super(FCHyperNet, self).__init__()

    # fmt: off
    in_dim                 = get_attr_kwargs(cfg, 'in_dim', default=2, **kwargs)
    hidden_dim             = get_attr_kwargs(cfg, 'hidden_dim', **kwargs)
    out_dim                = get_attr_kwargs(cfg, 'out_dim', **kwargs)
    num_layers             = get_attr_kwargs(cfg, 'num_layers', **kwargs)
    linear_cfg             = get_attr_kwargs(cfg, 'linear_cfg', **kwargs)
    act_cfg                = get_attr_kwargs(cfg, 'act_cfg', **kwargs)

    # fmt: on

    self.layers = nn.ModuleList()

    in_fc = build_layer(linear_cfg, in_features=in_dim, out_features=hidden_dim)
    in_act = build_layer(act_cfg)
    self.layers.append(in_fc)
    self.layers.append(in_act)

    for i in range(num_layers):
      fc = build_layer(linear_cfg, in_features=hidden_dim, out_features=hidden_dim)
      self.layers.append(fc)
      act = build_layer(act_cfg)
      self.layers.append(act)

    out_fc = build_layer(linear_cfg, in_features=hidden_dim, out_features=out_dim)
    self.layers.append(out_fc)
    pass

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x