import torch
import torch.nn as nn

from .build import MODEL_REGISTRY
from exp.comm.layers.build import build_layer

from template_lib.utils import get_attr_kwargs


@MODEL_REGISTRY.register()
class INRBlock(nn.Module):
  def __init__(self, cfg, **kwargs):
    super(INRBlock, self).__init__()
    # fmt: off
    in_features              = get_attr_kwargs(cfg, 'in_features', **kwargs)
    out_features             = get_attr_kwargs(cfg, 'out_features', **kwargs)
    c_dim                    = get_attr_kwargs(cfg, 'c_dim', **kwargs)
    norm_cfg                 = get_attr_kwargs(cfg, 'norm_cfg', **kwargs)
    act_cfg                  = get_attr_kwargs(cfg, 'act_cfg', **kwargs)
    linear_cfg               = get_attr_kwargs(cfg, 'linear_cfg', **kwargs)

    # fmt: on
    self.norm1 = build_layer(norm_cfg, in_dim=in_features, c_dim=c_dim)
    self.act1 = build_layer(act_cfg)
    self.fc1 = build_layer(linear_cfg, in_features=in_features, out_features=out_features)

    self.norm2 = build_layer(norm_cfg, in_dim=out_features, c_dim=c_dim)
    self.act2 = build_layer(act_cfg)
    self.fc2 = build_layer(linear_cfg, in_features=out_features, out_features=out_features)

    self.learnable_skip = in_features != out_features
    if self.learnable_skip:
      self.fc_skip = build_layer(linear_cfg, in_features=in_features, out_features=out_features)
    pass

  def forward(self, x, y):
    h = self.norm1(x, y)
    h = self.act1(h)
    h = self.fc1(h)

    h = self.norm2(h, y)
    h = self.act2(h)
    h = self.fc2(h)

    if self.learnable_skip:
      x = self.fc_skip(x)

    out = h + x
    return out

