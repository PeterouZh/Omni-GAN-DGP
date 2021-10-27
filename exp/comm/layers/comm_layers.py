import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import LAYER_REGISTRY, build_layer

from template_lib.d2.layers_v2 import comm_layers
from template_lib.utils import get_attr_kwargs


@LAYER_REGISTRY.register()
class ReLU(comm_layers.ReLU):
  pass


@LAYER_REGISTRY.register()
class LeakyReLU(comm_layers.LeakyReLU):
  pass


@LAYER_REGISTRY.register()
class CLN(nn.Module):
  def __init__(self, cfg, **kwargs):
    super(CLN, self).__init__()

    # fmt: off
    self.in_dim                  = get_attr_kwargs(cfg, 'in_dim', **kwargs)
    self.c_dim                   = get_attr_kwargs(cfg, 'c_dim', **kwargs)
    linear_cfg                   = get_attr_kwargs(cfg, 'linear_cfg', **kwargs)
    self.eps                     = get_attr_kwargs(cfg, 'eps', default=1e-5, **kwargs)

    # fmt: on

    # Prepare gain and bias layers
    self.gain = build_layer(linear_cfg, in_features=self.c_dim, out_features=self.in_dim)
    self.bias = build_layer(linear_cfg, in_features=self.c_dim, out_features=self.in_dim)

    # self.register_buffer('stored_mean', torch.zeros(output_size))
    # self.register_buffer('stored_var', torch.ones(output_size))
    pass

  def forward(self, x, y):
    """
    x: (b, num_seq, dim)
    y: (b, dim)
    """
    # Calculate class-conditional gains and biases
    gain = (1 + self.gain(y)).unsqueeze(1)
    bias = self.bias(y).unsqueeze(1)

    out = F.layer_norm(x, normalized_shape=(self.in_dim, ), weight=None, bias=None, eps=self.eps)

    out = out * gain + bias
    return out

  def extra_repr(self):
    s = 'in_dim: {in_dim}, c_dim: {c_dim}'
    return s.format(**self.__dict__)

  @staticmethod
  def test_case():

    in_dim = 128
    out_dim = 256
    num_seq = 10
    which_linear = nn.Linear

    x = torch.randn(2, num_seq, out_dim).requires_grad_()
    y = torch.randn(2, in_dim).requires_grad_()

    net = CLN(output_size=out_dim, input_size=in_dim, which_linear=which_linear)
    print(net)
    out = net(x, y)
    loss = out.mean()
    loss.backward()
    pass