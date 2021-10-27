import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.models_v2 import MODEL_REGISTRY
from template_lib.utils import get_attr_kwargs, get_dict_str


@MODEL_REGISTRY.register(name_prefix=__name__)
class CLN(nn.Module):
  def __init__(self,
               in_dim,
               style_dim,
               which_linear=nn.Linear,
               spectral_norm=False,
               eps=1e-5,
               verbose=False,
               **kwargs):
    super(CLN, self).__init__()

    self.in_dim, self.style_dim = in_dim, style_dim
    self.spectral_norm = spectral_norm
    # Prepare gain and bias layers
    self.gain = which_linear(style_dim, in_dim)
    self.bias = which_linear(style_dim, in_dim)
    if spectral_norm:
      self.gain = nn.utils.spectral_norm(self.gain)
      self.bias = nn.utils.spectral_norm(self.bias)

    self.eps = eps
    # self.register_buffer('stored_mean', torch.zeros(output_size))
    # self.register_buffer('stored_var', torch.ones(output_size))
    pass

  def forward(self, x, y):
    # Calculate class-conditional gains and biases
    gain = (1 + self.gain(y)).unsqueeze(1)
    bias = self.bias(y).unsqueeze(1)

    out = F.layer_norm(x, normalized_shape=(self.in_dim,), weight=None, bias=None, eps=self.eps)

    out = out * gain + bias
    return out

  def extra_repr(self):
    s = f'in_dim={self.in_dim}, style_dim={self.style_dim}, spectral_norm={self.spectral_norm}'
    return s

  @staticmethod
  def test_case():

    in_dim = 128
    out_dim = 256
    num_seq = 10
    which_linear = nn.Linear

    x = torch.randn(2, num_seq, out_dim).requires_grad_()
    y = torch.randn(2, in_dim).requires_grad_()

    net = CLN(in_dim=out_dim, style_dim=in_dim, which_linear=which_linear)
    print(net)
    out = net(x, y)
    loss = out.mean()
    loss.backward()
    pass