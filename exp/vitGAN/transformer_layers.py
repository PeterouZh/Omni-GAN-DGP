import torch
import torch.nn as nn
import torch.nn.functional as F


class CLN(nn.Module):
  def __init__(self, output_size, input_size, which_linear, use_sn=False, eps=1e-5, **kwargs):
    super(CLN, self).__init__()

    self.output_size, self.input_size = output_size, input_size
    # Prepare gain and bias layers
    self.gain = which_linear(input_size, output_size)
    self.bias = which_linear(input_size, output_size)
    if use_sn:
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

    out = F.layer_norm(x, normalized_shape=(self.output_size, ), weight=None, bias=None, eps=self.eps)

    out = out * gain + bias
    return out

  def extra_repr(self):
    s = 'out: {output_size}, in: {input_size}'
    return s.format(**self.__dict__)

  @staticmethod
  def test_case():


    pass