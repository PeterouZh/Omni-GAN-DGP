import os
import subprocess
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils
from template_lib.nni import update_nni_config_file


class FuseConvBN(unittest.TestCase):

  def test_register_forward_hook(self):
    """
    https://learnml.today/speeding-up-model-with-fusing-batch-normalization-and-convolution-3
    """
    import torch
    import torchvision
    from torch.nn.utils import fusion

    @torch.no_grad()
    def fuse(conv, bn):

      fused = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
      )

      # setting weights
      w_conv = conv.weight.clone().view(conv.out_channels, -1)
      w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))

      fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.size()))

      # setting bias
      if conv.bias is not None:
        b_conv = conv.bias
      else:
        b_conv = torch.zeros(conv.weight.size(0))

      b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps))
      fused.bias.copy_(w_bn @ b_conv + b_bn)
      # fused.bias.copy_(b_conv + b_bn)

      return fused

    # Testing
    # we need to turn off gradient calculation because we didn't write it
    torch.set_grad_enabled(False)
    x = torch.randn(16, 3, 256, 256)
    # resnet18 = torchvision.models.resnet18(pretrained=True)
    pretrained_net = torchvision.models.vgg11_bn(pretrained=True)
    # removing all learning variables, etc
    pretrained_net.eval()
    model = torch.nn.Sequential(
      pretrained_net.features[0],
      pretrained_net.features[1]
    )
    f1 = model.forward(x)
    print(model[0].weight.min(), model[0].weight.max(),
          model[0].bias.min(), model[0].bias.max())

    fused = fuse(model[0], model[1])
    print(fused.weight.min(), fused.weight.max(),
          fused.bias.min(), fused.bias.max())

    f2 = fused.forward(x)
    d = (f1 - f2).mean().item()
    print("error:", d)

    fused_2 = fusion.fuse_conv_bn_eval(model[0], model[1])
    f2 = fused_2.forward(x)
    d = (f1 - f2).mean().item()
    print("error:", d)

    pass




