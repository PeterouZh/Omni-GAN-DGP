from easydict import EasyDict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.layers.build import D2LAYER_REGISTRY, build_d2layer
from template_lib.utils import get_attr_kwargs

from .pagan_layers_utils import SN


@D2LAYER_REGISTRY.register()
class SNConv2d(nn.Conv2d, SN):
  """
  # 2D Conv layer with spectral norm
  """
  def __init__(self, cfg, **kwargs):

    self.in_channels                   = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels                  = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.kernel_size                   = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    self.stride                        = get_attr_kwargs(cfg, 'stride', default=1, **kwargs)
    self.padding                       = get_attr_kwargs(cfg, 'padding', default=0, **kwargs)
    self.dilation                      = get_attr_kwargs(cfg, 'dilation', default=1, **kwargs)
    self.groups                        = get_attr_kwargs(cfg, 'groups', default=1, **kwargs)
    self.bias                          = get_attr_kwargs(cfg, 'bias', default=True, **kwargs)
    self.num_svs                       = get_attr_kwargs(cfg, 'num_svs', default=1, **kwargs)
    self.num_itrs                      = get_attr_kwargs(cfg, 'num_itrs', default=1, **kwargs)
    self.eps                           = get_attr_kwargs(cfg, 'eps', default=1e-6, **kwargs)

    nn.Conv2d.__init__(self, self.in_channels, self.out_channels, self.kernel_size, self.stride,
                       self.padding, self.dilation, self.groups, self.bias)
    SN.__init__(self, self.num_svs, self.num_itrs, self.out_channels, eps=self.eps)

  def forward(self, x, *args):
    x = F.conv2d(x, self.W_(), self.bias, self.stride, self.padding, self.dilation, self.groups)
    return x


@D2LAYER_REGISTRY.register()
class Conv2d(nn.Conv2d):
  """
  # 2D Conv layer with spectral norm
  """
  def __init__(self, cfg, **kwargs):

    in_channels                   = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    out_channels                  = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    kernel_size                   = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    stride                        = get_attr_kwargs(cfg, 'stride', default=1, **kwargs)
    padding                       = get_attr_kwargs(cfg, 'padding', default=0, **kwargs)
    dilation                      = get_attr_kwargs(cfg, 'dilation', default=1, **kwargs)
    groups                        = get_attr_kwargs(cfg, 'groups', default=1, **kwargs)
    bias                          = get_attr_kwargs(cfg, 'bias', default=True, **kwargs)
    padding_mode                  = get_attr_kwargs(cfg, 'padding_mode', default='zeros', **kwargs)


    super(Conv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias,
                                 padding_mode=padding_mode)

  def forward(self, input, **kargs):
    x = super(Conv2d, self).forward(input)
    return x


@D2LAYER_REGISTRY.register()
class Conv2dOp(nn.Module):
  """
  """
  def __init__(self, cfg, **kwargs):
    super().__init__()

    self.stride                        = get_attr_kwargs(cfg, 'stride', default=1, **kwargs)
    self.padding                       = get_attr_kwargs(cfg, 'padding', default=0, **kwargs)
    self.dilation                      = get_attr_kwargs(cfg, 'dilation', default=1, **kwargs)
    self.groups                        = get_attr_kwargs(cfg, 'groups', default=1, **kwargs)

  def forward(self, x, weight, bias, **kargs):
    x = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding,
                 dilation=self.dilation, groups=self.groups)
    return x


@D2LAYER_REGISTRY.register()
class Linear(nn.Linear):
  """
  # 2D Conv layer with spectral norm
  """
  def __init__(self, cfg, **kwargs):

    in_features                   = get_attr_kwargs(cfg, 'in_features', **kwargs)
    out_features                  = get_attr_kwargs(cfg, 'out_features', **kwargs)
    bias                          = get_attr_kwargs(cfg, 'bias', default=True, **kwargs)

    super(Linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)


@D2LAYER_REGISTRY.register()
class DepthwiseSeparableConv2d(nn.Module):

  def __init__(self, cfg, **kwargs):
    super().__init__()

    in_channels                     = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    out_channels                    = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    kernel_size                     = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    stride                          = get_attr_kwargs(cfg, 'stride', default=1, **kwargs)
    padding                         = get_attr_kwargs(cfg, 'padding', default=0, **kwargs)
    dilation                        = get_attr_kwargs(cfg, 'dilation', default=1, **kwargs)
    bias                            = get_attr_kwargs(cfg, 'bias', default=True, **kwargs)
    padding_mode                    = get_attr_kwargs(cfg, 'padding_mode', default='zeros', **kwargs)

    depthwise_conv_cfg = EasyDict(
      in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
    dilation=dilation, groups=in_channels, bias=bias, padding_mode=padding_mode)
    depthwise_conv = Conv2d(depthwise_conv_cfg)

    pointwise_conv_cfg = EasyDict(
      in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
      dilation=1, groups=1, bias=bias, padding_mode=padding_mode)
    pointwise_conv = Conv2d(pointwise_conv_cfg)

    self.net = nn.Sequential(
      depthwise_conv,
      pointwise_conv
    )

  def forward(self, x, *args):
    return self.net(x)


@D2LAYER_REGISTRY.register()
class ActConv2d(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(ActConv2d, self).__init__()

    cfg_act                   = get_attr_kwargs(cfg, 'cfg_act', **kwargs)
    cfg_conv                  = get_attr_kwargs(cfg, 'cfg_conv', **kwargs)

    self.act = build_d2layer(cfg_act, **kwargs)
    self.conv = build_d2layer(cfg_conv, **kwargs)


  def forward(self, x, **kwargs):
    x = self.act(x)
    x = self.conv(x, **kwargs)
    return x


@D2LAYER_REGISTRY.register()
class Conv2dAct(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(Conv2dAct, self).__init__()

    cfg_act                   = get_attr_kwargs(cfg, 'cfg_act', **kwargs)
    cfg_conv                  = get_attr_kwargs(cfg, 'cfg_conv', **kwargs)

    self.conv = build_d2layer(cfg_conv, **kwargs)
    self.act = build_d2layer(cfg_act, **kwargs)


  def forward(self, x):
    x = self.conv(x)
    x = self.act(x)
    return x


@D2LAYER_REGISTRY.register()
class BNActConv2d(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(BNActConv2d, self).__init__()

    cfg_bn                    = get_attr_kwargs(cfg, 'cfg_bn', **kwargs)
    cfg_act                   = get_attr_kwargs(cfg, 'cfg_act', **kwargs)
    cfg_conv                  = get_attr_kwargs(cfg, 'cfg_conv', **kwargs)

    self.bn = build_d2layer(cfg_bn, num_features=kwargs['in_channels'], **kwargs)
    self.act = build_d2layer(cfg_act, **kwargs)
    self.conv = build_d2layer(cfg_conv, **kwargs)


  def forward(self, x):
    x = self.bn(x)
    x = self.act(x)
    x = self.conv(x)
    return x

@D2LAYER_REGISTRY.register()
class Conv2dBNAct(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(Conv2dBNAct, self).__init__()

    cfg_conv                  = get_attr_kwargs(cfg, 'cfg_conv', **kwargs)
    cfg_bn                    = get_attr_kwargs(cfg, 'cfg_bn', **kwargs)
    cfg_act                   = get_attr_kwargs(cfg, 'cfg_act', **kwargs)

    self.conv = build_d2layer(cfg_conv, **kwargs)
    self.bn = build_d2layer(cfg_bn, num_features=kwargs['out_channels'], **kwargs)
    self.act = build_d2layer(cfg_act, **kwargs)



  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.act(x)
    return x


@D2LAYER_REGISTRY.register()
class ActConv2dCBN(nn.Module):

  def __init__(self, cfg, **kwargs):
    super().__init__()

    cfg_act                   = get_attr_kwargs(cfg, 'cfg_act', **kwargs)
    cfg_conv                  = get_attr_kwargs(cfg, 'cfg_conv', **kwargs)
    cfg_cbn                   = get_attr_kwargs(cfg, 'cfg_cbn', **kwargs)

    self.act = build_d2layer(cfg_act, **kwargs)
    self.conv = build_d2layer(cfg_conv, **kwargs)
    self.cbn = build_d2layer(cfg_cbn, num_features=kwargs['in_channels'], **kwargs)


  def forward(self, x, cbn, **kwargs):
    x = self.act(x)
    x = self.conv(x)
    x = self.cbn(x, cbn)
    return x


