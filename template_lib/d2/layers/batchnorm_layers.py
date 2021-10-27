import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.layers.build import D2LAYER_REGISTRY, build_d2layer
from template_lib.utils import get_attr_kwargs


@D2LAYER_REGISTRY.register()
class BatchNorm2d(nn.BatchNorm2d):
  def __init__(self, cfg, **kwargs):

    self.num_features                  = get_attr_kwargs(cfg, 'num_features', **kwargs)
    self.eps                           = get_attr_kwargs(cfg, 'eps', default=1e-5, **kwargs)
    self.momentum                      = get_attr_kwargs(cfg, 'momentum', default=0.1, **kwargs)
    self.affine                        = get_attr_kwargs(cfg, 'affine', default=True, **kwargs)
    self.track_running_stats           = get_attr_kwargs(cfg, 'track_running_stats', default=True, **kwargs)

    super(BatchNorm2d, self).__init__(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                      affine=self.affine, track_running_stats=self.track_running_stats)
    pass
  
  def forward(self, input, *args):
    
    x = super(BatchNorm2d, self).forward(input)
    return x


@D2LAYER_REGISTRY.register()
class InstanceNorm2d(nn.InstanceNorm2d):
  def __init__(self, cfg, **kwargs):

    self.num_features                  = get_attr_kwargs(cfg, 'num_features', **kwargs)
    self.eps                           = get_attr_kwargs(cfg, 'eps', default=1e-5, **kwargs)
    self.momentum                      = get_attr_kwargs(cfg, 'momentum', default=0.1, **kwargs)
    self.affine                        = get_attr_kwargs(cfg, 'affine', default=False, **kwargs)
    self.track_running_stats           = get_attr_kwargs(cfg, 'track_running_stats', default=False, **kwargs)

    super(InstanceNorm2d, self).__init__(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                         affine=self.affine, track_running_stats=self.track_running_stats)
    pass


@D2LAYER_REGISTRY.register()
class NoNorm(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()

  def forward(self, x):
    return x


@D2LAYER_REGISTRY.register()
class CondBatchNorm2d(nn.Module):
  """
  # Class-conditional bn
  # output size is the number of channels, input size is for the linear layers
  # Andy's Note: this class feels messy but I'm not really sure how to clean it up
  # Suggestions welcome! (By which I mean, refactor this and make a pull request
  # if you want to make this more readable/usable).
  """
  def __init__(self, cfg, **kwargs):
    super(CondBatchNorm2d, self).__init__()

    self.in_features                = get_attr_kwargs(cfg, 'in_features', **kwargs)
    self.out_features               = get_attr_kwargs(cfg, 'out_features', **kwargs)
    self.eps                        = get_attr_kwargs(cfg, 'eps', default=1e-5, **kwargs)
    self.momentum                   = get_attr_kwargs(cfg, 'momentum', default=0.1, **kwargs)

    # Prepare gain and bias layers
    self.gain = build_d2layer(cfg.cfg_fc, in_features=self.in_features, out_features=self.out_features)
    self.bias = build_d2layer(cfg.cfg_fc, in_features=self.in_features, out_features=self.out_features)

    self.register_buffer('stored_mean', torch.zeros(self.out_features))
    self.register_buffer('stored_var', torch.ones(self.out_features))

  def forward(self, x, y):
    """

    :param x:
    :param y: feature [b, self.input_size]
    :return:
    """
    # Calculate class-conditional gains and biases
    gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
    bias = self.bias(y).view(y.size(0), -1, 1, 1)

    out = F.batch_norm(x, self.stored_mean, self.stored_var, weight=None, bias=None,
                       training=self.training, momentum=self.momentum, eps=self.eps)
    out = out * gain + bias
    return out

  def extra_repr(self):
    s = 'out: {out_features}, in: {in_features}'
    return s.format(**self.__dict__)


@D2LAYER_REGISTRY.register()
class CondInstanceNorm2d(nn.Module):
  """
  # Class-conditional bn
  # output size is the number of channels, input size is for the linear layers
  # Andy's Note: this class feels messy but I'm not really sure how to clean it up
  # Suggestions welcome! (By which I mean, refactor this and make a pull request
  # if you want to make this more readable/usable).
  """
  def __init__(self, cfg, **kwargs):
    super(CondInstanceNorm2d, self).__init__()

    self.in_features                = get_attr_kwargs(cfg, 'in_features', **kwargs)
    self.out_features               = get_attr_kwargs(cfg, 'out_features', **kwargs)
    self.eps                        = get_attr_kwargs(cfg, 'eps', default=1e-5, **kwargs)
    self.momentum                   = get_attr_kwargs(cfg, 'momentum', default=0.1, **kwargs)

    # Prepare gain and bias layers
    self.gain = build_d2layer(cfg.cfg_fc, in_features=self.in_features, out_features=self.out_features)
    self.bias = build_d2layer(cfg.cfg_fc, in_features=self.in_features, out_features=self.out_features)

    # self.register_buffer('stored_mean', torch.zeros(self.out_features))
    # self.register_buffer('stored_var', torch.ones(self.out_features))

  def forward(self, x, y):
    """

    :param x:
    :param y: feature [b, self.input_size]
    :return:
    """
    # Calculate class-conditional gains and biases
    gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
    bias = self.bias(y).view(y.size(0), -1, 1, 1)

    out = F.instance_norm(x, running_mean=None, running_var=None, weight=None, bias=None,
                          use_input_stats=True, momentum=self.momentum, eps=self.eps)
    out = out * gain + bias
    return out

  def extra_repr(self):
    s = 'out: {out_features}, in: {in_features}'
    return s.format(**self.__dict__)
