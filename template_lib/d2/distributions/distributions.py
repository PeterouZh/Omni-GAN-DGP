import torch
from torch import  distributions

from template_lib.utils import get_attr_kwargs
from .build import DISTRIBUTIONS_REGISTRY


@DISTRIBUTIONS_REGISTRY.register()
class Normal(distributions.normal.Normal):

  def __init__(self, cfg, **kwargs):
    # fmt: off
    loc                                 = get_attr_kwargs(cfg, 'loc', default=0, **kwargs)
    scale                               = get_attr_kwargs(cfg, 'scale', default=1, **kwargs)
    self.sample_shape                   = get_attr_kwargs(cfg, 'sample_shape', default=None, **kwargs)
    # fmt: on

    super(Normal, self).__init__(loc=loc, scale=scale)

    pass

  def sample(self, sample_shape=None):
    if sample_shape is None:
      sample = distributions.normal.Normal.sample(self, self.sample_shape)
      # sample = super(Normal, self).sample(sample_shape=self.sample_shape)
    else:
      sample = distributions.normal.Normal.sample(self, sample_shape)
      # sample = super(Normal, self).sample(sample_shape=sample_shape)
    return sample


@DISTRIBUTIONS_REGISTRY.register()
class Categorical(distributions.Categorical):

  def __init__(self, cfg, **kwargs):

    probs                                 = get_attr_kwargs(cfg, 'probs', default=None, **kwargs)
    logits                                = get_attr_kwargs(cfg, 'logits', default=None, **kwargs)
    validate_args                         = get_attr_kwargs(cfg, 'validate_args', default=None, **kwargs)
    self.sample_shape                     = get_attr_kwargs(cfg, 'sample_shape', default=None, **kwargs)

    if isinstance(self.sample_shape, int):
      self.sample_shape = [self.sample_shape, ]

    super(Categorical, self).__init__(probs=probs, logits=logits, validate_args=validate_args)
    pass

  def sample(self, sample_shape=None):
    if sample_shape is None:
      sample = super(Categorical, self).sample(sample_shape=self.sample_shape)
    else:
      sample = super(Categorical, self).sample(sample_shape=sample_shape)
    return sample


@DISTRIBUTIONS_REGISTRY.register()
class CategoricalUniform(distributions.Categorical):

  def __init__(self, cfg, **kwargs):
    # fmt: off
    n_classes                             = get_attr_kwargs(cfg, 'n_classes', **kwargs)
    validate_args                         = get_attr_kwargs(cfg, 'validate_args', default=None, **kwargs)
    self.sample_shape                     = get_attr_kwargs(cfg, 'sample_shape', default=None, **kwargs)
    # fmt: on

    if isinstance(self.sample_shape, int):
      self.sample_shape = [self.sample_shape, ]

    probs = torch.ones(n_classes) * 1./n_classes
    super(CategoricalUniform, self).__init__(probs=probs, logits=None, validate_args=validate_args)
    pass

  def sample(self, sample_shape=None):
    if sample_shape is None:
      sample = distributions.Categorical.sample(self, sample_shape=self.sample_shape)
      # sample = super(CategoricalUniform, self).sample(sample_shape=self.sample_shape)
    else:
      sample = distributions.Categorical.sample(self, sample_shape=sample_shape)
      # sample = super(CategoricalUniform, self).sample(sample_shape=sample_shape)
    return sample


@DISTRIBUTIONS_REGISTRY.register()
class ConstantValue(object):

  def __init__(self, cfg, **kwargs):

    self.constant                     = get_attr_kwargs(cfg, 'constant', **kwargs)
    self.sample_shape                 = get_attr_kwargs(cfg, 'sample_shape', default=None, **kwargs)

    if isinstance(self.sample_shape, int):
      self.sample_shape = [self.sample_shape, ]
    pass

  def sample(self, sample_shape=None):
    if sample_shape is None:
      sample = torch.empty(self.sample_shape, dtype=torch.int64).fill_(self.constant)
    else:
      sample = torch.empty(sample_shape, dtype=torch.int64).fill_(self.constant)
    return sample


@DISTRIBUTIONS_REGISTRY.register()
class Uniform(distributions.uniform.Uniform):

  def __init__(self, cfg, **kwargs):

    low                                 = get_attr_kwargs(cfg, 'low', default=0, **kwargs)
    high                                = get_attr_kwargs(cfg, 'high', default=1, **kwargs)
    self.sample_shape                   = get_attr_kwargs(cfg, 'sample_shape', default=None, **kwargs)

    super(Uniform, self).__init__(low=low, high=high)
    pass

  def sample(self, sample_shape=None):
    if sample_shape is None:
      sample = super(Uniform, self).sample(sample_shape=self.sample_shape)
    else:
      sample = super(Uniform, self).sample(sample_shape=sample_shape)
    return sample