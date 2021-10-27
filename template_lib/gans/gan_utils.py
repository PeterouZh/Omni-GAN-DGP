import torch


class _Distribution(torch.Tensor):
  """
  # A highly simplified convenience class for sampling from distributions
  # One could also use PyTorch's inbuilt distributions package.
  # Note that this class requires initialization to proceed as
  # x = Distribution(torch.randn(size))
  # x.init_distribution(dist_type, **dist_kwargs)
  # x = x.to(device,dtype)
  # This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
  """
  def init_distribution(self, dist_type, **kwargs):
    """ Init the params of the distribution"""
    self.dist_type = dist_type
    self.dist_kwargs = kwargs
    if self.dist_type == 'normal':
      self.mean, self.var = kwargs['mean'], kwargs['var']
    elif self.dist_type == 'categorical':
      self.num_categories = kwargs['num_categories']

  def sample_(self):
    if self.dist_type == 'normal':
      self.normal_(self.mean, self.var)
    elif self.dist_type == 'categorical':
      self.random_(0, self.num_categories)

  def to(self, *args, **kwargs):
    """ Silly hack: overwrite the to() method to wrap the new object in a distribution as well"""
    new_obj = _Distribution(self)
    new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    new_obj.data = super().to(*args, **kwargs)
    return new_obj


def z_normal(batch_size, dim_z, z_mean=0, z_var=1.0, device='cuda'):
  """Convenience function to prepare z"""
  z = _Distribution(torch.randn(batch_size, dim_z))
  z.init_distribution('normal', mean=z_mean, var=z_var)
  z = z.to(device, torch.float32)
  return z


def y_categorical(batch_size, nclasses, device='cuda'):
  y = _Distribution(torch.zeros(batch_size, requires_grad=False))
  y.init_distribution('categorical', num_categories=nclasses)
  y = y.to(device, torch.int64)
  return y


def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off

