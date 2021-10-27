import torch
import torch.nn as nn
import torch.nn.functional as F


# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs


# Spectral normalization base class
class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))

  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values;
  # note that these buffers are just for logging and are not used in training.
  @property
  def sv(self):
    return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

  # Compute the spectrally-normalized weight
  def W_(self):
    if not hasattr(self, 'kernel_size'):
      weight = self.weight
    else:
      max_ks = self.weight.size(-1)
      kernel_size = self.kernel_size \
        if isinstance(self.kernel_size, int) else self.kernel_size[0]
      start = (max_ks - kernel_size) // 2
      end = -start + max_ks
      weight = self.weight[:, :, start:end, start:end]
    # W_mat = weight.view(weight.size(0), -1)
    W_mat = weight.contiguous().view(weight.size(0), -1)
    # W_mat = weight.reshape(weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
      # Update the svs
    if self.training:
      with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv
    return weight / svs[0]


class SNConv2dFunc(nn.Module, SN):
  def __init__(self, weight, bias, kernel_size,
               num_svs=1, num_itrs=1, eps=1e-12,
               stride=1, padding=0, dilation=1, groups=1):
    nn.Module.__init__(self, )
    self.weight = weight
    self.bias = bias
    self.kernel_size = (kernel_size, kernel_size)
    self.out_channels = weight.size(0)
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    SN.__init__(self, num_svs, num_itrs, self.out_channels, eps=eps)

  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
