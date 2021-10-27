import yaml
from easydict import EasyDict
import logging
import sys
import functools
import os

import numpy as np
from scipy import linalg # For numpy FID
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.inception import inception_v3
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from template_lib.d2.utils import comm
from template_lib.v2.config import update_config
from template_lib.utils import get_attr_kwargs
from template_lib.v2.GAN.evaluation import GAN_METRIC_REGISTRY


__all__ = ['PyTorchFIDISScore']

# Module that wraps the inception network to enable use with dataparallel and
# returning pool features and logits.
class WrapInception(nn.Module):
  def __init__(self, net):
    super(WrapInception,self).__init__()
    self.net = net
    self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                  requires_grad=False)
    self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                 requires_grad=False)
  def forward(self, x):
    # Normalize x
    x = (x + 1.) / 2.0
    x = (x - self.mean) / self.std
    # Upsample if necessary
    if x.shape[2] != 299 or x.shape[3] != 299:
      if torch.__version__ in ['0.4.0']:
        x = F.upsample_bilinear(x, size=(299, 299))
      else:
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    # 299 x 299 x 3
    x = self.net.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.net.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.net.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.net.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.net.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.net.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.net.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.net.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.net.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6e(x)
    # 17 x 17 x 768
    # 17 x 17 x 768
    x = self.net.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.net.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.net.Mixed_7c(x)
    # 8 x 8 x 2048
    pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
    # 1 x 1 x 2048
    logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
    # 1000 (num_classes)
    return pool, logits


# A pytorch implementation of cov, from Modar M. Alfadly
# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt 
def sqrt_newton_schulz(A, numIters, dtype=None):
  with torch.no_grad():
    if dtype is None:
      dtype = A.type()
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    for i in range(numIters):
      T = 0.5*(3.0*I - Z.bmm(Y))
      Y = Y.bmm(T)
      Z = T.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
  return sA


# FID calculator from TTUR--consider replacing this with GPU-accelerated cov
# calculations using torch?
def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Numpy implementation of the Frechet Distance.
  Taken from https://github.com/bioinf-jku/TTUR
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
  Stable version by Dougal J. Sutherland.
  Params:
  -- mu1   : Numpy array containing the activations of a layer of the
             inception net (like returned by the function 'get_predictions')
             for generated samples.
  -- mu2   : The sample mean over activations, precalculated on an 
             representive data set.
  -- sigma1: The covariance matrix over activations for generated samples.
  -- sigma2: The covariance matrix over activations, precalculated on an 
             representive data set.
  Returns:
  --   : The Frechet Distance.
  """

  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)

  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  assert mu1.shape == mu2.shape, \
    'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, \
    'Training and test covariances have different dimensions'

  diff = mu1 - mu2

  # Product might be almost singular
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    msg = ('fid calculation produces singular product; '
           'adding %s to diagonal of cov estimates') % eps
    print(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # Numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
    print('wat')
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError('Imaginary component {}'.format(m))
    covmean = covmean.real  

  tr_covmean = np.trace(covmean) 

  out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
  return out


def torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Pytorch implementation of the Frechet Distance.
  Taken from https://github.com/bioinf-jku/TTUR
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
  Stable version by Dougal J. Sutherland.
  Params:
  -- mu1   : Numpy array containing the activations of a layer of the
             inception net (like returned by the function 'get_predictions')
             for generated samples.
  -- mu2   : The sample mean over activations, precalculated on an 
             representive data set.
  -- sigma1: The covariance matrix over activations for generated samples.
  -- sigma2: The covariance matrix over activations, precalculated on an 
             representive data set.
  Returns:
  --   : The Frechet Distance.
  """


  assert mu1.shape == mu2.shape, \
    'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, \
    'Training and test covariances have different dimensions'

  diff = mu1 - mu2
  # Run 50 itrs of newton-schulz to get the matrix sqrt of sigma1 dot sigma2
  covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze()  
  out = (diff.dot(diff) +  torch.trace(sigma1) + torch.trace(sigma2)
         - 2 * torch.trace(covmean))
  return out


# Calculate Inception Score mean + std given softmax'd logits and number of splits
def calculate_inception_score(pred, num_splits=10):
  scores = []
  for index in range(num_splits):
    pred_chunk = pred[index * (pred.shape[0] // num_splits): (index + 1) * (pred.shape[0] // num_splits), :]
    kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
    kl_inception = np.mean(np.sum(kl_inception, 1))
    scores.append(np.exp(kl_inception))
  return np.mean(scores), np.std(scores)


@GAN_METRIC_REGISTRY.register()
class PyTorchFIDISScore(object):

  def __init__(self, cfg, **kwargs):
    """

    """
    # fmt: off
    self.torch_fid_stat                   = cfg.torch_fid_stat
    self.num_inception_images             = get_attr_kwargs(cfg, 'num_inception_images', default=50000, **kwargs)
    self.IS_splits                        = get_attr_kwargs(cfg, 'IS_splits', default=10, **kwargs)
    self.calculate_FID_use_torch          = get_attr_kwargs(cfg, 'calculate_FID_use_torch', default=False, **kwargs)
    self.no_FID                           = get_attr_kwargs(cfg, 'no_FID', default=False, **kwargs)
    # fmt: on

    self.logger = logging.getLogger('tl')
    if os.path.isfile(self.torch_fid_stat):
      self.logger.info(f"Loading torch_fid_stat : {self.torch_fid_stat}")
      self.data_mu = np.load(self.torch_fid_stat)['mu']
      self.data_sigma = np.load(self.torch_fid_stat)['sigma']
    else:
      self.logger.warning(f"torch_fid_stat does not exist: {self.torch_fid_stat}")

    # Load inception_v3 network
    self.inception_net = self._load_inception_net()

    ws = comm.get_world_size()
    self.num_inception_images = self.num_inception_images // ws
    pass

  @staticmethod
  def _load_inception_net(device='cuda'):
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model = WrapInception(inception_model.eval()).cuda()
    inception_model.eval()
    net = inception_model.to(device)
    return net

  def _accumulate_inception_activations(
        self, sample_func, net, num_inception_images,
        show_process=True, as_numpy=False, stdout=sys.stdout):

    pool, logits = [], []
    count = 0
    net.eval()
    while (count) < num_inception_images:
      if show_process:
        print('\r',
              end=f'PyTorch FID IS Score forwarding: [{count}/{num_inception_images}]',
              file=stdout, flush=True)
      with torch.no_grad():
        try:
          images = sample_func()
        except StopIteration:
          break
        pool_val, logits_val = net(images.float())
        logits_val = F.softmax(logits_val, 1)

        if as_numpy:
          pool_val = pool_val.cpu().numpy()
          logits_val = logits_val.cpu().numpy()
        pool += [pool_val]
        logits += [logits_val]

        count += images.size(0)
    if show_process:
      print(f'rank {comm.get_rank()}', file=stdout)

    if as_numpy:
      pool, logits = np.concatenate(pool, 0), np.concatenate(logits, 0)
    else:
      pool, logits = torch.cat(pool, 0), torch.cat(logits, 0)
    return pool, logits

  def __call__(self, sample_func, return_fid_stat=False, num_inception_images=None, stdout=sys.stdout):
    start_time = time.time()

    if num_inception_images is None:
      num_inception_images = self.num_inception_images
    pool, logits = self._accumulate_inception_activations(
      sample_func, net=self.inception_net, num_inception_images=num_inception_images,
      as_numpy=True, stdout=stdout)

    pool = self._gather_data(pool[:num_inception_images], is_numpy=True)
    logits = self._gather_data(logits[:num_inception_images], is_numpy=True)

    if return_fid_stat:
      if comm.is_main_process():
        self.logger.info(f"Num of images: {len(pool)}")
        mu, sigma = self._get_FID_stat(pool=pool)
      else:
        mu, sigma = 0, 0
      return mu, sigma

    if comm.is_main_process():
      self.logger.info(f"Num of images: {len(pool)}")
      IS_mean_torch, IS_std_torch = calculate_inception_score(logits, num_splits=self.IS_splits)

      FID_torch = self._calculate_FID(pool=pool, no_fid=self.no_FID, use_torch=self.calculate_FID_use_torch)
    else:
      IS_mean_torch = IS_std_torch = FID_torch = 0

    elapsed_time = time.time() - start_time
    time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    self.logger.info('Elapsed time: %s' % (time_str))
    del pool, logits
    comm.synchronize()
    return FID_torch, IS_mean_torch, IS_std_torch

  def calculate_fid_stat_of_dataloader(
        self, data_loader, sample_func=None, return_fid_stat=False, num_images=float('inf'), save_fid_stat=True):
    if sample_func is None:
      class SampleClass(object):
        def __init__(self, data_loader):
          self.data_iter = iter(data_loader)
        def __call__(self, *args, **kwargs):
          """
          :return: images: [-1, 1]
          """
          inputs = next(self.data_iter)
          # images = [x["image"].to('cuda') for x in inputs]
          # images = torch.stack(images)
          images, labels = inputs
          images = images.to('cuda')
          return images
      sample_func = SampleClass(data_loader)

    data, label = next(iter(data_loader))
    num_inception_images = len(data) * len(data_loader)
    num_inception_images = min(num_images, num_inception_images)
    pool, logits = self._accumulate_inception_activations(
      sample_func, net=self.inception_net, num_inception_images=num_inception_images, as_numpy=True)

    pool = self._gather_data(pool[:num_inception_images], is_numpy=True)
    logits = self._gather_data(logits[:num_inception_images], is_numpy=True)

    if return_fid_stat:
      if comm.is_main_process():
        self.logger.info(f"Num of images: {len(pool)}")
        mu, sigma = self._get_FID_stat(pool=pool)
      else:
        mu, sigma = 0, 0
      return mu, sigma

    if comm.is_main_process():
      self.logger.info(f"Num of images: {len(pool)}")
      IS_mean, IS_std = calculate_inception_score(logits, self.IS_splits)
      self.logger.info(f'dataset IS_mean: {IS_mean:.3f} +- {IS_std}')

      if save_fid_stat:
        mu, sigma = self._get_FID_stat(pool=pool)
        self.logger.info(f'Saving torch_fid_stat to {self.torch_fid_stat}')
        os.makedirs(os.path.dirname(self.torch_fid_stat), exist_ok=True)
        np.savez(self.torch_fid_stat, **{'mu': mu, 'sigma': sigma})
    comm.synchronize()

  def _get_FID_stat(self, pool):
    mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
    return mu, sigma

  def _gather_data(self, data, is_numpy=False):
    data_list = comm.gather(data=data)
    if len(data_list) > 0:
      if is_numpy:
        data = np.concatenate(data_list, axis=0)
      else:
        data = torch.cat(data_list, dim=0).to('cuda')
    return data

  def _calculate_FID(self, pool, no_fid, use_torch=False,):

    if no_fid:
      FID = 0
    else:
      if use_torch:
        mu, sigma = torch.mean(pool, 0), torch_cov(pool, rowvar=False)
        FID = torch_calculate_frechet_distance(
          mu, sigma,
          torch.tensor(self.data_mu).float().cuda(),
          torch.tensor(self.data_sigma).float().cuda())
        FID = float(FID.cpu().numpy())
      else:
        mu, sigma = self._get_FID_stat(pool=pool)
        FID = numpy_calculate_frechet_distance(mu, sigma, self.data_mu, self.data_sigma)
    return FID

  def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
    return numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)

  def get_pool_and_logits(self, images):
    self.inception_net.eval()
    with torch.no_grad():
      pool, logits = self.inception_net(images.float())
      logits = F.softmax(logits, 1)

    pool = pool.cpu().numpy()
    logits = logits.cpu().numpy()
    return pool, logits

  def calculate_IS(self, logits, num_splits=10):
    scores = []
    for index in range(num_splits):
      pred_chunk = logits[index * (logits.shape[0] // num_splits): (index + 1) * (logits.shape[0] // num_splits), :]
      kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
      kl_inception = np.mean(np.sum(kl_inception, 1))
      scores.append(np.exp(kl_inception))
    return np.mean(scores), np.std(scores)

  @staticmethod
  def sample(G, z, parallel):
    """
    # Sample function for use with inception metrics
    :param z:
    :param parallel:
    :return:
    """
    with torch.no_grad():
      z.sample_()
      G.eval()

      if parallel:
        G_z = nn.parallel.data_parallel(G, (z,))
      else:
        G_z = G(z)

      G.train()
      return G_z

  @staticmethod
  def test_case_evaluate_FID_IS():
    import torch
    from template_lib.v2.GAN.evaluation import build_GAN_metric

    cfg_str = """
                update_cfg: true
                GAN_metric:
                  torch_fid_stat: "datasets/fid_stats_torch_cifar10.npz"
                  num_inception_images: 50000
                """
    config = EasyDict(yaml.safe_load(cfg_str))
    cfg = PyTorchFIDISScore.update_cfg(config)

    FID_IS_torch = build_GAN_metric(cfg.GAN_metric)

    class SampleFunc(object):
      def __init__(self, G, z):
        self.G = G
        self.z = z

      def __call__(self, *args, **kwargs):
        with torch.no_grad():
          z_sample = self.z.normal_(0, 1)
          # self.G.eval()
          # G_z = self.G(z_sample)
          G_z = self.G.normal_(0, 1)
        return G_z

    bs = 64
    z_dim = 128
    img_size = 32
    z = torch.empty((bs, z_dim)).cuda()
    G = torch.empty((bs, 3, img_size, img_size)).cuda()
    sample_func = SampleFunc(G=G, z=z)
    FID, IS_mean, IS_std = FID_IS_torch(sample_func=sample_func)
    logging.getLogger('tl').info(f'IS_mean_tf:{IS_mean:.3f} +- {IS_std:.3f}\n\tFID_tf: {FID:.3f}')
    pass


