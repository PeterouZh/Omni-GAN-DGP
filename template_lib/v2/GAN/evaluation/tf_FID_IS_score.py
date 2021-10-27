from __future__ import absolute_import, division, print_function

import argparse
import traceback

import yaml
from easydict import EasyDict
import logging
import os, sys
import pathlib
import warnings

import numpy as np

# import tensorflow as tf
try:
  import tensorflow as tf
except ImportError:
  import traceback
  traceback.print_exc()
  print('Cannot import tf.')

from scipy import linalg
from imageio import imread
import tarfile

from template_lib.d2.utils import comm
from template_lib.v2.config import update_config

from template_lib.v2.GAN.evaluation.build import GAN_METRIC_REGISTRY


__all__ = ['TFFIDISScore']


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class InvalidFIDException(Exception):
  pass


def create_inception_graph(pth):
  """Creates a graph from saved GraphDef file."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(pth, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')


# -------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
  """Prepares inception net for batched usage and returns pool_3 layer. """
  layername = 'FID_Inception_Net/pool_3:0'
  pool3 = sess.graph.get_tensor_by_name(layername)
  ops = pool3.graph.get_operations()
  for op_idx, op in enumerate(ops):
    for o in op.outputs:
      shape = o.get_shape()
      if shape._dims != []:
        shape = [s.value for s in shape]
        new_shape = []
        for j, s in enumerate(shape):
          if s == 1 and j == 0:
            new_shape.append(None)
          else:
            new_shape.append(s)
        o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
  return pool3


# -------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=50, verbose=True,
                    stdout=sys.stdout):
  """Calculates the activations of the pool_3 layer for all images.

  Params:
  -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                   must lie between 0 and 256.
  -- sess        : current session
  -- batch_size  : the images numpy array is split into batches with batch size
                   batch_size. A reasonable batch size depends on the disposable hardware.
  -- verbose    : If set to True and parameter out_step is given, the number of calculated
                   batches is reported.
  Returns:
  -- A numpy array of dimension (num images, 2048) that contains the
     activations of the given tensor when feeding inception with the query tensor.
  """
  inception_layer = _get_inception_layer(sess)
  d0 = images.shape[0]
  if batch_size > d0:
    print(
      "warning: batch size is bigger than the data size. setting batch size to data size")
    batch_size = d0
  n_batches = d0 // batch_size
  n_used_imgs = n_batches * batch_size
  pred_arr = np.empty((n_used_imgs, 2048))
  for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size
    if verbose:
      print('\r',
            end='FID forwarding [%d/%d]'%(start, n_used_imgs),
            file=stdout, flush=True)
    batch = images[start:end]
    pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
    pred_arr[start:end] = pred.reshape(batch_size, -1)
  if verbose:
    print('', file=stdout)
  return pred_arr


# -------------------------------------------------------------------------------



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Numpy implementation of the Frechet Distance.
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

  Stable version by Dougal J. Sutherland.

  Params:
  -- mu1 : Numpy array containing the activations of the pool_3 layer of the
           inception net ( like returned by the function 'get_predictions')
           for generated samples.
  -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
             on an representive data set.
  -- sigma1: The covariance matrix over activations of the pool_3 layer for
             generated samples.
  -- sigma2: The covariance matrix over activations of the pool_3 layer,
             precalcualted on an representive data set.

  Returns:
  --   : The Frechet Distance.
  """

  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)

  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
  assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

  diff = mu1 - mu2

  # product might be almost singular
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
    warnings.warn(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError("Imaginary component {}".format(m))
    covmean = covmean.real

  tr_covmean = np.trace(covmean)

  return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# -------------------------------------------------------------------------------


def calculate_activation_statistics(
      images, sess, batch_size=50, verbose=True, stdout=sys.stdout):
  """Calculation of the statistics used by the FID.
  Params:
  -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                   must lie between 0 and 255.
  -- sess        : current session
  -- batch_size  : the images numpy array is split into batches with batch size
                   batch_size. A reasonable batch size depends on the available hardware.
  -- verbose     : If set to True and parameter out_step is given, the number of calculated
                   batches is reported.
  Returns:
  -- mu    : The mean over samples of the activations of the pool_3 layer of
             the incption model.
  -- sigma : The covariance matrix of the activations of the pool_3 layer of
             the incption model.
  """
  act = get_activations(images, sess, batch_size, verbose, stdout=stdout)
  mu = np.mean(act, axis=0)
  sigma = np.cov(act, rowvar=False)
  return mu, sigma


# ------------------
# The following methods are implemented to obtain a batched version of the activations.
# This has the advantage to reduce memory requirements, at the cost of slightly reduced efficiency.
# - Pyrestone
# ------------------


def load_image_batch(files):
  """Convenience method for batch-loading images
  Params:
  -- files    : list of paths to image files. Images need to have same dimensions for all files.
  Returns:
  -- A numpy array of dimensions (num_images,hi, wi, 3) representing the image pixel values.
  """
  return np.array([imread(str(fn)).astype(np.float32) for fn in files])


def get_activations_from_files(files, sess, batch_size=50, verbose=False):
  """Calculates the activations of the pool_3 layer for all images.

  Params:
  -- files      : list of paths to image files. Images need to have same dimensions for all files.
  -- sess        : current session
  -- batch_size  : the images numpy array is split into batches with batch size
                   batch_size. A reasonable batch size depends on the disposable hardware.
  -- verbose    : If set to True and parameter out_step is given, the number of calculated
                   batches is reported.
  Returns:
  -- A numpy array of dimension (num images, 2048) that contains the
     activations of the given tensor when feeding inception with the query tensor.
  """
  inception_layer = _get_inception_layer(sess)
  d0 = len(files)
  if batch_size > d0:
    print(
      "warning: batch size is bigger than the data size. setting batch size to data size")
    batch_size = d0
  n_batches = d0 // batch_size
  n_used_imgs = n_batches * batch_size
  pred_arr = np.empty((n_used_imgs, 2048))
  for i in range(n_batches):
    if verbose:
      print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="",
            flush=True)
    start = i * batch_size
    end = start + batch_size
    batch = load_image_batch(files[start:end])
    pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
    pred_arr[start:end] = pred.reshape(batch_size, -1)
    del batch  # clean up memory
  if verbose:
    print(" done")
  return pred_arr


def calculate_activation_statistics_from_files(files, sess, batch_size=50,
                                               verbose=False):
  """Calculation of the statistics used by the FID.
  Params:
  -- files      : list of paths to image files. Images need to have same dimensions for all files.
  -- sess        : current session
  -- batch_size  : the images numpy array is split into batches with batch size
                   batch_size. A reasonable batch size depends on the available hardware.
  -- verbose     : If set to True and parameter out_step is given, the number of calculated
                   batches is reported.
  Returns:
  -- mu    : The mean over samples of the activations of the pool_3 layer of
             the incption model.
  -- sigma : The covariance matrix of the activations of the pool_3 layer of
             the incption model.
  """
  act = get_activations_from_files(files, sess, batch_size, verbose)
  mu = np.mean(act, axis=0)
  sigma = np.cov(act, rowvar=False)
  return mu, sigma


# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
# -------------------------------------------------------------------------------
def check_or_download_inception(inception_path):
  """ Checks if the path to the inception file is valid, or downloads
      the file if it is not present. """
  INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
  if inception_path is None:
    inception_path = '/tmp'
  inception_path = pathlib.Path(inception_path)
  model_file = inception_path / 'classify_image_graph_def.pb'
  if not model_file.exists():
    print("Downloading Inception model")
    from urllib import request
    import tarfile
    fn, _ = request.urlretrieve(INCEPTION_URL)
    with tarfile.open(fn, mode='r') as f:
      f.extract('classify_image_graph_def.pb', str(model_file.parent))
  return str(model_file)


def _handle_path(path, sess, low_profile=False, stdout=sys.stdout):
  if isinstance(path, str) and path.endswith('.npz'):
    path = os.path.expanduser(path)
    f = np.load(path)
    m, s = f['mu'][:], f['sigma'][:]
    f.close()
  elif isinstance(path, list):
    assert (type(path[0]) == np.ndarray)
    x = np.array(path)
    m, s = calculate_activation_statistics(x, sess, stdout=stdout)
  elif os.path.isdir(path):
    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    if low_profile:
      m, s = calculate_activation_statistics_from_files(files, sess)
    else:
      x = np.array([imread(str(fn)).astype(np.float32) for fn in files])
      m, s = calculate_activation_statistics(x, sess)
      del x  # clean up memory
  else:
    assert 0
  return m, s


def calculate_fid_given_paths(paths, inception_path, low_profile=False):
  """ Calculates the FID of two paths. """
  # inception_path = check_or_download_inception(inception_path)

  for p in paths:
    if not os.path.exists(p):
      raise RuntimeError("Invalid path: %s" % p)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    m1, s1 = _handle_path(paths[0], sess, low_profile=low_profile)
    m2, s2 = _handle_path(paths[1], sess, low_profile=low_profile)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
  sess.close()

  return fid_value


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None):
  import torch
  with torch.no_grad():
    if dtype is None:
      dtype = A.type()
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    for i in range(numIters):
      T = 0.5*(3.0*I - Z.bmm(Y))
      Y = Y.bmm(T)
      Z = T.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
  return sA


@GAN_METRIC_REGISTRY.register(name_prefix=__name__)
class TFFIDISScore(object):
  def __init__(self, cfg, **kwargs):

    # fmt: off
    self.tf_inception_model_dir       = cfg.tf_inception_model_dir
    self.tf_fid_stat                  = cfg.tf_fid_stat
    self.num_inception_images         = getattr(cfg, 'num_inception_images', 50000)
    self.IS_splits                    = getattr(cfg, 'IS_splits', 10)
    # fmt: on

    self.logger = logging.getLogger('tl')
    ws = comm.get_world_size()
    self.num_inception_images = self.num_inception_images // ws
    self.tf_graph_name = 'FID_IS_Inception_Net'
    if os.path.isfile(self.tf_fid_stat):
      self.logger.info(f'Loading tf_fid_stat: {self.tf_fid_stat}')
      f = np.load(self.tf_fid_stat)
      self.mu_data, self.sigma_data = f['mu'][:], f['sigma'][:]
      f.close()
    else:
      self.logger.warning(f"tf_fid_stat does not exist: {self.tf_fid_stat}")

    self.tf_inception_model_dir = os.path.expanduser(self.tf_inception_model_dir)
    inception_path = self._check_or_download_inception(self.tf_inception_model_dir)
    self.logger.info('Load tf inception model in %s', inception_path)
    self._create_inception_graph(inception_path, name=self.tf_graph_name)
    self._create_inception_net()
    comm.synchronize()

  def _create_inception_graph(self, pth, name):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(pth, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name=name)

  def _check_or_download_inception(self, tf_inception_model_dir):
    MODEL_DIR = os.path.expanduser(tf_inception_model_dir)
    if not os.path.exists(MODEL_DIR):
      os.makedirs(MODEL_DIR)
    DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    model_file = os.path.join(MODEL_DIR, 'classify_image_graph_def.pb')
    if not os.path.exists(model_file):
      if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
          sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
          sys.stdout.flush()
        from six.moves import urllib
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
      tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)

    return model_file

  def _get_inception_layers(self, sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = f'{self.tf_graph_name}/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
      for o in op.outputs:
        shape = o.get_shape()
        if shape._dims != []:
          shape = [s.value for s in shape]
          new_shape = []
          for j, s in enumerate(shape):
            if s == 1 and j == 0:
              new_shape.append(None)
            else:
              new_shape.append(s)
          o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

    w = sess.graph.get_operation_by_name(f"{self.tf_graph_name}/softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)

    FID_pool3 = pool3
    IS_softmax = softmax
    return FID_pool3, IS_softmax

  def __call__(self, sample_func, return_fid_stat=False, num_inception_images=None,
               return_fid_logit=False, stdout=sys.stdout):
    import torch

    class SampleClass(object):
      def __init__(self, sample_func):
        self.sample_func = sample_func

      def __call__(self, *args, **kwargs):
        """
        :return: images: [0, 255]
        """
        images = self.sample_func()
        images = images.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).type(torch.uint8)
        images = images.cpu().numpy()
        return images

    sample_func = SampleClass(sample_func)

    if num_inception_images is None:
      num_inception_images = self.num_inception_images
    else:
      num_inception_images = num_inception_images // comm.get_world_size()

    pred_FIDs, pred_ISs = self._get_activations_with_sample_func(
      sample_func=sample_func, num_inception_images=num_inception_images, stdout=stdout)

    if return_fid_stat:
      if comm.is_main_process():
        self.logger.info(f"Num of images: {len(pred_FIDs)}")
        mu, sigma = self._calculate_fid_stat(pred_FIDs=pred_FIDs)
      else:
        mu, sigma = 0, 0
      if return_fid_logit:
        return mu, sigma, pred_FIDs
      else:
        return mu, sigma

    if comm.is_main_process():
      self.logger.info(f"Num of images: {len(pred_FIDs)}")
      IS_mean_tf, IS_std_tf = self._calculate_IS(pred_ISs=pred_ISs, IS_splits=self.IS_splits)

      # calculate FID stat
      mu = np.mean(pred_FIDs, axis=0)
      sigma = np.cov(pred_FIDs, rowvar=False)
      FID_tf = calculate_frechet_distance(mu, sigma, self.mu_data, self.sigma_data)

    else:
      FID_tf = IS_mean_tf = IS_std_tf = 0

    del pred_FIDs, pred_ISs
    comm.synchronize()
    return FID_tf, IS_mean_tf, IS_std_tf

  def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6, use_torch=False):
    if use_torch:
      fid = self._torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps)
    else:
      fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps)
    return fid

  def _gather_numpy_array(self, data):
    data_list = comm.gather(data=data)
    if len(data_list) > 0:
      data = np.concatenate(data_list, axis=0)
    return data

  def _create_inception_net(self):
    # create tf session and specify the gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = f'{comm.get_rank()}'
    self.sess = tf.Session(config=config)
    self.sess.run(tf.global_variables_initializer())
    self.FID_pool3, self.IS_softmax = self._get_inception_layers(self.sess)

  def _get_activations_with_sample_func(self, sample_func, num_inception_images, stdout=sys.stdout, verbose=True):

    pred_FIDs = []
    pred_ISs = []
    count = 0

    while (count) < num_inception_images:
      if verbose and comm.is_main_process():
        print('\r', end=f'TF FID IS Score forwarding: [{count}/{num_inception_images}]', file=stdout, flush=True)
      try:
        batch = sample_func()
        # batch_list = comm.gather(data=batch)
        # if len(batch_list) > 0:
        #   batch = np.concatenate(batch_list, axis=0)
      except StopIteration:
        break
      try:
        pred_FID, pred_IS = self.sess.run([self.FID_pool3, self.IS_softmax],
                                          {f'{self.tf_graph_name}/ExpandDims:0': batch})
      except KeyboardInterrupt:
        exit(-1)
      except:
        print(traceback.format_exc())
        continue
      count += len(batch)
      pred_FIDs.append(pred_FID)
      pred_ISs.append(pred_IS)
    if verbose: print(f'rank: {comm.get_rank()}', file=stdout)

    pred_FIDs = np.concatenate(pred_FIDs, 0).squeeze()
    pred_ISs = np.concatenate(pred_ISs, 0)
    # sess.close()

    pred_FIDs = self._gather_numpy_array(pred_FIDs[:num_inception_images])
    pred_ISs = self._gather_numpy_array(pred_ISs[:num_inception_images])
    comm.synchronize()
    return pred_FIDs, pred_ISs

  def get_pool_and_logits(self, images):
    images = images.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1)
    images = images.cpu().numpy()
    images = images.astype(np.uint8)

    pool, logits = self.sess.run([self.FID_pool3, self.IS_softmax],
                                 {f'{self.tf_graph_name}/ExpandDims:0': images})
    return pool, logits

  def calculate_IS(self, logits, num_splits=10):
    IS_mean, IS_std = self._calculate_IS(pred_ISs=logits, IS_splits=num_splits)
    return IS_mean, IS_std

  def _calculate_IS(self, pred_ISs, IS_splits=10):
    # calculate IS
    scores = []
    for i in range(IS_splits):
      part = pred_ISs[(i * pred_ISs.shape[0] // IS_splits): ((i + 1) * pred_ISs.shape[0] // IS_splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    IS_mean, IS_std = np.mean(scores), np.std(scores)
    return IS_mean, IS_std

  def calculate_fid_stat_of_dataloader(self, data_loader, sample_func=None, return_fid_stat=False,
                                       num_images=float('inf'), stdout=sys.stdout):
    import torch

    if sample_func is None:
      class SampleClass(object):
        def __init__(self, data_loader):
          self.data_iter = iter(data_loader)

        def __call__(self, *args, **kwargs):
          inputs = next(self.data_iter)
          images = [x["image"].to('cuda') for x in inputs]
          images = torch.stack(images)
          images = images.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).type(torch.uint8)
          images = images.cpu().numpy()
          return images

      sample_func = SampleClass(data_loader)

    num_inception_images = len(next(iter(data_loader))) * len(data_loader)
    num_inception_images = min(num_images, num_inception_images)
    pred_FIDs, pred_ISs = self._get_activations_with_sample_func(
      sample_func=sample_func, num_inception_images=num_inception_images, stdout=stdout)

    if return_fid_stat:
      if comm.is_main_process():
        self.logger.warning(f"Num of images: {len(pred_FIDs)}")
        mu, sigma = self._calculate_fid_stat(pred_FIDs=pred_FIDs)
      else:
        mu, sigma = 0, 0
      return mu, sigma

    if comm.is_main_process():
      self.logger.info(f"Num of images: {len(pred_FIDs)}")
      IS_mean, IS_std = self._calculate_IS(pred_ISs=pred_ISs, IS_splits=self.IS_splits)
      self.logger.info(f'dataset IS_mean: {IS_mean:.3f} +- {IS_std}')

      # calculate FID stat
      mu, sigma = self._calculate_fid_stat(pred_FIDs=pred_FIDs)
      self.logger.info(f'Saving tf_fid_stat to {self.tf_fid_stat}')
      os.makedirs(os.path.dirname(self.tf_fid_stat), exist_ok=True)
      np.savez(self.tf_fid_stat, **{'mu': mu, 'sigma': sigma})
    comm.synchronize()

  def _calculate_fid_stat(self, pred_FIDs):
    mu = np.mean(pred_FIDs, axis=0)
    sigma = np.cov(pred_FIDs, rowvar=False)
    return mu, sigma

  @staticmethod
  def _torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
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
    import torch

    mu1 = torch.from_numpy(mu1).cuda()
    sigma1 = torch.from_numpy(sigma1).cuda()
    mu2 = torch.from_numpy(mu2).cuda()
    sigma2 = torch.from_numpy(sigma2).cuda()

    diff = mu1 - mu2
    # Run 50 itrs of newton-schulz to get the matrix sqrt of sigma1 dot sigma2
    covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze()
    out = (diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2)
           - 2 * torch.trace(covmean))
    return out

  @staticmethod
  def test_case_calculate_fid_stat_CIFAR():
    from template_lib.d2.data import build_dataset_mapper
    from template_lib.d2template.trainer.base_trainer import build_detection_test_loader
    from template_lib.v2.GAN.evaluation import build_GAN_metric
    from template_lib.d2.utils.d2_utils import D2Utils
    from template_lib.v2.config_cfgnode import global_cfg

    from detectron2.utils import logger
    logger.setup_logger('d2')

    cfg = D2Utils.create_cfg()
    cfg.update(global_cfg)
    global_cfg.merge_from_dict(cfg)

    # fmt: off
    dataset_name                 = cfg.dataset_name
    IMS_PER_BATCH                = cfg.IMS_PER_BATCH
    img_size                     = cfg.img_size
    dataset_mapper_cfg           = cfg.dataset_mapper_cfg
    GAN_metric                   = cfg.GAN_metric
    # fmt: on

    num_workers = comm.get_world_size()
    batch_size = IMS_PER_BATCH // num_workers

    dataset_mapper = build_dataset_mapper(dataset_mapper_cfg, img_size=img_size)
    data_loader = build_detection_test_loader(
      cfg, dataset_name=dataset_name, batch_size=batch_size, mapper=dataset_mapper)

    FID_IS_tf = build_GAN_metric(GAN_metric)
    FID_IS_tf.calculate_fid_stat_of_dataloader(data_loader=data_loader)

    comm.synchronize()

    pass

  @staticmethod
  def test_case_evaluate_FID_IS():
    import torch
    from template_lib.v2.GAN.evaluation import build_GAN_metric

    cfg_str = """
              update_cfg: true
              GAN_metric:
                tf_fid_stat: "datasets/fid_stats_tf_cifar10.npz"
                tf_inception_model_dir: "datasets/GAN_eval/tf_inception_model"
                num_inception_images: 5000
              """
    config = EasyDict(yaml.safe_load(cfg_str))
    cfg = TFFIDISScore.update_cfg(config)

    FID_IS_tf = build_GAN_metric(cfg.GAN_metric)

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
    try:
      FID_tf, IS_mean_tf, IS_std_tf = FID_IS_tf(sample_func=sample_func, num_inception_images=5000)
      print(f'IS_mean_tf:{IS_mean_tf:.3f} +- {IS_std_tf:.3f}\n\tFID_tf: {FID_tf:.3f}')
    except:
      print("Error FID_IS_tf.")
      import traceback
      print(traceback.format_exc())
    pass

  @staticmethod
  def update_cfg(cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
              GAN_metric:
                name: TFFIDISScore
                tf_fid_stat: "datasets/tf_fid_stat_{dataset_name}_{img_size}.npz"
                tf_inception_model_dir: "datasets/tf_inception_model"
                num_inception_images: 50000

              """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg


if __name__ == '__main__':
  # from template_lib.v2.ddp import main
  # args = main()

  from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, build_parser
  parser = update_parser_defaults_from_yaml(parser=None, use_cfg_as_args=True)
  args = parser.parse_args()
  eval(args.run_func)()
  pass



