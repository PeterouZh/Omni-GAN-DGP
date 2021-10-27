#!/usr/bin/env python3
""" Calculates the Frechet Inception Distance (FID) to evaluate GANs.

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.
"""

from __future__ import absolute_import, division, print_function

import logging
import os, sys
import pathlib
import warnings

import numpy as np

try:
  import tensorflow as tf
except ImportError:
  print('Cannot import tf.')

from scipy import linalg
from imageio import imread
import tarfile

from .build import GAN_METRIC_REGISTRY

__all__ = ['FIDScore', 'TFFIDScore']


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


@GAN_METRIC_REGISTRY.register()
class TFFIDScore(object):
  def __init__(self, cfg):

    self.tf_inception_model_dir = cfg.GAN_metric.tf_inception_model_dir

    self.logger = logging.getLogger('tl')
    self.logger.info('Load tf inception model in %s', self.tf_inception_model_dir)

    self.tf_inception_model_dir = os.path.expanduser(self.tf_inception_model_dir)
    inception_path = self.check_or_download_inception(self.tf_inception_model_dir)
    create_inception_graph(inception_path)
    pass

  def check_or_download_inception(self, tf_inception_model_dir):
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

  def calculate_fid_given_paths(self, fid_buffer, fid_stat,
                                low_profile=False, stdout=sys.stdout):
    """Calculates the FID of two paths.

    :param fid_buffer: dir containing images or list of images
                       images: (h, w, c), [0, 255]
    :param fid_stat: *.npz
    :param low_profile:
    :return:
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())
      m1, s1 = _handle_path(fid_buffer, sess, low_profile=low_profile,
                            stdout=stdout)
      m2, s2 = _handle_path(fid_stat, sess, low_profile=low_profile)
      fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    sess.close()

    return fid_value

  def get_activations_from_pytorch_dataloader(
          self, dataloader, sess, transform_to_uint8=False, stdout=sys.stdout):
    """Calculates the activations of the pool_3 layer for all images.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """

    import tqdm
    pbar = tqdm.tqdm(dataloader, file=stdout,
                     desc='get_activations_from_pytorch_dataloader')

    inception_layer = _get_inception_layer(sess)
    pred_arr_list = []
    for b_imgs, _ in pbar:
      if transform_to_uint8:
        batch = b_imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0) \
          .permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
      else:
        batch = b_imgs.cpu().numpy()
      try:
        pred = sess.run(inception_layer,
                        {'FID_Inception_Net/ExpandDims:0': batch})
      except:
        print('\nException when forwarding inception net')
        continue
      pred_arr = pred.reshape(pred.shape[0], -1)
      pred_arr_list.append(pred_arr)
      del batch  # clean up memory
    pred_arr_list = np.concatenate(pred_arr_list)
    print('Num of images: %d'%pred_arr_list.shape[0])
    return pred_arr_list

  def calculate_fid_stat_for_pytorch_dataset(
          self, dataloader, fid_stat, transform_to_uint8=False, stdout=sys.stdout):
    """Calculates the FID of two paths.

    :param dataloader: images: uint8 [0, 255], (b, h, w, c)
    :param fid_stat: *.npz
    :return:
    """
    fid_stat = os.path.expanduser(fid_stat)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())
      act = self.get_activations_from_pytorch_dataloader(
        dataloader=dataloader, sess=sess,
        transform_to_uint8=transform_to_uint8, stdout=stdout)

    sess.close()
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    np.savez(fid_stat, mu=mu, sigma=sigma)

FIDScore = TFFIDScore