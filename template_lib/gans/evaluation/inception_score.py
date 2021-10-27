# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import os.path
import sys
import tarfile

import numpy as np
import tensorflow as tf
from six.moves import urllib

from .build import GAN_METRIC_REGISTRY

__all__ = ['InceptionScore', 'TFInceptionScore']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

config = tf.ConfigProto(device_count = {'GPU': 0})
config.gpu_options.allow_growth = True


@GAN_METRIC_REGISTRY.register()
class TFInceptionScore(object):
  def __init__(self, cfg):

    self.tf_inception_model_dir = cfg.GAN_metric.tf_inception_model_dir

    self.logger = logging.getLogger('tl')
    self.logger.info('Load tf inception model in %s', self.tf_inception_model_dir)

    global softmax
    MODEL_DIR = os.path.expanduser(self.tf_inception_model_dir)
    if not os.path.exists(MODEL_DIR):
      os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]

    filepath = os.path.join(MODEL_DIR, filename)
    pb_file = os.path.join(MODEL_DIR, 'classify_image_graph_def.pb')
    if not os.path.exists(pb_file):
      if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
          sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
          sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
      tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='')
    # Works with an arbitrary minibatch size.
    with tf.Session(config=config) as sess:
      pool3 = sess.graph.get_tensor_by_name('pool_3:0')
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
      w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
      logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
      softmax = tf.nn.softmax(logits)
      sess.close()
    pass

  def get_inception_score(self, images, splits=10,
                          bs=100, stdout=sys.stdout):
    """
    # Call this function with list of images. Each of elements should be a
    # numpy array with values ranging from 0 to 255.
    :param images: [img, ], range: [0, 255]
    :param stdout:
    :return:
    """
    assert (type(images) == list)
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
      img = img.astype(np.float32)
      inps.append(np.expand_dims(img, 0))
    with tf.Session(config=config) as sess:
      preds = []
      n_batches = int(math.ceil(float(len(inps)) / float(bs)))
      # for i in tqdm(range(n_batches), desc="Calculate inception score",
      #               file=stdout):
      for i in range(n_batches):
        print('\r',
              end='IS forwarding [%d/%d]' % (i*bs, n_batches*bs),
              file=stdout, flush=True)
        # sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
      print('', file=stdout)
      preds = np.concatenate(preds, 0)
      scores = []
      for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):(
                  (i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

      sess.close()
    return np.mean(scores), np.std(scores)

  def get_inception_score_dataloader(self, dataloader, splits=10,
                                     stdout=sys.stdout):
    """
    # Call this function with list of images. Each of elements should be a
    # numpy array with values ranging from 0 to 255.
    :param images: [img, ], range: [0, 255]
    :param stdout:
    :return:
    """

    with tf.Session(config=config) as sess:
      preds = []
      bs = len(next(iter(dataloader))[0])
      n_batches = len(dataloader)
      for i, (img, _) in enumerate(dataloader):
        print('\r',
              end='Calculate inception score [%d/%d]' % (
              i * bs, n_batches * bs),
              file=stdout, flush=True)
        inp = self.img_float_to_uint(img)

        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
      print('', file=stdout)
      preds = np.concatenate(preds, 0)
      scores = []
      for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):(
                (i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

      sess.close()
    return np.mean(scores), np.std(scores)

  @staticmethod
  def img_float_to_uint(imgs):
    import torch
    uimgs = imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0) \
      .permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    return uimgs

InceptionScore = TFInceptionScore