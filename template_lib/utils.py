import unittest
import os, sys
import numpy as np


def shuffle_list(*ls):
  """ shuffle multiple list at the same time

  :param ls:
  :return:
  """
  from random import shuffle
  l = list(zip(*ls))
  shuffle(l)
  return zip(*l)


def save_args_txt(path, args):
  with open(os.path.join(path), 'w') as f:
    for k, v in sorted(args.__dict__.items()):
      f.write('{}={}\n'.format(k, v))


def save_args_pickle_and_txt(obj, path):
  import pickle
  with open(path, 'wb') as f:
    pickle.dump(obj, f)
  root, ext = os.path.splitext(path)
  save_args_txt(root + '.txt', obj)


def load_args_pickle(path):
  import pickle
  with open(path, 'rb') as f:
    args = pickle.load(f)
  return args


class BatchImages(object):
  def __init__(self):
    pass

  @staticmethod
  def split(num_images):
    assert type(num_images) == int
    t = int(np.floor(np.sqrt(num_images)))
    for row in range(t, 0, -1):
      if num_images % row == 0:
        return row, int(num_images / row)

  @staticmethod
  def merge_batch_images_to_one_auto(b_images, keep_dim=False, ticks=False):
    """

    :param b_images: (b, h, w, c)
    :return: np.array: merged image
    """
    b, h, w, c = b_images.shape
    row, col = BatchImages.split(b)
    b_images = np.reshape(b_images, [row, col, h, w, c])
    b_images = np.transpose(b_images, [0, 2, 1, 3, 4])
    merged_image = np.reshape(b_images, [row * h, col * w, c])

    if keep_dim:
      merged_image = np.expand_dims(merged_image, axis=0)

    if ticks:
      yticks = [i * h for i in range(1, row)]
      xticks = [i * w for i in range(1, col)]
      return merged_image, xticks, yticks

    return merged_image

  @staticmethod
  def merge_batch_images_to_one(images, row, col, keep_dim=True):
    """ merge images into an image with (row * h) * (col * w)

    :param images: (b, h, w, c)
    :param row:
    :param col:
    :return:
    """
    b, h, w, c = images.shape
    assert row * col >= b
    merged_img = np.zeros((h * row, w * col, c))
    for idx, image in enumerate(images):
      j = idx // col
      i = idx % col
      merged_img[j * h: j * h + h, i * w: i * w + w, ...] = image

    if keep_dim:
      merged_img = np.expand_dims(merged_img, axis=0)
    return merged_img

  @staticmethod
  def plt_grid_show(x):
    """ Show batch images
    :param x: (b, h, w, c), dtype: np.uint8
    :return:
    """
    import matplotlib.pyplot as plt
    assert x.dtype == np.uint8

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, xticks, yticks = BatchImages.merge_batch_images_to_one_auto(x, keep_dim=False, ticks=False)
    if x.shape[-1] == 3:
      ax.imshow(x)
    elif x.shape[-1] == 1:
      x = np.squeeze(x, axis=2)
      ax.imshow(x, cmap='gray')

    ax.set_xticks(xticks, minor=True)
    ax.set_yticks(yticks, minor=True)
    ax.xaxis.grid(True, which='minor', color='w')
    ax.yaxis.grid(True, which='minor', color='w')
    plt.xticks([])
    plt.yticks([])
    fig.show()
    return fig

  @staticmethod
  def normalize_img_to_prob(imgs):
    assert imgs.dtype == np.uint8
    imgs = imgs.astype(np.float64)
    imgs = np.clip(imgs, a_min=0, a_max=255)
    if len(imgs.shape) == 4:
      # (b, h, w, c)
      imgs /= imgs.sum(axis=(1, 2, 3), keepdims=True)
    elif len(imgs.shape) == 3 and imgs.shape[-1] == 1:
      # (h, w, c)
      imgs /= imgs.sum(axis=(0, 1), keepdims=True)
    elif len(imgs.shape) == 2:
      # (h, w)
      imgs /= imgs.sum(axis=(0, 1), keepdims=True)
    return imgs

  @staticmethod
  def compute_batch_images_emd(b_img1, b_img2, eps=0):
    """

    :param b_img1:
    :param b_img2:
    :return:
    """
    import ot
    assert b_img1.shape[-1] == 1 and b_img2.shape[-1] == 1

    b_img1 = b_img1.astype(np.float64)
    b_img2 = b_img2.astype(np.float64)
    # eps: avoid emd be zero
    b_img1 = np.clip(b_img1, a_min=eps, a_max=255)
    b_img1 /= b_img1.sum(axis=(1, 2, 3), keepdims=True)
    b_img2 = np.clip(b_img2, a_min=eps, a_max=255)
    b_img2 /= b_img2.sum(axis=(1, 2, 3), keepdims=True)

    b, h, w, c = b_img1.shape
    b_img1 = b_img1.reshape((b, -1))
    b_img2 = b_img2.reshape((b, -1))

    xx, yy = np.meshgrid(np.arange(h), np.arange(w))
    xy = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    M = ot.dist(xy, xy)
    emd = np.zeros((b, 1))

    for idx in range(b):
      xapp1 = b_img1[idx]
      xapp2 = b_img2[idx]
      dist = ot.emd2(xapp1, xapp2, M)
      assert dist > 0
      emd[idx] = dist

    return emd


class AverageMeter():
  """ Computes and stores the average and current value """

  def __init__(self):
    self.reset()

  def reset(self):
    """ Reset all statistics """
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    """ Update statistics """
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class Testing(unittest.TestCase):
  """

  """

  def test_plt_grid_show(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=1,2,3,4;
        python -c "import test_train; \
        test_train.Testing().test_demo()"
    :return:
    """
    batch_data = np.ones((32, 28, 28, 1)).astype('uint8')
    BatchImages.plt_grid_show(batch_data)
