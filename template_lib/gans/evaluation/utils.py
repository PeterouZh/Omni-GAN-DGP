import numpy as np
import torch
import sys
from template_lib.d2.utils import comm

__all__ = ['get_sample_imgs_list', "get_sample_imgs_list_ddp"]

def get_sample_imgs_list(sample_func, num_imgs=50000, as_numpy=True, stdout=sys.stdout):
  """

  :param sample_func: return imgs (b, c, h, w), range [-1, 1]
  :param num_imgs:
  :param stdout:
  :return:
  """

  imgs_list = list()
  imgs = sample_func()
  bs = len(imgs)
  eval_iter = (num_imgs - 1) // bs + 1
  for iter_idx in range(eval_iter):
    print('\r', end='sample images [%d/%d]' % (iter_idx * bs, eval_iter * bs), file=stdout, flush=True)

    # Generate a batch of images
    imgs = sample_func()
    if as_numpy:
      imgs = imgs.cpu().numpy()
    imgs_list.append(imgs)
  if as_numpy:
    imgs = np.concatenate(imgs_list, axis=0)
  else:
    imgs = torch.cat(imgs_list, dim=0)
  print('', file=stdout)
  return imgs


def get_sample_imgs_list_ddp(sample_func, num_imgs=50000, as_numpy=True, stdout=sys.stdout):
  """

  :param sample_func:
  :param num_imgs:
  :param stdout:
  :return:
  """
  import torch
  from template_lib.d2.utils import comm

  ws = comm.get_world_size()
  num_imgs = num_imgs // ws
  imgs = get_sample_imgs_list(sample_func=sample_func, num_imgs=num_imgs,
                              as_numpy=as_numpy, stdout=stdout)

  imgs_list = comm.gather(imgs)
  if len(imgs_list) > 0:
    if as_numpy:
      imgs = np.concatenate(imgs_list, axis=0)
    else:
      imgs = torch.cat(imgs_list, dim=0)
  return imgs


