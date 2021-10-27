import matplotlib.pylab  as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import functools
import os
import numpy as np
import random
import copy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from detectron2.data import DatasetCatalog, MetadataCatalog

from template_lib.utils import get_attr_kwargs
from .build import DATASET_MAPPER_REGISTRY


@DATASET_MAPPER_REGISTRY.register()
class PointsDatasetMapper(object):
  """
  A callable which takes a dataset dict in Detectron2 Dataset format,
  and map it into a format used by the model.

  This is the default callable to be used to map your dataset dict into training data.
  You may need to follow it to implement your own one for customized logic.

  The callable currently does the following:

  1. Read the image from "file_name"
  2. Applies cropping/geometric transforms to the image and annotations
  3. Prepare data and annotations to Tensor and :class:`Instances`
  """
  def build_transform(self, img_size):

    return None

  def __init__(self, cfg, **kwargs):

    # self.img_size             = get_attr_kwargs(cfg, 'img_size', **kwargs)

    # self.transform = self.build_transform(img_size=self.img_size)
    pass

  def __call__(self, dataset_dict):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

    Returns:
        dict: a format that builtin models in detectron2 accept
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    points = dataset_dict['points']
    dataset_dict['points'] = points.permute(1, 0)

    return dataset_dict


def _normalize_data(data):
  N, c = data.shape
  min_scale = 1
  for i in range(c):
    vmin = data[:, i].min()
    vmax = data[:, i].max()
    scale = 2. / (vmax - vmin)
    if scale < min_scale:
      min_scale = scale
  data = data * min_scale

  for i in range(c):
    vmin = data[:, i].min()
    vmax = data[:, i].max()
    bias = (vmax + vmin) / 2.
    data[:, i] = data[:, i] - bias
  return data


def plot_3d_point_cloud(x, y, z, in_u_sphere=False, marker='.', s=8, alpha=.8,
                        elev=10, azim=240, axes=None, colors='b', *args, **kwargs):

  sc = axes.scatter(x, y, z, marker=marker, s=s, alpha=alpha, c=colors, *args, **kwargs)
  axes.view_init(elev=elev, azim=azim)

  if in_u_sphere:
    axes.set_xlim3d(-0.5, 0.5)
    axes.set_ylim3d(-0.5, 0.5)
    axes.set_zlim3d(-0.5, 0.5)
  else:
    miv = 1.2 * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
    mav = 1.2 * np.max([np.max(x), np.max(y), np.max(z)])
    axes.set_xlim(miv, mav)
    axes.set_ylim(miv, mav)
    axes.set_zlim(miv, mav)
    plt.tight_layout()

  if 'c' in kwargs:
    plt.colorbar(sc)

  return axes

def plot_points(points, show=True):
  fig = plt.figure(figsize=(5, 5))
  fig.tight_layout()
  points = points.squeeze()
  ax = fig.add_subplot(111, projection='3d')

  points = points.cpu().numpy()
  plot_3d_point_cloud(points[:, 0], points[:, 1], points[:, 2],
                      axes=ax, in_u_sphere=False)
  if show:
    fig.show()
  # plt.close(fig)
  pass


def get_dict_swiss_roll(name, n_samples, noise, **kwargs):
  from sklearn.datasets.samples_generator import make_swiss_roll

  data, t = make_swiss_roll(n_samples, noise)
  # Make it thinner
  data[:, 1] *= 0.5
  # Normalize to range of [-1, 1]
  data = _normalize_data(data)

  data = data.astype(np.float32)
  pclouds = torch.from_numpy(data)
  pclouds = pclouds.contiguous()

  meta_dict = {}
  meta_dict['num_images'] = 1
  MetadataCatalog.get(name).set(**meta_dict)

  dataset_dicts = []

  record = {}

  record["id"] = 0
  record["num_samples"] = n_samples
  record["dim"] = 3
  record["points"] = pclouds
  dataset_dicts.append(record)

  return dataset_dicts




data_path = "datasets/points/"
registed_name_list = [
  'swiss_roll_5000',
]

registed_func_list = [
  get_dict_swiss_roll,
]

kwargs_list = [
  {'n_samples': 5000, 'noise': 0.05},
]

for name, func, kwargs in zip(registed_name_list, registed_func_list, kwargs_list):
  # warning : lambda must specify keyword arguments
  DatasetCatalog.register(name, (lambda name=name, func=func, data_path=data_path, kwargs=kwargs:
                                 func(name=name, data_path=data_path, **kwargs)))


pass