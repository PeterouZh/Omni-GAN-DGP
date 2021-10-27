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
from template_lib.d2.data.build import DATASET_MAPPER_REGISTRY


@DATASET_MAPPER_REGISTRY.register()
class LSUNDatasetMapper(object):
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
    t = transforms.Compose([
      transforms.Scale(img_size),
      transforms.CenterCrop(img_size),
      transforms.ToTensor(),
      transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return t

  def __init__(self, cfg, **kwargs):

    self.img_size             = get_attr_kwargs(cfg, 'img_size', **kwargs)
    self.dataset_name         = get_attr_kwargs(cfg, 'dataset_name', **kwargs)

    self.transform = self.build_transform(img_size=self.img_size)

  def __call__(self, dataset_dict):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

    Returns:
        dict: a format that builtin models in detectron2 accept
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    image_index = dataset_dict['image_index']

    metadata = MetadataCatalog.get(self.dataset_name)
    dataset = metadata.get('dataset')

    image, label = dataset[image_index]
    dataset_dict['image'] = self.transform(image)
    dataset_dict['label'] = label
    return dataset_dict


def get_dict(name, data_path, classes, **kwargs):


  dataset = datasets.LSUN(root=data_path, classes=classes)

  meta_dict = {}
  meta_dict['dataset'] = dataset
  meta_dict['num_samples'] = len(dataset)
  MetadataCatalog.get(name).set(**meta_dict)

  dataset_dicts = []
  data_iter = range(len(dataset))
  for idx, index in enumerate(data_iter):
    record = {}

    record["image_id"] = idx
    record["image_index"] = index
    dataset_dicts.append(record)
  return dataset_dicts


data_path = "datasets/lsun/"
registed_name_list = [
  'lsun_bedroom_train',
  'lsun_bedroom_val',
]

registed_func_list = [
  get_dict,
  get_dict,
]

kwargs_list = [
  {'classes': ['bedroom_train'], },
  {'classes': ['bedroom_val'], },
]

for name, func, kwargs in zip(registed_name_list, registed_func_list, kwargs_list):
  # warning : lambda must specify keyword arguments
  DatasetCatalog.register(name, (lambda name=name, func=func, data_path=data_path, kwargs=kwargs:
                                 func(name=name, data_path=data_path, **kwargs)))


pass