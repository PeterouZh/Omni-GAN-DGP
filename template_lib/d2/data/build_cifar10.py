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
class CIFAR10DatasetMapper(object):
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
    transform = transforms.Compose([
      transforms.Resize(img_size),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform

  def __init__(self, cfg, **kwargs):

    self.img_size             = get_attr_kwargs(cfg, 'img_size', **kwargs)

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
    image = dataset_dict['image']
    dataset_dict['image'] = self.transform(image)
    return dataset_dict


@DATASET_MAPPER_REGISTRY.register()
class CIFAR10ClassificationTrainMapper(object):

  def __init__(self, cfg, **kwargs):
    # self.img_size = get_attr_kwargs(cfg.dataset, 'img_size', kwargs=kwargs)

    self.transform = self.build_transform()
    pass

  def build_transform(self, ):

    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf)
    return train_transform

  def __call__(self, dataset_dict):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

    Returns:
        dict: a format that builtin models in detectron2 accept
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    image = dataset_dict['image']
    dataset_dict['image'] = self.transform(image)
    return dataset_dict


@DATASET_MAPPER_REGISTRY.register()
class CIFAR10ClassificationTestMapper(object):

  def __init__(self, cfg, **kwargs):
    # self.img_size = get_attr_kwargs(cfg.dataset, 'img_size', kwargs=kwargs)

    self.transform = self.build_transform()
    pass

  def build_transform(self, ):

    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
      transforms.ToTensor(),
      transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf)
    return train_transform

  def __call__(self, dataset_dict):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

    Returns:
        dict: a format that builtin models in detectron2 accept
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    image = dataset_dict['image']
    dataset_dict['image'] = self.transform(image)
    return dataset_dict



def get_dict(name, data_path, subset, **kwargs):
  if subset.lower() == 'train':
    train = True
  elif subset.lower() == 'test':
    train = False
  c10_dataset = datasets.CIFAR10(root=data_path, train=train, download=True)

  meta_dict = {}
  meta_dict['num_samples'] = len(c10_dataset)
  meta_dict['class_to_idx'] = c10_dataset.class_to_idx
  meta_dict['classes'] = c10_dataset.classes
  MetadataCatalog.get(name).set(**meta_dict)

  dataset_dicts = []
  data_iter = iter(c10_dataset)
  for idx, (img, label) in enumerate(data_iter):
    record = {}

    record["image_id"] = idx
    record["height"] = img.height
    record["width"] = img.width
    record["image"] = img
    record["label"] = int(label)
    dataset_dicts.append(record)
  return dataset_dicts


def get_sampled_dict(name, data_path, sampler, **kwargs):
  sampler = list(sampler)
  train = True
  c10_dataset = datasets.CIFAR10(root=data_path, train=train, download=True)

  meta_dict = {}
  meta_dict['num_samples'] = len(sampler)
  meta_dict['class_to_idx'] = c10_dataset.class_to_idx
  meta_dict['classes'] = c10_dataset.classes
  MetadataCatalog.get(name).set(**meta_dict)

  dataset_dicts = []
  data_iter = iter(c10_dataset)
  for idx, (img, label) in enumerate(data_iter):
    if not idx in sampler:
      continue
    record = {}

    record["image_id"] = idx
    record["height"] = img.height
    record["width"] = img.width
    record["image"] = img
    record["label"] = int(label)
    dataset_dicts.append(record)
  return dataset_dicts



data_path = "datasets/cifar10/"
registed_name_list = [
  'cifar10_train',
  'cifar10_test',
  'cifar10_0-25000',
  'cifar10_25000-50000',
]

registed_func_list = [
  get_dict,
  get_dict,
  get_sampled_dict,
  get_sampled_dict,
]

kwargs_list = [
  {'subset': 'train', },
  {'subset': 'test', },
  {'sampler': range(25000)},
  {'sampler': range(25000, 50000)},
]

for name, func, kwargs in zip(registed_name_list, registed_func_list, kwargs_list):
  # warning : lambda must specify keyword arguments
  DatasetCatalog.register(name, (lambda name=name, func=func, data_path=data_path, kwargs=kwargs:
                                 func(name=name, data_path=data_path, **kwargs)))


pass