import functools
import tqdm
import os
import numpy as np
import random
import copy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle
import json

from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils import comm
from detectron2.config import CfgNode

from template_lib.d2.data.build import DATASET_MAPPER_REGISTRY


def get_dict(registed_name, class_name, data_path, subset, **kwargs):
  if subset.lower() == 'train':
    train = True
  elif subset.lower() == 'test':
    train = False
  c10_dataset = datasets.CIFAR10(root=data_path, train=train, download=True)

  dataset_dicts = []
  data_iter = iter(c10_dataset)
  idx = 0
  for (img, label) in data_iter:
    if label != c10_dataset.class_to_idx[class_name]:
      continue
    record = {}
    record["image_id"] = idx
    idx += 1
    record["height"] = img.height
    record["width"] = img.width
    record["image"] = img
    record["label"] = int(label)
    dataset_dicts.append(record)

  meta_dict = {}
  meta_dict['num_images'] = len(dataset_dicts)
  meta_dict['class_to_idx'] = c10_dataset.class_to_idx
  meta_dict['classes'] = c10_dataset.classes
  meta_dict['class'] = class_name
  MetadataCatalog.get(registed_name).set(**meta_dict)

  return dataset_dicts


def find_classes(data_path):
  c10_dataset = datasets.CIFAR10(root=data_path, train=True, download=True)
  classes = c10_dataset.classes
  class_to_idx = c10_dataset.class_to_idx
  return classes, class_to_idx


data_path = "datasets/cifar10/"
registed_name_list = [
  'cifar10_train_per_class',
]
registed_func_list = [
  get_dict,
]
kwargs_list = [
  {'subset': 'train'},
]

classes, class_to_idx = find_classes(data_path)
meta_dict = {}
meta_dict['class_to_idx'] = class_to_idx
meta_dict['classes'] = classes
MetadataCatalog.get(registed_name_list[0]).set(**meta_dict)

for name, registed_func, kwargs in zip(registed_name_list, registed_func_list, kwargs_list):
  # warning : lambda must specify keyword arguments
  for class_name, idx in tqdm.tqdm(class_to_idx.items(), desc=f"Registering cifar10 per class: {name}"):
    registed_name = f'{name}_{class_name}'

    # registed_func(registed_name=registed_name, class_name=class_name, data_path=data_path, **kwargs)

    DatasetCatalog.register(
      registed_name,
      (lambda registed_func=registed_func, registed_name=registed_name, class_name=class_name,
              data_path=data_path, kwargs=kwargs:
       registed_func(registed_name=registed_name, class_name=class_name, data_path=data_path, **kwargs)))


  # DatasetCatalog.register(
  #   name, (lambda name=name, data_path=data_path, kwargs=kwargs:
  #          get_dict(name=name, data_path=data_path, **kwargs)))
  # Save index json file
  # get_dict(name=name, data_path=data_path, images_per_class=images_per_class, show_bar=True)


