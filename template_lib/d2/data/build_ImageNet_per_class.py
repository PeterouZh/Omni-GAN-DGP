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

from template_lib.d2.data.BigGAN import default_loader, find_classes, is_image_file
from template_lib.d2.data.build import DATASET_MAPPER_REGISTRY

__all__ = ['ImageNetDatasetPerClassMapper']


class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    return transforms.functional.center_crop(img, min(img.size))

  def __repr__(self):
    return self.__class__.__name__


@DATASET_MAPPER_REGISTRY.register()
class ImageNetDatasetPerClassMapper:
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
  def __init__(self, cfg):
    self.img_size          = cfg.dataset.img_size
    self.load_in_memory    = getattr(cfg.dataset, "load_in_memory", False)

    self.transform = self.build_transform(img_size=self.img_size)

  def build_transform(self, img_size):
    transform = transforms.Compose([
      CenterCropLongEdge(),
      transforms.Resize(img_size),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform

  def __call__(self, dataset_dict):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

    Returns:
        dict: a format that builtin models in detectron2 accept
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    if self.load_in_memory:
      return dataset_dict
    # USER: Write your own image loading if it's not from a file
    image_path = dataset_dict['image_path']
    image = default_loader(image_path)
    dataset_dict['image'] = self.transform(image)
    return dataset_dict


def get_dict(cache_dir, registed_name, data_path, class_idx, images_per_class=np.inf, use_cache=True):
  rank = comm.get_rank()
  index_filename = f'{cache_dir}/{registed_name}_index_rank_{rank}.json'

  if os.path.exists(index_filename) and use_cache:
    # print('Loading pre-saved Index file %s...' % index_filename)
    with open(index_filename, 'r') as fp:
      dataset_dicts = json.load(fp)

  else:
    # print('Saving Index file %s...\n' % index_filename)
    dataset_dicts = []

    data_path = os.path.expanduser(data_path)
    num_imgs = 0
    for root, _, fnames in sorted(os.walk(data_path)):
      for fname in sorted(fnames):
        if is_image_file(fname):
          if num_imgs >= images_per_class:
            break
          record = {}

          record["image_id"] = num_imgs
          num_imgs += 1
          # record["height"] = img.height
          # record["width"] = img.width
          image_path = os.path.join(root, fname)
          record["image_path"] = image_path
          record["label"] = class_idx

          dataset_dicts.append(record)
    os.makedirs(os.path.dirname(index_filename), exist_ok=True)
    with open(index_filename, 'w') as fp:
      json.dump(dataset_dicts, fp)
    # print('Saved Index file %s.' % index_filename)

  meta_dict = {}
  meta_dict['num_images'] = len(dataset_dicts)
  MetadataCatalog.get(registed_name).set(**meta_dict)


  return dataset_dicts


registed_names = ['imagenet_train_per_class',
                  'imagenet_train_12x1k_per_class',
                  ]
data_paths = ["datasets/imagenet/train",] * len(registed_names)
kwargs_list = [{'images_per_class': np.inf, 'use_cache': True},
               {'images_per_class': 12, 'use_cache': True},
               ]


classes, class_to_idx = find_classes(data_paths[0])
for name, root_path, kwargs in zip(registed_names, data_paths, kwargs_list):
  # warning : lambda must specify keyword arguments
  for class_path, idx in tqdm.tqdm(class_to_idx.items(), desc=f"Registering ImageNet per class: {name}"):
    registed_name = f'{name}_{class_path}'
    data_path = os.path.join(root_path, class_path)
    DatasetCatalog.register(
      registed_name,
      (lambda name=name, registed_name=registed_name, data_path=data_path, idx=idx, kwargs=kwargs:
       get_dict(cache_dir=name, registed_name=registed_name, data_path=data_path, class_idx=idx, **kwargs)))


  # DatasetCatalog.register(
  #   name, (lambda name=name, data_path=data_path, kwargs=kwargs:
  #          get_dict(name=name, data_path=data_path, **kwargs)))
  # Save index json file
  # get_dict(name=name, data_path=data_path, images_per_class=images_per_class, show_bar=True)


