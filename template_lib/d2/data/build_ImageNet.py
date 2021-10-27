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

__all__ = ['ImageNetDatasetMapper']


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
class ImageNetDatasetMapper:
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
      CenterCropLongEdge(),
      transforms.Resize(img_size),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform

  def __init__(self, cfg):

    self.img_size               = cfg.dataset.img_size
    self.load_in_memory         = getattr(cfg.dataset, "load_in_memory", False)

    self.transform = self.build_transform(img_size=self.img_size)


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


def get_dict(name, data_path, images_per_class=np.inf, show_bar=False,
             load_in_memory=False, img_size=None):
  rank = comm.get_rank()
  index_filename = f'{name}/{name}_index_rank_{rank}.json'

  if os.path.exists(index_filename):
    print('Loading pre-saved Index file %s...' % index_filename)
    with open(index_filename, 'r') as fp:
      dataset_dicts = json.load(fp)

  else:
    print('Saving Index file %s...\n' % index_filename)
    dataset_dicts = []
    classes, class_to_idx = find_classes(data_path)
    data_path = os.path.expanduser(data_path)
    pbar = sorted(os.listdir(data_path))
    if show_bar:
      pbar = tqdm.tqdm(pbar, desc=f'get_dict, {index_filename}')

    idx = 0
    for target in pbar:
      d = os.path.join(data_path, target)
      if not os.path.isdir(d):
        continue
      num_imgs = 0
      for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
          if is_image_file(fname):
            if num_imgs >= images_per_class:
              break
            num_imgs += 1
            record = {}

            record["image_id"] = idx
            idx += 1
            # record["height"] = img.height
            # record["width"] = img.width
            image_path = os.path.join(root, fname)
            record["image_path"] = image_path
            record["label"] = class_to_idx[target]

            dataset_dicts.append(record)
    os.makedirs(os.path.dirname(index_filename), exist_ok=True)
    with open(index_filename, 'w') as fp:
      json.dump(dataset_dicts, fp)
    print('Saved Index file %s.' % index_filename)

  meta_dict = {}
  meta_dict['num_images'] = len(dataset_dicts)
  MetadataCatalog.get(name).set(**meta_dict)

  if load_in_memory:
    data_filename = f'{name}/{name}_data_dicts_rank_{rank}.pickle'
    if os.path.exists(data_filename):
      print(f'Loading dataset_dicts_with_images: {data_filename}')
      with open(data_filename, 'rb') as f:
        dataset_dicts_with_images = pickle.load(f)
    else:
      cfg = CfgNode()
      assert img_size is not None
      cfg.dataset = CfgNode()
      cfg.dataset.img_size = img_size
      mapper = ImageNetDatasetMapper(cfg)
      dataset_dicts_with_images = []
      print(f'Saving dataset_dicts_with_images to {data_filename}\n')
      pbar = dataset_dicts
      if show_bar:
        pbar = tqdm.tqdm(pbar, desc='Saving dataset_dicts_with_images')
      for dataset_dict in pbar:
        dataset_dicts_with_images.append(mapper(dataset_dict))

      os.makedirs(os.path.dirname(data_filename), exist_ok=True)
      with open(data_filename, 'wb') as f:
        pickle.dump(dataset_dicts_with_images, f, pickle.HIGHEST_PROTOCOL)
        print(f'Saved dataset_dicts_with_images to {data_filename}')
    return dataset_dicts_with_images

  return dataset_dicts


registed_names = ['imagenet_train',
                  'imagenet_train_100x1k',
                  'imagenet_train_5x1k',
                  'imagenet_train_2x1k_size_48_in_memory',
                  'imagenet_train_100x1k_size_48_in_memory',
                  'imagenet_train_size_32_in_memory'
                  ]
data_paths = ["datasets/imagenet/train",
              "datasets/imagenet/train",
              "datasets/imagenet/train",
              "datasets/imagenet/train",
              "datasets/imagenet/train",
              "datasets/imagenet/train",
              ]
images_per_class_list = [np.inf,
                         100,
                         5,
                         2,
                         100,
                         np.inf,
                         ]
kwargs_list = [{},
               {},
               {},
               {'load_in_memory': True, 'img_size': 48},
               {'load_in_memory': True, 'img_size': 48},
               {'load_in_memory': True, 'img_size': 32},
               ]

for name, data_path, images_per_class, kwargs in zip(registed_names, data_paths, images_per_class_list, kwargs_list):
  # warning : lambda must specify keyword arguments
  DatasetCatalog.register(
    name, (lambda name=name, data_path=data_path, images_per_class=images_per_class, kwargs=kwargs:
           get_dict(name=name, data_path=data_path, images_per_class=images_per_class, **kwargs)))
  # Save index json file
  # get_dict(name=name, data_path=data_path, images_per_class=images_per_class, show_bar=True)


if __name__ == '__main__':
  import matplotlib.pylab as plt
  dataset_dicts = get_dict(name=registed_names[-1], data_path=data_paths[-1],
                           images_per_class=images_per_class_list[-1], show_bar=True, **(kwargs_list[-1]))

  metadata = MetadataCatalog.get(registed_names[-1])
  for d in random.sample(dataset_dicts, 3):
    image = default_loader(d['image_path'])
    img = np.asarray(image)
    visualizer = Visualizer(img, metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    file_name = os.path.basename(d['image_path'])
    saved_dir = 'results/build_ImageNet'
    os.makedirs(saved_dir, exist_ok=True)
    vis.save(os.path.join(saved_dir, file_name))
    # plt.imshow(vis.get_image())
    # plt.show()
    # cv2_imshow(vis.get_image()[:, :, ::-1])