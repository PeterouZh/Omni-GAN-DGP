import functools
import os
import numpy as np
import random
import copy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from detectron2.utils.visualizer import Visualizer

from .build import DATASET_MAPPER_REGISTRY


@DATASET_MAPPER_REGISTRY.register()
class STL10DatasetMapper:
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

  def __init__(self, cfg, is_train=True):
    img_size = cfg.dataset.img_size
    self.transform = self.build_transform(img_size=img_size)
    self.is_train = is_train

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


def get_dict(name, data_path, split, labels_path=None):
  stldataset = datasets.STL10(root=data_path, split=split, download=True)
  if labels_path is not None:
    labels_path = os.path.expanduser(labels_path)
    stldataset.labels = np.load(labels_path)['plabels']

  meta_dict = {}
  meta_dict['num_images'] = len(stldataset)
  MetadataCatalog.get(name).set(**meta_dict)

  dataset_dicts = []
  data_iter = iter(stldataset)
  for idx, (img, label) in enumerate(data_iter):
    record = {}

    record["image_id"] = idx
    record["height"] = img.height
    record["width"] = img.width
    record["image"] = img
    record["label"] = label
    dataset_dicts.append(record)
  return dataset_dicts


from detectron2.data import DatasetCatalog, MetadataCatalog

data_path = "datasets/stl10/"
registed_name = ['stl10_train+unlabeled_predicted_labels',
                 'stl10_test']
split_list = ['train+unlabeled',
              'test']
labels_path_list = ['datasets/stl10_plabels.npz',
                    None]


for name, split, labels_path in zip(registed_name, split_list, labels_path_list):
  # warning : lambda must specify keyword arguments
  DatasetCatalog.register(
    name, (lambda name=name, data_path=data_path, split=split, labels_path=labels_path:
           get_dict(name=name, data_path=data_path, split=split, labels_path=labels_path)))


if __name__ == '__main__':
  import matplotlib.pylab as plt
  dataset_dicts = get_dict(data_path=data_path, split=split_list[0],
                           labels_path=labels_path_list[0])
  stl10_metadata = MetadataCatalog.get(registed_name[0])
  for d in random.sample(dataset_dicts, 3):
    img = np.asarray(d["image"])
    visualizer = Visualizer(img, metadata=stl10_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    plt.imshow(vis.get_image())
    plt.show()
    # cv2_imshow(vis.get_image()[:, :, ::-1])