import functools
import os
import numpy as np
import random
import copy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from detectron2.data import DatasetCatalog, MetadataCatalog

from nasbench import api

from template_lib.utils import get_attr_kwargs
from .build import DATASET_MAPPER_REGISTRY




@DATASET_MAPPER_REGISTRY.register()
class NASBench101_Arc2Seq_DatasetMapper(object):
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
  ops2idx = {
    'nc': 1,
    'c': 2,
    'conv1x1-bn-relu': 3,
    'conv3x3-bn-relu': 4,
    'maxpool3x3'     : 5,
    'output'         : 6
  }
  idx2ops = {v: k for k, v in ops2idx.items()}

  @staticmethod
  def convert_arch_to_seq(matrix, ops):

    seq = []
    n = len(matrix)
    assert n == len(ops)

    # from node1 to the output node
    for col in range(1, n):
      # for edges connected to node of col
      for row in range(col):
        # connect: 2; no connect: 1
        edge_idx = matrix[row][col] + 1
        seq.append(edge_idx)

      if ops[col] != 'input':
        ops_idx = NASBench101_Arc2Seq_DatasetMapper.ops2idx[ops[col]]
        seq.append(ops_idx)
      # if ops[col] == CONV1X1:
      #   seq.append(3)
      # elif ops[col] == CONV3X3:
      #   seq.append(4)
      # elif ops[col] == MAXPOOL3X3:
      #   seq.append(5)
      # if ops[col] == OUTPUT:
      #   seq.append(6)
    assert len(seq) == (n + 2) * (n - 1) / 2 # num of edges and nodes
    return seq

  def build_transform(self, img_size):

    return None

  def __init__(self, cfg, **kwargs):

    # self.img_size             = get_attr_kwargs(cfg, 'img_size', **kwargs)

    # self.transform = self.build_transform(img_size=self.img_size)
    pass

  def __call__(self, dataset_dict):
    """
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file

    arc_seq = self.convert_arch_to_seq(dataset_dict['module_adjacency'], ops=dataset_dict['module_operations'])

    dataset_dict['idx2ops'] = self.idx2ops
    dataset_dict['arc_seq'] = arc_seq
    return dataset_dict


def get_dict(name, data_path, nasbench_file, num_samples, num_operations, val_acc_threshold=0., seed=1234, **kwargs):

  nasbench_file_abs = os.path.join(data_path, nasbench_file)
  print(f'Loading nasbench101: {nasbench_file_abs}')
  nasbench = api.NASBench(nasbench_file_abs, num_samples=num_samples, seed=seed)

  archs = []
  seqs = []
  valid_accs = []
  all_keys = list(nasbench.hash_iterator())
  dataset_dicts = []

  min_val_acc = float('inf')
  max_val_acc = 0
  min_data = max_data = None
  for idx, key in enumerate(all_keys):
    fixed_stat, computed_stat = nasbench.get_metrics_from_hash(key)
    if len(fixed_stat['module_operations']) not in num_operations:
      continue

    arch = api.ModelSpec(matrix=fixed_stat['module_adjacency'], ops=fixed_stat['module_operations'])
    data = nasbench.query(arch)

    if data['validation_accuracy'] < val_acc_threshold:
      continue

    if min_val_acc > data['validation_accuracy']:
      min_val_acc = data['validation_accuracy']
      min_data = data
    if max_val_acc < data['validation_accuracy']:
      max_val_acc = data['validation_accuracy']
      max_data = data

    data["id"] = idx
    dataset_dicts.append(data)

  meta_dict = {}
  meta_dict['num_samples'] = len(dataset_dicts)
  meta_dict['num_operations'] = num_operations
  meta_dict['min_val_acc'] = min_val_acc
  meta_dict['max_val_acc'] = max_val_acc
  meta_dict['min_data'] = min_data
  meta_dict['max_data'] = max_data
  print(f'min_val_acc: {min_val_acc}, max_val_acc: {max_val_acc}')
  if name not in MetadataCatalog.list():
    MetadataCatalog.get(name).set(**meta_dict)

  return dataset_dicts


data_path = "datasets/nasbench/"
registed_name_list = [
  'nasbench_only108_ops_7_num_9999',
  'nasbench_only108_ops_7_all',
  'nasbench_only108_num_9999',
  'nasbench_only108_all',
  'nasbench_only108_ops_7_acc_th_08_num_9999',
  'nasbench_only108_ops_7_acc_th_08_all',
]

registed_func_list = [
  get_dict,
  get_dict,
  get_dict,
  get_dict,
  get_dict,
  get_dict,
]

kwargs_list = [
  {'nasbench_file': 'nasbench_only108.tfrecord', 'num_samples': 9999, 'num_operations': [7, ]},
  {'nasbench_file': 'nasbench_only108.tfrecord', 'num_samples': float('inf'), 'num_operations': [7, ]},
  {'nasbench_file': 'nasbench_only108.tfrecord', 'num_samples': 9999, 'num_operations': list(range(2, 8))},
  {'nasbench_file': 'nasbench_only108.tfrecord', 'num_samples': float('inf'), 'num_operations': list(range(2, 8))},
  {'nasbench_file': 'nasbench_only108.tfrecord', 'num_samples': 9999, 'num_operations': [7, ],
   'val_acc_threshold': 0.8},
  {'nasbench_file': 'nasbench_only108.tfrecord', 'num_samples': float('inf'), 'num_operations': [7, ],
   'val_acc_threshold': 0.8},
]

for name, func, kwargs in zip(registed_name_list, registed_func_list, kwargs_list):
  # warning : lambda must specify keyword arguments
  DatasetCatalog.register(name, (lambda name=name, func=func, data_path=data_path, kwargs=kwargs:
                                 func(name=name, data_path=data_path, **kwargs)))


pass