import torch
import h5py
import numpy as np
from numpy.random import shuffle as shffle
from torch_geometric.data import Data, DataLoader
import functools
import os
import numpy as np
import random
import copy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch_geometric.data import DataLoader

from detectron2.data import DatasetCatalog, MetadataCatalog

from template_lib.utils import get_attr_kwargs
from .build import DATASET_MAPPER_REGISTRY


@DATASET_MAPPER_REGISTRY.register()
class NASBench101_GAE_DatasetMapper(object):
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
    # image = dataset_dict['image']
    # dataset_dict['image'] = self.transform(image)
    return dataset_dict


def _gauss(n):
  n = int(n)
  return n * (n + 1) // 2

def _get_edges(edge_list, max_num_nodes):
  edge_list_sorted = np.array(sorted(edge_list, key=lambda edge: tuple(edge[::-1])))
  edges = np.zeros(_gauss(max_num_nodes - 1), dtype=np.float32)
  for edge in edge_list_sorted:
    idx = edge[0] + _gauss(edge[1] - 1)
    edges[idx] = 1.
  return edges, edge_list_sorted

def _get_edge_list_list(edge_list_sorted, max_num_nodes):
  edge_list_list = [np.array([[0, 1]]) for _ in range(max_num_nodes - 1)]
  dst = 1
  for idx, edge in enumerate(edge_list_sorted):
    if edge[1] != dst:
      dst = edge[1]
      # for the nodes in front of dst
      edge_list_list[dst - 2] = edge_list_sorted[:idx]
  # for the last node
  edge_list_list[dst - 1] = edge_list_sorted
  return edge_list_list


def preprocess_batch_graphs(batched_inputs, device):

  batch_size = len(batched_inputs)
  batch_list = list()
  for graph_batch in zip(*batched_inputs):
    for i in range(batch_size):
      graph_batch[i].to(device)
    loader = DataLoader(graph_batch, batch_size, False)
    batch_list.append(loader.__iter__().__next__().to(device))

  return batch_list


def get_dict(name, data_path, data_file, max_num_nodes, aggr='sum', device='cpu', **kwargs):
  # from vs_gae.utils.prepData import get_edges, get_edge_list_list, gauss

  device = torch.device(device)
  data_list = list()
  data_file_abs = os.path.join(data_path, data_file)
  print(f"Loading {data_file_abs}")
  data_loaded = torch.load(data_file_abs)

  for graph in data_loaded:
    edges, edge_list = _get_edges(graph.edge_index.numpy().transpose(), max_num_nodes)
    edge_list_list = _get_edge_list_list(edge_list, max_num_nodes)
    node_atts = graph.node_atts.numpy()
    num_nodes = node_atts.size
    node_atts_padded = np.ones(max_num_nodes, dtype=np.int32)
    node_atts_padded[:num_nodes - 1] = node_atts[1:]
    nodes = np.zeros(max_num_nodes - 1, dtype=np.float32)
    nodes[:num_nodes - 1] = 1
    if aggr == 'mean':
      nodes /= num_nodes
    acc = graph.acc.numpy().item()
    #                 test_acc=graph.test_acc.numpy().item()
    #                 training_time=graph.training_time.numpy().item()
    data = Data(edge_index=torch.LongTensor(np.transpose(edge_list)).to(device),
                num_nodes=num_nodes,
                node_atts=torch.LongTensor(node_atts).to(device),
                acc=torch.tensor([acc]).to(device),
                #                             test_acc=torch.tensor([test_acc]).to(device),
                #                             training_time=torch.tensor([training_time]).to(device),
                nodes=torch.tensor(nodes).unsqueeze(0).to(device),
                )

    data_full = [data]
    for idx in range(max_num_nodes - 1):
      num_nodes = idx + 2
      data = Data(edge_index=torch.LongTensor(np.transpose(edge_list_list[idx])).to(device),
                  num_nodes=num_nodes,
                  node_atts=torch.LongTensor([node_atts_padded[idx]]).to(device),
                  edges=torch.tensor(edges[_gauss(num_nodes - 2):_gauss(num_nodes - 1)]).unsqueeze(0).to(device)
                  )
      data_full.append(data)
    data_list.append(tuple(data_full))

  dataset_dicts = data_list

  meta_dict = {}
  meta_dict['num_images'] = len(dataset_dicts)
  meta_dict['node_atts_idx2name'] = {1: 'input',
                                     2: 'conv3x3-bn-relu', 3: 'conv1x1-bn-relu', 4: 'maxpool3x3',
                                     0: 'output'}
  MetadataCatalog.get(name).set(**meta_dict)

  # for idx, (img, label) in enumerate(data_iter):
  #   record = {}
  #
  #   record["image_id"] = idx
  #   record["height"] = img.height
  #   record["width"] = img.width
  #   record["image"] = img
  #   record["label"] = int(label)
  #   dataset_dicts.append(record)
  return dataset_dicts


data_path = "datasets/vs_gae/"
registed_name_list = [
  'validation_data_10',
  'validation_data_1000_samples',
  'training_data_90',
]

registed_func_list = [
  get_dict,
  get_dict,
  get_dict,
]

kwargs_list = [
  {'data_file': 'validation_data_10.pth', 'max_num_nodes': 7},
  {'data_file': 'validation_data_1000_samples.pth', 'max_num_nodes': 7},
  {'data_file': 'training_data_90.pth', 'max_num_nodes': 7},
]

for name, func, kwargs in zip(registed_name_list, registed_func_list, kwargs_list):
  # warning : lambda must specify keyword arguments
  DatasetCatalog.register(name, (lambda name=name, func=func, data_path=data_path, kwargs=kwargs:
                                 func(name=name, data_path=data_path, **kwargs)))


pass