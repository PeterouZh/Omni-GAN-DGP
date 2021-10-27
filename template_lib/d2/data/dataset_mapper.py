from detectron2.data import DatasetCatalog, MetadataCatalog
import copy
from .build import DATASET_MAPPER_REGISTRY


@DATASET_MAPPER_REGISTRY.register()
class NoneMapper(object):

  def __init__(self, cfg, **kwargs):
    pass

  def __call__(self, dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    return dataset_dict


DatasetCatalog.register('NoneDataset', (lambda : [{'test': 'test'}]*10))
if 'NoneDataset' not in MetadataCatalog.list():
  MetadataCatalog.get('NoneDataset').set(**{'num_samples': 10})
