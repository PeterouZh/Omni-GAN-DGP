import logging

import torch
from torch.utils.data import DataLoader as DataLoader_base
from torchvision import datasets
from torchvision import transforms

from template_lib.d2.data_v2 import DATASET_REGISTRY, DATALOADER_REGISTRY, build_dataset, build_dataloader
from template_lib.utils import get_attr_kwargs

from .cifar10 import DDPDataLoaderCIFAR10


@DATASET_REGISTRY.register()
class CIFAR100(datasets.CIFAR100):
  def __init__(self, cfg, **kwargs):
    # fmt: off
    root                = get_attr_kwargs(cfg, 'root', **kwargs)
    train               = get_attr_kwargs(cfg, 'train', default=True, **kwargs)
    transform           = get_attr_kwargs(cfg, 'transform', default=None, **kwargs)
    download            = get_attr_kwargs(cfg, 'download', default=True, **kwargs)

    # fmt: on
    super(CIFAR100, self).__init__(root=root, train=train, transform=transform, download=download)
    logging.getLogger('tl').info(f"number images of CIFAR100: {len(self)}")
    pass


@DATALOADER_REGISTRY.register()
class DDPDataLoaderCIFAR100(DDPDataLoaderCIFAR10):
  pass


