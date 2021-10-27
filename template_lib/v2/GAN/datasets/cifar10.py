import logging

import torch
from torch.utils.data import DataLoader as DataLoader_base
from torchvision import datasets
from torchvision import transforms

from template_lib.d2.data_v2 import DATASET_REGISTRY, DATALOADER_REGISTRY, build_dataset, build_dataloader
from template_lib.utils import get_attr_kwargs
from template_lib.v2.utils import get_dict_str


@DATASET_REGISTRY.register()
class CIFAR10(datasets.CIFAR10):
  def __init__(self, cfg, **kwargs):
    # fmt: off
    root                = get_attr_kwargs(cfg, 'root', **kwargs)
    train               = get_attr_kwargs(cfg, 'train', default=True, **kwargs)
    transform           = get_attr_kwargs(cfg, 'transform', default=None, **kwargs)
    download            = get_attr_kwargs(cfg, 'download', default=True, **kwargs)

    # fmt: on
    super(CIFAR10, self).__init__(root=root, train=train, transform=transform, download=download)
    logging.getLogger('tl').info(f"number images of CIFAR10: {len(self)}")
    pass


@DATALOADER_REGISTRY.register()
class DDPDataLoaderCIFAR10(DataLoader_base):
  norm_mean = [0.5, 0.5, 0.5]
  norm_std = [0.5, 0.5, 0.5]

  def __init__(self, cfg, **kwargs):
    """
    sampler.set_epoch(epoch)
    """
    # fmt: off
    distributed               = get_attr_kwargs(cfg, 'distributed', default=False, **kwargs)
    batch_size                = get_attr_kwargs(cfg, 'batch_size', **kwargs)
    shuffle                   = get_attr_kwargs(cfg, 'shuffle', default=True, **kwargs)
    seed                      = get_attr_kwargs(cfg, 'seed', default=0, **kwargs)
    drop_last                 = get_attr_kwargs(cfg, 'drop_last', default=False, **kwargs)
    num_workers               = get_attr_kwargs(cfg, 'num_workers', default=0, **kwargs)
    pin_memory                = get_attr_kwargs(cfg, 'pin_memory', default=True, **kwargs)

    # fmt: on
    kwargs_dict = dict(distributed=distributed, batch_size=batch_size, shuffle=shuffle)
    logging.getLogger('tl').info(f"kwargs: \n {get_dict_str(kwargs_dict)}")

    train_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(self.norm_mean, self.norm_std)])

    dataset = build_dataset(cfg.dataset_cfg, transform=train_transform)

    if distributed:
      self.ddp_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=shuffle, seed=seed, drop_last=drop_last)
      super(DDPDataLoaderCIFAR10, self).__init__(
        dataset, batch_size=batch_size, shuffle=False, sampler=self.ddp_sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    else:
      super(DDPDataLoaderCIFAR10, self).__init__(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    pass






