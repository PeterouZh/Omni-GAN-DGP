import os
import numpy as np
import PIL.Image as Image
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def CelebA64(datadir, batch_size, shuffle, num_workers, seed,
             get_dataset=False):
  crop_size = 108
  re_size = 64
  offset_height = (218 - crop_size) // 2
  offset_width = (178 - crop_size) // 2
  crop = lambda x: x[:, offset_height:offset_height + crop_size,
                   offset_width:offset_width + crop_size]

  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(crop),
     transforms.ToPILImage(),
     transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

  imagenet_data = dsets.ImageFolder(datadir, transform=transform)

  if get_dataset:
    return imagenet_data

  def _init_fn(worker_id):
    np.random.seed(seed + worker_id)

  data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            worker_init_fn=_init_fn)
  return data_loader


def CelebaHQ(datadir, batch_size, shuffle, num_workers, seed,
             get_dataset=False):
  datadir = os.path.expanduser(datadir)
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

  imagenet_data = dsets.ImageFolder(datadir, transform=transform)

  if get_dataset:
    return imagenet_data

  def _init_fn(worker_id):
    np.random.seed(seed + worker_id)

  data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            worker_init_fn=_init_fn)
  return data_loader