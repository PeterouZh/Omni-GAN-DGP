import random
from tqdm import tqdm
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as trans_f

from BigGAN_Pytorch_lib.datasets import default_loader, find_classes, make_dataset, IMG_EXTENSIONS


class VarImageTransform(object):

  def __init__(self):
    self.norm_mean = [0.5, 0.5, 0.5]
    self.norm_std = [0.5, 0.5, 0.5]
    pass

  def __call__(self, img_pil, size):
    out = trans_f.center_crop(img_pil, min(img_pil.size))
    out = trans_f.resize(out, size=size)
    out = trans_f.to_tensor(out)
    out = trans_f.normalize(out, mean=self.norm_mean, std=self.norm_std)
    return out

var_image_transform = VarImageTransform()

def my_collate(batch, min_size, max_size):
  img_size = random.randint(min_size, max_size)
  data = [var_image_transform(item[0], img_size) for item in batch]
  target = [item[1] for item in batch]

  data = torch.stack(data, dim=0)
  target = torch.LongTensor(target)

  return [data, target]


class ImageFolderVar(data.Dataset):

  def __init__(self, root, target_transform=None, loader=default_loader, load_in_mem=False,
               index_filename='imagenet_imgs.npz', **kwargs):
    classes, class_to_idx = find_classes(root)
    # Load pre-computed image directory walk
    if os.path.exists(index_filename):
      print('Loading pre-saved Index file %s...' % index_filename)
      imgs = np.load(index_filename)['imgs']
    # If first time, walk the folder directory and save the
    # results to a pre-computed file.
    else:
      print('Generating  Index file %s...' % index_filename)
      imgs = make_dataset(root, class_to_idx)
      os.makedirs(os.path.dirname(index_filename), exist_ok=True)
      np.savez_compressed(index_filename, **{'imgs' : imgs})
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n" 
              "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx

    self.transform = None
    self.target_transform = target_transform
    self.loader = loader
    self.load_in_mem = load_in_mem

    if self.load_in_mem:
      print('Loading all images into memory...')
      self.data, self.labels = [], []
      for index in tqdm(range(len(self.imgs))):
        path, target = imgs[index][0], imgs[index][1]
        if self.transform is not None:
          self.data.append(self.transform(self.loader(path)))
        else:
          self.data.append(self.loader(path))
        self.labels.append(target)
    pass

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    if self.load_in_mem:
      img = self.data[index]
      target = self.labels[index]
    else:
      path, target = self.imgs[index]
      img = self.loader(str(path))
      if self.transform is not None:
        img = self.transform(img, size=img_size)

    if self.target_transform is not None:
      target = self.target_transform(target)

    # print(img.size(), target)
    return img, int(target)

  def __len__(self):
    return len(self.imgs)

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str
