import shutil
import os
from PIL import Image
import glob
import unittest
import tqdm
from multiprocessing import Pool
from functools import partial

from template_lib.utils import logging_utils


def remove_corrupt_image(img_path, corrupt_dir=None):
  try:
    img = Image.open(img_path)  # open the image file
    img.verify()  # verify that it is, in fact an image
  except (IOError, SyntaxError) as e:
    print('Bad file:', img_path)  # print out the names of corrupt files
    if corrupt_dir:
      print('Moving %s \n to %s'%(img_path, corrupt_dir))
      shutil.move(img_path, corrupt_dir)


def remove_corrupt_images(imgs_dir, corrupt_dir=None, img_ext=None):
  imgs_list = get_imgs_list(imgs_dir, img_ext=img_ext)
  if corrupt_dir:
    corrupt_dir = os.path.expanduser(corrupt_dir)
    os.makedirs(corrupt_dir, exist_ok=True)
  pool = Pool()
  remove_corrupt_image_func = partial(remove_corrupt_image,
                                      corrupt_dir=corrupt_dir)
  num_every = 10000
  step = (len(imgs_list) + num_every - 1) // num_every
  for idx in tqdm.tqdm(range(step)):
    sub_imgs_list = imgs_list[idx*num_every : (idx+1)*num_every]
    pool.map(remove_corrupt_image_func, sub_imgs_list)


def get_imgs_list(imgs_dir, img_ext=None):
  imgs_dir = os.path.expanduser(imgs_dir)
  if not img_ext:
    img_ext = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
    img_ext = img_ext + list(map(lambda x: x.upper(), img_ext))
  if not isinstance(img_ext, list):
    img_ext = [img_ext]
  imgs_list = []
  for ext in img_ext:
    imgs_list += glob.glob(os.path.join(imgs_dir, '**/*' + ext))
  imgs_list = sorted(imgs_list)
  return imgs_list


class TestingUnit(unittest.TestCase):

  def test_ffhq_thumbnails128x128(self):
    imgs_dir = '/media/shhs/Peterou2/user/code/ffhq-dataset/thumbnails128x128'
    remove_corrupt_images(imgs_dir)

  def test_ImageNet_train(self):
    """
    export PYTHONPATH=../..
    python -c "import remove_corrupt_images; \
        remove_corrupt_images.TestingUnit().test_ImageNet_train()"

    """
    imgs_dir = '~/ZhouPeng/dataset/ILSVRC/Data/CLS-LOC/train'
    corrupt_dir = '~/ZhouPeng/dataset/ILSVRC/Data/CLS-LOC/corrupt'
    remove_corrupt_images(imgs_dir, corrupt_dir=corrupt_dir,
                          img_ext='.JPEG')