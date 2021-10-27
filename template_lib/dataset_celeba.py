import os
from glob import glob
from PIL import Image


def celeba_read_one_image_crop_128(imgpath, out_size=None):
  """

  :param imgpath: img_align_celeba
  :param out_size: 64
  :return:
  """
  off_x, off_y = 25, 60
  crop_size = 128

  image = Image.open(imgpath)

  # crop and resize images
  area = (off_x, off_y, off_x + crop_size, off_y + crop_size)
  cropped_img = image.crop(area)
  if out_size:
    cropped_img.thumbnail((out_size, out_size), Image.ANTIALIAS)
  return cropped_img


def crop_celeba_64(data_dir, save_dir):
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  off_x, off_y = 25, 60
  crop_size = 128
  out_size = 64

  files = glob(os.path.join(data_dir, "*.jpg"))
  num_images = len(files)
  for iter, img_path in enumerate(files):
    image = Image.open(img_path)

    # crop and resize images
    area = (off_x, off_y, off_x + crop_size, off_y + crop_size)
    cropped_img = image.crop(area)
    cropped_img.thumbnail((out_size, out_size), Image.ANTIALIAS)
    # cropped_img.show()
    # save cropped images
    image_name = os.path.basename(img_path)
    img_save_path = os.path.join(save_dir, image_name)
    cropped_img.save(img_save_path)
    print("save images [%d/%d]: %s" % (iter, num_images, img_save_path), end='\r')


import unittest
import os, sys


class Testing(unittest.TestCase):
  """

  """

  def test_crop_celeba_64(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=1,2,3,4;
        python -c "import test_train; \
        test_train.Testing().test_demo()"
    :return:
    """
    data_dir = os.path.expanduser('~/.keras/mydata/celeba/img_align_celeba')
    save_dir = os.path.expanduser('~/.keras/mydata/celeba/img_align_celeba_64')

    crop_celeba_64(data_dir=data_dir, save_dir=save_dir)

  def test_celeba_read_one_image_crop_128(self):
    imgpath = os.path.expanduser('~/.keras/celeba/img_align_celeba/000001.jpg')
    celeba_read_one_image_crop_128(imgpath)
