import sys
import os
import unittest

import torch
import torch.utils.data as data_utils
import torchvision.transforms as tv_trans
import torchvision.transforms.functional as trans_f

from template_lib.proj import pil_utils


class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """

  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    return trans_f.center_crop(img, min(img.size))

  def __repr__(self):
    return self.__class__.__name__



class ImageListDataset(data_utils.Dataset):

  def __init__(self,
               meta_file,
               load_image=False,
               root_dir=None,
               transform=None,
               image_size=(128, 128),
               normalize=True):
    self.load_image = load_image
    self.root_dir = root_dir

    if transform is not None:
      self.transform = transform
    else:
      norm_mean = [0.5, 0.5, 0.5]
      norm_std = [0.5, 0.5, 0.5]
      if normalize:
        self.transform = tv_trans.Compose([
          tv_trans.Resize(image_size),
          tv_trans.ToTensor(),
          tv_trans.Normalize(norm_mean, norm_std)
        ])
      else:
        self.transform = tv_trans.Compose([
          tv_trans.Resize(image_size),
          tv_trans.ToTensor()
        ])

    with open(meta_file) as f:
      lines = f.readlines()
    print("building dataset from %s" % meta_file)
    self.num = len(lines)
    self.metas = []

    for line in lines:
      line_split = line.rstrip().split()
      self.metas.append(line_split)

    print("read meta done")

  def __len__(self):
    return self.num

  def __getitem__(self, idx):
    filename = self.metas[idx][0]
    if self.root_dir is not None:
      filename = self.root_dir + '/' + filename

    if self.load_image:
      img = pil_utils.pil_open_rgb(filename)
      # transform
      if self.transform is not None:
        img = self.transform(img)
      # cls = self.metas[idx][1]
      return img
    else:
      return filename


class Testing_Dataset(unittest.TestCase):

  def test_ImageListDataset(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str, return_cfg=True)

    """
    python3 template_lib/proj/ubuntu/get_data_list.py \
      --source_dir template_lib/proj/pytorch/examples/cam_img  \
      --outfile template_lib/proj/pytorch/examples/cam_img/image_list.txt --ext *.png
    
    """

    image_list_file = "template_lib/proj/pytorch/examples/cam_img/image_list.txt"
    num_workers = 0

    dataset = ImageListDataset(meta_file=image_list_file, )
    img = dataset[0]

    if False and 'dist':
      sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
      sampler = None

    train_loader = data_utils.DataLoader(
      dataset,
      batch_size=1,
      shuffle=(sampler is None),
      sampler=sampler,
      num_workers=num_workers,
      pin_memory=False)

    data_iter = iter(train_loader)
    data = next(data_iter)

    pass



