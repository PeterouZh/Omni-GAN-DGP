import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from template_lib.proj.pytorch.pytorch_hook import VerboseModel
from template_lib.proj.pil.pil_utils import merge_image_np


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc1_relu = nn.ReLU()
    self.fc2 = nn.Linear(120, 84)
    self.fc2_relu = nn.ReLU()
    self.fc3 = nn.Linear(84, 10)

    self.fmap_block = list()
    self.grad_block = list()
    pass

  def forward_hook(self):

    def forward_hook_(module, input, output):
      self.fmap_block.append(output)
    return forward_hook_

  def backward_hook(self):

    def backward_hook_(module, grad_in, grad_out):
      self.grad_block.append(grad_out[0].detach())
    return backward_hook_

  def forward(self, x):
    x = self.pool1(self.relu1(self.conv1(x)))
    x = self.pool2(self.relu2(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = self.fc1_relu(self.fc1(x))
    x = self.fc2_relu(self.fc2(x))
    x = self.fc3(x)
    return x


def img_transform(img_in, transform):
  """
  将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
  :param img_roi: np.array
  :return:
  """
  img = img_in.copy()
  img = Image.fromarray(np.uint8(img))
  img = transform(img)
  img = img.unsqueeze(0)  # C*H*W --> B*C*H*W
  return img


def img_preprocess(img_in):
  """
  读取图片，转为模型可读的形式
  :param img_in: ndarray, [H, W, C]
  :return: PIL.image
  """
  img = img_in.copy()
  img = cv2.resize(img, (32, 32))
  img = img[:, :, ::-1]  # BGR --> RGB
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
  ])
  img_input = img_transform(img, transform)
  return img_input








def show_cam_on_image(img, mask, out_dir):
  heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
  heatmap = np.float32(heatmap) / 255
  cam = heatmap + np.float32(img)
  cam = cam / np.max(cam)

  path_cam_img = os.path.join(out_dir, "cam.jpg")
  path_raw_img = os.path.join(out_dir, "raw.jpg")
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  cv2.imwrite(path_cam_img, np.uint8(255 * cam))
  cv2.imwrite(path_raw_img, np.uint8(255 * img))
  return img[:,:, ::-1], cam[:,:,::-1]


def comp_class_vec(ouput_vec, index=None):
  """
  计算类向量
  :param ouput_vec: tensor
  :param index: int，指定类别
  :return: tensor
  """
  if not index:
    index = np.argmax(ouput_vec.cpu().data.numpy())
  else:
    index = np.array(index)
  index = index[np.newaxis, np.newaxis]
  index = torch.from_numpy(index)
  one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
  one_hot.requires_grad = True
  class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

  return class_vec


def gen_cam(feature_map, grads):
  """
  依据梯度和特征图，生成cam
  :param feature_map: np.array， in [C, H, W]
  :param grads: np.array， in [C, H, W]
  :return: np.array, [H, W]
  """
  cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

  weights = np.mean(grads, axis=(1, 2))  #

  for i, w in enumerate(weights):
    cam += w * feature_map[i, :, :]

  cam = np.maximum(cam, 0)
  cam = cv2.resize(cam, (32, 32))
  cam -= np.min(cam)
  cam /= np.max(cam)

  return cam


if __name__ == '__main__':
  from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, global_cfg
  update_parser_defaults_from_yaml(parser=None)

  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  path_img = os.path.join(BASE_DIR, "cam_img", "test_img_8.png")
  path_net = os.path.join(BASE_DIR, "cam_img", "net_params_72p.pkl")
  output_dir = global_cfg.tl_outdir

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


  # 图片读取；网络加载
  img = cv2.imread(path_img, 1)  # H*W*C
  img_input = img_preprocess(img)
  net = Net()

  ret = net.load_state_dict(torch.load(path_net))

  net_verbose = VerboseModel(net)
  net_verbose(img_input)
  del net_verbose

  # 注册hook
  ret = net.conv2.register_forward_hook(net.forward_hook())
  ret = net.conv2.register_backward_hook(net.backward_hook())

  # forward
  output = net(img_input)
  idx = np.argmax(output.cpu().data.numpy())
  print("predict: {}".format(classes[idx]))

  # backward
  net.zero_grad()
  # class_loss = comp_class_vec(output)
  class_loss = output[0, idx]
  class_loss.backward()

  # 生成cam
  grads_val = net.grad_block[0].cpu().data.numpy().squeeze()
  fmap = net.fmap_block[0].cpu().data.numpy().squeeze()
  cam = gen_cam(fmap, grads_val)

  # 保存cam图片
  img_show = np.float32(cv2.resize(img, (32, 32))) / 255
  raw_img, img_cam = show_cam_on_image(img_show, cam, output_dir)


  merged_img = merge_image_np([raw_img, img_cam], nrow=2, pad=1, range01=True)

  plt.imshow(merged_img)
  plt.show()
  pass

