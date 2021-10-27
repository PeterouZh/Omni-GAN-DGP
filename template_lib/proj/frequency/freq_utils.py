from PIL import Image
import numpy as np
from matplotlib import cm


def gray_to_cm_pil(gray_np, log=False, fftshift=False, cmap='viridis'):

  if log:
    gray_np = 20 * np.log(np.abs(gray_np))

  if fftshift:
    gray_np = np.fft.fftshift(gray_np)

  gray_np_norm = (gray_np - gray_np.min()) / (gray_np.max() - gray_np.min())
  # img_pil = Image.fromarray(np.uint8(cm.gist_earth(gray_np_norm) * 255))

  if cmap == 'viridis':
    img_cmap = cm.viridis(gray_np_norm)
  elif cmap.lower() in ['greys', 'grey', 'gray']:
    img_cmap = cm.Greys(1 - gray_np_norm)

  img_pil = Image.fromarray(np.uint8(img_cmap * 255))
  return img_pil


def rgb_to_cm_pil(img_np, channel_first=False, axis=[0, ], log=False, fftshift=False, cmap='viridis'):
  if channel_first:
    img_np = img_np.transpose(1, 2, 0)

  # gray_np = img_np.mean(axis=2)
  img_pil_list = []
  for idx in axis:
    gray_np = img_np[:, :, idx]
    img_pil = gray_to_cm_pil(gray_np, log=log, fftshift=fftshift, cmap=cmap)
    img_pil_list.append(img_pil)

  return img_pil_list


