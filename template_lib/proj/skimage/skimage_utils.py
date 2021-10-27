import numpy as np
from PIL import Image
import skimage.metrics as sk_metrics


def sk_psnr(image_true_pil, image_test_pil):
  """
  Peak Signal to Noise Ratio, PSNR
  """
  image_true_np = np.array(image_true_pil)
  image_test_np = np.array(image_test_pil)

  psnr = sk_metrics.peak_signal_noise_ratio(image_true=image_true_np, image_test=image_test_np, data_range=255)
  return psnr


def sk_ssim(image_true_pil, image_test_pil, multichannel=True):
  """
  structural similarity index，SSIM
  """
  image_true_np = np.array(image_true_pil)
  image_test_np = np.array(image_test_pil)

  psnr = sk_metrics.structural_similarity(
    im1=image_true_np, im2=image_test_np, multichannel=multichannel, data_range=255)
  return psnr


class LPIPS():
  def __init__(self, net='vgg', device='cuda'):
    '''
    net: ['alex', 'vgg']
    References:
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

    '''
    import lpips

    self.device = device

    ## Initializing the model
    self.loss_fn = lpips.LPIPS(net=net)
    self.loss_fn = self.loss_fn.to(device).eval()
    pass

  def calc_lpips(self, img0_pil, img1_pil):
    '''
    Returns
    dist01 : torch.Tensor, Learned Perceptual Image Patch Similarity, LPIPS
    References:
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    '''
    import torch
    import lpips
    # Load images
    img0 = np.array(img0_pil)
    img1 = np.array(img1_pil)

    with torch.no_grad():
      img0 = lpips.im2tensor(img0)  # RGB image from [-1,1]
      img1 = lpips.im2tensor(img1)

      img0 = img0.to(self.device)
      img1 = img1.to(self.device)
      dist01 = self.loss_fn.forward(img0, img1)

    lpips_value = round(dist01.item(), 4)
    return lpips_value
