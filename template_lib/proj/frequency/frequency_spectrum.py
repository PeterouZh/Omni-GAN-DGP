from PIL import Image
import numpy as np
from scipy import ndimage

from template_lib.proj.pil import pil_utils


class FrequencySpectrum(object):
  def __init__(self):

    self.mean_gray_spectrum = None
    self.sum_gray_spectrum = None
    self.count = 0
    pass

  def to_pil(self, gray_np):
    from matplotlib import cm
    gray_np_norm = (gray_np - gray_np.min()) / (gray_np.max() - gray_np.min())
    # img_pil = Image.fromarray(np.uint8(cm.gist_earth(gray_np_norm) * 255))
    img_pil = Image.fromarray(np.uint8(cm.viridis(gray_np_norm) * 255))
    return img_pil

  def get_spectrum_pil(self, text=None):
    spec_pil = self.to_pil(self.mean_gray_spectrum)
    if text is not None:
      pil_utils.add_text(spec_pil, text=text, size=spec_pil.size[0]//18)
    return spec_pil

  def get_frequency_spectrum(self, image_pil, ):
    """

    """

    image_np = np.array(image_pil)

    H, W, C = np.shape(image_np)
    real_r, real_g, real_b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    real_gray = 0.2989 * real_r + 0.5870 * real_g + 0.1140 * real_b

    real_gray_f = np.fft.fft2(real_gray - ndimage.median_filter(real_gray, size=H // 8))
    real_gray_f_shifted = np.fft.fftshift(real_gray_f)

    real_gray_spectrum = 20 * np.log(np.abs(real_gray_f_shifted))

    return real_gray_spectrum

  def update(self, gray_spectrum):
    if self.mean_gray_spectrum is None:
      self.mean_gray_spectrum = gray_spectrum
      self.sum_gray_spectrum = gray_spectrum
      self.count += 1
    else:
      self.sum_gray_spectrum = self.sum_gray_spectrum + gray_spectrum
      self.count += 1
      self.mean_gray_spectrum = self.sum_gray_spectrum / self.count
    pass

  def get_spectrum_and_update(self, image_pil):
    image_pil_spectrum = self.get_frequency_spectrum(image_pil)
    self.update(image_pil_spectrum)

  def reset(self):
    self.mean_gray_spectrum = None
    self.sum_gray_spectrum = None
    self.count = 0

