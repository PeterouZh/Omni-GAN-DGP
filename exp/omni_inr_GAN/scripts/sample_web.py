from pathlib import Path
import tqdm
import os
import math
import streamlit as st
from PIL import Image
import numpy as np
import itertools

import torch
import torchvision
import torchvision.transforms.functional as trans_f

from template_lib.d2.models_v2 import MODEL_REGISTRY, build_model
from template_lib.d2.utils.checkpoint import VisualModelCkpt
from template_lib.proj.streamlit.utils import number_input, selectbox, parse_list_from_st_text_input,\
  text_input
from template_lib.utils import generate_random_string, merge_image_pil, get_time_str
from template_lib.proj.cv2.cv2_utils import ImageioVideoWriter
from template_lib.proj.pil.pil_utils import add_text

from BigGAN_Pytorch_lib import utils


@MODEL_REGISTRY.register(name_prefix=__name__)
class SampleWeb(object):

  def __init__(self,
               G_weight,
               G_cfg,
               batch_size,
               z_var,
               y,
               y_choice,
               mode,
               saved_size,
               video_cfg,
               n_classes=1000,
               **kwargs):

    self.G_weight = G_weight
    self.G_cfg = G_cfg
    self.batch_size = batch_size
    self.z_var = z_var
    self.y = y
    self.y_choice = y_choice
    self.mode = mode
    self.saved_size = saved_size
    self.video_cfg = video_cfg
    self.n_classes = n_classes

    self.G_weight = text_input(label="G_weight", value=self.G_weight)
    self.z_var = number_input(label='z_var', min_value=0., value=z_var, step=0.05)
    self.y = number_input(label='y', min_value=0, value=y, step=1)
    self.batch_size = number_input(label='bs', min_value=1, value=batch_size, step=1)

    self.select_mode = selectbox(label='mode', options=mode, index=0)

    if self.select_mode == 'sample_SR':
      self.saved_size = parse_list_from_st_text_input(label='saved_size', value=self.saved_size)
      self.video_cfg.steps = parse_list_from_st_text_input(label='ratio_steps', value=self.video_cfg.steps)

    st.subheader('y_choice')
    st.write(y_choice)
    pass

  @st.cache(allow_output_mutation=True, suppress_st_warning=True)
  def build_model(self):
    self.G = build_model(self.G_cfg).cuda()
    self.G.eval()
    ckpt = VisualModelCkpt(self.G)
    ckpt.load_from_path(self.G_weight)
    del ckpt
    return self.G

  @torch.no_grad()
  def sample(self, outdir, stem):
    self.G = self.build_model()

    if self.select_mode == 'sample_batch':
      self.sample_batch(outdir=outdir)
    elif self.select_mode == 'sample_SR':
      self.sample_SR(outdir=outdir)
    elif self.select_mode == 'sample_interpolation':
      self.sample_interpolation(outdir=outdir)
    pass

  def sample_batch(self, outdir):
    z_, y_ = utils.prepare_z_y(self.batch_size, self.G.dim_z, self.n_classes, z_var=self.z_var)
    z_.sample_()
    y_.fill_(self.y)

    self.G.eval()
    img = self.G.inference(z_, self.G.shared(y_))
    merged_img = torchvision.utils.make_grid(
      img, nrow=int(math.sqrt(self.batch_size)), padding=0, normalize=True, scale_each=True)
    img_pil = trans_f.to_pil_image(merged_img)
    img_pil.save(f"{outdir}/sample_class_{self.y:04d}.png")

    st.image(img_pil, caption=f'{img.shape}', use_column_width=True)
    pass

  def sample_SR(self, outdir):
    z_, y_ = utils.prepare_z_y(1, self.G.dim_z, self.n_classes, z_var=self.z_var)
    z_.sample_()
    y_.fill_(self.y)

    self.G.eval()

    images = []
    # rand_suffix = generate_random_string()
    rand_suffix = get_time_str()
    saved_sampledir = f"{outdir}/class_{self.y:04d}_{rand_suffix}"
    os.makedirs(saved_sampledir)
    saved_sampledir_png = f"{saved_sampledir}_png"
    os.makedirs(saved_sampledir_png, exist_ok=True)
    max_size = max(self.saved_size)
    for size in self.saved_size:
      img = self.G.inference(z_, self.G.shared(y_), shape=(size, size), max_points_every=90000)

      img = torchvision.utils.make_grid(img, nrow=1, padding=0, normalize=True, scale_each=True)
      img_pil = trans_f.to_pil_image(img)
      img_pil.save(f"{saved_sampledir}/{size:04d}.jpg")
      img_pil.save(f"{saved_sampledir_png}/{size:04d}.png")

      img_pil = img_pil.resize((max_size, max_size), Image.NEAREST)
      images.append(img_pil)

    merged_img = merge_image_pil(
      image_list=images, nrow=len(images),
      saved_file=f"{outdir}/class_{self.y:04d}_{size:04d}_{rand_suffix}.jpg")
    st.image(merged_img, caption=f'{merged_img.size}', use_column_width=True)

    st_image = st.empty()
    st_progress = st.empty()
    cfg = self.video_cfg
    max_size = self.G.resolution * cfg.max_ratio
    pbar = list(itertools.chain(
      np.arange(cfg.start_ratio, cfg.mid_ratio, cfg.steps[0]),
      np.arange(cfg.mid_ratio, cfg.max_ratio, cfg.steps[1])))
    pbar.append(max_size / self.G.resolution)
    pbar = tqdm.tqdm(pbar)

    video_path = f"{outdir}/class_{self.y:04d}_{rand_suffix}.mp4"
    video_f = ImageioVideoWriter(outfile=video_path, fps=cfg.fps)
    for r in pbar:
      st_progress.write(str(pbar))
      cur_shape = [int(self.G.resolution * r)] * 2
      img = self.G.inference(z_, self.G.shared(y_), shape=cur_shape, max_points_every=90000)
      img = torchvision.utils.make_grid(img, nrow=1, padding=0, normalize=True, scale_each=True)
      img_pil = trans_f.to_pil_image(img)

      img_frame = Image.new(mode='RGB', size=(max_size, max_size), color='black')
      img_frame.paste(img_pil, ((max_size - img_pil.size[0]) // 2, (max_size - img_pil.size[1]) // 2))
      add_text(img_frame, text=f"{cur_shape[0]}x{cur_shape[1]}", size=max_size//18)
      video_f.write(img_frame)
      st_image.image(img_frame)
    video_f.release()

    st_video = st.empty()
    st_video.video(video_path)
    pass

  def sample_interpolation(self, outdir):

    self.G.eval()

    images = []
    rand_suffix = get_time_str()
    saved_sampledir = Path(f"{outdir}/class_{self.y:04d}_{rand_suffix}")
    os.makedirs(saved_sampledir, exist_ok=True)

    from BigGAN_Pytorch_lib.utils import interp

    if fix_z:  # If fix Z, only sample 1 z per row
      z_, y_ = utils.prepare_z_y(1, self.G.dim_z, self.n_classes, z_var=self.z_var)
      z_.sample_()
      y_.fill_(self.y)

      zs = zs.repeat(1, num_midpoints + 2, 1).view(-1, self.G.dim_z)
    else:
      zs = interp(torch.randn(1, self.G.dim_z, device=device),
                  torch.randn(1, self.G.dim_z, device=device),
                  num_midpoints).view(-1, self.G.dim_z)
    if fix_y:  # If fix y, only sample 1 z per row
      ys = sample_1hot(num_per_sheet, num_classes)
      ys = G.shared(ys).view(num_per_sheet, 1, -1)
      ys = ys.repeat(1, num_midpoints + 2, 1).view(num_per_sheet * (num_midpoints + 2), -1)
    else:
      ys = interp(G.shared(sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                  G.shared(sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                  num_midpoints).view(num_per_sheet * (num_midpoints + 2), -1)


    merged_img = merge_image_pil(
      image_list=images, nrow=len(images),
      saved_file=f"{outdir}/class_{self.y:04d}_{size:04d}_{rand_suffix}.jpg")
    st.image(merged_img, caption=f'{merged_img.size}', use_column_width=True)








