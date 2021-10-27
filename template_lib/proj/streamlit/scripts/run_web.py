from pathlib import Path
import logging
import os
import sys
from PIL import Image
import streamlit as st

sys.path.insert(0, os.getcwd())

from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, global_cfg, get_dict_str
from template_lib.d2.models_v2 import build_model
from template_lib.proj.streamlit import SessionState
from template_lib.v2.logger.logger import get_file_logger
import template_lib.proj.streamlit.utils as st_utils

# sys.path.insert(0, f"{os.getcwd()}/DGP_lib")
# from DGP_lib import utils
# sys.path.pop(0)


def build_sidebar():
  st.sidebar.text(global_cfg.sidebar.sidebar_name)
  st.sidebar.text(f"{global_cfg.tl_outdir}")
  pass


class STModel(object):
  def __init__(self):

    pass

  def show_video(self,
                 mode,
                 outdir,
                 num_video,
                 saved_suffix_state=None,
                 **kwargs):
    from template_lib.proj.streamlit import st_utils

    num_video = st_utils.number_input('num_video', num_video, sidebar=True)
    for idx in range(num_video):
      tag = st_utils.text_input(f"tag {idx}", "", sidebar=True)
      video_path = st_utils.text_input(f"video {idx} ", "", sidebar=True)
      if video_path and os.path.isfile(video_path):
        st.subheader(f"tag {idx}: {tag}")
        if video_path.endswith(('.jpg', '.png')):
          st.image(video_path)
        elif video_path.endswith('.mp4'):
          st.video(video_path)
        st.write(video_path)

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    pass

  def image_list_files(self,
                       mode,
                       outdir,
                       cfg,
                       saved_suffix_state=None,
                       **kwargs):
    import collections
    from template_lib.proj.streamlit import st_utils
    from template_lib.proj.pil import pil_utils

    image_list_kwargs = collections.defaultdict(dict)
    for k, v in cfg.image_list_files.items():
      header = f"{k}_s"
      image_path = st_utils.parse_image_list(image_list_file=v.image_list_file, header=header, )
      image_list_kwargs[header]['image_path'] = image_path
      header = f"{k}_t"
      image_path = st_utils.parse_image_list(image_list_file=v.image_list_file, header=header, )
      image_list_kwargs[header]['image_path'] = image_path
    source_k = st_utils.radio('source', options=image_list_kwargs.keys(), index=0, sidebar=True)
    target_k = st_utils.radio('target', options=image_list_kwargs.keys(), index=1, sidebar=True)

    image_path_s = image_list_kwargs[source_k]['image_path']
    image_path_t = image_list_kwargs[target_k]['image_path']

    img_pil_s = Image.open(image_path_s)
    img_pil_t = Image.open(image_path_t)
    img_pil_s_t = pil_utils.merge_image_pil([img_pil_s, img_pil_t], nrow=2, pad=1, dst_size=2048)
    st.image(img_pil_s_t, caption=f"source: {img_pil_s.size}, target: {img_pil_t.size}", use_column_width=True)

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    pass


# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
def build_model_st():
  # model = build_model(global_cfg.model_cfg, cfg_to_kwargs=True)
  model = STModel()
  return model


def main():
  # sys.path.insert(0, f"{os.getcwd()}/BigGAN_Pytorch_lib")

  update_parser_defaults_from_yaml(parser=None)

  # sidebar
  build_sidebar()

  # outdir
  kwargs = {}
  if global_cfg.start_web:
    saved_suffix_state = SessionState.get(saved_suffix=0)
    saved_suffix = saved_suffix_state.saved_suffix
    kwargs.update({'saved_suffix_state': saved_suffix_state})
  else:
    saved_suffix = 0
  st_saved_suffix = st.empty()
  st.sidebar.header(f"Outdir: ")
  saved_suffix = st_saved_suffix.number_input(label="Saved dir suffix: ", min_value=0,
                                              value=saved_suffix)

  outdir = f"{global_cfg.tl_outdir}/exp/{saved_suffix:04d}"
  kwargs['outdir'] = outdir
  st.sidebar.write(outdir)
  os.makedirs(outdir, exist_ok=True)

  get_file_logger(filename=f"{outdir}/log.txt", logger_names=['st'])
  logger = logging.getLogger('st')
  logger.info(f"global_cfg: \n{get_dict_str(global_cfg, use_pprint=False)}")

  try:
    # image list
    image_list = st_utils.read_image_list_and_show_in_st(
      image_list_file=global_cfg.image_list.image_list_file, columns=global_cfg.image_list.columns,
      show_dataframe=global_cfg.image_list.show_dataframe)

    default_index = st.number_input(
      f"default index (0~{len(image_list) - 1})",
      value=global_cfg.image_list.default_index,
      min_value=0, max_value=len(image_list) - 1)
    image_path = image_list[default_index][0]
    image_path = Path(image_path)

    kwargs['image_path'] = image_path
    image_pil = Image.open(image_path)
    st.image(image_pil, caption=f"{image_path.name, image_pil.size}", use_column_width=False)
    st.write(f"{image_path}")
    image_pil.save(f"{outdir}/{image_path.name}")
  except:
    pass

  st_model = build_model_st()

  mode = st_utils.selectbox(label='mode', options=global_cfg.mode, sidebar=True)
  getattr(st_model, mode)(mode=mode, cfg=global_cfg.get(mode, {}), **global_cfg.get(mode, {}), **kwargs)

  pass

if __name__ == '__main__':
  main()
