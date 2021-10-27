from pathlib import Path
import logging
import os
import sys
from PIL import Image
import streamlit as st

import torch
import torchvision

sys.path.insert(0, os.getcwd())

from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, global_cfg, get_dict_str
from template_lib.v2.logger import TextLogger
from template_lib.d2.models_v2 import build_model
from template_lib.d2.utils.checkpoint import D2Checkpointer
from template_lib.proj.streamlit.utils import parse_list_from_st_text_input
from template_lib.proj.streamlit import SessionState
from template_lib.v2.logger.logger import get_file_logger
from template_lib.proj.streamlit.utils import read_image_list_and_show_in_st

# sys.path.insert(0, f"{os.getcwd()}/DGP_lib")
# from DGP_lib import utils
# sys.path.pop(0)


def build_sidebar():
  st.sidebar.text(global_cfg.sidebar.sidebar_name)
  st.sidebar.text(f"{global_cfg.tl_outdir}")
  pass


# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
def build_model_st():
  model = build_model(cfg=global_cfg.sample_cfg, cfg_to_kwargs=True)
  return model


def main():
  # sys.path.insert(0, f"{os.getcwd()}/BigGAN_Pytorch_lib")

  update_parser_defaults_from_yaml(parser=None)

  # sidebar
  build_sidebar()

  # image list
  image_list = read_image_list_and_show_in_st(
    image_list_file=global_cfg.image_list.image_list_file, columns=global_cfg.image_list.columns)

  default_index = st.sidebar.number_input(
    f"default index (0~{len(image_list) - 1})", value=global_cfg.image_list.default_index,
    min_value=0, max_value=len(image_list) - 1)
  image_path = image_list[default_index][0]
  image_path = Path(image_path)

  image_pil = Image.open(image_path)
  st.image(image_pil, caption=f"{image_path.name, image_pil.size}", use_column_width=False)
  st.write(f"{image_path}")

  # outdir
  if not global_cfg.tl_debug:
    saved_suffix_state = SessionState.get(saved_suffix=0)
    saved_suffix = saved_suffix_state.saved_suffix
  else:
    saved_suffix = 0
  st_saved_suffix = st.empty()
  st.sidebar.header(f"Outdir: ")
  saved_suffix = st_saved_suffix.number_input(label="Saved dir suffix: ", min_value=0,
                                              value=saved_suffix)

  outdir = f"{global_cfg.tl_outdir}/exp/{image_path.stem}_{saved_suffix:04d}"
  st.sidebar.write(outdir)
  os.makedirs(outdir, exist_ok=True)
  image_pil.save(f"{outdir}/{image_path.name}")
  get_file_logger(filename=f"{outdir}/log.txt", logger_names=['st'])
  logger = logging.getLogger('tl')
  logger.info(f"global_cfg: \n{get_dict_str(global_cfg, use_pprint=False)}")

  st_model = build_model_st()

  if st.button("Run") or global_cfg.tl_debug:
    if not global_cfg.tl_debug:
      saved_suffix_state.saved_suffix = saved_suffix + 1

    textlogger = TextLogger(log_root=f"{outdir}/textdir")
    state_dict = {'itr': 0, "outdir": outdir, "textlogger": textlogger}

    # st_model.build_model()
    st_model.sample(outdir=outdir, stem=image_path.stem)

  pass

if __name__ == '__main__':
  main()
