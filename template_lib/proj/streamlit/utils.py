import numpy as np
from PIL import Image
from pathlib import Path
import logging
import json
import ast
import streamlit as st
import pandas as pd

from template_lib.utils import read_image_list_from_files
from template_lib.utils import generate_random_string
from . import SessionState

def is_init():
  try:
    saved_suffix_state = SessionState.get(saved_suffix=0)
  except:
    return False
  return True


def radio(label, options, index=0, sidebar=False):
  if sidebar:
    ret = st.sidebar.radio(label=label, options=list(options), index=index)
  else:
    ret = st.radio(label=label, options=list(options), index=index)
  logging.getLogger('st').info(f"{label}={ret}")
  print(f"{label}={ret}")
  return ret



class LineChart_deprecated(object):
  def __init__(self, x_label, y_label):
    self.x_label = x_label
    self.y_label = y_label

    self.pd_data = pd.DataFrame(columns=[x_label, y_label])
    self.st_line_chart = st.empty()
    self.st_init = is_init()
    pass

  def write(self, x, y):
    if not self.st_init:
      return
    self.pd_data = self.pd_data.append({self.x_label: x, self.y_label: y}, ignore_index=True)
    pd_data = self.pd_data.set_index(self.x_label)
    self.st_line_chart.line_chart(pd_data)
    pass


class LineChart(object):
  def __init__(self, x_label, y_label):
    self.x_label = x_label
    self.y_label = y_label

    # self.pd_data = pd.DataFrame(columns=[x_label, y_label])
    self.st_line_chart = st.line_chart()
    self.st_init = is_init()
    pass

  def write(self, x, y):
    # if not self.st_init:
    #   return
    data = {self.x_label: [x], self.y_label: [y]}
    pd_data = pd.DataFrame(data=data).set_index(self.x_label)

    # self.pd_data = self.pd_data.append(data, ignore_index=True)
    # pd_data = pd_data.set_index(self.x_label)

    # data = np.array([[x, y]])
    self.st_line_chart.add_rows(pd_data)
    pass


def multiselect(label, options, default=None, sidebar=False):
  if sidebar:
    ret = st.sidebar.multiselect(label=label, options=options, default=default)
  else:
    ret = st.multiselect(label=label, options=options, default=default)
  logging.getLogger('st').info(f"{label}={ret}")
  print(f"{label}={ret}")
  return ret


def selectbox(label, options, index=0, sidebar=False):
  options = list(options)
  if sidebar:
    ret = st.sidebar.selectbox(label=label, options=options, index=index)
  else:
    ret = st.selectbox(label=label, options=options, index=index)
  logging.getLogger('st').info(f"{label}={ret}")
  st.write(f"{label}={ret}")
  print(f"{label}={ret}")
  return ret

def number_input(label,
                 value,
                 min_value=None,
                 step=None,
                 format=None, # "%.8f"
                 sidebar=False,
                 **kwargs):
  if sidebar:
    st_empty = st.sidebar.empty()
  else:
    st_empty = st.empty()
  ret = st_empty.number_input(label=f"{label}: {value}", value=value, min_value=min_value,
                              step=step, format=format, **kwargs)
  logging.getLogger('st').info(f"{label}={ret}")
  print(f"{label}={ret}")
  return ret


def checkbox(label, value, sidebar=True):
  if sidebar:
    st_empty = st.sidebar.empty()
  else:
    st_empty = st.empty()
  ret = st_empty.checkbox(label=f"{label}: {value}", value=value)
  logging.getLogger('st').info(f"{label}={ret}")
  print(f"{label}={ret}")
  return ret


def text_input(label,
               value,
               sidebar=False,
               **kwargs):
  if sidebar:
    ret = st.sidebar.text_input(label=f"{label}: {value}", value=value, key=label)
  else:
    ret = st.text_input(label=f"{label}: {value}", value=value, key=label)
  logging.getLogger('st').info(f"{label}={ret}")
  print(f"{label}={ret}")
  return ret


def parse_list_from_st_text_input(label, value, sidebar=False):
  """
  return: list
  """
  value = str(value)
  if sidebar:
    st_text_input = st.sidebar.empty()
  else:
    st_text_input = st.empty()
  st_value = st_text_input.text_input(label=f"{label}: {value}", value=value, key=label)

  parsed_value = ast.literal_eval(st_value)
  print(f"{label}: {parsed_value}")
  logging.getLogger('st').info(f"label: {parsed_value}")
  return parsed_value


def read_image_list_and_show_in_st(image_list_file, columns=['path', 'class_id'], header=None, show_dataframe=True):
  if not isinstance(image_list_file, (list, tuple)):
    image_list_file = [image_list_file, ]

  if not header:
    header = "Image list file: "

  if len(image_list_file) > 0 and image_list_file[0]:
    st.header(header)
  for image_file in image_list_file:
    st.write(image_file)

  all_image_list = read_image_list_from_files(image_list_file)
  if show_dataframe:
    image_list_df = pd.DataFrame(all_image_list, columns=columns)
    st.dataframe(image_list_df)
  return all_image_list


def parse_image_list(image_list_file, header='selected index', columns=['path', ], default_index=0):
  image_list = read_image_list_and_show_in_st(image_list_file=image_list_file, columns=columns, header=header,
                                              show_dataframe=False)

  default_index = st.sidebar.number_input(f"{header} (0~{len(image_list) - 1})", value=default_index,
                                          min_value=0, max_value=len(image_list) - 1, key=header)
  image_path = image_list[default_index][0]
  image_path = Path(image_path)

  image_pil = Image.open(image_path)
  st.image(image_pil, caption=f"{image_path.name, image_pil.size}", width=256)
  st.write(f"{image_path}")
  return image_path


def parse_dict_from_st_text_input(label, value):
  to_list = False
  if isinstance(value, list):
    value = {str(k): v for k, v in enumerate(value)}
    to_list = True

  if isinstance(value, dict):
    value = json.dumps(value)
  st_text_input = st.empty()
  st_value = st_text_input.text_input(label=f"{label}: {value}", value=value, key=label)
  parse_value = json.loads(st_value)
  print(f"{label}: {parse_value}")
  logging.getLogger('st').info(f"label: {parse_value}")
  if to_list:
    parse_value = list(parse_value.values())
    print(f"{label}={parse_value}")
  return parse_value











