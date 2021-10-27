from collections import defaultdict
import time

import numpy as np
import logging, os
import datetime
import sys

from template_lib.v2.utils.plot_utils import plot_figure, plot_defaultdict2figure
from template_lib.v2.utils import get_prefix_abb

class TextLogger(object):
  """
  # Logstyle is either:
  # '%#.#f' for floating point representation in text
  # '%#.#e' for exponent representation in text
  """
  def __init__(self, log_root, reinitialize=False, logstyle='%10.6f'):
    self.root = log_root
    if log_root and not os.path.exists(self.root):
      os.makedirs(self.root)
    self.reinitialize = reinitialize
    self.metrics = []
    # One of '%3.3f' or like '%3.3e'
    self.logstyle = logstyle
    pass

  def update(self, textlogger):
    self.root = textlogger.root
    self.reinitialize = textlogger.reinitialize
    self.logstyle = textlogger.logstyle
    pass

  def reinit(self, item):
    """
      Delete log if re-starting and log already exists
    """
    if os.path.exists('%s/%s.log' % (self.root, item)):
      os.remove('%s/%s.log' % (self.root, item))


  def log(self, itr, **kwargs):
    """
    Log in plaintext;
    """
    for arg in kwargs:
      file_path = '%s/%s.log' % (self.root, arg)
      if arg not in self.metrics:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if self.reinitialize:
          self.reinit(arg)
        self.metrics += [arg]
      with open(file_path, 'a') as f:
        f.write('%3.4f: %s\n' % (itr, self.logstyle % kwargs[arg]))

  def log_axes(self, **kwargs):
    names = []
    filepaths = []
    for arg in kwargs:
      filename = '%s/%s.log' % (self.root, arg)
      names.append(arg)
      filepaths.append(filename)

    plot_figure(names=names, filepaths=filepaths,
                outdir=self.root, in_one_axes=False)

  def log_ax(self, **kwargs):
    names = []
    filepaths = []
    for arg in kwargs:
      filename = '%s/%s.log' % (self.root, arg)
      names.append(arg)
      filepaths.append(filename)

    plot_figure(names=names, filepaths=filepaths,
                outdir=self.root, in_one_axes=True)

  def _get_filepath_from_dictlist(self, dict_list, in_one_figure, MAXLEN=100, ext='.png'):
    filepaths = []
    if in_one_figure:
      labels = []
      for d in dict_list:
        labels += d.keys()
      filename = '__'.join(labels)[:MAXLEN]
      filepath = os.path.join(self.root, '0plot__' + filename + ext)
      filepaths.append(filepath)
    else:
      for d in dict_list:
        labels = d.keys()
        filename = '__'.join(labels)[:MAXLEN]
        filepath = os.path.join(self.root, '0plot__' + filename + ext)
        filepaths.append(filepath)
    return filepaths

  def log_defaultdict2figure(self, default_dict, in_one_figure=True, save_fig_sec=300):
    label2filepaths_list = []
    for _, v in default_dict.items():
      label2filepaths_list.append({subk : '%s/%s.log' % (self.root, subk) for subk, _ in v.items()})
    filepaths = self._get_filepath_from_dictlist(dict_list=label2filepaths_list, in_one_figure=in_one_figure)
    # save figure after a while
    time_str = '_'.join(filepaths)
    now = time.time()
    last_time = getattr(TextLogger, time_str, 0)
    if now - last_time > save_fig_sec:
      plot_defaultdict2figure(label2filepaths_list=label2filepaths_list, filepaths=filepaths,
                              in_one_figure=in_one_figure)
      setattr(TextLogger, time_str, now)


  def logstr(self, itr, **kwargs):
    # if not comm.is_main_process():
    #   return
    for arg in kwargs:
      if arg not in self.metrics:
        if self.reinitialize:
          self.reinit(arg)
        self.metrics += [arg]
      with open('%s/%s.log' % (self.root, arg), 'a') as f:
        f.write('%3d: %s\n' % (itr, kwargs[arg]))


def summary_defaultdict2txtfig(default_dict, prefix, step,
                               textlogger=None, in_one_figure=True,
                               log_txt=True, log_fig=True, save_fig_sec=100, is_main_process=True):
  if not is_main_process:
    return
  if textlogger is not None:
    prefix_abb = get_prefix_abb(prefix=prefix)
    default_dict_copy = defaultdict(dict)
    # add prefix_abb and key to subkey
    for k, v in default_dict.items():
      default_dict_copy[k] = {prefix_abb + '.' + k + '.' + subk: subv for subk, subv in v.items()}
    default_dict = default_dict_copy

    if log_txt:
      for k, v in default_dict.items():
        textlogger.log(step, **v)
    if log_fig:
      textlogger.log_defaultdict2figure(default_dict, in_one_figure=in_one_figure, save_fig_sec=save_fig_sec)
  else:
    print('textlogger are None!')


def summary_dict2txtfig(dict_data, prefix, step,
                        textlogger=None, in_one_axe=False,
                        log_txt=True, log_fig=True, save_fig_sec=100, is_main_process=True):
  if not is_main_process:
    return
  new_key_dict_data = {}
  for k, v in dict_data.items():
    new_k = k.replace('/', '--')
    new_key_dict_data[new_k] = v
  dict_data = new_key_dict_data

  if in_one_axe:
    default_dict = defaultdict(dict)
    keys = 'sa'
    default_dict[keys] = dict_data
  else:
    default_dict = defaultdict(dict)
    keys = 'ma%d'
    for i, (k, v) in enumerate(dict_data.items()):
      default_dict[keys%i] = {k: v}
  summary_defaultdict2txtfig(default_dict=default_dict, prefix=prefix, step=step,
                             textlogger=textlogger, in_one_figure=True,
                             log_txt=log_txt, log_fig=log_fig, save_fig_sec=save_fig_sec)


global_textlogger = TextLogger(log_root=None)

def set_global_textlogger(textlogger):
  global global_textlogger
  global_textlogger.update(textlogger)
  pass