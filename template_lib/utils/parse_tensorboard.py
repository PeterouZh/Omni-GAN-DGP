import sys
import pprint
from collections import OrderedDict
import numpy as np
import os
import glob

import tensorflow as tf

from . import plot_utils


class SummaryReader(object):
  def __init__(self, tbdir):
    self.tbdir = tbdir
    self.event_paths = sorted(glob.glob(os.path.join(tbdir, "**/event*"),
                                        recursive=True))

  def get_tags(self, stop_step=10000):
    tags = set()
    for event_path in self.event_paths:
      try:
        for e in tf.train.summary_iterator(event_path):
          for v in e.summary.value:
            tags.add(v.tag)
          if e.step > stop_step:
            break
      except:
        print("Exception occured!", sys.exc_info()[0])
    tags = sorted(tags)
    return tags

  def get_scalar(self, tag, use_dump=True, dump_name=None):
    if dump_name:
      data_path = os.path.join(self.tbdir, dump_name + '.pkl')
    else:
      data_name = tag.replace('/', '_')
      data_path = os.path.join(self.tbdir, data_name + '.pkl')

    if os.path.exists(data_path) and use_dump:
      data = np.load(data_path, allow_pickle=True)
    else:
      data = []
      for event_path in self.event_paths:
        try:
          for e in tf.train.summary_iterator(event_path):
            for v in e.summary.value:
              if v.tag == tag:
                step = e.step
                data.append([step, v.simple_value])
        except:
          print("Exception occured!", sys.exc_info()[0])
      data = np.array(data).T
      data.dump(data_path)
    return data


def parse_tensorboard(args, myargs):
  config = getattr(myargs.config, args.command)
  print(pprint.pformat(OrderedDict(config)))
  data_dict = {}

  for label, line in config.lines.items():
    print("Parsing: %s"%line.tbdir)
    summary_reader = SummaryReader(tbdir=line.tbdir)
    tags = summary_reader.get_tags()
    print(pprint.pformat(tags))
    data = summary_reader.get_scalar(tag=config.tag, use_dump=config.use_dump)
    data_dict[label] = data

  matplot = plot_utils.MatPlot()
  fig, ax = matplot.get_fig_and_ax()
  for label, line in config.lines.items():
    data = data_dict[label]
    ax.plot(data[0], data[1], **{"label": label,
                                 **getattr(line, 'property', {})})
  ax.legend()
  ax.set_ylim(config.ylim)
  ax.set_xlim(config.xlim)
  ax.set_title(config.title, fontdict={'fontsize': 10})

  fig_name = config.tag.replace('/', '_') + '.png'
  filepath = os.path.join(args.outdir, fig_name)
  matplot.save_to_png(fig=fig, filepath=filepath)

  fig_name = config.tag.replace('/', '_') + '.pdf'
  filepath = os.path.join(args.outdir, fig_name)
  matplot.save_to_pdf(fig=fig, filepath=filepath)

  pass


