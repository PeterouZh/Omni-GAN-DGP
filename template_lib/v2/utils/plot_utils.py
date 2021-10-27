import numpy as np
import math
import re, os
import multiprocessing


class MatPlot(object):
  def __init__(self, style='ggplot'):
    """
      plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue'])
    :param style: [classic, ggplot]
    """
    import matplotlib.pyplot as plt
    # R style
    plt.style.use(style)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
      color=['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'orange', 'lime', 'tan', 'salmon', 'gold', 'darkred',
             'darkblue'])

    pass

  def get_fig_and_ax(self, nrows=1, ncols=1, ravel=False, fig_w_h=(6.4, 4.8)):
    """
    ax.legend(loc='best')
    """
    import matplotlib.pyplot as plt
    figsize = (fig_w_h[0] * ncols, fig_w_h[1] * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if ravel:
      if ncols == 1 and nrows == 1:
        axes = [axes]
      else:
        axes = axes.ravel()
    return fig, axes

  def save_to_png(self, fig, filepath, dpi=1000, bbox_inches='tight',
                  pad_inches=0.1):
    assert filepath.endswith('.png')
    fig.savefig(
      filepath, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)

  def save_to_pdf(self, fig, filepath):
    fig.savefig(filepath, bbox_inches='tight', pad_inches=0)

  def parse_logfile_using_re(self, logfile, re_str):
    """
    import re

    """
    with open(logfile) as f:
      logstr = f.read()
      val = [float(x) for x in re_str.findall(logstr)]
      idx = range(len(val))
    return (idx, val)


def parse_logfile(args, myargs):
  config = getattr(myargs.config, args.command)
  matplot = MatPlot()
  fig, ax = matplot.get_fig_and_ax()
  if len(config.logfiles) == 1:
    logfiles = config.logfiles * len(config.re_strs)
  for logfile, re_str in zip(logfiles, config.re_strs):
    RE_STR = re.compile(re_str)
    (idx, val) = matplot.parse_logfile_using_re(logfile=logfile, re_str=RE_STR)
    ax.plot(idx, val, label=re_str)
  ax.legend()
  matplot.save_to_png(
    fig, filepath=os.path.join(args.outdir, config.title + '.png'))
  pass


def _plot_figure(names, datas, outdir, in_one_axes=False):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  assert len(datas) == len(names)
  filename = os.path.join(outdir, 'plot_' + '__'.join(names) + '.png')
  matplot = MatPlot()
  if not in_one_axes:
    ncols = math.ceil(math.sqrt(len(names)))
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = matplot.get_fig_and_ax(nrows=nrows, ncols=ncols)
    if ncols == 1 and nrows == 1:
      axes = [axes]
    else:
      axes = axes.ravel()
  else:
    ncols = 1
    nrows = 1
    fig, axes = matplot.get_fig_and_ax(nrows=nrows, ncols=ncols)
    axes = [axes] * len(names)

  for idx, (label, data) in enumerate(zip(names, datas)):
    data = data.reshape(-1, 2)
    axes[idx].plot(data[:, 0], data[:, 1], marker='.', label=label, alpha=0.7)
    axes[idx].legend(loc='best')

  matplot.save_to_png(fig=fig, filepath=filename, dpi=None, bbox_inches=None)
  plt.close(fig)
  pass


class PlotFigureProcessing(multiprocessing.Process):
  """
    worker = PlotFigureProcessing(args=(s, d, copytree))
    worker.start()
    worker.join()
  """
  def run(self):
    names, filepaths, outdir, in_one_axes = self._args
    datas = []
    for filepath in filepaths:
      data = np.loadtxt(filepath, delimiter=':')
      datas.append(data)
    _plot_figure(
      names=names, datas=datas, outdir=outdir, in_one_axes=in_one_axes)
    pass

def plot_figure(names, filepaths, outdir, in_one_axes, join=False):
  worker = PlotFigureProcessing(args=(names, filepaths, outdir, in_one_axes))
  worker.start()

  if join:
    worker.join()
  pass


class PlotDefaultdict2figure(multiprocessing.Process):
  """
    worker = PlotDefaultdict2figure(args=(s, d, copytree))
    worker.start()
    worker.join()
  """
  def run(self):
    label2filepaths_list, filepaths, in_one_figure = self._args
    # load data
    label2datas_list = []
    for label2filepaths in label2filepaths_list:
      label2datas_list.append({k: np.loadtxt(filepath, delimiter=':') for k, filepath in label2filepaths.items()})
    self._plot_figure(label2datas_list=label2datas_list, filepaths=filepaths, in_one_figure=in_one_figure)
    pass

  def _plot_figure(self, label2datas_list, filepaths, in_one_figure=False):
    import matplotlib
    # matplotlib.use(arg='Agg', warn=False)
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    if in_one_figure:
      self._plot_in_one_figure(label2datas_list, filepaths)
    else:
      self._plot_in_multi_figures(label2datas_list=label2datas_list, filepaths=filepaths)

  def _plot_in_one_figure(self, label2datas_list, filepaths):
    import matplotlib.pyplot as plt
    assert len(filepaths) == 1
    matplot = MatPlot()
    # ncols = math.ceil(math.sqrt(len(label2datas_list)))
    ncols = 2
    nrows = (len(label2datas_list) + ncols - 1) // ncols
    fig, axes = matplot.get_fig_and_ax(nrows=nrows, ncols=ncols, ravel=True)


    for idx, label2datas in enumerate(label2datas_list):
      for label, data in label2datas.items():
        data = data.reshape(-1, 2)
        axes[idx].plot(data[:, 0], data[:, 1], marker='.', label=label, alpha=0.7)
      axes[idx].legend(loc='best')

    matplot.save_to_png(fig=fig, filepath=filepaths[0], dpi=None, bbox_inches='tight')
    plt.close(fig)
    pass

  def _plot_in_multi_figures(self, label2datas_list, filepaths):
    import matplotlib.pyplot as plt
    assert len(filepaths) == len(label2datas_list)
    matplot = MatPlot()
    for idx, label2datas in enumerate(label2datas_list):
      fig, axes = matplot.get_fig_and_ax(nrows=1, ncols=1)

      for label, data in label2datas.items():
        data = data.reshape(-1, 2)
        axes.plot(data[:, 0], data[:, 1], marker='.', label=label, alpha=0.7)
      axes.legend(loc='best')

      matplot.save_to_png(fig=fig, filepath=filepaths[idx], dpi=None, bbox_inches='tight')
      plt.close(fig)
    pass


def plot_defaultdict2figure(label2filepaths_list, filepaths, in_one_figure, join=False):
  worker = PlotDefaultdict2figure(args=(label2filepaths_list, filepaths, in_one_figure))
  worker.start()

  if join:
    worker.join()
  pass