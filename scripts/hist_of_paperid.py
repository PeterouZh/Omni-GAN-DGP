import numpy as np

from tl2.proj.matplot import plt_utils


def get_fig_ax(nrows=1,
               ncols=1,
               style='seaborn-paper',
               usetex=False,
               add_grid=True):
  """

  :param nrows:
  :param ncols:
  :param style:  seaborn-paper seaborn-whitegrid
  :param usetex:
  :return:
  """
  import matplotlib.pyplot as plt

  plt.style.use(style)

  plt_utils.set_times_new_roman_font(usetex=usetex)
  fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.4*ncols, 4.4*nrows))

  if nrows * ncols > 1:
    ax = list(ax)
  if add_grid:
    plt_utils.ax_add_grid(axs=ax)

  return fig, ax


def main():

  n_bins = 100
  fig, axs = get_fig_ax(ncols=2)


  ax = axs[0]
  data_path = "datasets/cvpr2021_accepted_paper_ids.txt"
  data = np.loadtxt(data_path)
  ax.hist(data, bins=n_bins)
  xticks = list(range(0, 12000))
  plt_utils.ax_set_xticks(ax, ticks=xticks, num_ticks=12, fontsize=8)
  plt_utils.ax_set_xlabel(ax, 'CVPR2021 accepted paper id')
  plt_utils.ax_set_ylabel(ax, 'Freq')

  ax = axs[1]
  data_path = "datasets/iccv2021_final_accepts_publish.txt"
  data = np.loadtxt(data_path)
  ax.hist(data, bins=n_bins)
  xticks = list(range(0, 12000))
  plt_utils.ax_set_xticks(ax, ticks=xticks, num_ticks=12, fontsize=8)
  plt_utils.ax_set_xlabel(ax, 'ICCV2021 accepted paper id')
  plt_utils.ax_set_ylabel(ax, 'Freq')

  fig.show()
  pass

if __name__ == '__main__':
  main()





