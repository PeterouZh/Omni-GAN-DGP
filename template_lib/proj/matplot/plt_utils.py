import matplotlib.pyplot as plt
from pathlib import Path


colors_dict = {
  'dark_red': '#FF0000',
  'beauty_green': '#33E69C',
  'blue': '#2B99F0',
  'black': '#020202',
  'pink': '#FD4BD7',

  'red': '#FE2224',
  'green': '#17CE11',
  'dark_green': '#17CE11',
  'blue_violet': '#7241BE',
  'grey': '#A6A6A6',
  'peach': '#FEAA92',
  'yellow': '#FEFF00',
  'purple': '#A40190',
  'beauty_blue': '#1F5CFA',
  'beauty_light_blue': '#0CE6DA',
  'beauty_red': '#F84D4D',
  'beauty_orange': '#FF743E',
}


def set_times_new_roman_font():
  plt.rcParams["font.family"] = "Times New Roman"
  plt.rcParams["mathtext.fontset"] = "cm"


def set_non_gui_backend(plt, ):
  try:
    fig = plt.figure()
  except:
    plt.switch_backend('agg')


def ax_legend(ax, font_size, loc='lower right', ncol=1, framealpha=1, zorder=100):
  legend = ax.legend(prop={'size': font_size}, ncol=ncol, loc=loc, framealpha=framealpha)
  legend.set_zorder(zorder)
  pass


def savefig(saved_file, fig, pad_inches=0.0):
  print(f'Saved to {saved_file}')
  saved_file = Path(saved_file)
  saved_file_png = f"{saved_file.parent}/{saved_file.stem}.png"

  fig.savefig(saved_file, bbox_inches='tight', pad_inches=pad_inches)
  fig.savefig(saved_file_png, bbox_inches='tight', pad_inches=0.1)
  pass