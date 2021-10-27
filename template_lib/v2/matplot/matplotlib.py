import matplotlib.pyplot as plt


def set_font():
  plt.rcParams["font.family"] = "Times New Roman"
  plt.rcParams["mathtext.fontset"] = "cm"


def set_non_gui_backend(plt, ):
  try:
    fig = plt.figure()
  except:
    plt.switch_backend('agg')







