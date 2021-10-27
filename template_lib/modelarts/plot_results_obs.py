from IPython.core.debugger import set_trace


class PlotResultsObs(object):

  def __init__(self, ):
    self.root_obs_dict = {
      'beijing': 's3://bucket-cv-competition-bj4/ZhouPeng',
      'huanan': 's3://bucket-1892/ZhouPeng',
      'huabei': 's3://bucket-cv-competition',
      '7001': "s3://bucket-7001/ZhouPeng"
    }
    PlotResultsObs.setup_env()

    pass

  @staticmethod
  def setup_env():
    import os
    try:
      import mpld3
    except:
      os.system('pip install mpld3')

  def get_last_md_inter_time(self, filepath_obs):
    import moxing as mox
    from datetime import datetime, timedelta

    statbuf = mox.file.stat(filepath_obs)
    modi_time = datetime.fromtimestamp(statbuf.mtime_nsec / 1e9) + timedelta(hours=8)
    modi_inter = datetime.now() - modi_time
    modi_minutes = modi_inter.total_seconds() // 60
    return int(modi_minutes)

  def get_fig_axes(self, rows, cols, figsize_wh=(15, 7)):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
      color=['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'orange', 'lime', 'tan', 'salmon', 'gold', 'darkred',
             'darkblue'])
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_wh[0] * cols, figsize_wh[1] * rows))
    if rows * cols > 1:
      axes = axes.ravel()
    else:
      axes = [axes]
    return fig, axes

  def get_itr_val_str(self, data, ismax):
    if ismax:
      itr = int(data[:, 0][data[:, 1].argmax()])
      val = data[:, 1].max()
      return f'itr.{itr:06d}_maxv.{val:.3f}'
    else:
      itr = int(data[:, 0][data[:, 1].argmin()])
      val = data[:, 1].min()
      return f'itr.{itr:06d}_minv.{val:.3f}'

  def _data_load_func(self, filepath):
    import numpy as np
    data = np.loadtxt(filepath, delimiter=':')
    data = data.reshape(-1, 2)
    return data

  def plot_defaultdicts(self, default_dicts, show_max=True, bucket='huanan', figsize_wh=(15, 8), legend_size=12,
                        data_load_func=None):
    import matplotlib.pyplot as plt
    % matplotlib inline
    import numpy as np
    import mpld3
    mpld3.enable_notebook()
    import os
    import moxing as mox
    import tempfile
    assert isinstance(default_dicts, dict)

    if not isinstance(show_max, list):
      show_max = [show_max]
    assert len(show_max) == len(default_dicts)

    fig, axes = self.get_fig_axes(rows=len(default_dicts), cols=1, figsize_wh=figsize_wh)

    if data_load_func is None:
      data_load_func_list = [self._data_load_func, ] * len(default_dicts)
    elif not isinstance(data_load_func, (list, tuple)):
      data_load_func_list = [data_load_func, ] * len(default_dicts)
    else:
      data_load_func_list = data_load_func

    bucket = self.root_obs_dict[bucket]
    root_dir = os.path.expanduser('~/results')

    label2datas_list = {}
    for idx, (dict_name, default_dict) in enumerate(default_dicts.items()):
      data_xlim = None
      axes_prop = default_dict.get('properties')
      if axes_prop is not None:
        if 'xlim' in axes_prop:
          data_xlim = axes_prop['xlim'][-1]

      label2datas = {}
      # for each result dir
      for (result_dir, label2file) in default_dict.items():
        if result_dir == 'properties':
          continue
        # for each texlog file
        for label, file in label2file.items():
          filepath = os.path.join(root_dir, result_dir, file)
          filepath_obs = os.path.join(bucket, result_dir, file)
          if not mox.file.exists(filepath_obs):
            print("=> Not exist: '%s'" % filepath_obs)
            continue
          mox.file.copy(filepath_obs, filepath)
          # get modified time
          modi_minutes = self.get_last_md_inter_time(filepath_obs)

          data = data_load_func_list[idx](filepath)
          # data = np.loadtxt(filepath, delimiter=':')
          # data = data.reshape(-1, 2)
          # limit x in a range
          if data_xlim:
              data = data[data[:, 0] <= data_xlim]

          itr_val_str = self.get_itr_val_str(data, show_max[idx])
          label_str = f'{itr_val_str}' + f'-{modi_minutes:03d}m---' + label

          axes[idx].plot(data[:, 0], data[:, 1], label=label_str, marker='.', linewidth='5', markersize='15', alpha=0.5)
          label2datas[label] = data
      axes[idx].legend(prop={'size': legend_size})
      axes[idx].set(**default_dict['properties'])
      axes[idx].grid(b=True, which='major', color='#666666', linestyle='--', alpha=0.2)

      label2datas_list[dict_name] = label2datas

    return label2datas_list


import unittest
class Testing_plot_results_obs(unittest.TestCase):

  def test_plot_results(self, ):
    import collections, os, functools

    default_dicts = collections.OrderedDict()
    show_max = []

    FID = collections.defaultdict(dict)
    title = 'FID'
    log_file = 'textdir/evaltorch.ma2.FID.log'
    dd = eval(title)
    dd['results/Omni-GAN-ImageNet/OmniInrGAN_ImageNet256/train_ImageNet256-20210126_161550_248'] = \
      {'20210126_161550_248-OmniInrGAN256-Gwd.1e-4-nd.2-bs.128x2': log_file, }

    dd['properties'] = {'title': title, }
    default_dicts[title] = dd
    show_max.append(False)


    plotobs = PlotResultsObs()
    label2datas_list = plotobs.plot_defaultdicts(default_dicts=default_dicts, show_max=show_max, bucket='7001',
                                                 figsize_wh=(16, 7.2))
    pass

  def test_save_results_list(self):
    import moxing as mox
    import pickle

    obs_path = "s3://bucket-7001/ZhouPeng/results/Omni-GAN-ImageNet/data"
    saved_data = 'OmniGAN_ImageNet128_results.pkl'
    with open(saved_data, 'wb') as f:
      pickle.dump(label2datas_list, f)
    mox.file.copy(saved_data, f'{obs_path}/{saved_data}')
    pass