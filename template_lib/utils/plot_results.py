import pickle
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import unittest


class PlotResults(object):
    
    def __init__(self, ):
        # PlotResults.setup_env()
        
        pass
    
    @staticmethod
    def setup_env():
        import os
        try:
            import mpld3
        except:
            os.system('pip install mpld3')
    
    def get_last_md_inter_time(self, filepath):
        from datetime import datetime, timedelta

        modi_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        modi_inter = datetime.now() - modi_time
        modi_minutes = modi_inter.total_seconds() // 60
        return int(modi_minutes)
    
    def get_fig_axes(self, rows, cols, figsize_wh=(15, 7)):
        import matplotlib.pyplot as plt
        # plt.style.use('ggplot')
        plt.style.use('seaborn-whitegrid')
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            color=['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'orange', 'lime', 'tan', 'salmon', 'gold', 'darkred', 'darkblue'])
        fig, axes = plt.subplots(rows, cols, figsize=(figsize_wh[0]*cols, figsize_wh[1]*rows))
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
        data = np.loadtxt(filepath, delimiter=':')
        data = data.reshape(-1, 2)
        return data

    def plot_defaultdicts(self, outfigure, default_dicts, show_max=True, figsize_wh=(15, 8), legend_size=12,
                          dpi=500, data_load_func=None):

        import tempfile
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
                    filepath = os.path.join(result_dir, file)
                    if not os.path.exists(filepath):
                      print(f'Not exist {filepath}, skip.')
                      continue
                    # get modified time
                    modi_minutes = self.get_last_md_inter_time(filepath)

                    data = data_load_func_list[idx](filepath)
                    # data = np.loadtxt(filepath, delimiter=':')
                    # data = data.reshape(-1, 2)
                    # limit x in a range
                    if data_xlim:
                      data = data[data[:, 0] <= data_xlim]
                    
                    itr_val_str = self.get_itr_val_str(data, show_max[idx])
                    label_str = f'{itr_val_str}' + f'-{modi_minutes:03d}m---' + label
                    
                    axes[idx].plot(data[:, 0], data[:, 1], label=label_str, marker='.', linewidth='5', markersize='10', alpha=0.5)
                    label2datas[label] = data
            axes[idx].legend(prop={'size': legend_size})
            axes[idx].set(**default_dict['properties'])
            axes[idx].grid(b=True, which='major', color='#666666', linestyle='--', alpha=0.2)
                    
            label2datas_list[dict_name] = label2datas
        fig.show()
        fig.savefig(outfigure, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        return label2datas_list


class TestingPlot(unittest.TestCase):

  def test_plot_figures_template(self):
    """
    python -c "from exp.tests.test_styleganv2 import Testing_stylegan2_style_position;\
      Testing_stylegan2_style_position().test_plot_FID_cifar10_style_position()"
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                    --tl_config_file none
                    --tl_command none
                    --tl_outdir {outdir}
                    """
    args = setup_outdir_and_yaml(argv_str)
    outdir = args.tl_outdir

    from template_lib.utils.plot_results import PlotResults
    import collections
    import pickle

    outfigure = os.path.join(outdir, 'IS.jpg')
    default_dicts = collections.OrderedDict()
    show_max = []

    IS_GAN_cGANs_CIFAR100 = collections.defaultdict(dict)
    title = 'IS_GAN_cGANs_CIFAR100'
    log_file = 'textdir/evaltf.ma1.IS_mean_tf.log'
    dd = eval(title)
    dd['results/stylegan2/train_cifar100-20201030_2234_241'] = \
        {'20201030_2234_241-stylegan': "textdir/eval.ma1.IS_mean_tf.log", }

    dd['properties'] = {'title': title, }
    default_dicts[title] = dd
    show_max.append(True)

    plotobs = PlotResults()
    label2datas_list = plotobs.plot_defaultdicts(
        outfigure=outfigure, default_dicts=default_dicts, show_max=show_max, figsize_wh=(16, 7.2))
    print(f'Save to {outfigure}.')

    saved_data = '__'.join(outdir.split('/')[-2:])
    saved_data = f"{outdir}/{saved_data}.pkl"
    with open(saved_data, 'wb') as f:
        pickle.dump(label2datas_list, f)
    print(f"Save data to {saved_data}")
    pass

