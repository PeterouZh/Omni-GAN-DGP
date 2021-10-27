import os
import subprocess
import sys
import unittest
import argparse

from torch.optim import SGD
import torch
from torch import nn

from template_lib.examples import test_bash
from template_lib import utils
from template_lib.nni import update_nni_config_file


class Testing_optim(unittest.TestCase):

  def test_base_usage(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import torch
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')  # ignore warnings

    x = torch.linspace(-np.pi, np.pi, 2000)
    y = torch.sin(x)

    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    model = torch.nn.Sequential(
      torch.nn.Linear(3, 1),
      torch.nn.Flatten(0, 1)
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    for t in range(1, 1001):
      y_pred = model(xx)
      loss = loss_fn(y_pred, y)
      if t % 100 == 0:
        print('No.{: 5d}, loss: {:.6f}'.format(t, loss.item()))
      optimizer.zero_grad()  # 梯度清零
      loss.backward()  # 反向传播计算梯度
      optimizer.step()  # 梯度下降法更新参数

    pass

  def test_diff_lr(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str, return_cfg=True)

    from torch.optim import SGD
    from torch import nn

    class DummyModel(nn.Module):
      def __init__(self, class_num=10):
        super(DummyModel, self).__init__()
        self.base = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, class_num)

      def forward(self, x):
        x = self.base(x)
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    model = DummyModel().cuda()

    optimizer = SGD([
      {'params': model.base.parameters()},
      {'params': model.fc.parameters(), 'lr': 1e-3}  # 对 fc的参数设置不同的学习率
    ], lr=1e-2, momentum=0.9)

    pass

  def test_step_closure(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str, return_cfg=True)

    from torch.nn import CrossEntropyLoss

    class DummyModel(nn.Module):
      def __init__(self, class_num=10):
        super(DummyModel, self).__init__()
        self.base = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, class_num)

      def forward(self, x):
        x = self.base(x)
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    dummy_model = DummyModel().cuda()

    optimizer = SGD(dummy_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    # 定义loss
    loss_fn = CrossEntropyLoss()
    # 定义数据
    batch_size = 2
    data = torch.randn(64, 3, 64, 128).cuda()  # 制造假数据shape=64 * 3 * 64 * 128
    data_label = torch.randint(0, 10, size=(64,), dtype=torch.long).cuda()  # 制造假的label

    for batch_index in range(10):
      batch_data = data[batch_index * batch_size: batch_index * batch_size + batch_size]
      batch_label = data_label[batch_index * batch_size: batch_index * batch_size + batch_size]

      def closure():
        optimizer.zero_grad()  # 清空梯度
        output = dummy_model(batch_data)  # forward
        loss = loss_fn(output, batch_label)  # 计算loss
        loss.backward()  # backward
        print('No.{: 2d} loss: {:.6f}'.format(batch_index, loss.item()))
        return loss

      optimizer.step(closure=closure)  # 更新参数

    pass


class Testing_scheduler(unittest.TestCase):

  def test_StepLR(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import torch.nn as nn
    from torch.optim import lr_scheduler
    from matplotlib import pyplot as plt

    model = nn.Linear(3, 64)
    def create_optimizer():
      return SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    def plot_lr(scheduler, title='', labels=['base'], nrof_epoch=100):
      lr_li = [[] for _ in range(len(labels))]
      epoch_li = list(range(nrof_epoch))
      for epoch in epoch_li:
        scheduler.step()  # 调用step()方法,计算和更新optimizer管理的参数基于当前epoch的学习率
        lr = scheduler.get_last_lr()  # 获取当前epoch的学习率
        for i in range(len(labels)):
          lr_li[i].append(lr[i])
      for lr, label in zip(lr_li, labels):
        plt.plot(epoch_li, lr, label=label)
      plt.grid()
      plt.xlabel('epoch')
      plt.ylabel('lr')
      plt.title(title)
      plt.legend()
      plt.show()

    ## StepLR 可视化学习率
    optimizer = create_optimizer()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    plot_lr(scheduler, title='StepLR')
    pass

  def test_MultiplicativeLR(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import torch.nn as nn
    from torch.optim import lr_scheduler
    from matplotlib import pyplot as plt

    def plot_lr(scheduler, title='', labels=['base'], nrof_epoch=100):
      lr_li = [[] for _ in range(len(labels))]
      epoch_li = list(range(nrof_epoch))
      for epoch in epoch_li:
        scheduler.step()  # 调用step()方法,计算和更新optimizer管理的参数基于当前epoch的学习率
        lr = scheduler.get_last_lr()  # 获取当前epoch的学习率
        for i in range(len(labels)):
          lr_li[i].append(lr[i])
      for lr, label in zip(lr_li, labels):
        plt.plot(epoch_li, lr, label=label)
      plt.grid()
      plt.xlabel('epoch')
      plt.ylabel('lr')
      plt.title(title)
      plt.legend()
      plt.show()

    ## StepLR 可视化学习率

    base = nn.Linear(3, 32)
    fc = nn.Linear(32, 10)
    optimizer = SGD([
      {'params': base.parameters()},
      {'params': fc.parameters(), 'lr': 0.05}  # 对 fc的参数设置不同的学习率
    ], lr=0.1, momentum=0.9)
    lambda_base = lambda epoch: 0.5 if epoch % 10 == 0 else 1
    lambda_fc = lambda epoch: 0.8 if epoch % 10 == 0 else 1
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, [lambda_base, lambda_fc])
    plot_lr(scheduler, title='MultiplicativeLR', labels=['base', 'fc'])
    pass

  def test_LambdaLR(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import torch.nn as nn
    from torch.optim import lr_scheduler
    from matplotlib import pyplot as plt

    model = nn.Linear(3, 64)

    def create_optimizer():
      return SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    def plot_lr(scheduler, title='', labels=['base'], nrof_epoch=100):
      lr_li = [[] for _ in range(len(labels))]
      epoch_li = list(range(nrof_epoch))
      for epoch in epoch_li:
        scheduler.step()  # 调用step()方法,计算和更新optimizer管理的参数基于当前epoch的学习率
        lr = scheduler.get_last_lr()  # 获取当前epoch的学习率
        for i in range(len(labels)):
          lr_li[i].append(lr[i])
      for lr, label in zip(lr_li, labels):
        plt.plot(epoch_li, lr, label=label)
      plt.grid()
      plt.xlabel('epoch')
      plt.ylabel('lr')
      plt.title(title)
      plt.legend()
      plt.show()

    def lambda_foo(epoch):
      if epoch < 10:
        return (epoch + 1) * 1e-3
      elif epoch < 40:
        return 1e-2
      else:
        return 1e-3

    optimizer = create_optimizer()
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_foo)
    plot_lr(scheduler, title='LambdaLR')
    pass

  def test_ExponentialLR(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import torch.nn as nn
    from torch.optim import lr_scheduler
    from matplotlib import pyplot as plt

    model = nn.Linear(3, 64)

    def create_optimizer():
      return SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    def plot_lr(scheduler, title='', labels=['base'], nrof_epoch=100):
      lr_li = [[] for _ in range(len(labels))]
      epoch_li = list(range(nrof_epoch))
      for epoch in epoch_li:
        scheduler.step()  # 调用step()方法,计算和更新optimizer管理的参数基于当前epoch的学习率
        lr = scheduler.get_last_lr()  # 获取当前epoch的学习率
        for i in range(len(labels)):
          lr_li[i].append(lr[i])
      for lr, label in zip(lr_li, labels):
        plt.plot(epoch_li, lr, label=label)
      plt.grid()
      plt.xlabel('epoch')
      plt.ylabel('lr')
      plt.title(title)
      plt.legend()
      plt.show()

    optimizer = create_optimizer()
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    plot_lr(scheduler, title='ExponentialLR')
    pass

  def test_CosineAnnealingLR(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import torch.nn as nn
    from torch.optim import lr_scheduler
    from matplotlib import pyplot as plt

    model = nn.Linear(3, 64)

    def create_optimizer():
      return SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    def plot_lr(scheduler, title='', labels=['base'], nrof_epoch=100):
      lr_li = [[] for _ in range(len(labels))]
      epoch_li = list(range(nrof_epoch))
      for epoch in epoch_li:
        scheduler.step()  # 调用step()方法,计算和更新optimizer管理的参数基于当前epoch的学习率
        lr = scheduler.get_last_lr()  # 获取当前epoch的学习率
        for i in range(len(labels)):
          lr_li[i].append(lr[i])
      for lr, label in zip(lr_li, labels):
        plt.plot(epoch_li, lr, label=label)
      plt.grid()
      plt.xlabel('epoch')
      plt.ylabel('lr')
      plt.title(title)
      plt.legend()
      plt.show()

    optimizer = create_optimizer()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    plot_lr(scheduler, title='CosineAnnealingLR')

    pass

  def test_CosineAnnealingWarmRestarts(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import torch.nn as nn
    from torch.optim import lr_scheduler
    from matplotlib import pyplot as plt

    model = nn.Linear(3, 64)

    def create_optimizer():
      return SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    def plot_lr(scheduler, title='', labels=['base'], nrof_epoch=100):
      lr_li = [[] for _ in range(len(labels))]
      epoch_li = list(range(nrof_epoch))
      for epoch in epoch_li:
        scheduler.step()  # 调用step()方法,计算和更新optimizer管理的参数基于当前epoch的学习率
        lr = scheduler.get_last_lr()  # 获取当前epoch的学习率
        for i in range(len(labels)):
          lr_li[i].append(lr[i])
      for lr, label in zip(lr_li, labels):
        plt.plot(epoch_li, lr, label=label)
      plt.grid()
      plt.xlabel('epoch')
      plt.ylabel('lr')
      plt.title(title)
      plt.legend()
      plt.show()

    optimizer = create_optimizer()
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    plot_lr(scheduler, title='CosineAnnealingWarmRestarts')

    pass

  def test_CyclicLR(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import torch.nn as nn
    from torch.optim import lr_scheduler
    from matplotlib import pyplot as plt

    model = nn.Linear(3, 64)

    def create_optimizer():
      return SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    def plot_lr(scheduler, title='', labels=['base'], nrof_epoch=100):
      lr_li = [[] for _ in range(len(labels))]
      epoch_li = list(range(nrof_epoch))
      for epoch in epoch_li:
        scheduler.step()  # 调用step()方法,计算和更新optimizer管理的参数基于当前epoch的学习率
        lr = scheduler.get_last_lr()  # 获取当前epoch的学习率
        for i in range(len(labels)):
          lr_li[i].append(lr[i])
      for lr, label in zip(lr_li, labels):
        plt.plot(epoch_li, lr, label=label)
      plt.grid()
      plt.xlabel('epoch')
      plt.ylabel('lr')
      plt.title(title)
      plt.legend()
      plt.show()

    optimizer = create_optimizer()
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=25, step_size_down=10)
    plot_lr(scheduler, title='CyclicLR')

    pass
  
  def test_OneCycleLR(self, debug=True):
    """
    Usage:
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/pypi/torch1_7_0 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        pip install -r requirements.txt

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import torch.nn as nn
    from torch.optim import lr_scheduler
    from matplotlib import pyplot as plt

    model = nn.Linear(3, 64)

    def create_optimizer():
      return SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    def plot_lr(scheduler, title='', labels=['base'], nrof_epoch=100):
      lr_li = [[] for _ in range(len(labels))]
      epoch_li = list(range(nrof_epoch))
      for epoch in epoch_li:
        scheduler.step()  # 调用step()方法,计算和更新optimizer管理的参数基于当前epoch的学习率
        lr = scheduler.get_last_lr()  # 获取当前epoch的学习率
        for i in range(len(labels)):
          lr_li[i].append(lr[i])
      for lr, label in zip(lr_li, labels):
        plt.plot(epoch_li, lr, label=label)
      plt.grid()
      plt.xlabel('epoch')
      plt.ylabel('lr')
      plt.title(title)
      plt.legend()
      plt.show()

    optimizer = create_optimizer()
    scheduler = lr_scheduler.OneCycleLR(optimizer, 0.1, total_steps=100)
    plot_lr(scheduler, title='OneCycleLR')
    pass