import torch
from easydict import EasyDict
import os
import sys
import unittest
from template_lib import utils


class TestingLayers(unittest.TestCase):

  def test_SNConv2d(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers import build_d2layer

    cfg_str = """
      SNConv2d_3x3:
        name: "SNConv2d"
        in_channels: "kwargs['in_channels']"
        out_channels: "kwargs['out_channels']"
        kernel_size: 3
        padding: 1
    """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    op = build_d2layer(cfg.SNConv2d_3x3, in_channels=8, out_channels=8)

    op.cuda()
    x = torch.randn(2, 8, 32, 32).cuda()
    y = op(x)
    pass

  def test_Conv2d(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers import build_d2layer

    cfg_str = """
      Conv2d_3x3:
        name: "Conv2d"
        in_channels: "kwargs['in_channels']"
        out_channels: "kwargs['out_channels']"
        kernel_size: 3
        padding: 1
    """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    op = build_d2layer(cfg.Conv2d_3x3, in_channels=8, out_channels=8)

    op.cuda()
    x = torch.randn(2, 8, 32, 32).cuda()
    y = op(x)
    pass

  def test_MixedLayerCond(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers import build_d2layer

    cfg_str = """
      layer:
        name: "MixedLayerCond"
        in_channels: "kwargs['in_channels']"
        out_channels: "kwargs['out_channels']"
        cfg_ops: "kwargs['cfg_ops']"
        cfg_bn:
          name: "BatchNorm2d"
          num_features: "kwargs['num_features']"
          affine: true
          track_running_stats: true
        cfg_act:
          name: "ReLU"
      cfg_ops:
        SNConv2d_3x3:
          name: "SNConv2d"
          in_channels: "kwargs['in_channels']"
          out_channels: "kwargs['out_channels']"
          kernel_size: 3
          padding: 1
        Conv2d_3x3:
          name: "Conv2d"
          in_channels: "kwargs['in_channels']"
          out_channels: "kwargs['out_channels']"
          kernel_size: 3
          padding: 1
    """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    op = build_d2layer(cfg.layer, in_channels=8, out_channels=8, cfg_ops=cfg.cfg_ops)
    num_classes = 2
    bs = num_classes
    num_ops = 2

    op.cuda()
    x = torch.randn(bs, 8, 32, 32).cuda()
    y = torch.arange(bs).cuda()
    sample_arc = torch.arange(num_ops).cuda()
    x = op(x, y, sample_arc)
    pass

  def test_CondBatchNorm2d(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers import build_d2layer

    cfg_str = """
      cfg_bn:
          name: "CondBatchNorm2d"
          in_features: "kwargs['in_features']"
          out_features: "kwargs['out_features']"
          cfg_fc:
            name: "Linear"
            in_features: "kwargs['in_features']"
            out_features: "kwargs['out_features']"
    """
    cfg = EasyDict(yaml.safe_load(cfg_str))

    bs = 2
    in_features = out_features = 8

    op = build_d2layer(cfg.cfg_bn, in_features=in_features, out_features=out_features)
    op.cuda()
    x = torch.randn(bs, in_features, 32, 32).cuda()
    y = torch.randn(bs, in_features).cuda()

    x = op(x, y)
    pass

  def test_DepthwiseSeparableConv2d(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers import build_d2layer

    cfg_str = """
        name: "DepthwiseSeparableConv2d"
        in_channels: 256
        out_channels: 256
        kernel_size: 7
        padding: 3
    """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    op = build_d2layer(cfg)

    op.cuda()
    x = torch.randn(2, 256, 32, 32).cuda()
    y = op(x)

    import torchviz
    g = torchviz.make_dot(y)
    g.view()
    pass

  def test_CondInstanceNorm2d(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers import build_d2layer

    cfg_str = """
          name: "CondInstanceNorm2d"
          in_features: "kwargs['in_features']"
          out_features: "kwargs['out_features']"
          cfg_fc:
            name: "Linear"
            in_features: "kwargs['in_features']"
            out_features: "kwargs['out_features']"
    """
    cfg = EasyDict(yaml.safe_load(cfg_str))

    bs = 2
    in_features = out_features = 8

    op = build_d2layer(cfg, in_features=in_features, out_features=out_features)
    op.cuda()
    x = torch.randn(bs, in_features, 32, 32).cuda()
    y = torch.randn(bs, in_features).cuda()

    x = op(x, y)

    import torchviz
    g = torchviz.make_dot(x)
    g.view()
    pass

  def test_DenseCell(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers import build_d2layer

    cfg_str = """
          name: "DenseCell"
          update_cfg: true
          in_channels: 3
          out_channels: 32
          
    """
    cfg = EasyDict(yaml.safe_load(cfg_str))

    op = build_d2layer(cfg, in_channels=3, out_channels=32)
    op.cuda()
    out = op.test_case()


    import torchviz
    g = torchviz.make_dot(out)
    g.view()
    pass

  def test_DenseBlock(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers import build_d2layer

    from template_lib.d2.layers import DenseBlock

    DenseBlock.test_case()

    pass

  def test_StyleLayer(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers import build_d2layer

    cfg_str = """
          name: "StyleLayer"
          update_cfg: true
    """
    cfg = EasyDict(yaml.safe_load(cfg_str))

    op = build_d2layer(cfg)
    op.cuda()
    out = op.test_case()

    import torchviz
    g = torchviz.make_dot(out)
    g.view()
    pass

  def test_StyleV2Conv(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers import build_d2layer

    cfg_str = """
          name: "StyleV2Conv"
          update_cfg: true
    """
    cfg = EasyDict(yaml.safe_load(cfg_str))

    op = build_d2layer(cfg, in_channels=256, out_channels=256)
    op.cuda()
    out = op.test_case(in_channels=256, out_channels=256)

    import torchviz
    g = torchviz.make_dot(out)
    g.view()
    pass


class TestingLayers_v2(unittest.TestCase):

  def test_DenseBlockWithArc(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers_v2.nas_layers import DenseBlockWithArc

    DenseBlockWithArc.test_case()

    pass

  def test_ModulatedConv2d(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers_v2.convs import ModulatedConv2d

    ModulatedConv2d.test_case()
    pass

  def test_EmbeddingModulatedConv2d(self):
    """
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    import yaml
    from template_lib.d2.layers_v2.convs import EmbeddingModulatedConv2d

    EmbeddingModulatedConv2d.test_case()
    pass

