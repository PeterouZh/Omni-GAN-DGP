from easydict import EasyDict
import yaml
import os
import sys
import unittest
import argparse

from template_lib import utils


class TestingGenerator(unittest.TestCase):

  def test_PathAwareResNetGenCBN(self):
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

    from template_lib.gans.networks import build_generator
    cfg_str = """
          generator:
            name: "PathAwareResNetGenCBN"
            update_cfg: true
        """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    G = build_generator(cfg.generator, n_classes=10, img_size=32).cuda()
    out = G.test_case()

    import torchviz
    g = torchviz.make_dot(out)
    g.view()

    pass


class TestingDiscriminator(unittest.TestCase):

  def test_BigGANDisc(self):
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

    from template_lib.gans.networks import build_discriminator
    cfg_str = """
          name: "BigGANDisc"
          update_cfg: true
        """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    D = build_discriminator(cfg, n_classes=10, img_size=32).cuda()
    out = D.test_case()

    import torchviz
    g = torchviz.make_dot(out)
    g.view()

    pass

  def test_AutoGANCIFAR10ADiscriminator(self):
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

    from template_lib.gans.networks import build_discriminator
    cfg_str = """
          name: "AutoGANCIFAR10ADiscriminator"
          update_cfg: true
        """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    D = build_discriminator(cfg).cuda()
    out = D.test_case()

    import torchviz
    g = torchviz.make_dot(out)
    g.view()

    pass

  def test_DenseDiscriminator_v1(self):
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

    from template_lib.gans.networks import build_discriminator
    cfg_str = """
          name: "DenseDiscriminator_v1"
          update_cfg: true
        """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    D = build_discriminator(cfg).cuda()
    out = D.test_case()

    import torchviz
    g = torchviz.make_dot(out)
    g.view()

    pass


class TestingController(unittest.TestCase):

  def test_PAGANFairController(self):
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

    from template_lib.d2.models import build_d2model
    cfg_str = """
            name: "PAGANFairController"
            update_cfg: true
        """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    controller = build_d2model(cfg, n_classes=10, num_layers=6, num_branches=8).cuda()
    out = controller.test_case()

    pass

  def test_PAGANRLController(self):
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

    from template_lib.d2.models import build_d2model
    cfg_str = """
            name: "PAGANRLController"
            update_cfg: true
        """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    controller = build_d2model(cfg, num_layers=6, num_branches=4).cuda()
    out = controller.test_case()

    pass
