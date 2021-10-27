import unittest
import os
import sys
import yaml
from easydict import EasyDict
import torch

from template_lib import utils


class TestingNoise(unittest.TestCase):

  def test_Normal(self):
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

    from template_lib.d2.distributions import build_d2distributions
    cfg_str = """
            name: "Normal"
            loc: 0
            scale: 1
            sample_shape: 
              - 20
              - 300000
        """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    noise = build_d2distributions(cfg)
    z = noise.sample()
    print(z.mean(), z.std())
    pass

  def test_CategoricalUniform(self):
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

    from template_lib.d2.distributions import build_d2distributions
    cfg_str = """
      name: "CategoricalUniform"
      n_classes: "kwargs['n_classes']"
      sample_shape: "kwargs['sample_shape']"
    """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    noise = build_d2distributions(cfg, n_classes=1000, sample_shape=4)
    y = noise.sample()
    y = y.type(torch.float32)
    print(y.mean(), y.std())
    pass


class TestingFairNAS_noise_sample(unittest.TestCase):

  def test_FairNASNormal(self):
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

    from template_lib.d2.distributions import build_d2distributions
    cfg_str = """
            name: "FairNASNormal"
            loc: 0
            scale: 1
            sample_shape: 
              - 2
              - 3
            num_ops: 8
        """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    noise = build_d2distributions(cfg)
    z = noise.sample()
    print(z.mean(), z.std())
    pass

  def test_FairNASCategoricalUniform(self):
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

    from template_lib.d2.distributions import build_d2distributions
    cfg_str = """
      name: "FairNASCategoricalUniform"
      n_classes: "kwargs['n_classes']"
      sample_shape: "kwargs['sample_shape']"
      num_ops: 2
    """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    noise = build_d2distributions(cfg, n_classes=8, sample_shape=10)
    y = noise.sample()
    y = y.type(torch.float32)
    print(y.mean(), y.std())
    pass