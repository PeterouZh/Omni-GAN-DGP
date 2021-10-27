import logging

import tqdm
import collections
from collections import OrderedDict
from easydict import EasyDict
import yaml
import functools
import numpy as np
import statistics

import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from template_lib.utils import get_attr_kwargs, get_ddp_attr, AverageMeter, get_prefix_abb
from template_lib.v2.config import update_config
from template_lib.d2.utils import comm
from template_lib.d2.layers import build_d2layer
from template_lib.d2.models.build import D2MODEL_REGISTRY
from template_lib.trainer.base_trainer import summary_defaultdict2txtfig

from .pagan import layers, build_layer
from .pagan.BigGAN import G_arch, D_arch
from .pagan.ops import \
  (MixedLayer, UpSample, DownSample, Identity,
   MixedLayerSharedWeights, MixedLayerCondSharedWeights,
   SinglePathLayer)

from .build import GENERATOR_REGISTRY

__all__ = ['PathAwareResNetGen']


@GENERATOR_REGISTRY.register()
class PathAwareResNetGen(nn.Module):
  def __init__(self, cfg, **kwargs):
    super(PathAwareResNetGen, self).__init__()

    self.img_size                          = get_attr_kwargs(cfg.model.generator, 'img_size', kwargs=kwargs)
    self.ch                                = cfg.model.generator.ch
    self.attention                         = cfg.model.generator.attention
    self.use_sn                            = cfg.model.generator.use_sn
    self.dim_z                             = cfg.model.generator.dim_z
    self.bottom_width                      = cfg.model.generator.bottom_width
    self.track_running_stats               = cfg.model.generator.track_running_stats
    self.share_conv_weights                = cfg.model.generator.share_conv_weights
    self.single_path_layer                 = cfg.model.generator.single_path_layer
    self.share_bias                        = cfg.model.generator.share_bias
    self.output_type                       = cfg.model.generator.output_type
    self.bn_type                           = cfg.model.generator.bn_type
    self.ops                               = cfg.model.generator.ops
    self.init                              = cfg.model.generator.init
    self.use_sync_bn                       = getattr(cfg.model.generator, 'use_sync_bn', False)

    self.arch = G_arch(self.ch, self.attention)[self.img_size]

    if self.use_sn:
      self.which_linear = functools.partial(
        layers.SNLinear, num_svs=1, num_itrs=1, eps=1e-6)
      self.which_conv = functools.partial(
        layers.SNConv2d, kernel_size=3, padding=1,
        num_svs=1, num_itrs=1, eps=1e-6)
      self.which_conv_1x1 = functools.partial(
        layers.SNConv2d, kernel_size=1, padding=0,
        num_svs=1, num_itrs=1, eps=1e-6)
    else:
      self.which_linear = nn.Linear
      self.which_conv = functools.partial(
        nn.Conv2d, kernel_size=3, padding=1)
      self.which_conv_1x1 = functools.partial(
        nn.Conv2d, kernel_size=1, padding=0)

    # First linear layer
    self.linear = self.which_linear(
      self.dim_z, self.arch['in_channels'][0] * (self.bottom_width ** 2))

    num_conv_in_block = 2
    self.num_layers = len(self.arch['in_channels']) * num_conv_in_block
    self.upsample_layer_idx = \
      [num_conv_in_block * l
        for l in range(0, self.num_layers//num_conv_in_block)]

    self.layers = nn.ModuleList([])
    self.layers_para_list = []
    self.skip_layers = nn.ModuleList([])
    bn_type = getattr(self, 'bn_type', 'bn').lower()

    for layer_id in range(self.num_layers):
      block_in = self.arch['in_channels'][layer_id//num_conv_in_block]
      block_out = self.arch['out_channels'][layer_id//num_conv_in_block]
      if layer_id % num_conv_in_block == 0:
        in_channels = block_in
        out_channels = block_out
      else:
        in_channels = block_out
        out_channels = block_out
      upsample = (UpSample()
                  if (self.arch['upsample'][layer_id//num_conv_in_block] and
                      layer_id in self.upsample_layer_idx)
                  else None)
      if getattr(self, 'share_conv_weights', False):
        if getattr(self, 'single_path_layer', False):
          layer = SinglePathLayer(
            layer_id=layer_id, in_planes=in_channels, out_planes=out_channels,
            ops=self.ops, track_running_stats=self.track_running_stats,
            scalesample=upsample, bn_type=bn_type, share_bias=self.share_bias)
        else:
          layer = MixedLayerSharedWeights(
            layer_id=layer_id, in_planes=in_channels, out_planes=out_channels,
            ops=self.ops, track_running_stats=self.track_running_stats,
            scalesample=upsample, bn_type=bn_type)
      else:
        layer = MixedLayer(
          layer_id, in_channels, out_channels,
          ops=self.ops, track_running_stats=self.track_running_stats,
          scalesample=upsample, bn_type=bn_type)
      self.layers.append(layer)
      self.layers_para_list.append(layer.num_para_list)
      if layer_id in self.upsample_layer_idx:
        skip_layers = []
        if self.arch['upsample'][layer_id//num_conv_in_block]:
          skip_layers.append(('upsample_%d'%layer_id, UpSample()))
        # if in_channels != out_channels:
          conv_1x1 = self.which_conv_1x1(in_channels, out_channels,
                                         kernel_size=1, padding=0)
          skip_layers.append(('upsample_%d_conv_1x1'%layer_id, conv_1x1))
        else:
          identity = Identity()
          skip_layers.append(('skip_%d_identity' % layer_id, identity))
        skip_layers = nn.Sequential(OrderedDict(skip_layers))
        self.skip_layers.append(skip_layers)

    self.layers_para_matrix = np.array(self.layers_para_list).T
    # output layer
    self.output_type = getattr(self, 'output_type', 'snconv')
    self.output_sample_arc = False
    if self.output_type == 'snconv':
      if self.use_sync_bn:
        from detectron2.layers import NaiveSyncBatchNorm
        self.output_layer = nn.Sequential(
          NaiveSyncBatchNorm(self.arch['out_channels'][-1], affine=True, track_running_stats=self.track_running_stats),
          nn.ReLU(),
          self.which_conv(self.arch['out_channels'][-1], 3))
      else:
        self.output_layer = nn.Sequential(
          nn.BatchNorm2d(self.arch['out_channels'][-1], affine=True, track_running_stats=self.track_running_stats),
          nn.ReLU(),
          self.which_conv(self.arch['out_channels'][-1], 3))
    elif self.output_type == 'MixedLayer':
      self.output_sample_arc = True
      if getattr(self, 'share_conv_weights', False):
        self.output_conv = MixedLayerSharedWeights(
          layer_id + 1, self.arch['out_channels'][-1], 3, ops=self.ops,
          track_running_stats=self.track_running_stats, scalesample=None,
          bn_type=bn_type)
      else:
        self.output_conv = MixedLayer(
          layer_id + 1, self.arch['out_channels'][-1], 3, ops=self.ops,
          track_running_stats=self.track_running_stats, scalesample=None,
          bn_type=bn_type)
    else:
      assert 0

    self.init_weights()
    pass

  def forward(self, x, sample_arcs, *args, **kwargs):
    """

    :param x:
    :param sample_arcs: (b, num_layers)
    :return:
    """

    x = self.linear(x)
    x = x.view(x.size(0), -1, self.bottom_width, self.bottom_width)

    upsample_layer = 0
    for layer_id in range(self.num_layers):
      if layer_id == 0:
        prev_layer = x
      sample_arc = sample_arcs[:, layer_id]
      x = self.layers[layer_id](x, sample_arc)

      if layer_id - 1 in self.upsample_layer_idx:
        x_up = self.skip_layers[upsample_layer](prev_layer)
        upsample_layer += 1
        x = x + x_up
        prev_layer = x

    if self.output_type == 'snconv':
      x = self.output_layer(x)
    elif self.output_type == 'MixedLayer':
      sample_arc = sample_arcs[:, -1]
      x = self.output_conv(x, sample_arc)
    else:
      assert 0
    x = torch.tanh(x)
    return x

  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
              or isinstance(module, nn.Linear)
              or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
      if (isinstance(module, MixedLayerSharedWeights)):
        if self.init == 'ortho':
          for k, w in module.conv_weights.items():
            init.orthogonal_(w)
        else:
          assert 0
      if (isinstance(module, SinglePathLayer)):
        if self.init == 'ortho':
          init.orthogonal_(module.conv_weights_space)
        else:
          assert 0
      pass


@GENERATOR_REGISTRY.register()
class PathAwareResNetGenCBN(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(PathAwareResNetGenCBN, self).__init__()

    cfg = self.update_cfg(cfg)

    self.n_classes                 = get_attr_kwargs(cfg, 'n_classes', **kwargs)
    self.img_size                  = get_attr_kwargs(cfg, 'img_size', **kwargs)
    self.ch                        = get_attr_kwargs(cfg, 'ch', default=8, **kwargs)
    self.attention                 = get_attr_kwargs(cfg, 'attention', default='0', **kwargs)
    self.dim_z                     = get_attr_kwargs(cfg, 'dim_z', default=256, **kwargs)
    self.hier                      = get_attr_kwargs(cfg, 'hier', default=True, **kwargs)
    self.embedding_dim             = get_attr_kwargs(cfg, 'embedding_dim', default=128, **kwargs)
    self.bottom_width              = get_attr_kwargs(cfg, 'bottom_width', **kwargs)
    self.init                      = get_attr_kwargs(cfg, 'init', default='ortho', **kwargs)
    self.cfg_first_fc              = cfg.cfg_first_fc
    self.cfg_bn                    = cfg.cfg_bn
    self.cfg_act                   = cfg.cfg_act
    self.cfg_mix_layer             = cfg.cfg_mix_layer
    self.cfg_upsample              = cfg.cfg_upsample
    self.cfg_conv_1x1              = cfg.cfg_conv_1x1
    self.cfg_out_bn                = cfg.cfg_out_bn
    self.cfg_out_conv              = cfg.cfg_out_conv
    self.cfg_ops                   = cfg.cfg_ops

    self.cfg = cfg
    self.arch = G_arch(self.ch, self.attention)[self.img_size]
    self.num_branches = len(self.cfg_ops)
    self.device = torch.device(f'cuda:{comm.get_rank()}')

    if self.hier:
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      self.dim_z_input = self.z_chunk_size
      self.cbn_in_features = self.embedding_dim + self.z_chunk_size
    else:
      self.num_slots = 1
      self.z_chunk_size = 0
      self.cbn_in_features = self.embedding_dim

    # Prepare class embedding
    self.class_embedding = nn.Embedding(self.n_classes, self.embedding_dim)

    # First linear layer
    self.linear = build_d2layer(cfg.cfg_first_fc, in_features=self.dim_z_input,
                                out_features=self.arch['in_channels'][0] * (self.bottom_width ** 2))

    self.num_conv_in_block = 2
    self.num_layers = len(self.arch['in_channels']) * self.num_conv_in_block
    self.upsample_layer_idx = [self.num_conv_in_block * l for l in range(0, self.num_layers//self.num_conv_in_block)]

    self.bns = nn.ModuleList()
    self.acts = nn.ModuleList()
    self.mix_layers = nn.ModuleList([])
    self.upsamples = nn.ModuleList()

    self.skip_layers = nn.ModuleList([])

    for layer_id in range(self.num_layers):
      block_in = self.arch['in_channels'][layer_id//self.num_conv_in_block]
      block_out = self.arch['out_channels'][layer_id//self.num_conv_in_block]
      if layer_id % self.num_conv_in_block == 0:
        in_channels = block_in
        out_channels = block_out
      else:
        in_channels = block_out
        out_channels = block_out

      # bn relu mix_layer upsample
      bn = build_d2layer(self.cfg_bn, in_features=self.cbn_in_features, out_features=in_channels)
      self.bns.append(bn)

      act = build_d2layer(self.cfg_act)
      self.acts.append(act)

      mix_layer = build_d2layer(self.cfg_mix_layer, in_channels=in_channels, out_channels=out_channels,
                                cfg_ops=self.cfg_ops)
      self.mix_layers.append(mix_layer)

      if layer_id in self.upsample_layer_idx:
        upsample = build_d2layer(self.cfg_upsample)
      else:
        upsample = build_d2layer(EasyDict(name="Identity"))
      self.upsamples.append(upsample)

      # skip branch
      if layer_id in self.upsample_layer_idx:
        skip_layers = []

        skip_conv_1x1 = build_d2layer(self.cfg_conv_1x1, in_channels=in_channels, out_channels=out_channels)
        skip_layers.append((f'skip_conv_1x1_{layer_id}', skip_conv_1x1))

        skip_upsample = build_d2layer(self.cfg_upsample)
        skip_layers.append((f'skip_upsample_{layer_id}', skip_upsample))

        skip_layers = nn.Sequential(OrderedDict(skip_layers))
        self.skip_layers.append(skip_layers)

    out_bn = build_d2layer(self.cfg_out_bn, num_features=self.arch['out_channels'][-1])
    out_act = build_d2layer(self.cfg_act)
    out_conv = build_d2layer(self.cfg_out_conv, in_channels=self.arch['out_channels'][-1])
    self.output_layer = nn.Sequential(OrderedDict([
      ('out_bn', out_bn),
      ('out_act', out_act),
      ('out_conv', out_conv)
    ]))

    self.init_weights()
    pass

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "PathAwareResNetGenCBN"
      n_classes: "kwargs['n_classes']"
      img_size: "kwargs['img_size']"
      ch: 64
      dim_z: 256
      bottom_width: 4
      init: 'ortho'
      cfg_first_fc:
        name: "Linear"
        in_features: "kwargs['in_features']"
        out_features: "kwargs['out_features']"
      cfg_bn:
        name: "CondBatchNorm2d"
        in_features: "kwargs['in_features']"
        out_features: "kwargs['out_features']"
        cfg_fc:
          name: "Linear"
          in_features: "kwargs['in_features']"
          out_features: "kwargs['out_features']"
      cfg_act:
        name: "ReLU"
      cfg_mix_layer:
        name: "MixedLayerCond"
        in_channels: "kwargs['in_channels']"
        out_channels: "kwargs['out_channels']"
        cfg_ops: "kwargs['cfg_ops']"
      cfg_upsample:
        name: "UpSample"
        mode: "bilinear"
      cfg_conv_1x1:
        name: "Conv2d"
        in_channels: "kwargs['in_channels']"
        out_channels: "kwargs['out_channels']"
        kernel_size: 1
      cfg_ops:
        Identity:
          name: "Identity"
        DepthwiseSeparableConv2d_3x3:
          name: "DepthwiseSeparableConv2d"
          in_channels: "kwargs['in_channels']"
          out_channels: "kwargs['out_channels']"
          kernel_size: 3
          padding: 1
        DepthwiseSeparableConv2d_5x5:
          name: "DepthwiseSeparableConv2d"
          in_channels: "kwargs['in_channels']"
          out_channels: "kwargs['out_channels']"
          kernel_size: 5
          padding: 2
        DepthwiseSeparableConv2d_7x7:
          name: "DepthwiseSeparableConv2d"
          in_channels: "kwargs['in_channels']"
          out_channels: "kwargs['out_channels']"
          kernel_size: 7
          padding: 3
        Conv2d_3x3:
          name: "Conv2d"
          in_channels: "kwargs['in_channels']"
          out_channels: "kwargs['out_channels']"
          kernel_size: 3
          padding: 1
        Conv2d_5x5:
          name: "Conv2d"
          in_channels: "kwargs['in_channels']"
          out_channels: "kwargs['out_channels']"
          kernel_size: 5
          padding: 2
        Conv2d_7x7:
          name: "Conv2d"
          in_channels: "kwargs['in_channels']"
          out_channels: "kwargs['out_channels']"
          kernel_size: 7
          padding: 3
      cfg_out_bn:
        name: "BatchNorm2d"
        num_features: "kwargs['num_features']"
      cfg_out_conv:
        name: "Conv2d"
        in_channels: "kwargs['in_channels']"
        out_channels: 3
        kernel_size: 3
        padding: 1
    """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg

  def test_case(self):
    bs = num_ops = len(self.cfg.cfg_ops)
    z = torch.randn(bs, self.cfg.dim_z).cuda()
    y = torch.arange(num_ops).cuda()
    sample_arcs = torch.arange(bs).view(-1, 1).repeat(1, self.num_layers).cuda()
    out = self(z, y, sample_arcs)
    return out

  def forward(self, z, y, batched_arcs):
    """

    :param sample_arcs: (b, num_layers)
    :return:
    """
    z = z.to(self.device)
    y = y.to(self.device)
    batched_arcs = batched_arcs.to(self.device)

    y = self.class_embedding(y)
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.arch['in_channels'])

    x = self.linear(z)
    x = x.view(x.size(0), -1, self.bottom_width, self.bottom_width)

    upsample_layer = 0
    for layer_id in range(self.num_layers):
      if layer_id == 0:
        prev_layer = x
      sample_arc = batched_arcs[:, layer_id]

      x = self.bns[layer_id](x, ys[layer_id // self.num_conv_in_block])
      x = self.acts[layer_id](x)
      x = self.mix_layers[layer_id](x=x, y=ys[layer_id // self.num_conv_in_block], sample_arc=sample_arc)
      x = self.upsamples[layer_id](x)

      if layer_id - 1 in self.upsample_layer_idx:
        x_up = self.skip_layers[upsample_layer](prev_layer)
        upsample_layer += 1
        x = x + x_up
        prev_layer = x

    x = self.output_layer(x)

    x = torch.tanh(x)
    return x

  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
              or isinstance(module, nn.Linear)
              or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')

      if (isinstance(module, (MixedLayerCondSharedWeights,
                              MixedLayerSharedWeights))):
        if self.init == 'ortho':
          for k, w in module.conv_weights.items():
            init.orthogonal_(w)
        else:
          assert 0




@D2MODEL_REGISTRY.register()
class PAGANFairController(nn.Module):
  '''
  '''
  def __init__(self, cfg, **kwargs):
    super(PAGANFairController, self).__init__()

    cfg = self.update_cfg(cfg)

    self.num_layers              = get_attr_kwargs(cfg, 'num_layers', **kwargs)
    self.num_branches            = get_attr_kwargs(cfg, 'num_branches', **kwargs)

    pass

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "PAGANRLController"
      n_classes: "kwargs['n_classes']"
      num_layers: "kwargs['num_layers']"
      num_branches: "kwargs['num_branches']"
    """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg

  def test_case(self):
    bs = 4
    out = self(bs)
    return out

  def forward(self, bs):
    """

    :param batch_imgs:
    :return: (bs x num_branches, num_layers)
    """
    arcs = []
    for l in range(self.num_layers):
      layer_arcs = torch.randperm(self.num_branches).view(-1, 1)
      arcs.append(layer_arcs)
    arcs = torch.cat(arcs, dim=1)
    batched_arcs = arcs.repeat(bs, 1)
    return batched_arcs

  def fairnas_repeat_tensor(self, sample):
    repeat_arg = [1] * (sample.dim() + 1)
    repeat_arg[1] = self.num_branches
    sample = sample.unsqueeze(1).repeat(repeat_arg)
    sample = sample.view(-1, *sample.shape[2:])
    return sample

@D2MODEL_REGISTRY.register()
class PAGANRLControllerLSTM(nn.Module):
  '''
  https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py
  '''

  def __init__(self, cfg, **kwargs):
    super(PAGANRLControllerLSTM, self).__init__()

    cfg = self.update_cfg(cfg)

    self.FID_IS                  = kwargs['FID_IS']
    self.myargs                  = kwargs['myargs']
    self.num_layers              = get_attr_kwargs(cfg, 'num_layers', **kwargs)
    self.num_branches            = get_attr_kwargs(cfg, 'num_branches', **kwargs)
    self.search_whole_channels   = get_attr_kwargs(cfg, 'search_whole_channels', default=True, **kwargs)
    self.lstm_size               = get_attr_kwargs(cfg, 'lstm_size', default=64, **kwargs)
    self.lstm_num_layers         = get_attr_kwargs(cfg, 'lstm_num_layers', default=2, **kwargs)
    self.tanh_constant           = get_attr_kwargs(cfg, 'tanh_constant', default=1.5, **kwargs)
    self.temperature             = get_attr_kwargs(cfg, 'temperature', default=-1, **kwargs)
    self.num_aggregate           = get_attr_kwargs(cfg, 'num_aggregate', **kwargs)
    self.entropy_weight          = get_attr_kwargs(cfg, 'entropy_weight', **kwargs)
    self.bl_dec                  = get_attr_kwargs(cfg, 'bl_dec', **kwargs)
    self.child_grad_bound        = get_attr_kwargs(cfg, 'child_grad_bound', **kwargs)
    self.log_every_iter          = get_attr_kwargs(cfg, 'log_every_iter', default=50, **kwargs)
    self.cfg_ops                 = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)

    self.device = torch.device(f'cuda:{comm.get_rank()}')
    self.baseline = None
    self.logger = logging.getLogger('tl')
    self._create_params()
    self._reset_params()

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "PAGANRLController"
      num_layers: "kwargs['num_layers']"
      num_branches: "kwargs['num_branches']"
      lstm_size: 64
      lstm_num_layers: 2
      temperature: -1
      num_aggregate: 10
      entropy_weight: 0.0001
      bl_dec: 0.99
      child_grad_bound: 5.0
    """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg

  def test_case(self):
    out = self()
    return out


  def _create_params(self):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L83
    '''
    self.w_lstm = nn.LSTM(input_size=self.lstm_size,
                          hidden_size=self.lstm_size,
                          num_layers=self.lstm_num_layers)

    if self.search_whole_channels:
      self.w_emb = nn.Embedding(self.num_layers, self.lstm_size)
      self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=True)
    else:
      assert False, "Not implemented error: search_whole_channels = False"

    # self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
    # self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
    # self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)

  def _reset_params(self):
    for m in self.modules():
      if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)

    nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
    nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

  def forward(self, bs, determine_sample=False):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L126
    '''
    h0 = None  # setting h0 to None will initialize LSTM state with 0s
    arc_seq = []
    entropys = []
    log_probs = []

    self.op_dist = []
    for layer_id in range(self.num_layers):
      if self.search_whole_channels:
        inputs = self.w_emb.weight[[layer_id]]
        inputs = inputs.unsqueeze(dim=0)
        output, hn = self.w_lstm(inputs, h0)
        output = output.squeeze(dim=0)
        h0 = hn

        logit = self.w_soft(output)
        if self.temperature > 0:
          logit /= self.temperature
        if self.tanh_constant is not None:
          logit = self.tanh_constant * torch.tanh(logit)

        branch_id_dist = Categorical(logits=logit)
        self.op_dist.append(branch_id_dist)

        if determine_sample:
          branch_id = logit.argmax(dim=1)
        else:
          branch_id = branch_id_dist.sample()

        arc_seq.append(branch_id)

        log_prob = branch_id_dist.log_prob(branch_id)
        log_probs.append(log_prob.view(-1))
        entropy = branch_id_dist.entropy()
        entropys.append(entropy.view(-1))

      else:
        # https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L171
        assert False, "Not implemented error: search_whole_channels = False"

      # inputs = self.w_emb(branch_id)

    self.sample_arc = torch.stack(arc_seq, dim=1)

    self.sample_entropy = torch.stack(entropys, dim=1)

    self.sample_log_prob = torch.stack(log_probs, dim=1)
    self.sample_prob = self.sample_log_prob.exp()

    batched_arcs = self.sample_arc.repeat((bs, 1))
    return batched_arcs

  def get_sampled_arc(self):
    sampled_arc = self.sample_arc.detach()
    return sampled_arc

  def train_controller(self, G, z, y, controller, controller_optim, iteration, pbar):
    """

    :param controller: for ddp training
    :return:
    """
    if comm.is_main_process() and iteration % 1000 == 0:
      pbar.set_postfix_str("PAGANRLCondControllerLSTM")

    meter_dict = {}

    G.eval()
    controller.train()

    controller.zero_grad()

    z_samples = z.sample()
    batched_arcs = controller(bs=len(z_samples))
    sample_entropy = get_ddp_attr(controller, 'sample_entropy')
    sample_log_prob = get_ddp_attr(controller, 'sample_log_prob')

    pool_list, logits_list = [], []
    for i in range(self.num_aggregate):
      z_samples = z.sample().to(self.device)
      y_samples = y.sample().to(self.device)
      with torch.set_grad_enabled(False):
        x = G(z=z_samples, y=y_samples, batched_arcs=batched_arcs)

      pool, logits = self.FID_IS.get_pool_and_logits(x)

      pool_list.append(pool)
      logits_list.append(logits)

    pool = np.concatenate(pool_list, 0)
    logits = np.concatenate(logits_list, 0)

    reward_g, _ = self.FID_IS.calculate_IS(logits)
    meter_dict['reward_g'] = reward_g

    # detach to make sure that gradients aren't backpropped through the reward
    reward = torch.tensor(reward_g).cuda()
    sample_entropy_mean = sample_entropy.mean()
    meter_dict['sample_entropy'] = sample_entropy_mean.item()
    reward += self.entropy_weight * sample_entropy_mean

    if self.baseline is None:
      baseline = torch.tensor(reward_g)
    else:
      baseline = self.baseline - (1 - self.bl_dec) * (self.baseline - reward)
      # detach to make sure that gradients are not backpropped through the baseline
      baseline = baseline.detach()

    sample_log_prob_mean = sample_log_prob.mean()
    meter_dict['sample_log_prob'] = sample_log_prob_mean.item()
    loss = -1 * sample_log_prob_mean * (reward - baseline)

    meter_dict['reward'] = reward.item()
    meter_dict['baseline'] = baseline.item()
    meter_dict['loss'] = loss.item()

    loss.backward(retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), self.child_grad_bound)
    meter_dict['grad_norm'] = grad_norm

    controller_optim.step()

    baseline_list = comm.all_gather(baseline)
    baseline_mean = sum(map(lambda v: v.item(), baseline_list)) / len(baseline_list)
    baseline.fill_(baseline_mean)
    self.baseline = baseline

    if iteration % self.log_every_iter == 0:
      if comm.is_main_process():
        tqdm.tqdm.write("*******In PAGANRLCondControllerLSTM*******", file=self.myargs.stdout)
      default_dicts = collections.defaultdict(dict)
      for meter_k, meter in meter_dict.items():
        if meter_k in ['reward', 'baseline']:
          default_dicts['reward_baseline'][meter_k] = meter
        else:
          default_dicts[meter_k][meter_k] = meter
      summary_defaultdict2txtfig(default_dict=default_dicts,
                                 prefix='train_controller',
                                 step=iteration,
                                 textlogger=self.myargs.textlogger)
    comm.synchronize()
    return

  def print_distribution(self, iteration):
    # remove formats
    org_formatters = []
    for handler in self.logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("%(message)s"))

    default_dict = collections.defaultdict(dict)
    self.logger.info("####### distribution #######")
    searched_arc = []
    for layer_id, op_dist in enumerate(self.op_dist):
      prob = op_dist.probs
      max_op_id = prob.argmax().item()
      searched_arc.append(max_op_id)
      for op_id, op_name in enumerate(self.cfg_ops.keys()):
        op_prob = prob[0][op_id]
        default_dict[f'L{layer_id}'][get_prefix_abb(op_name)] = op_prob.item()

      self.logger.info(prob)
    searched_arc = np.array(searched_arc)
    self.logger.info('\nsearched arcs: \n%s' % searched_arc)
    self.myargs.textlogger.logstr(iteration,
                                  searched_arc='\n' + np.array2string(searched_arc, threshold=np.inf))

    summary_defaultdict2txtfig(default_dict=default_dict, prefix='', step=iteration,
                               textlogger=self.myargs.textlogger)
    self.logger.info("#####################")

    # restore formats
    for handler, formatter in zip(self.logger.handlers, org_formatters):
      handler.setFormatter(formatter)


@D2MODEL_REGISTRY.register()
class PAGANRLCondControllerLSTMFair(PAGANRLControllerLSTM):

  def __init__(self, cfg, **kwargs):
    super(PAGANRLCondControllerLSTMFair, self).__init__(cfg=cfg, **kwargs)

  def forward(self, fair_arcs):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L126
    '''
    fair_arcs = fair_arcs.to(self.device)
    h0 = None  # setting h0 to None will initialize LSTM state with 0s

    fair_entropys = []
    fair_log_probs = []

    self.op_dist = []
    for layer_id in range(self.num_layers):
      inputs = self.w_emb.weight[[layer_id]]
      inputs = inputs.unsqueeze(dim=0)
      output, hn = self.w_lstm(inputs, h0)
      output = output.squeeze(dim=0)
      h0 = hn

      logit = self.w_soft(output)
      if self.temperature > 0:
        logit /= self.temperature
      if self.tanh_constant is not None:
        logit = self.tanh_constant * torch.tanh(logit)

      op_dist = Categorical(logits=logit)
      self.op_dist.append(op_dist)

      log_prob = op_dist.log_prob(fair_arcs[:, layer_id])
      fair_log_probs.append(log_prob.view(-1, 1))
      entropy = op_dist.entropy()
      fair_entropys.append(entropy.view(-1, 1))

      # inputs = self.w_emb(branch_id)

    self.sample_entropy = torch.cat(fair_entropys, dim=1)
    self.sample_log_prob = torch.cat(fair_log_probs, dim=1)

    return

  def get_fair_path(self, bs):
    """

    :param batch_imgs:
    :return: (bs x num_branches, num_layers)
    """
    arcs = []
    for l in range(self.num_layers):
      layer_arcs = torch.randperm(self.num_branches).view(-1, 1)
      arcs.append(layer_arcs)
    arcs = torch.cat(arcs, dim=1)
    batched_arcs = arcs.repeat(bs, 1)

    fair_arcs = arcs
    return batched_arcs, fair_arcs

  def get_sampled_arc(self):
    _, sampled_arc = self.get_fair_path(bs=1)
    return sampled_arc

  def train_controller(self, G, z, y, controller, controller_optim, iteration, pbar):
    """

    :param controller: for ddp training
    :return:
    """
    if comm.is_main_process() and iteration % 1000 == 0:
      pbar.set_postfix_str("PAGANRLCondControllerLSTMFair")
    meter_dict = {}

    G.eval()
    controller.train()

    controller.zero_grad()

    z_samples = z.sample()
    bs = len(z_samples) // self.num_branches

    batched_arcs, fair_arcs = self.get_fair_path(bs=bs)

    controller(fair_arcs=fair_arcs)
    sample_entropy = get_ddp_attr(controller, 'sample_entropy')
    sample_log_prob = get_ddp_attr(controller, 'sample_log_prob')

    pool_list, logits_list = [], []
    for i in range(self.num_aggregate):
      z_samples = z.sample().to(self.device)
      y_samples = y.sample().to(self.device)
      with torch.set_grad_enabled(False):
        x = G(z=z_samples, y=y_samples, batched_arcs=batched_arcs)

      pool, logits = self.FID_IS.get_pool_and_logits(x)

      # pool_list.append(pool)
      logits_list.append(logits)

    # pool = np.concatenate(pool_list, 0)
    logits = np.concatenate(logits_list, 0)

    fair_reward_g = []
    num_arcs = len(fair_arcs)
    for i in range(num_arcs):
      logits_i = logits[i::num_arcs]
      reward_g, _ = self.FID_IS.calculate_IS(logits_i)
      fair_reward_g.append(reward_g)

    meter_dict['fair_reward_g_mean'] = sum(fair_reward_g) / num_arcs

    # detach to make sure that gradients aren't backpropped through the reward
    fair_reward = torch.tensor(fair_reward_g).cuda().view(-1, 1)

    sample_entropy_mean = sample_entropy.mean()
    meter_dict['sample_entropy'] = sample_entropy_mean.item()
    fair_reward += self.entropy_weight * sample_entropy_mean.view(1, 1)

    if self.baseline is None:
      baseline = torch.tensor(fair_reward).mean().view(1, 1).to(self.device)
    else:
      baseline = self.baseline - (1 - self.bl_dec) * (self.baseline - fair_reward)
      # detach to make sure that gradients are not backpropped through the baseline
      baseline = baseline.mean().view(1, 1).detach()

    sample_log_prob_mean = sample_log_prob.mean(dim=1, keepdim=True)
    meter_dict['sample_log_prob'] = sample_log_prob_mean.mean().item()
    loss = -1 * sample_log_prob_mean * (fair_reward - baseline)
    loss = loss.mean()

    meter_dict['reward'] = fair_reward.mean().item()
    meter_dict['baseline'] = baseline.item()
    meter_dict['loss'] = loss.item()

    loss.backward(retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), self.child_grad_bound)
    meter_dict['grad_norm'] = grad_norm

    controller_optim.step()

    baseline_list = comm.all_gather(baseline)
    baseline_mean = sum(map(lambda v: v.item(), baseline_list)) / len(baseline_list)
    baseline.fill_(baseline_mean)
    self.baseline = baseline

    if iteration % self.log_every_iter == 0:
      if comm.is_main_process():
        tqdm.tqdm.write("*******In PAGANRLCondControllerLSTMFair*******", file=self.myargs.stdout)
      default_dicts = collections.defaultdict(dict)
      for meter_k, meter in meter_dict.items():
        if meter_k in ['reward', 'baseline']:
          default_dicts['reward_baseline'][meter_k] = meter
        else:
          default_dicts[meter_k][meter_k] = meter
      summary_defaultdict2txtfig(default_dict=default_dicts,
                                 prefix='train_controller',
                                 step=iteration,
                                 textlogger=self.myargs.textlogger)
    comm.synchronize()
    return


@D2MODEL_REGISTRY.register()
class PAGANRLControllerAlphaFair(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(PAGANRLControllerAlphaFair, self).__init__()

    self.FID_IS                  = kwargs['FID_IS']
    self.myargs                  = kwargs['myargs']
    self.num_layers              = get_attr_kwargs(cfg, 'num_layers', **kwargs)
    self.num_branches            = get_attr_kwargs(cfg, 'num_branches', **kwargs)
    self.tanh_constant           = get_attr_kwargs(cfg, 'tanh_constant', default=1.5, **kwargs)
    self.temperature             = get_attr_kwargs(cfg, 'temperature', default=-1, **kwargs)
    self.num_aggregate           = get_attr_kwargs(cfg, 'num_aggregate', **kwargs)
    self.entropy_weight          = get_attr_kwargs(cfg, 'entropy_weight', default=0.0001, **kwargs)
    self.bl_dec                  = get_attr_kwargs(cfg, 'bl_dec', **kwargs)
    self.child_grad_bound        = get_attr_kwargs(cfg, 'child_grad_bound', **kwargs)
    self.log_every_iter          = get_attr_kwargs(cfg, 'log_every_iter', default=50, **kwargs)
    self.cfg_ops                 = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)

    self.device = torch.device(f'cuda:{comm.get_rank()}')
    self.baseline = None
    self.logger = logging.getLogger('tl')

    self.alpha = nn.ParameterList()
    for i in range(self.num_layers):
      self.alpha.append(nn.Parameter(1e-4 * torch.randn(1, self.num_branches)))

  def forward(self, fair_arcs):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L126
    '''
    fair_arcs = fair_arcs.to(self.device)
    h0 = None  # setting h0 to None will initialize LSTM state with 0s

    fair_entropys = []
    fair_log_probs = []

    self.op_dist = []
    for layer_id in range(self.num_layers):
      logit = self.alpha[layer_id]
      # if self.temperature > 0:
      #   logit /= self.temperature
      # if self.tanh_constant is not None:
      #   logit = self.tanh_constant * torch.tanh(logit)

      op_dist = Categorical(logits=logit)
      self.op_dist.append(op_dist)

      log_prob = op_dist.log_prob(fair_arcs[:, layer_id])
      fair_log_probs.append(log_prob.view(-1, 1))
      entropy = op_dist.entropy()
      fair_entropys.append(entropy.view(-1, 1))

      # inputs = self.w_emb(branch_id)

    self.sample_entropy = torch.cat(fair_entropys, dim=1)
    self.sample_log_prob = torch.cat(fair_log_probs, dim=1)

    return

  def get_fair_path(self, bs):
    """

    :param batch_imgs:
    :return: (bs x num_branches, num_layers)
    """
    arcs = []
    for l in range(self.num_layers):
      layer_arcs = torch.randperm(self.num_branches).view(-1, 1)
      arcs.append(layer_arcs)
    arcs = torch.cat(arcs, dim=1)
    batched_arcs = arcs.repeat(bs, 1)

    fair_arcs = arcs
    return batched_arcs, fair_arcs

  def get_sampled_arc(self):
    sampled_arc = []
    for layer_id, op_dist in enumerate(self.op_dist):
      sampled_op = op_dist.sample().view(-1, 1)
      sampled_arc.append(sampled_op)

    sampled_arc = torch.cat(sampled_arc, dim=1)
    return sampled_arc

  def train_controller(self, G, z, y, controller, controller_optim, iteration, pbar):
    """

    :param controller: for ddp training
    :return:
    """
    if comm.is_main_process() and iteration % 1000 == 0:
      pbar.set_postfix_str("PAGANRLControllerAlphaFair")
    meter_dict = {}

    G.eval()
    controller.train()

    controller.zero_grad()

    z_samples = z.sample()
    bs = len(z_samples) // self.num_branches

    batched_arcs, fair_arcs = self.get_fair_path(bs=bs)

    controller(fair_arcs=fair_arcs)
    sample_entropy = get_ddp_attr(controller, 'sample_entropy')
    sample_log_prob = get_ddp_attr(controller, 'sample_log_prob')

    pool_list, logits_list = [], []
    for i in range(self.num_aggregate):
      z_samples = z.sample().to(self.device)
      y_samples = y.sample().to(self.device)
      with torch.set_grad_enabled(False):
        x = G(z=z_samples, y=y_samples, batched_arcs=batched_arcs)

      pool, logits = self.FID_IS.get_pool_and_logits(x)

      # pool_list.append(pool)
      logits_list.append(logits)

    # pool = np.concatenate(pool_list, 0)
    logits = np.concatenate(logits_list, 0)

    fair_reward_g = []
    num_arcs = len(fair_arcs)
    for i in range(num_arcs):
      logits_i = logits[i::num_arcs]
      reward_g, _ = self.FID_IS.calculate_IS(logits_i)
      fair_reward_g.append(reward_g)

    meter_dict['fair_reward_g_mean'] = sum(fair_reward_g) / num_arcs

    # detach to make sure that gradients aren't backpropped through the reward
    fair_reward = torch.tensor(fair_reward_g).cuda().view(-1, 1)

    sample_entropy_mean = sample_entropy.mean()
    meter_dict['sample_entropy'] = sample_entropy_mean.item()
    fair_reward += self.entropy_weight * sample_entropy_mean.view(1, 1)

    if self.baseline is None:
      baseline = torch.tensor(fair_reward).mean().view(1, 1).to(self.device)
    else:
      baseline = self.baseline - (1 - self.bl_dec) * (self.baseline - fair_reward)
      # detach to make sure that gradients are not backpropped through the baseline
      baseline = baseline.mean().view(1, 1).detach()

    sample_log_prob_mean = sample_log_prob.mean(dim=1, keepdim=True)
    meter_dict['sample_log_prob'] = sample_log_prob_mean.mean().item()
    loss = -1 * sample_log_prob_mean * (fair_reward - baseline)
    loss = loss.mean()

    meter_dict['reward'] = fair_reward.mean().item()
    meter_dict['baseline'] = baseline.item()
    meter_dict['loss'] = loss.item()

    loss.backward(retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), self.child_grad_bound)
    meter_dict['grad_norm'] = grad_norm

    controller_optim.step()

    baseline_list = comm.all_gather(baseline)
    baseline_mean = sum(map(lambda v: v.item(), baseline_list)) / len(baseline_list)
    baseline.fill_(baseline_mean)
    self.baseline = baseline

    if iteration % self.log_every_iter == 0:
      default_dicts = collections.defaultdict(dict)
      for meter_k, meter in meter_dict.items():
        if meter_k in ['reward', 'baseline']:
          default_dicts['reward_baseline'][meter_k] = meter
        else:
          default_dicts[meter_k][meter_k] = meter
      summary_defaultdict2txtfig(default_dict=default_dicts,
                                 prefix='train_controller',
                                 step=iteration,
                                 textlogger=self.myargs.textlogger)
    comm.synchronize()
    return

  def print_distribution(self, iteration):
    # remove formats
    org_formatters = []
    for handler in self.logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("%(message)s"))

    default_dict = collections.defaultdict(dict)
    self.logger.info("####### distribution #######")
    searched_arc = []
    for layer_id, op_dist in enumerate(self.op_dist):
      prob = op_dist.probs
      max_op_id = prob.argmax().item()
      searched_arc.append(max_op_id)
      for op_id, op_name in enumerate(self.cfg_ops.keys()):
        op_prob = prob[0][op_id]
        default_dict[f'L{layer_id}'][get_prefix_abb(op_name)] = op_prob.item()

      self.logger.info(prob)
    searched_arc = np.array(searched_arc)
    self.logger.info('\nsearched arcs: \n%s' % searched_arc)
    self.myargs.textlogger.logstr(iteration,
                                  searched_arc='\n' + np.array2string(searched_arc, threshold=np.inf))

    summary_defaultdict2txtfig(default_dict=default_dict, prefix='', step=iteration,
                               textlogger=self.myargs.textlogger)
    self.logger.info("#####################")

    # restore formats
    for handler, formatter in zip(self.logger.handlers, org_formatters):
      handler.setFormatter(formatter)


@D2MODEL_REGISTRY.register()
class PAGANRLNoBaselineControllerAlphaFair(PAGANRLControllerAlphaFair):

  def train_controller(self, G, z, y, controller, controller_optim, iteration, pbar):
    """

    :param controller: for ddp training
    :return:
    """
    if comm.is_main_process() and iteration % 1000 == 0:
      pbar.set_postfix_str("PAGANRLNoBaselineControllerAlphaFair")
    meter_dict = {}

    G.eval()
    controller.train()

    controller.zero_grad()

    z_samples = z.sample()
    bs = len(z_samples) // self.num_branches

    batched_arcs, fair_arcs = self.get_fair_path(bs=bs)

    controller(fair_arcs=fair_arcs)
    sample_entropy = get_ddp_attr(controller, 'sample_entropy')
    sample_log_prob = get_ddp_attr(controller, 'sample_log_prob')

    pool_list, logits_list = [], []
    for i in range(self.num_aggregate):
      z_samples = z.sample().to(self.device)
      y_samples = y.sample().to(self.device)
      with torch.set_grad_enabled(False):
        x = G(z=z_samples, y=y_samples, batched_arcs=batched_arcs)

      pool, logits = self.FID_IS.get_pool_and_logits(x)

      # pool_list.append(pool)
      logits_list.append(logits)

    # pool = np.concatenate(pool_list, 0)
    logits = np.concatenate(logits_list, 0)

    fair_reward_g = []
    num_arcs = len(fair_arcs)
    for i in range(num_arcs):
      logits_i = logits[i::num_arcs]
      reward_g, _ = self.FID_IS.calculate_IS(logits_i)
      fair_reward_g.append(reward_g)

    meter_dict['fair_reward_g_mean'] = sum(fair_reward_g) / num_arcs

    # detach to make sure that gradients aren't backpropped through the reward
    fair_reward = torch.tensor(fair_reward_g).cuda().view(-1, 1)

    sample_entropy_mean = sample_entropy.mean()
    meter_dict['sample_entropy'] = sample_entropy_mean.item()
    fair_reward += self.entropy_weight * sample_entropy_mean.view(1, 1)

    sample_log_prob_mean = sample_log_prob.mean(dim=1, keepdim=True)
    meter_dict['sample_log_prob'] = sample_log_prob_mean.mean().item()
    loss = -1 * sample_log_prob_mean * (fair_reward)
    loss = loss.mean()

    meter_dict['reward'] = fair_reward.mean().item()
    meter_dict['loss'] = loss.item()

    loss.backward(retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), self.child_grad_bound)
    meter_dict['grad_norm'] = grad_norm

    controller_optim.step()

    if iteration % self.log_every_iter == 0:
      default_dicts = collections.defaultdict(dict)
      for meter_k, meter in meter_dict.items():
        if meter_k in ['reward', 'baseline']:
          default_dicts['reward_baseline'][meter_k] = meter
        else:
          default_dicts[meter_k][meter_k] = meter
      summary_defaultdict2txtfig(default_dict=default_dicts,
                                 prefix='train_controller',
                                 step=iteration,
                                 textlogger=self.myargs.textlogger)
    comm.synchronize()
    return


class _FairController(nn.Module):
  def __init__(self):
    super(_FairController, self).__init__()
    self.num_layers = 0
    self.num_branches = 0

  def get_fair_path(self, bs):
    """

    :param batch_imgs:
    :return: (bs x num_branches, num_layers)
    """
    arcs = []
    for l in range(self.num_layers):
      layer_arcs = torch.randperm(self.num_branches).view(-1, 1)
      arcs.append(layer_arcs)
    arcs = torch.cat(arcs, dim=1)
    batched_arcs = arcs.repeat(bs, 1)

    fair_arcs = arcs
    return batched_arcs, fair_arcs

  def fairnas_repeat_tensor(self, sample):
    repeat_arg = [1] * (sample.dim() + 1)
    repeat_arg[1] = self.num_branches
    sample = sample.unsqueeze(1).repeat(repeat_arg)
    sample = sample.view(-1, *sample.shape[2:])
    return sample



@D2MODEL_REGISTRY.register()
class PAGANRLControllerAlpha(_FairController):

  def __init__(self, cfg, **kwargs):
    super(PAGANRLControllerAlpha, self).__init__()

    self.FID_IS                  = kwargs['FID_IS']
    self.myargs                  = kwargs['myargs']
    self.num_layers              = get_attr_kwargs(cfg, 'num_layers', **kwargs)
    self.num_branches            = get_attr_kwargs(cfg, 'num_branches', **kwargs)
    self.tanh_constant           = get_attr_kwargs(cfg, 'tanh_constant', default=1.5, **kwargs)
    self.temperature             = get_attr_kwargs(cfg, 'temperature', default=-1, **kwargs)
    self.num_aggregate           = get_attr_kwargs(cfg, 'num_aggregate', **kwargs)
    self.entropy_weight          = get_attr_kwargs(cfg, 'entropy_weight', default=0.0001, **kwargs)
    self.bl_dec                  = get_attr_kwargs(cfg, 'bl_dec', **kwargs)
    self.child_grad_bound        = get_attr_kwargs(cfg, 'child_grad_bound', **kwargs)
    self.log_every_iter          = get_attr_kwargs(cfg, 'log_every_iter', default=50, **kwargs)
    self.cfg_ops                 = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)

    self.device = torch.device(f'cuda:{comm.get_rank()}')
    self.baseline = None
    self.logger = logging.getLogger('tl')

    self.alpha = nn.ParameterList()
    for i in range(self.num_layers):
      self.alpha.append(nn.Parameter(1e-4 * torch.randn(1, self.num_branches)))

  def forward(self, ):
    '''
    '''

    entropys = []
    log_probs = []
    sampled_arcs = []

    self.op_dist = []
    for layer_id in range(self.num_layers):
      logit = self.alpha[layer_id]
      # if self.temperature > 0:
      #   logit /= self.temperature
      # if self.tanh_constant is not None:
      #   logit = self.tanh_constant * torch.tanh(logit)

      op_dist = Categorical(logits=logit)
      self.op_dist.append(op_dist)

      sampled_op = op_dist.sample()
      sampled_arcs.append(sampled_op.view(-1, 1))

      log_prob = op_dist.log_prob(sampled_op)
      log_probs.append(log_prob.view(-1, 1))
      entropy = op_dist.entropy()
      entropys.append(entropy.view(-1, 1))

      # inputs = self.w_emb(branch_id)

    self.sampled_arcs = torch.cat(sampled_arcs, dim=1)
    self.sample_entropy = torch.cat(entropys, dim=1)
    self.sample_log_prob = torch.cat(log_probs, dim=1)

    return self.sampled_arcs

  def get_sampled_arc(self):
    sampled_arc = []
    for layer_id, op_dist in enumerate(self.op_dist):
      sampled_op = op_dist.sample().view(-1, 1)
      sampled_arc.append(sampled_op)

    sampled_arc = torch.cat(sampled_arc, dim=1)
    return sampled_arc

  def train_controller(self, G, z, y, controller, controller_optim, iteration, pbar):
    """

    :param controller: for ddp training
    :return:
    """
    if comm.is_main_process() and iteration % 1000 == 0:
      pbar.set_postfix_str("PAGANRLControllerAlpha")

    meter_dict = {}

    G.eval()
    controller.train()

    controller.zero_grad()

    sampled_arcs = controller()
    sample_entropy = get_ddp_attr(controller, 'sample_entropy')
    sample_log_prob = get_ddp_attr(controller, 'sample_log_prob')

    z_samples = z.sample()
    bs = len(z_samples)
    batched_arcs = sampled_arcs.repeat(bs, 1)

    pool_list, logits_list = [], []
    for i in range(self.num_aggregate):
      z_samples = z.sample().to(self.device)
      y_samples = y.sample().to(self.device)
      with torch.set_grad_enabled(False):
        x = G(z=z_samples, y=y_samples, batched_arcs=batched_arcs)

      pool, logits = self.FID_IS.get_pool_and_logits(x)

      # pool_list.append(pool)
      logits_list.append(logits)

    # pool = np.concatenate(pool_list, 0)
    logits = np.concatenate(logits_list, 0)

    reward_g, _ = self.FID_IS.calculate_IS(logits)
    meter_dict['reward_g'] = reward_g

    # detach to make sure that gradients aren't backpropped through the reward
    reward = torch.tensor(reward_g).cuda()
    sample_entropy_mean = sample_entropy.mean()
    meter_dict['sample_entropy'] = sample_entropy_mean.item()
    reward += self.entropy_weight * sample_entropy_mean

    if self.baseline is None:
      baseline = torch.tensor(reward_g)
    else:
      baseline = self.baseline - (1 - self.bl_dec) * (self.baseline - reward)
      # detach to make sure that gradients are not backpropped through the baseline
      baseline = baseline.detach()

    sample_log_prob_mean = sample_log_prob.mean()
    meter_dict['sample_log_prob'] = sample_log_prob_mean.item()
    loss = -1 * sample_log_prob_mean * (reward - baseline)

    meter_dict['reward'] = reward.item()
    meter_dict['baseline'] = baseline.item()
    meter_dict['loss'] = loss.item()

    loss.backward(retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), self.child_grad_bound)
    meter_dict['grad_norm'] = grad_norm

    controller_optim.step()

    baseline_list = comm.all_gather(baseline)
    baseline_mean = sum(map(lambda v: v.item(), baseline_list)) / len(baseline_list)
    baseline.fill_(baseline_mean)
    self.baseline = baseline

    if iteration % self.log_every_iter == 0:
      get_ddp_attr(controller, 'print_distribution')(iteration=iteration)
      default_dicts = collections.defaultdict(dict)
      for meter_k, meter in meter_dict.items():
        if meter_k in ['reward', 'baseline']:
          default_dicts['reward_baseline'][meter_k] = meter
        else:
          default_dicts[meter_k][meter_k] = meter
      summary_defaultdict2txtfig(default_dict=default_dicts,
                                 prefix='train_controller',
                                 step=iteration,
                                 textlogger=self.myargs.textlogger)
    comm.synchronize()
    return

  def print_distribution(self, iteration):
    # remove formats
    org_formatters = []
    for handler in self.logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("%(message)s"))

    default_dict = collections.defaultdict(dict)
    self.logger.info("####### distribution #######")
    searched_arc = []
    for layer_id, op_dist in enumerate(self.op_dist):
      prob = op_dist.probs
      max_op_id = prob.argmax().item()
      searched_arc.append(max_op_id)
      for op_id, op_name in enumerate(self.cfg_ops.keys()):
        op_prob = prob[0][op_id]
        default_dict[f'L{layer_id}'][get_prefix_abb(op_name)] = op_prob.item()

      self.logger.info(prob)
    searched_arc = np.array(searched_arc)
    self.logger.info('\nsearched arcs: \n%s' % searched_arc)
    self.myargs.textlogger.logstr(iteration,
                                  searched_arc='\n' + np.array2string(searched_arc, threshold=np.inf))

    summary_defaultdict2txtfig(default_dict=default_dict, prefix='', step=iteration,
                               textlogger=self.myargs.textlogger)
    self.logger.info("#####################")

    # restore formats
    for handler, formatter in zip(self.logger.handlers, org_formatters):
      handler.setFormatter(formatter)





@D2MODEL_REGISTRY.register()
class PAGANRLCondController(nn.Module):
  '''
  https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py
  '''

  def __init__(self, cfg, **kwargs):
    super(PAGANRLCondController, self).__init__()

    cfg = self.update_cfg(cfg)

    self.n_classes               = get_attr_kwargs(cfg, 'n_classes', **kwargs)
    self.num_layers              = get_attr_kwargs(cfg, 'num_layers', **kwargs)
    self.num_branches            = get_attr_kwargs(cfg, 'num_branches', **kwargs)
    self.search_whole_channels   = get_attr_kwargs(cfg, 'search_whole_channels', default=True, **kwargs)
    self.lstm_size               = get_attr_kwargs(cfg, 'lstm_size', default=64, **kwargs)
    self.lstm_num_layers         = get_attr_kwargs(cfg, 'lstm_num_layers', default=2, **kwargs)
    self.tanh_constant           = get_attr_kwargs(cfg, 'tanh_constant', default=1.5, **kwargs)
    self.temperature             = get_attr_kwargs(cfg, 'temperature', default=-1, **kwargs)

    self._create_params()
    self._reset_params()

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "PAGANRLController"
      n_classes: "kwargs['n_classes']"
      num_layers: "kwargs['num_layers']"
      num_branches: "kwargs['num_branches']"
      lstm_size: 64
      lstm_num_layers: 2
      temperature: -1
    """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg

  def test_case(self):
    bs = 2
    y = list(range(bs))
    out = self(class_ids=y)
    return out


  def _create_params(self):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L83
    '''
    self.w_lstm = nn.LSTM(input_size=self.lstm_size,
                          hidden_size=self.lstm_size,
                          num_layers=self.lstm_num_layers)
    # Learn the starting input
    self.g_emb = nn.Embedding(self.n_classes, self.lstm_size)

    if self.search_whole_channels:
      self.w_emb = nn.Embedding(self.num_branches, self.lstm_size)
      self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=False)
    else:
      assert False, "Not implemented error: search_whole_channels = False"

    # self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
    # self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
    # self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)

  def _reset_params(self):
    for m in self.modules():
      if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)

    nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
    nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

  def forward(self, class_ids, determine_sample=False):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L126
    '''
    h0 = None  # setting h0 to None will initialize LSTM state with 0s
    arc_seq = []
    entropys = []
    log_probs = []
    if isinstance(class_ids, int):
      class_ids = [class_ids]
    if isinstance(class_ids, list):
      class_ids = torch.tensor(class_ids, dtype=torch.int64)
    class_ids = class_ids.type(torch.int64)
    inputs = self.g_emb.weight[class_ids]

    for layer_id in range(self.num_layers):
      if self.search_whole_channels:
        inputs = inputs.unsqueeze(dim=0)
        output, hn = self.w_lstm(inputs, h0)
        output = output.squeeze(dim=0)
        h0 = hn

        logit = self.w_soft(output)
        if self.temperature > 0:
          logit /= self.temperature
        if self.tanh_constant is not None:
          logit = self.tanh_constant * torch.tanh(logit)

        branch_id_dist = Categorical(logits=logit)
        if determine_sample:
          branch_id = logit.argmax(dim=1)
        else:
          branch_id = branch_id_dist.sample()

        arc_seq.append(branch_id)

        log_prob = branch_id_dist.log_prob(branch_id)
        log_probs.append(log_prob.view(-1))
        entropy = branch_id_dist.entropy()
        entropys.append(entropy.view(-1))

      else:
        # https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L171
        assert False, "Not implemented error: search_whole_channels = False"

      # Calculate average of class and branch embedding
      # and use it as input for next step
      inputs = self.w_emb(branch_id) + self.g_emb.weight[class_ids]
      inputs /= 2


    self.sample_arc = torch.stack(arc_seq, dim=1)

    self.sample_entropy = torch.stack(entropys, dim=1)

    self.sample_log_prob = torch.stack(log_probs, dim=1)
    self.sample_prob = self.sample_log_prob.exp()
    return self.sample_arc

  def forward_G(self, G, z, gy, train_C=False, train_G=True,
              fixed_arc=None, same_in_batch=True,
              return_sample_entropy=False,
              return_sample_log_prob=False,
              determine_sample=False,
              return_sample_arc=False,
              cbn=False):

    if fixed_arc is not None:
      fixed_arc = torch.from_numpy(fixed_arc).cuda()
      self.sample_arc = fixed_arc[gy]
    else:
      with torch.set_grad_enabled(train_C):
        if same_in_batch:
          y_range = torch.arange(
            0, get_ddp_attr(self, 'n_classes'), dtype=torch.int64)
          self(y_range, determine_sample=determine_sample)
          self.sample_arc = get_ddp_attr(self, 'sample_arc')[gy]
        else:
          self(gy, determine_sample=determine_sample)
          self.sample_arc = get_ddp_attr(self, 'sample_arc')

    with torch.set_grad_enabled(train_G):
      if cbn:
        x = G(z, gy, self.sample_arc)
      else:
        # x = nn.parallel.data_parallel(G, (z, self.sample_arc))
        x = G(z, self.sample_arc)

    out = x
    ret_out = (out,)
    if return_sample_entropy:
      sample_entropy = get_ddp_attr(self, 'sample_entropy')
      ret_out = ret_out + (sample_entropy,)
    if return_sample_log_prob:
      sample_log_prob = get_ddp_attr(self, 'sample_log_prob')
      ret_out = ret_out + (sample_log_prob,)
    if return_sample_arc:
      ret_out = ret_out + (self.sample_arc,)
    if len(ret_out) == 1:
      return ret_out[0]
    return ret_out


@D2MODEL_REGISTRY.register()
class PAGANFairControllerFC(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(PAGANFairControllerFC, self).__init__()

    cfg = self.update_cfg(cfg)

    self.FID_IS                  = kwargs['FID_IS']
    self.myargs                  = kwargs['myargs']
    self.episode                 = get_attr_kwargs(cfg, 'episode', **kwargs)
    self.in_dim                  = get_attr_kwargs(cfg, 'in_dim', **kwargs)
    self.out_dim                 = get_attr_kwargs(cfg, 'out_dim', **kwargs)
    self.hidden_dim              = get_attr_kwargs(cfg, 'hidden_dim', default=64, **kwargs)
    self.num_hidden_layer        = get_attr_kwargs(cfg, 'num_hidden_layer', default=2, **kwargs)
    self.tanh_constant           = get_attr_kwargs(cfg, 'tanh_constant', default=1.5, **kwargs)
    self.temperature             = get_attr_kwargs(cfg, 'temperature', default=-1, **kwargs)
    self.num_aggregate           = get_attr_kwargs(cfg, 'num_aggregate', default=20, **kwargs)
    self.entropy_weight          = get_attr_kwargs(cfg, 'entropy_weight', default=0.0001, **kwargs)
    self.bl_dec                  = get_attr_kwargs(cfg, 'bl_dec', default=0.99, **kwargs)
    self.child_grad_bound        = get_attr_kwargs(cfg, 'child_grad_bound', default=5.0, **kwargs)
    self.log_every_iter          = get_attr_kwargs(cfg, 'log_every_iter', default=50, **kwargs)

    self.device = torch.device(f'cuda:{comm.get_rank()}')
    self.logger = logging.getLogger('tl')
    self.baseline = None
    self._create_params()
    self._reset_params()
    pass

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "PAGANRLController"
      num_layers: "kwargs['num_layers']"
      num_branches: "kwargs['num_branches']"
      lstm_size: 64
      lstm_num_layers: 2
      temperature: -1
      num_aggregate: 10
      entropy_weight: 0.0001
      bl_dec: 0.99
      child_grad_bound: 5.0
    """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg

  def test_case(self):
    out = self()
    return out


  def _create_params(self):

    layers = []
    in_fc = nn.Linear(self.in_dim, self.hidden_dim, bias=True)
    layers.append((f'in_fc', in_fc))
    layers.append((f'in_fc_act', nn.ReLU()))
    for i in range(self.num_hidden_layer):
      hidden_layer = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
      layers.append((f'hidden_{i}', hidden_layer))
      layers.append((f'act_{i}', nn.ReLU()))
    out_fc = nn.Linear(self.hidden_dim, self.out_dim, bias=True)
    layers.append((f'out_fc', out_fc))
    self.net = nn.Sequential(OrderedDict(layers))

    self.w_emb = nn.Embedding(self.episode, self.in_dim)
    # self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=True)

    # self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
    # self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
    # self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)

  def _reset_params(self):
    for m in self.modules():
      if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)

  def get_fair_path(self, bs):
    """

    :param batch_imgs:
    :return: (bs x num_branches, num_layers)
    """
    arcs = []
    for l in range(self.episode):
      layer_arcs = torch.randperm(self.out_dim).view(-1, 1)
      arcs.append(layer_arcs)
    arcs = torch.cat(arcs, dim=1)
    batched_arcs = arcs.repeat(bs, 1)

    fair_arcs = arcs
    return batched_arcs, fair_arcs

  def fairnas_repeat_tensor(self, sample):
    repeat_arg = [1] * (sample.dim() + 1)
    repeat_arg[1] = self.out_dim
    sample = sample.unsqueeze(1).repeat(repeat_arg)
    sample = sample.view(-1, *sample.shape[2:])
    return sample

  def forward(self, fair_arcs):

    arc_seq = []
    entropys = []
    log_probs = []

    inputs = self.w_emb.weight
    logits = self.net(inputs)

    if self.temperature > 0:
      logits /= self.temperature
    if self.tanh_constant is not None:
      logits = self.tanh_constant * torch.tanh(logits)

    batched_dist = Categorical(logits=logits)

    for arc in fair_arcs:
      log_prob = batched_dist.log_prob(arc)
      log_probs.append(log_prob.view(-1))

    entropy = batched_dist.entropy()
    entropys.append(entropy.view(-1))

    self.sample_entropy = torch.stack(entropys, dim=0)

    self.sample_log_prob = torch.stack(log_probs, dim=0)
    self.sample_prob = self.sample_log_prob.exp()

    return self.sample_log_prob

  def train_controller(self, G, z, y, controller, controller_optim, fair_arcs, iteration):
    """

    :param controller: for ddp training
    :return:
    """
    meter_dict = {}

    G.eval()
    controller.train()

    controller.zero_grad()

    controller(fair_arcs=fair_arcs)
    sample_entropy = get_ddp_attr(controller, 'sample_entropy')
    sample_log_prob = get_ddp_attr(controller, 'sample_log_prob')

    z_samples = z.sample()
    repeat_times = len(z_samples) // len(fair_arcs)
    batched_arcs = fair_arcs.repeat(repeat_times, 1)

    pool_list, logits_list = [], []
    for i in range(self.num_aggregate):
      z_samples = z.sample().to(self.device)
      y_samples = y.sample().to(self.device)
      with torch.set_grad_enabled(False):
        x = G(z=z_samples, y=y_samples, batched_arcs=batched_arcs)

      _, logits = self.FID_IS.get_pool_and_logits(x)

      # pool_list.append(pool)
      logits_list.append(logits)

      # pool = np.concatenate(pool_list, 0)
      logits = np.concatenate(logits_list, 0)

    rewards_g = []
    num_arcs = len(fair_arcs)
    for i in range(num_arcs):
      logits_i = logits[i::num_arcs]
      reward_g, _ = self.FID_IS.calculate_IS(logits_i)
      rewards_g.append(reward_g)

    meter_dict['reward_g_mean'] = sum(rewards_g)/num_arcs

    # detach to make sure that gradients aren't backpropped through the reward
    reward = torch.tensor(rewards_g).cuda().view(-1, 1)

    sample_entropy_mean = sample_entropy.mean()
    meter_dict['sample_entropy'] = sample_entropy_mean.item()
    reward += self.entropy_weight * sample_entropy_mean.view(1, 1)

    if self.baseline is None:
      baseline = torch.tensor(rewards_g).mean().view(-1, 1).to(self.device)
    else:
      baseline = self.baseline - (1 - self.bl_dec) * (self.baseline - reward)
      # detach to make sure that gradients are not backpropped through the baseline
      baseline = baseline.mean().view(-1, 1).detach()

    sample_log_prob_mean = sample_log_prob.mean(dim=1, keepdim=True)
    meter_dict['sample_log_prob_mean'] = sample_log_prob_mean.mean().item()
    losses = -1 * sample_log_prob_mean * (reward - baseline)
    loss = losses.mean()

    meter_dict['reward'] = reward.mean().item()
    meter_dict['baseline'] = baseline.item()
    meter_dict['loss'] = loss.item()

    loss.backward(retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), self.child_grad_bound)
    meter_dict['grad_norm'] = grad_norm

    controller_optim.step()

    baseline_list = comm.all_gather(baseline)
    baseline_mean = sum(map(lambda v: v.item(), baseline_list)) / len(baseline_list)
    baseline.fill_(baseline_mean)
    self.baseline = baseline

    if iteration % self.log_every_iter == 0:
      default_dicts = collections.defaultdict(dict)
      for meter_k, meter in meter_dict.items():
        if meter_k in ['reward', 'baseline']:
          default_dicts['reward_baseline'][meter_k] = meter
        else:
          default_dicts[meter_k][meter_k] = meter
      summary_defaultdict2txtfig(default_dict=default_dicts,
                                 prefix='train_controller',
                                 step=iteration,
                                 textlogger=self.myargs.textlogger)
    comm.synchronize()
    return

  def print_distribution(self, ):
    # remove formats
    org_formatters = []
    for handler in self.logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("%(message)s"))

    self.logger.info("####### distribution #######")
    with torch.no_grad():
      inputs = self.w_emb.weight
      logits = self.net(inputs)
      self.logger.info(F.softmax(logits, dim=-1))

    self.logger.info("#####################")

    # restore formats
    for handler, formatter in zip(self.logger.handlers, org_formatters):
      handler.setFormatter(formatter)










