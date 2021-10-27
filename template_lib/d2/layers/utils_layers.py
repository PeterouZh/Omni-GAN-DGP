import math
import yaml
from easydict import EasyDict
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.layers.build import D2LAYER_REGISTRY, build_d2layer
from template_lib.utils import get_attr_kwargs
from template_lib.v2.config import update_config


@D2LAYER_REGISTRY.register()
class UpSample(nn.Module):

  def __init__(self, cfg, scale_factor=2, **kwargs):
    super(UpSample, self).__init__()

    self.scale_factor                   = get_attr_kwargs(cfg, 'scale_factor', default=2, **kwargs)
    self.mode                           = get_attr_kwargs(cfg, 'mode', default='bilinear',
                                                          choices=['bilinear', 'nearest'], **kwargs)
    self.align_corners                  = get_attr_kwargs(cfg, 'align_corners', default=None, **kwargs)


  def forward(self, x):
    x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
    return x


@D2LAYER_REGISTRY.register()
class AvgPool2d(nn.AvgPool2d):

  def __init__(self, cfg, **kwargs):
    kernel_size              = get_attr_kwargs(cfg, 'kernel_size', default=2, **kwargs)
    stride                   = get_attr_kwargs(cfg, 'stride', default=None, **kwargs)
    padding                  = get_attr_kwargs(cfg, 'padding', default=0, **kwargs)
    ceil_mode                = get_attr_kwargs(cfg, 'ceil_mode', default=False, **kwargs)
    count_include_pad        = get_attr_kwargs(cfg, 'count_include_pad', default=True, **kwargs)
    # divisor_override         = get_attr_kwargs(cfg, 'divisor_override', default=None, **kwargs)

    super().__init__(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                     count_include_pad=count_include_pad)


@D2LAYER_REGISTRY.register()
class Identity(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()

  def forward(self, x):
    return x


@D2LAYER_REGISTRY.register()
class D2None(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()

    pass

  def forward(self, x, **kwargs):

    return x * 0.0


@D2LAYER_REGISTRY.register()
class FactorizedReduce(nn.Module):
  """
  Reduce feature map size by factorized pointwise(stride=2).
  """

  def __init__(self, cfg, **kwargs):
    super().__init__()

    self.in_channels       = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels      = get_attr_kwargs(cfg, 'out_channels', **kwargs)


    # self.bn = nn.BatchNorm2d(self.out_channels, affine=True)
    # self.relu = nn.ReLU()
    self.conv1 = nn.Conv2d(self.in_channels, self.out_channels   // 2, 1, stride=2, padding=0, bias=False)
    self.conv2 = nn.Conv2d(self.in_channels, self.out_channels   // 2, 1, stride=2, padding=0, bias=False)

  def forward(self, x):
    # x = self.relu(x)
    x = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
    # x = self.bn(x)
    return x


@D2LAYER_REGISTRY.register()
class DenseBlock(nn.Module):

  def __init__(self, cfg, **kwargs):
    super().__init__()

    cfg = self.update_cfg(cfg)

    self.in_channels           = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.n_nodes               = get_attr_kwargs(cfg, 'n_nodes', **kwargs)
    self.cfg_mix_layer         = get_attr_kwargs(cfg, 'cfg_mix_layer', **kwargs)
    self.cfg_ops               = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)
    self.cell_op_idx           = get_attr_kwargs(cfg, 'cell_op_idx', default=None, **kwargs)

    self.num_edges = self.get_edges(self.n_nodes)
    self.cfg_keys = list(self.cfg_ops.keys())
    self.out_channels = self.in_channels

    assert (self.in_channels) % self.n_nodes == 0
    self.internal_c = self.in_channels // self.n_nodes

    self.in_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.internal_c,
                             kernel_size=1, stride=1, padding=0)

    # generate dag
    edge_idx = 0
    self.dag = nn.ModuleList()
    for i in range(self.n_nodes):
      self.dag.append(nn.ModuleList())
      for j in range(1 + i):
        if self.cell_op_idx is not None:
          op_key = self.cfg_keys[self.cell_op_idx[edge_idx]]
          cfg_ops = EasyDict({op_key: self.cfg_ops[op_key]})
          edge_idx += 1
          op = build_d2layer(self.cfg_mix_layer, **{**kwargs, "in_channels": self.internal_c,
                                                    "out_channels"         : self.internal_c,
                                                    "cfg_ops"              : cfg_ops})
        else:
          op = build_d2layer(self.cfg_mix_layer, **{**kwargs, "in_channels": self.internal_c,
                                                    "out_channels"         : self.internal_c,
                                                    "cfg_ops"              : self.cfg_ops})
        self.dag[i].append(op)
    pass

  def forward(self, x, batched_arcs, **kwargs):
    skip = x
    x = self.in_conv(x)

    states = [x, ]
    idx_start = 0
    for edges in self.dag:
      edges_arcs = batched_arcs[:, idx_start:idx_start + len(edges)]
      idx_start += len(edges)
      s_cur = sum(edge(s, edges_arcs[:, i], **kwargs) for i, (edge, s) in enumerate(zip(edges, states)))

      states.append(s_cur)

    x = torch.cat(states[1:], dim=1)
    x = x + skip
    return x

  @staticmethod
  def get_edges(n_nodes):
    num_edges = (n_nodes + 1) * n_nodes // 2
    return num_edges

  @staticmethod
  def test_case():
    cfg_str = """
              name: "DenseBlock"
              update_cfg: true
              in_channels: 144
              """
    cfg = EasyDict(yaml.safe_load(cfg_str))

    op = build_d2layer(cfg)
    op.cuda()

    bs = 2
    num_ops = len(op.cfg_ops)
    x = torch.randn(bs, 144, 8, 8).cuda()
    batched_arcs = torch.tensor([
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]).cuda()
    out = op(x, batched_arcs)

    import torchviz
    g = torchviz.make_dot(out)
    g.view()
    pass

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "DenseCellCat"
      n_nodes: 4
      cfg_mix_layer:
        name: "MixedLayer"
      cfg_ops:
        Identity:
          name: "Identity"
        Conv2d_3x3:
          name: "Conv2dAct"
          cfg_act:
            name: "ReLU"
          cfg_conv:
            name: "Conv2d"
            kernel_size: 3
            padding: 1
        None:
          name: "D2None"

      """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg


@D2LAYER_REGISTRY.register()
class DenseBlockReZero(nn.Module):

  def __init__(self, cfg, **kwargs):
    super().__init__()

    cfg = self.update_cfg(cfg)

    self.in_channels           = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.n_nodes               = get_attr_kwargs(cfg, 'n_nodes', **kwargs)
    self.cfg_mix_layer         = get_attr_kwargs(cfg, 'cfg_mix_layer', **kwargs)
    self.cfg_ops               = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)
    self.cell_op_idx           = get_attr_kwargs(cfg, 'cell_op_idx', default=None, **kwargs)

    self.num_edges = self.get_edges(self.n_nodes)
    self.cfg_keys = list(self.cfg_ops.keys())
    self.out_channels = self.in_channels

    assert (self.in_channels) % self.n_nodes == 0
    self.internal_c = self.in_channels // self.n_nodes

    self.resweight = nn.Parameter(torch.zeros(1), requires_grad=True)
    self.in_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.internal_c,
                             kernel_size=1, stride=1, padding=0)

    # generate dag
    edge_idx = 0
    self.dag = nn.ModuleList()
    for i in range(self.n_nodes):
      self.dag.append(nn.ModuleList())
      for j in range(1 + i):
        if self.cell_op_idx is not None:
          op_key = self.cfg_keys[self.cell_op_idx[edge_idx]]
          cfg_ops = EasyDict({op_key: self.cfg_ops[op_key]})
          edge_idx += 1
          op = build_d2layer(self.cfg_mix_layer, **{**kwargs, "in_channels": self.internal_c,
                                                    "out_channels"         : self.internal_c,
                                                    "cfg_ops"              : cfg_ops})
        else:
          op = build_d2layer(self.cfg_mix_layer, **{**kwargs, "in_channels": self.internal_c,
                                                    "out_channels"         : self.internal_c,
                                                    "cfg_ops"              : self.cfg_ops})
        self.dag[i].append(op)
    pass

  def forward(self, x, batched_arcs):
    skip = x
    x = self.in_conv(x)

    states = [x, ]
    idx_start = 0
    for edges in self.dag:
      edges_arcs = batched_arcs[:, idx_start:idx_start + len(edges)]
      idx_start += len(edges)
      s_cur = sum(edge(s, edges_arcs[:, i]) for i, (edge, s) in enumerate(zip(edges, states)))

      states.append(s_cur)

    x = torch.cat(states[1:], dim=1)
    x = self.resweight * x + skip
    return x

  @staticmethod
  def get_edges(n_nodes):
    num_edges = (n_nodes + 1) * n_nodes // 2
    return num_edges

  def test_case(self):
    bs = 2
    num_ops = len(self.cfg_ops)
    x = torch.randn(bs, 144, 8, 8).cuda()
    batched_arcs = torch.tensor([
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]).cuda()
    out = self(x, batched_arcs)
    return out

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "DenseCellCat"
      n_nodes: 4
      cfg_mix_layer:
        name: "MixedLayer"
      cfg_ops:
        Identity:
          name: "Identity"
        Conv2d_3x3:
          name: "Conv2dAct"
          cfg_act:
            name: "ReLU"
          cfg_conv:
            name: "Conv2d"
            kernel_size: 3
            padding: 1
        None:
          name: "D2None"

      """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg


@D2LAYER_REGISTRY.register()
class DenseBlockV1(nn.Module):

  def __init__(self, cfg, **kwargs):
    super().__init__()

    cfg = self.update_cfg(cfg)

    self.in_channels           = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.n_nodes               = get_attr_kwargs(cfg, 'n_nodes', **kwargs)
    self.cfg_mix_layer         = get_attr_kwargs(cfg, 'cfg_mix_layer', **kwargs)
    self.cfg_ops               = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)
    self.cell_op_idx           = get_attr_kwargs(cfg, 'cell_op_idx', default=None, **kwargs)

    self.num_edges = self.get_edges(self.n_nodes)
    self.cfg_keys = list(self.cfg_ops.keys())
    self.out_channels = self.in_channels

    assert (self.in_channels) % self.n_nodes == 0
    self.internal_c = self.in_channels // self.n_nodes

    # self.act = nn.ReLU()
    self.in_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.internal_c,
                             kernel_size=1, stride=1, padding=0)

    # generate dag
    edge_idx = 0
    self.dag = nn.ModuleList()
    for i in range(self.n_nodes):
      self.dag.append(nn.ModuleList())
      for j in range(1 + i):
        if self.cell_op_idx is not None:
          op_key = self.cfg_keys[self.cell_op_idx[edge_idx]]
          cfg_ops = EasyDict({op_key: self.cfg_ops[op_key]})
          edge_idx += 1
          op = build_d2layer(self.cfg_mix_layer, **{**kwargs, "in_channels": self.internal_c,
                                                    "out_channels"         : self.internal_c,
                                                    "cfg_ops"              : cfg_ops})
        else:
          op = build_d2layer(self.cfg_mix_layer, **{**kwargs, "in_channels": self.internal_c,
                                                    "out_channels"         : self.internal_c,
                                                    "cfg_ops"              : self.cfg_ops})
        self.dag[i].append(op)
    pass

  def forward(self, x, batched_arcs):
    skip = x
    # x = self.act(x)
    x = self.in_conv(x)

    states = [x, ]
    idx_start = 0
    for edges in self.dag:
      edges_arcs = batched_arcs[:, idx_start:idx_start + len(edges)]
      idx_start += len(edges)
      s_cur = sum(edge(s, edges_arcs[:, i]) for i, (edge, s) in enumerate(zip(edges, states)))

      states.append(s_cur)

    x = torch.cat(states[1:], dim=1)
    x = x + skip
    return x

  @staticmethod
  def get_edges(n_nodes):
    num_edges = (n_nodes + 1) * n_nodes // 2
    return num_edges

  def test_case(self):
    bs = 2
    num_ops = len(self.cfg_ops)
    x = torch.randn(bs, 144, 8, 8).cuda()
    batched_arcs = torch.tensor([
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]).cuda()
    out = self(x, batched_arcs)
    return out

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "DenseCellCat"
      n_nodes: 4
      cfg_mix_layer:
        name: "MixedLayer"
      cfg_ops:
        Identity:
          name: "Identity"
        Conv2d_3x3:
          name: "Conv2dAct"
          cfg_act:
            name: "ReLU"
          cfg_conv:
            name: "Conv2d"
            kernel_size: 3
            padding: 1
        None:
          name: "D2None"

      """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg


class PixelNorm(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, input):
    return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class NoiseInjection(nn.Module):
  def __init__(self, channel):
    super().__init__()

    self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

  def forward(self, image, noise):
    return image + self.weight * noise

@D2LAYER_REGISTRY.register()
class StyleLayer(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(StyleLayer, self).__init__()

    cfg = self.update_cfg(cfg)

    self.z_dim                  = get_attr_kwargs(cfg, 'z_dim', **kwargs)
    self.n_mlp                  = get_attr_kwargs(cfg, 'n_mlp', **kwargs)
    self.num_features           = get_attr_kwargs(cfg, 'num_features', **kwargs)
    self.eps                    = get_attr_kwargs(cfg, 'eps', default=1e-5, **kwargs)
    self.momentum               = get_attr_kwargs(cfg, 'momentum', default=0.1, **kwargs)

    self.conv = nn.Conv2d(self.num_features, self.num_features, 3, 1, 1)
    # Prepare gain and bias layers
    layers = [PixelNorm()]
    for i in range(self.n_mlp):
      layers.append(nn.Linear(self.z_dim, self.z_dim))
      layers.append(nn.LeakyReLU(0.2))
    self.style = nn.Sequential(*layers)

    self.gain = nn.Linear(in_features=self.z_dim, out_features=self.num_features)
    self.bias = nn.Linear(in_features=self.z_dim, out_features=self.num_features)

    self.noise = NoiseInjection(self.num_features)
    self.lrelu = nn.LeakyReLU(0.2)
    pass

  def forward(self, x):
    """

    :param x:
    :param y: feature [b, self.input_size]
    :return:
    """
    x = self.conv(x)
    b, c, h, w = x.size()
    noise = torch.randn(b, 1, h, w, device=x.device)
    x = self.noise(x, noise)
    x = self.lrelu(x)

    z = torch.randn(b, self.z_dim, device=x.device)
    w_code = self.style(z)
    gain = (1 + self.gain(w_code)).view(b, -1, 1, 1)
    bias = self.bias(w_code).view(b, -1, 1, 1)
    x = F.instance_norm(x, running_mean=None, running_var=None, weight=None, bias=None,
                        use_input_stats=True, momentum=self.momentum, eps=self.eps)
    out = x * gain + bias
    return out

  def test_case(self):
    bs = 2
    x = torch.randn(bs, 256, 8, 8).cuda()
    out = self(x)
    return out

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
        name: "StyleLayer"
        z_dim: 128
        n_mlp: 1
        num_features: 256

      """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg


@D2LAYER_REGISTRY.register()
class ModulatedConv2d(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()

    self.in_channels               = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels              = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.kernel_size               = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    self.style_dim                 = get_attr_kwargs(cfg, 'style_dim', **kwargs)
    self.demodulate                = get_attr_kwargs(cfg, 'demodulate', default=True, **kwargs)

    fan_in = self.in_channels * self.kernel_size ** 2
    self.scale = 1 / math.sqrt(fan_in)
    self.padding = self.kernel_size // 2

    self.weight = nn.Parameter(torch.randn(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
    self.modulation = nn.Linear(self.style_dim, self.in_channels)
    pass

  def forward(self, input, style):
    batch, in_channel, height, width = input.shape

    style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
    weight = self.scale * self.weight * style

    if self.demodulate:
      demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
      weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

    weight = weight.view(batch * self.out_channels, in_channel, self.kernel_size, self.kernel_size)

    input = input.view(1, batch * in_channel, height, width)
    out = F.conv2d(input, weight, padding=self.padding, groups=batch)
    _, _, height, width = out.shape
    out = out.view(batch, self.out_channels, height, width)

    return out

  def __repr__(self):
    return (f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, {self.kernel_size}')


@D2LAYER_REGISTRY.register()
class ModulatedConv2dOp(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()

    self.in_channels               = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels              = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.kernel_size               = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    self.style_dim                 = get_attr_kwargs(cfg, 'style_dim', **kwargs)
    self.demodulate                = get_attr_kwargs(cfg, 'demodulate', default=True, **kwargs)

    fan_in = self.in_channels * self.kernel_size ** 2
    self.scale = 1 / math.sqrt(fan_in)
    self.padding = self.kernel_size // 2

    # self.weight = nn.Parameter(torch.randn(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
    self.modulation = nn.Linear(self.style_dim, self.in_channels)
    pass

  def forward(self, input, style, weight, **kwargs):
    batch, in_channel, height, width = input.shape

    style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
    weight = weight.unsqueeze(0)
    weight = self.scale * weight * style

    if self.demodulate:
      demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
      weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

    weight = weight.view(batch * self.out_channels, in_channel, self.kernel_size, self.kernel_size)

    input = input.view(1, batch * in_channel, height, width)
    out = F.conv2d(input, weight, padding=self.padding, groups=batch)
    _, _, height, width = out.shape
    out = out.view(batch, self.out_channels, height, width)

    return out

  def __repr__(self):
    return (f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, {self.kernel_size}')


class NoiseInjectionV2(nn.Module):
  def __init__(self):
    super().__init__()

    self.weight = nn.Parameter(torch.zeros(1))

  def forward(self, image, noise=None):
    if noise is None:
      batch, _, height, width = image.shape
      noise = image.new_empty(batch, 1, height, width).normal_()
    return image + self.weight * noise


@D2LAYER_REGISTRY.register()
class StyleV2Conv(nn.Module):

  def __init__(self, cfg, **kwargs):
    super().__init__()

    cfg = self.update_cfg(cfg)

    self.cfg_modconv                   = get_attr_kwargs(cfg, 'cfg_modconv', **kwargs)
    self.add_noise                     = get_attr_kwargs(cfg, 'add_noise', default=True, **kwargs)

    self.cfg = cfg

    self.activate = nn.ReLU()
    self.conv = build_d2layer(self.cfg_modconv, **kwargs)
    self.noise = NoiseInjectionV2()

  def forward(self, x, style, noise=None, **kwargs):
    x = self.activate(x)
    x = self.conv(x, style, **kwargs)
    if self.add_noise:
      out = self.noise(x, noise=noise)
    else:
      out = x
    return out

  def test_case(self, in_channels, out_channels):
    bs = 2
    x = torch.randn(bs, in_channels, 8, 8).cuda()
    style = torch.randn(bs, self.cfg.cfg_modconv.style_dim).cuda()
    out = self(x, style)
    return out

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
        name: "StyleV2Conv"
        cfg_modconv:
          name: "ModulatedConv2d"
          kernel_size: 3
          style_dim: 192

      """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg







  