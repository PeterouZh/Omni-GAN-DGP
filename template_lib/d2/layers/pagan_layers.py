import yaml
from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.utils import get_attr_kwargs
from template_lib.v2.config import update_config

from .build import D2LAYER_REGISTRY, build_d2layer


@D2LAYER_REGISTRY.register()
class MixedLayerCond(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(MixedLayerCond, self).__init__()

    self.in_channels = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.cfg_ops = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)

    self.num_branch = len(self.cfg_ops)

    self.branches = nn.ModuleList()
    for name, cfg_op in self.cfg_ops.items():
      branch = build_d2layer(cfg_op, in_channels=self.in_channels, out_channels=self.out_channels)
      self.branches.append(branch)
    pass

  def forward(self, x, y, sample_arc):
    bs = len(sample_arc)
    sample_arc = sample_arc.type(torch.int64)

    sample_arc_onehot = torch.zeros(bs, self.num_branch).cuda()
    sample_arc_onehot[torch.arange(bs), sample_arc] = 1
    sample_arc_onehot = sample_arc_onehot.view(bs, self.num_branch, 1, 1, 1)

    x = [branch(x, y).unsqueeze(1) if idx in sample_arc else \
           (torch.zeros(bs, self.out_channels, x.size(-1), x.size(-1)).cuda().requires_grad_(False).unsqueeze(1)) \
         for idx, branch in enumerate(self.branches)]
    # x = [branch(x, y).unsqueeze(1) for branch in self.branches]
    x = torch.cat(x, 1)
    x = sample_arc_onehot * x
    x = x.sum(dim=1)

    return x


@D2LAYER_REGISTRY.register()
class MixedLayer(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(MixedLayer, self).__init__()

    self.in_channels = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.cfg_ops = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)

    self.num_branch = len(self.cfg_ops)

    self.branches = nn.ModuleList()
    for name, cfg_op in self.cfg_ops.items():
      branch = build_d2layer(cfg_op, in_channels=self.in_channels, out_channels=self.out_channels)
      self.branches.append(branch)
    pass

  def forward(self, x, sample_arc, **kwargs):
    bs = len(sample_arc)
    sample_arc = sample_arc.type(torch.int64)

    arc_unique = sample_arc.unique()
    if len(arc_unique) == 1:
      x = self.branches[arc_unique](x, **kwargs)
      return x

    sample_arc_onehot = torch.zeros(bs, self.num_branch).cuda()
    sample_arc_onehot[torch.arange(bs), sample_arc] = 1
    sample_arc_onehot = sample_arc_onehot.view(bs, self.num_branch, 1, 1, 1)

    x = [branch(x, **kwargs).unsqueeze(1) if idx in sample_arc else \
           (torch.zeros(bs, self.out_channels, x.size(-1), x.size(-1)).cuda().requires_grad_(False).unsqueeze(1)) \
         for idx, branch in enumerate(self.branches)]
    # x = [branch(x, y).unsqueeze(1) for branch in self.branches]
    x = torch.cat(x, 1)
    x = sample_arc_onehot * x
    x = x.sum(dim=1)

    return x


@D2LAYER_REGISTRY.register()
class MixedLayerShareWeights(nn.Module):

  def __init__(self, cfg, **kwargs):
    super().__init__()

    self.in_channels            = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels           = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.kernel_size            = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    self.bias                   = get_attr_kwargs(cfg, 'bias', **kwargs)
    self.cfg_ops                = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)

    self.num_branch = len(self.cfg_ops)

    self.shared_weights = nn.Parameter(
      torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
    nn.init.orthogonal_(self.shared_weights.data)
    if self.bias:
      self.shared_bias = nn.Parameter(torch.randn(self.out_channels))
    else:
      self.shared_bias = None

    self.branches = nn.ModuleList()
    for name, cfg_op in self.cfg_ops.items():
      branch = build_d2layer(cfg_op, in_channels=self.in_channels, out_channels=self.out_channels)
      self.branches.append(branch)
    pass

  def forward(self, x, sample_arc, **kwargs):
    bs = len(sample_arc)
    sample_arc = sample_arc.type(torch.int64)

    arc_unique = sample_arc.unique()
    if len(arc_unique) == 1:
      x = self.branches[arc_unique](x, weight=self.shared_weights, bias=self.shared_bias, **kwargs)
      return x

    sample_arc_onehot = torch.zeros(bs, self.num_branch).cuda()
    sample_arc_onehot[torch.arange(bs), sample_arc] = 1
    sample_arc_onehot = sample_arc_onehot.view(bs, self.num_branch, 1, 1, 1)

    x = [branch(x, weight=self.shared_weights, bias=self.shared_bias, **kwargs).unsqueeze(1) if idx in sample_arc else \
           (torch.zeros(bs, self.out_channels, x.size(-1), x.size(-1)).cuda().requires_grad_(False).unsqueeze(1)) \
         for idx, branch in enumerate(self.branches)]
    # x = [branch(x, y).unsqueeze(1) for branch in self.branches]
    x = torch.cat(x, 1)
    x = sample_arc_onehot * x
    x = x.sum(dim=1)

    return x



@D2LAYER_REGISTRY.register()
class MixedActLayer(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(MixedActLayer, self).__init__()

    self.out_channels = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.cfg_ops = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)

    self.num_branch = len(self.cfg_ops)

    self.branches = nn.ModuleList()
    for name, cfg_op in self.cfg_ops.items():
      branch = build_d2layer(cfg_op, **kwargs)
      self.branches.append(branch)
    pass

  def forward(self, x, sample_arc):
    bs = len(sample_arc)
    sample_arc = sample_arc.type(torch.int64)

    sample_arc_onehot = torch.zeros(bs, self.num_branch).cuda()
    sample_arc_onehot[torch.arange(bs), sample_arc] = 1
    sample_arc_onehot = sample_arc_onehot.view(bs, self.num_branch, 1, 1, 1)

    x = [branch(x).unsqueeze(1) if idx in sample_arc else \
           (torch.zeros(bs, self.out_channels, x.size(-1), x.size(-1)).cuda().requires_grad_(False).unsqueeze(1)) \
         for idx, branch in enumerate(self.branches)]
    # x = [branch(x, y).unsqueeze(1) for branch in self.branches]
    x = torch.cat(x, 1)
    x = sample_arc_onehot * x
    x = x.sum(dim=1)

    return x


@D2LAYER_REGISTRY.register()
class DenseCell(nn.Module):

  def __init__(self, cfg, **kwargs):
    super().__init__()

    cfg = self.update_cfg(cfg)

    self.in_channels            = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels           = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.n_nodes                = get_attr_kwargs(cfg, 'n_nodes', **kwargs)
    self.cfg_mix_layer          = get_attr_kwargs(cfg, 'cfg_mix_layer', **kwargs)
    self.cfg_ops                = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)
    self.cell_op_idx            = get_attr_kwargs(cfg, 'cell_op_idx', default=None, **kwargs)

    self.num_edges = (self.n_nodes + 1) * self.n_nodes // 2
    self.cfg_keys = list(self.cfg_ops.keys())

    self.in_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                             kernel_size=1, stride=1, padding=0)
    self.out_conv = nn.Conv2d(in_channels=self.out_channels*self.n_nodes, out_channels=self.out_channels,
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
          op = build_d2layer(self.cfg_mix_layer, **{**kwargs, "in_channels": self.out_channels,
                                                    "out_channels"         : self.out_channels,
                                                    "cfg_ops"              : cfg_ops})
        else:
          op = build_d2layer(self.cfg_mix_layer, **{**kwargs, "in_channels": self.out_channels,
                                                    "out_channels"         : self.out_channels,
                                                    "cfg_ops"              : self.cfg_ops})
        self.dag[i].append(op)
    pass

  def forward(self, x, batched_arcs):
    x = self.in_conv(x)

    states = [x, ]
    idx_start = 0
    for edges in self.dag:
      edges_arcs = batched_arcs[:, idx_start:idx_start+len(edges)]
      idx_start += len(edges)
      s_cur = sum(edge(s, edges_arcs[:, i]) for i, (edge, s) in enumerate(zip(edges, states)))

      states.append(s_cur)

    cat_out = torch.cat(states[1:], dim=1)
    out = self.out_conv(cat_out)
    return out

  def test_case(self):
    bs = 2
    num_ops = len(self.cfg_ops)
    x = torch.randn(bs, 3, 8, 8).cuda()
    batched_arcs = torch.tensor([
      [0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 1],
    ]).cuda()
    out = self(x, batched_arcs)
    return out

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "DenseCell"
      n_nodes: 3
      cfg_mix_layer:
        name: "MixedLayer"
      cfg_ops:
        Identity:
          name: "Identity"
        Conv2d_3x3:
          name: "ActConv2d"
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
