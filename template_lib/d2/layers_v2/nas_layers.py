import math
import yaml
from easydict import EasyDict
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.layers_v2.build import D2LAYERv2_REGISTRY, build_d2layer_v2
from template_lib.utils import get_attr_kwargs
from template_lib.v2.config import update_config


@D2LAYERv2_REGISTRY.register()
class DenseBlockWithArc(nn.Module):

  def __init__(self, cfg, **kwargs):
    super().__init__()

    cfg = self.update_cfg(cfg)

    # fmt: off
    self.in_channels           = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.n_nodes               = get_attr_kwargs(cfg, 'n_nodes', **kwargs) # include input node
    self.cfg_mix_layer         = get_attr_kwargs(cfg, 'cfg_mix_layer', **kwargs)
    self.cfg_ops               = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)
    self.cell_op_idx           = get_attr_kwargs(cfg, 'cell_op_idx', default=None, **kwargs)
    # fmt: on

    self.num_edges = self.get_edges(self.n_nodes)
    if self.cell_op_idx is not None:
      assert self.num_edges == len(self.cell_op_idx)

    self.cfg_keys = list(self.cfg_ops.keys())
    self.out_channels = self.in_channels

    # generate dag
    edge_idx = 0
    self.dag = nn.ModuleList()
    for i in range(1, self.n_nodes):
      self.dag.append(nn.ModuleList())
      for j in range(i):
        if self.cell_op_idx is not None:
          op_key = self.cfg_keys[self.cell_op_idx[edge_idx]]
          cfg_ops = EasyDict({op_key: self.cfg_ops[op_key]})
          edge_idx += 1
          op = build_d2layer_v2(self.cfg_mix_layer, **{**kwargs, "in_channels": self.in_channels,
                                                       "out_channels"         : self.out_channels,
                                                       "cfg_ops"              : cfg_ops})
        else:
          op = build_d2layer_v2(self.cfg_mix_layer, **{**kwargs, "in_channels": self.in_channels,
                                                       "out_channels"         : self.out_channels,
                                                       "cfg_ops"              : self.cfg_ops})
        self.dag[i-1].append(op)
    pass

  def forward(self, x, batched_arcs=None, **kwargs):
    if self.cell_op_idx is not None and batched_arcs is None:
      batched_arcs = torch.zeros((x.shape[0], self.num_edges), dtype=torch.int64).cuda()

    states = [x, ]
    idx_start = 0
    for edges in self.dag:
      edges_arcs = batched_arcs[:, idx_start:idx_start + len(edges)]
      idx_start += len(edges)
      s_cur = sum(edge(s, edges_arcs[:, i], **kwargs) for i, (edge, s) in enumerate(zip(edges, states)))

      states.append(s_cur)
    x = states[-1]
    return x

  @staticmethod
  def get_edges(n_nodes):
    num_edges = (n_nodes - 1) * n_nodes // 2
    return num_edges

  @staticmethod
  def test_case():
    from template_lib.d2.layers_v2.convs import Conv2d

    cfg_str = """
              name: "DenseBlockWithArc"
              update_cfg: true
              n_nodes: 4
              in_channels: 16
              """
    cfg = EasyDict(yaml.safe_load(cfg_str))

    op = build_d2layer_v2(cfg)
    op.cuda()

    bs = 2
    num_ops = len(op.cfg_ops)
    x = torch.randn(bs, 16, 8, 8).cuda().requires_grad_(True)
    batched_arcs = torch.tensor([
      [0, 0, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1],
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
      name: ""
      n_nodes: 4
      cfg_mix_layer:
        name: "MixedLayerWithArc"
      cfg_ops:        
        Conv2d_1x1:
          name: "Conv2d"
          kernel_size: 1
          padding: 0
        Conv2d_3x3:
          name: "Conv2d"
          kernel_size: 3
          padding: 1

      """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg


@D2LAYERv2_REGISTRY.register()
class MixedLayerWithArc(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(MixedLayerWithArc, self).__init__()

    # fmt: off
    self.in_channels             = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels            = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.cfg_ops                 = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)
    # fmt: on

    self.num_branch = len(self.cfg_ops)

    self.branches = nn.ModuleList()
    for name, cfg_op in self.cfg_ops.items():
      branch = build_d2layer_v2(
        cfg_op, **{**kwargs, 'in_channels': self.in_channels, 'out_channels': self.out_channels})
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
