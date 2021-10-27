from einops import rearrange
import logging
import functools
import numpy as np

import torch
import torch.optim as optim
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from template_lib.d2.models_v2 import MODEL_REGISTRY, build_model
from template_lib.utils import get_attr_kwargs, get_dict_str

from .conditional_layer_norm import CLN
from .reduction_attention import ReductionAttention
from exp.omniGAN.BigGAN_omni import Discriminator, G_D


def split_last(x, shape):
  "split the last dimension to given shape"
  shape = list(shape)
  assert shape.count(-1) <= 1
  if -1 in shape:
    shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
  return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
  "merge the last n_dims to a dimension"
  s = x.size()
  assert n_dims > 1 and n_dims < len(s)
  return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
  """Multi-Headed Dot Product Attention"""

  def __init__(self, dim, num_heads, dropout):
    super().__init__()
    self.proj_q = nn.Linear(dim, dim)
    self.proj_k = nn.Linear(dim, dim)
    self.proj_v = nn.Linear(dim, dim)
    self.drop = nn.Dropout(dropout)
    self.n_heads = num_heads
    self.scores = None  # for visualization

  def forward(self, x, mask):
    """
    x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(in_dim))
    mask : (B(batch_size) x S(seq_len))
    * split D(in_dim) into (H(n_heads), W(width of head)) ; D = H * W
    """
    # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
    q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
    q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
    # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
    scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
    if mask is not None:
      mask = mask[:, None, None, :].float()
      scores -= 10000.0 * (1.0 - mask)
    scores = self.drop(F.softmax(scores, dim=-1))
    # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
    h = (scores @ v).transpose(1, 2).contiguous()
    # -merge-> (B, S, D)
    h = merge_last(h, 2)
    self.scores = scores
    return h


class PositionWiseFeedForward(nn.Module):
  """FeedForward Neural Networks for each position"""

  def __init__(self, dim, ff_dim, use_sn):
    super().__init__()
    self.fc1 = nn.Linear(dim, ff_dim)
    self.fc2 = nn.Linear(ff_dim, dim)
    if use_sn:
      self.fc1 = nn.utils.spectral_norm(self.fc1)
      self.fc2 = nn.utils.spectral_norm(self.fc2)
    self.act_gelu = nn.GELU()
    pass

  def forward(self, x):
    # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
    h = self.fc1(x)
    h = self.act_gelu(h)
    h = self.fc2(h)
    return h


class Block(nn.Module):
  """Transformer Block"""

  def __init__(self,
               in_dim,
               out_dim,
               num_heads,
               ff_dim,
               dropout,
               which_norm,
               sr_ratio,
               height,
               width,
               use_sn=False,
               eps=1e-6):
    super().__init__()

    self.height = height
    self.width = width

    self.cln_1 = which_norm(in_dim, eps=eps)
    self.r_attn = ReductionAttention(dim=in_dim, num_heads=num_heads, sr_ratio=sr_ratio)
    # self.attn = MultiHeadedSelfAttention(in_dim, num_heads, dropout)
    self.r_atten_fc = nn.Linear(in_dim, in_dim)
    if use_sn:
      self.r_atten_fc = nn.utils.spectral_norm(self.r_atten_fc)
    self.drop1 = nn.Dropout(dropout)

    self.cln_2 = which_norm(out_dim, eps=eps)
    self.pwff = PositionWiseFeedForward(out_dim, ff_dim, use_sn=use_sn)
    self.drop2 = nn.Dropout(dropout)
    pass

  def forward(self, x, y):

    h = self.cln_1(x, y)
    h = self.r_attn(h, H=self.height, W=self.width)
    h = self.r_atten_fc(h)
    h = self.drop1(h)
    x = x + h

    h = self.cln_2(x, y)
    h = self.pwff(h)
    h = self.drop2(h)
    x = x + h
    return x


class Transformer(nn.Module):
  """Transformer with Self-Attentive Blocks"""

  def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
    super().__init__()
    self.blocks = nn.ModuleList([
      Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

  def forward(self, x, mask=None):
    for block in self.blocks:
      x = block(x, mask)
    return x


class AddPositionEmbedding(nn.Module):
  def __init__(self,
               N,
               dim):
    super(AddPositionEmbedding, self).__init__()
    self.pos_emb = nn.Parameter(torch.zeros(1, N, dim))
    pass

  def forward(self, x):
    out = x + self.pos_emb
    return out


@MODEL_REGISTRY.register(name_prefix=__name__)
class Generator(nn.Module):

  def __init__(self,
               cfg=None,
               no_optim=False,
               G_lr=1e-4,
               G_B1=0.0,
               G_B2=0.999,
               adam_eps=1e-8,
               verbose=True,
               **kwargs):
    super(Generator, self).__init__()
    kwargs['tl_ret_kwargs'] = {}
    # fmt: off
    self.dim_z              = get_attr_kwargs(cfg, 'dim_z', default=120, **kwargs)
    self.n_classes          = get_attr_kwargs(cfg, 'n_classes', default=1000, **kwargs)
    self.G_shared           = get_attr_kwargs(cfg, 'G_shared', default=True, **kwargs)
    self.shared_dim         = get_attr_kwargs(cfg, 'shared_dim', default=128, **kwargs)
    self.hier               = get_attr_kwargs(cfg, 'hier', default=True, **kwargs)
    self.bottom_width       = get_attr_kwargs(cfg, 'bottom_width', default=4, **kwargs)
    self.channel            = get_attr_kwargs(cfg, 'channel', default=3, **kwargs)
    self.use_sn             = get_attr_kwargs(cfg, 'use_sn', default=True, **kwargs)
    self.num_heads          = get_attr_kwargs(cfg, 'num_heads', default=[], **kwargs)
    self.sr_ratios          = get_attr_kwargs(cfg, 'sr_ratios', default=[], **kwargs)
    self.block_repeat       = get_attr_kwargs(cfg, 'block_repeat', default=[], **kwargs)
    self.embed_dims         = get_attr_kwargs(cfg, 'embed_dims', default=[], **kwargs)
    self.ff_dim_mul         = get_attr_kwargs(cfg, 'ff_dim_mul', default=4, **kwargs)
    self.dropout            = get_attr_kwargs(cfg, 'dropout', default=0, **kwargs)
    self.weight_decay       = get_attr_kwargs(cfg, 'weight_decay', default=0.001, **kwargs)
    inr_net_cfg             = get_attr_kwargs(cfg, 'inr_net_cfg', **kwargs)

    # fmt: on
    if verbose:
      logging.getLogger('tl').info(f"  {self.__class__.__name__} kwargs: {get_dict_str(kwargs['tl_ret_kwargs'], use_pprint=False)}")

    self.fp16 = False

    if self.hier:
      # Number of places z slots into
      self.num_slots = len(self.num_heads) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      self.dim_z = self.z_chunk_size * self.num_slots
    else:
      self.num_slots = 1
      self.z_chunk_size = 0


    bn_linear = (functools.partial(nn.Linear, bias=False) if self.G_shared else nn.Embedding)
    self.which_norm = functools.partial(
      CLN, which_linear=bn_linear, spectral_norm=self.use_sn,
      style_dim=(self.shared_dim + self.z_chunk_size if self.G_shared else self.n_classes))
    self.shared = nn.Embedding(self.n_classes, self.shared_dim) if self.G_shared else nn.Identity()

    self.linear = nn.Linear(self.dim_z // self.num_slots, self.channel * (self.bottom_width ** 2))
    if self.use_sn:
      self.linear = nn.utils.spectral_norm(self.linear)

    self.module_names = []

    in_dim = self.channel
    height, width = self.bottom_width, self.bottom_width
    for idx_s in range(len(self.block_repeat)):
      out_dim = self.embed_dims[idx_s]

      name = f"stage{idx_s}_up"
      upsample = nn.UpsamplingNearest2d(scale_factor=2)
      self.add_module(name, upsample)
      self.module_names.append(name)
      height = height * 2
      width = width * 2

      name = f"stage{idx_s}_conv"
      conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1)
      if self.use_sn:
        conv = nn.utils.spectral_norm(conv)
      self.add_module(name, conv)
      self.module_names.append(name)

      name = f"stage{idx_s}_pos_embed"
      pos_emb = AddPositionEmbedding(N=height*width, dim=self.embed_dims[idx_s])
      self.add_module(name, pos_emb)

      in_dim = out_dim

      sr_ratio = self.sr_ratios[idx_s]
      for idx_b in range(self.block_repeat[idx_s]):
        name = f"stage{idx_s}_b{idx_b}"
        block = Block(in_dim=out_dim, out_dim=out_dim, num_heads=self.num_heads[idx_s],
                      ff_dim=out_dim * self.ff_dim_mul, dropout=self.dropout,
                      which_norm=self.which_norm, use_sn=self.use_sn,
                      sr_ratio=sr_ratio,
                      height=height, width=width)
        self.add_module(name, block)
        self.module_names.append(name)

    self.inr_net = build_model(
      cfg=inr_net_cfg, kwargs_priority=True, in_dim=self.embed_dims[-1],
      spectral_norm=self.use_sn, verbose=verbose)

    self.out_layer = nn.Tanh()

    # If this is an EMA copy, no need for an optim, so just return now
    if no_optim:
      return
    self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps

    self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                            betas=(self.B1, self.B2), weight_decay=self.weight_decay,
                            eps=self.adam_eps)
    pass

  def forward(self, z, y):
    # If hierarchical, concatenate zs and ys
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * self.num_slots

    # First linear layer
    h = self.linear(z)
    h = rearrange(h, "b (h w c) -> b (h w) c", h=self.bottom_width, w=self.bottom_width)

    height, width = self.bottom_width, self.bottom_width
    for idx_s in range(len(self.block_repeat)):
      h = rearrange(h, "b (h w) c -> b c h w", h=height)

      name = f"stage{idx_s}_up"
      h = getattr(self, name)(h)
      height = height * 2
      width = width * 2

      name = f"stage{idx_s}_conv"
      h = getattr(self, name)(h)

      h = rearrange(h, "b c h w -> b (h w) c", h=height, w=width)
      name = f"stage{idx_s}_pos_embed"
      h = getattr(self, name)(h)

      for idx_b in range(self.block_repeat[idx_s]):
        name = f"stage{idx_s}_b{idx_b}"
        h = getattr(self, name)(h, ys[idx_s])

    h = rearrange(h, "b (h w) c -> b c h w", h=height)

    h = self.inr_net(h, shape=(height, width))

    out = self.out_layer(h)
    return out


