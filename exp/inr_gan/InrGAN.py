import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

import layers
from sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d

from exp.inr_gan.models.build import build_model
from exp.comm.layers.build import build_layer

from template_lib.v2.config_cfgnode import global_cfg
from template_lib.utils import get_attr_kwargs, print_number_params


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch[512] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
               'upsample' : [True] * 7,
               'resolution' : [8, 16, 32, 64, 128, 256, 512],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,10)}}
  arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
  arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
  arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

  return arch

class Generator(nn.Module):
  def __init__(self, G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
               G_mixed_precision=False, G_init='ortho', skip_init=False, no_optim=False,
               **kwargs):
    super(Generator, self).__init__()

    cfg = kwargs['cfg']
    # fmt: off
    hyper_net_cfg                = get_attr_kwargs(cfg, 'hyper_net_cfg', **kwargs)
    inr_block_cfg                = get_attr_kwargs(cfg, 'inr_block_cfg', **kwargs)
    in_channels                  = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    out_channels                 = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    num_classes                  = get_attr_kwargs(cfg, 'num_classes', **kwargs)
    embedding_dim                = get_attr_kwargs(cfg, 'embedding_dim', **kwargs)
    w_dim                        = get_attr_kwargs(cfg, 'w_dim', **kwargs)
    out_layer_cfg                = get_attr_kwargs(cfg, 'out_layer_cfg', **kwargs)
    self.img_size                = get_attr_kwargs(cfg, 'img_size', **kwargs)

    # fmt: on
    assert len(in_channels) == len(out_channels)
    self.dim_z = hyper_net_cfg.in_dim
    self.init = G_init

    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    self.shared = nn.Embedding(num_classes, embedding_dim)
    # self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared
    #                 else layers.identity())

    num_blocks = len(out_channels)
    self.hyper_net = build_model(cfg=hyper_net_cfg, kwargs_priority=True,
                                 hidden_dim=w_dim*num_blocks, out_dim=w_dim*num_blocks)
    print_number_params({'hyper_net': self.hyper_net})

    self.blocks = []
    for index in range(len(out_channels)):
      block = build_model(inr_block_cfg, kwargs_priority=True, in_features=in_channels[index],
                          out_features=out_channels[index])
      self.blocks += [[block]]

    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    print_number_params({'inr_net': self.blocks})

    self.output_layer = build_layer(out_layer_cfg, in_features=out_channels[-1])

    # Initialize weights. Optionally skip init for testing.
    if not skip_init:
      self.init_weights()

    cfg = global_cfg.Generator
    # fmt: off
    weight_decay                = cfg.weight_decay
    # fmt: on
    # Set up optimizer
    # If this is an EMA copy, no need for an optim, so just return now
    if no_optim:
      return
    self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
    if G_mixed_precision:
      print('Using fp16 adam in G...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=weight_decay,
                           eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=weight_decay,
                           eps=self.adam_eps)

    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
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
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
  def forward(self, z, y):

    w = self.hyper_net(z)

    ws = torch.split(w, w.shape[-1]//len(self.blocks), 1)
    ys = [torch.cat([y, item], 1) for item in ws]

    coord = self.make_coord(self.img_size).cuda().unsqueeze(0)
    cell = self.make_cell(coord_shape=coord.shape, h=self.img_size[0], w=self.img_size[1]).cuda()

    h = torch.cat([coord, cell], dim=-1)
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        h = block(h, ys[index])
        
    out = torch.tanh(self.output_layer(h))
    out = out.view(out.shape[0], self.img_size[0], self.img_size[1], out.shape[-1])
    out = out.permute(0, 3, 1, 2)
    return out

  @staticmethod
  def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
      if ranges is None:
        v0, v1 = -1, 1
      else:
        v0, v1 = ranges[i]
      r = (v1 - v0) / (2 * n)
      seq = v0 + r + (2 * r) * torch.arange(n).float()
      coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
      ret = ret.view(-1, ret.shape[-1])
    return ret

  @staticmethod
  def make_cell(coord_shape, h, w):
    cell = torch.ones(*coord_shape)
    cell[..., 0] *= 2 / h
    cell[..., 1] *= 2 / w
    return cell


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64',ksize='333333', dilation='111111'):
  arch = {}
  arch[256] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[128] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[64]  = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,7)}}
  arch[32]  = {'in_channels' :  [3] + [item * ch for item in [4, 4, 4]],
               'out_channels' : [item * ch for item in [4, 4, 4, 4]],
               'downsample' : [True, True, False, False],
               'resolution' : [16, 16, 16, 16],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,6)}}
  return arch

class Discriminator(nn.Module):

  def __init__(self, D_ch=64, D_wide=True, resolution=128,
               D_kernel_size=3, D_attn='64', n_classes=1000,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
               D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
               SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
               D_init='ortho', skip_init=False, D_param='SN', **kwargs):
    super(Discriminator, self).__init__()
    # Width multiplier
    self.ch = D_ch
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = D_wide
    # Resolution
    self.resolution = resolution
    # Kernel size
    self.kernel_size = D_kernel_size
    # Attention?
    self.attention = D_attn
    # Number of classes
    self.n_classes = n_classes
    # Activation
    self.activation = D_activation
    # Initialization style
    self.init = D_init
    # Parameterization style
    self.D_param = D_param
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # Fp16?
    self.fp16 = D_fp16
    # Architecture
    self.arch = D_arch(self.ch, self.attention)[resolution]

    # Which convs, batchnorms, and linear layers to use
    # No option to turn off SN in D right now
    if self.D_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_embedding = functools.partial(layers.SNEmbedding,
                              num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                              eps=self.SN_eps)
    # Prepare model
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       wide=self.D_wide,
                       activation=self.activation,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                             self.which_conv)]
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    cfg = global_cfg.Discriminator
    # fmt: off
    output_dim              = cfg.output_dim
    weight_decay            = cfg.weight_decay
    # fmt: on
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output
    self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

    # Initialize weights
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
    if D_mixed_precision:
      print('Using fp16 adam in D...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=weight_decay, eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=weight_decay, eps=self.adam_eps)
    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
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
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x, y=None):
    # Stick x into h for cleaner for loops without flow control
    h = x
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3])
    out = self.linear(h)
    return out

# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
  def __init__(self, G, D):
    super(G_D, self).__init__()
    self.G = G
    self.D = D

  def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
              split_D=False):              
    # If training G, enable grad tape
    with torch.set_grad_enabled(train_G):
      # Get Generator output given noise
      G_z = self.G(z, self.G.shared(gy))
      # Cast as necessary
      if self.G.fp16 and not self.D.fp16:
        G_z = G_z.float()
      if self.D.fp16 and not self.G.fp16:
        G_z = G_z.half()
    # Split_D means to run D once with real data and once with fake,
    # rather than concatenating along the batch dimension.
    if split_D:
      D_fake = self.D(G_z, gy)
      if x is not None:
        D_real = self.D(x, dy)
        return D_fake, D_real
      else:
        if return_G_z:
          return D_fake, G_z
        else:
          return D_fake
    # If real data is provided, concatenate it with the Generator's output
    # along the batch dimension for improved efficiency.
    else:
      D_input = torch.cat([G_z, x], 0) if x is not None else G_z
      D_class = torch.cat([gy, dy], 0) if dy is not None else gy
      # Get Discriminator output
      D_out = self.D(D_input, D_class)
      if x is not None:
        return torch.split(D_out, [G_z.shape[0], x.shape[0]]) # D_fake, D_real
      else:
        if return_G_z:
          return D_out, G_z
        else:
          return D_out
