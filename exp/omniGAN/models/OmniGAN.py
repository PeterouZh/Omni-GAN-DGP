import logging
import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

from template_lib.v2.config_cfgnode import global_cfg
from template_lib.d2.models_v2 import MODEL_REGISTRY
from template_lib.utils import get_dict_str, get_attr_kwargs

import layers


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


@MODEL_REGISTRY.register(name=f"{__name__}.Generator")
class Generator(nn.Module):
  def __init__(self,
               cfg,
               G_activation=nn.ReLU(inplace=True),
               verbose=True,
               **kwargs):
    super(Generator, self).__init__()
    kwargs['tl_ret_kwargs'] = {}
    # fmt: off
    # Channel width mulitplier
    self.ch                         = get_attr_kwargs(cfg, 'G_ch', default=96, **kwargs)
    # Dimensionality of the latent space
    self.dim_z                      = get_attr_kwargs(cfg, 'dim_z', default=120, **kwargs)
    # The initial spatial dimensions
    self.bottom_width               = get_attr_kwargs(cfg, 'bottom_width', default=4, **kwargs)
    # Resolution of the output
    self.resolution                 = get_attr_kwargs(cfg, 'resolution', default=128, **kwargs)
    # Kernel size?
    self.kernel_size                = get_attr_kwargs(cfg, 'G_kernel_size', default=3, **kwargs)
    # Attention?
    self.attention                  = get_attr_kwargs(cfg, 'G_attn', default='64', **kwargs)
    # number of classes, for use in categorical conditional generation
    self.n_classes                  = get_attr_kwargs(cfg, 'n_classes', default=1000, **kwargs)
    # Use shared embeddings?
    self.G_shared                   = get_attr_kwargs(cfg, 'G_shared', default=True, **kwargs)
    # Dimensionality of the shared embedding? Unused if not using G_shared
    shared_dim                      = get_attr_kwargs(cfg, 'shared_dim', default=128, **kwargs)
    self.shared_dim = shared_dim if shared_dim > 0 else self.dim_z
    # Hierarchical latent space?
    self.hier                       = get_attr_kwargs(cfg, 'hier', default=True, **kwargs)
    # Cross replica batchnorm?
    self.cross_replica              = get_attr_kwargs(cfg, 'cross_replica', default=False, **kwargs)
    # Use my batchnorm?
    self.mybn                       = get_attr_kwargs(cfg, 'mybn', default=False, **kwargs)
    # nonlinearity for residual blocks
    self.activation = G_activation
    # Initialization style
    self.init                       = get_attr_kwargs(cfg, 'G_init', default='ortho', **kwargs)
    # Parameterization style
    self.G_param                    = get_attr_kwargs(cfg, 'G_param', default='SN', **kwargs)
    # Normalization style
    self.norm_style                 = get_attr_kwargs(cfg, 'norm_style', default='bn', **kwargs)
    # Epsilon for BatchNorm?
    self.BN_eps                     = get_attr_kwargs(cfg, 'BN_eps', default=1e-5, **kwargs)
    # Epsilon for Spectral Norm?
    self.SN_eps                     = get_attr_kwargs(cfg, 'SN_eps', default=1e-6, **kwargs)
    # fp16?
    self.fp16                       = get_attr_kwargs(cfg, 'G_fp16', default=False, **kwargs)

    num_G_SVs                       = get_attr_kwargs(cfg, 'num_G_SVs', default=1, **kwargs)
    num_G_SV_itrs                   = get_attr_kwargs(cfg, 'num_G_SV_itrs', default=1, **kwargs)
    G_lr                            = get_attr_kwargs(cfg, 'G_lr', default=1e-4, **kwargs)
    G_B1                            = get_attr_kwargs(cfg, 'G_B1', default=0., **kwargs)
    G_B2                            = get_attr_kwargs(cfg, 'G_B2', default=0.999, **kwargs)
    weight_decay                    = get_attr_kwargs(cfg, 'weight_decay', default=0.001, **kwargs)
    adam_eps                        = get_attr_kwargs(cfg, 'adam_eps', default=1e-6, **kwargs)
    G_mixed_precision               = get_attr_kwargs(cfg, 'G_mixed_precision', default=False, **kwargs)
    skip_init                       = get_attr_kwargs(cfg, 'skip_init', default=False, **kwargs)
    no_optim                        = get_attr_kwargs(cfg, 'no_optim', default=False, **kwargs)

    # fmt: on
    if verbose:
      logging.getLogger('tl').info(f"  G kwargs: \n{get_dict_str(kwargs['tl_ret_kwargs'])}")

    # Architecture dict
    self.arch = G_arch(self.ch, self.attention)[self.resolution]

    # If using hierarchical latents, adjust z
    if self.hier:
      # Number of places z slots into
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      self.dim_z = self.z_chunk_size *  self.num_slots
    else:
      self.num_slots = 1
      self.z_chunk_size = 0

    # Which convs, batchnorms, and linear layers to use
    if self.G_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear
      
    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    self.which_embedding = nn.Embedding
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                 else self.which_embedding)
    self.which_bn = functools.partial(layers.ccbn,
                          which_linear=bn_linear,
                          cross_replica=self.cross_replica,
                          mybn=self.mybn,
                          input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                      else self.n_classes),
                          norm_style=self.norm_style,
                          eps=self.BN_eps)


    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
    self.shared = (self.which_embedding(self.n_classes, self.shared_dim) if self.G_shared
                    else layers.identity())
    # First linear layer
    self.linear = self.which_linear(self.dim_z // self.num_slots,
                                    self.arch['in_channels'][0] * (self.bottom_width **2))

    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    # while the inner loop is over a given block
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             which_conv=self.which_conv,
                             which_bn=self.which_bn,
                             activation=self.activation,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] else None))]]

      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here
    self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                cross_replica=self.cross_replica,
                                                mybn=self.mybn),
                                    self.activation,
                                    self.which_conv(self.arch['out_channels'][-1], 3))

    # Initialize weights. Optionally skip init for testing.
    if not skip_init:
      self.init_weights()

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
    pass

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
    # If hierarchical, concatenate zs and ys
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks)
      
    # First linear layer
    h = self.linear(z)
    # Reshape
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
    
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        h = block(h, ys[index])
        
    # Apply batchnorm-relu-conv-tanh at output
    return torch.tanh(self.output_layer(h))


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


@MODEL_REGISTRY.register(name=f"{__name__}.Discriminator")
class Discriminator(nn.Module):
  def __init__(self, cfg,
               D_activation=nn.ReLU(inplace=True),
               verbose=True,
               **kwargs):
    super(Discriminator, self).__init__()
    kwargs['tl_ret_kwargs'] = {}
    # fmt: off
    # Width multiplier
    self.ch                            = get_attr_kwargs(cfg, 'D_ch', default=96, **kwargs)
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide                        = get_attr_kwargs(cfg, 'D_wide', default=True, **kwargs)
    # Resolution
    self.resolution                    = get_attr_kwargs(cfg, 'resolution', default=128, **kwargs)
    # Kernel size
    self.kernel_size                   = get_attr_kwargs(cfg, 'D_kernel_size', default=3, **kwargs)
    # Attention?
    self.attention                     = get_attr_kwargs(cfg, 'D_attn', default='64', **kwargs)
    # Number of classes
    self.n_classes                     = get_attr_kwargs(cfg, 'n_classes', default=1000, **kwargs)
    # Activation
    self.activation = D_activation
    # Initialization style
    self.init                          = get_attr_kwargs(cfg, 'D_init', default='ortho', **kwargs)
    # Parameterization style
    self.D_param                       = get_attr_kwargs(cfg, 'D_param', default='SN', **kwargs)
    # Epsilon for Spectral Norm?
    self.SN_eps                        = get_attr_kwargs(cfg, 'SN_eps', default=1e-6, **kwargs)
    # Fp16?
    self.fp16                          = get_attr_kwargs(cfg, 'D_fp16', default=False, **kwargs)

    num_D_SVs                          = get_attr_kwargs(cfg, 'num_D_SVs', default=1, **kwargs)
    num_D_SV_itrs                      = get_attr_kwargs(cfg, 'num_D_SV_itrs', default=1, **kwargs)
    D_lr                               = get_attr_kwargs(cfg, 'D_lr', default=4e-4, **kwargs)
    D_B1                               = get_attr_kwargs(cfg, 'D_B1', default=0., **kwargs)
    D_B2                               = get_attr_kwargs(cfg, 'D_B2', default=0.999, **kwargs)
    weight_decay                       = get_attr_kwargs(cfg, 'weight_decay', default=1e-5, **kwargs)
    adam_eps                           = get_attr_kwargs(cfg, 'adam_eps', default=1e-6, **kwargs)
    D_mixed_precision                  = get_attr_kwargs(cfg, 'D_mixed_precision', default=False, **kwargs)
    skip_init                          = get_attr_kwargs(cfg, 'skip_init', default=False, **kwargs)
    output_dim                         = get_attr_kwargs(cfg, 'output_dim', default=1002, **kwargs)

    # fmt: on
    if verbose:
      logging.getLogger('tl').info(f"  D kwargs: \n{get_dict_str(kwargs['tl_ret_kwargs'])}")

    # Architecture
    self.arch = D_arch(self.ch, self.attention)[self.resolution]

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
