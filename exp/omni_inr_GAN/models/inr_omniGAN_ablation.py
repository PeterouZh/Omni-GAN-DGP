import logging
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F

from template_lib.utils import get_attr_kwargs, get_dict_str
from template_lib.d2.models_v2 import MODEL_REGISTRY, build_model

import layers
from exp.omni_inr_GAN.comm.utils import make_coord, make_cell
from .inr_omniGAN import G_arch


@MODEL_REGISTRY.register(name=f"{__name__}.Generator")
class Generator(nn.Module):
  def __init__(self, cfg, G_activation=nn.ReLU(inplace=True), verbose=True, **kwargs):
    super(Generator, self).__init__()
    kwargs['tl_ret_kwargs'] = {}

    # fmt: off
    # Channel width mulitplier
    self.ch                 = get_attr_kwargs(cfg, 'G_ch', default=64, **kwargs)
    # Dimensionality of the latent space
    self.dim_z              = get_attr_kwargs(cfg, 'dim_z', default=128, **kwargs)
    # The initial spatial dimensions
    self.bottom_width       = get_attr_kwargs(cfg, 'bottom_width', default=4, **kwargs)
    # Resolution of the output
    self.resolution         = get_attr_kwargs(cfg, 'resolution', default=128, **kwargs)
    # Kernel size?
    self.kernel_size        = get_attr_kwargs(cfg, 'G_kernel_size', default=3, **kwargs)
    # Attention?
    self.attention          = get_attr_kwargs(cfg, 'G_attn', default='64', **kwargs)
    # number of classes, for use in categorical conditional generation
    self.n_classes          = get_attr_kwargs(cfg, 'n_classes', default=1000, **kwargs)
    # Use shared embeddings?
    self.G_shared           = get_attr_kwargs(cfg, 'G_shared', default=True, **kwargs)
    # Dimensionality of the shared embedding? Unused if not using G_shared
    shared_dim              = get_attr_kwargs(cfg, 'shared_dim', default=0, **kwargs)
    self.shared_dim = shared_dim if shared_dim > 0 else self.dim_z
    # Hierarchical latent space?
    self.hier               = get_attr_kwargs(cfg, 'hier', default=False, **kwargs)
    # Cross replica batchnorm?
    self.cross_replica      = get_attr_kwargs(cfg, 'cross_replica', default=False, **kwargs)
    # Use my batchnorm?
    self.mybn               = get_attr_kwargs(cfg, 'mybn', default=False, **kwargs)
    # nonlinearity for residual blocks
    self.activation         = G_activation
    # Initialization style
    self.init               = get_attr_kwargs(cfg, 'G_init', default='ortho', **kwargs)
    # Parameterization style
    self.G_param            = get_attr_kwargs(cfg, 'G_param', default='SN', **kwargs)
    # Normalization style
    self.norm_style         = get_attr_kwargs(cfg, 'norm_style', default='bn', **kwargs)
    # Epsilon for BatchNorm?
    self.BN_eps             = get_attr_kwargs(cfg, 'BN_eps', default=1e-5, **kwargs)
    # Epsilon for Spectral Norm?
    self.SN_eps             = get_attr_kwargs(cfg, 'SN_eps', default=1e-12, **kwargs)
    # fp16?
    self.fp16               = get_attr_kwargs(cfg, 'G_fp16', default=False, **kwargs)
    # Architecture dict
    self.arch = G_arch(self.ch, self.attention)[self.resolution]

    num_G_SVs               = get_attr_kwargs(cfg, 'num_G_SVs', default=1, **kwargs)
    num_G_SV_itrs           = get_attr_kwargs(cfg, 'num_G_SV_itrs', default=1, **kwargs)
    G_mixed_precision       = get_attr_kwargs(cfg, 'G_mixed_precision', default=False, **kwargs)
    skip_init               = get_attr_kwargs(cfg, 'skip_init', default=False, **kwargs)
    no_optim                = get_attr_kwargs(cfg, 'no_optim', default=False, **kwargs)

    inr_net_cfg             = get_attr_kwargs(cfg, 'inr_net_cfg', **kwargs)
    optim_cfg               = get_attr_kwargs(cfg, 'optim_cfg', default={}, **kwargs)

    # fmt: on
    if verbose:
      logging.getLogger('tl').info(f"  G kwargs: \n{get_dict_str(kwargs['tl_ret_kwargs'])}")

    # If using hierarchical latents, adjust z
    if self.hier:
      # Number of places z slots into
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      self.dim_z = self.z_chunk_size * self.num_slots
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
                                    self.arch['in_channels'][0] * (self.bottom_width ** 2))

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
    # self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
    #                                             cross_replica=self.cross_replica,
    #                                             mybn=self.mybn),
    #                                 self.activation,
    #                                 self.which_conv(self.arch['out_channels'][-1], 3))

    # Initialize weights. Optionally skip init for testing.
    if not skip_init:
      self.init_weights()

    in_dim = self._get_inr_net_in_dim()
    self.inr_net = build_model(cfg=inr_net_cfg, which_linear=self.which_linear, in_dim=in_dim, out_dim=3,
                               verbose=verbose)



    # Set up optimizer
    # If this is an EMA copy, no need for an optim, so just return now
    if no_optim:
      return
    # self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
    if G_mixed_precision:
      print('Using fp16 adam in G...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                betas=(self.B1, self.B2), weight_decay=0,
                                eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), **optim_cfg)

    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0
    pass

  def _get_inr_net_in_dim(self):
    # unfold + coords + cell
    in_dim = self.arch['out_channels'][-1] * 9 + 2 + 2
    return in_dim

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
  def forward(self, z, y, shape=None):
    if shape is None:
      shape = [self.resolution, ] * 2
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

    out = self._inr_net_forward(h=h, shape=shape)
    # Apply batchnorm-relu-conv-tanh at output
    return out

  def _inr_net_forward(self, h, shape):
    coord = make_coord(shape=shape, flatten=True).cuda()
    coord = coord.unsqueeze(0).expand(h.shape[0], *coord.shape[-2:])
    cell = make_cell(shape).cuda()
    cell = cell.unsqueeze(0).expand(h.shape[0], *cell.shape[-2:])

    feat = F.unfold(h, 3, padding=1).view(h.shape[0], h.shape[1] * 9, h.shape[2], h.shape[3])

    sampled_feat = F.grid_sample(
      feat, coord.flip(-1).unsqueeze(1), mode='bilinear', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

    cell = cell.clone()
    cell[:, :, 0] *= feat.shape[-2]
    cell[:, :, 1] *= feat.shape[-1]
    inp = torch.cat([sampled_feat, coord, cell], dim=-1)

    bs, q = inp.shape[:2]
    pred = self.inr_net(inp.view(bs * q, -1)).view(bs, q, -1)
    pred = pred.permute(0, 2, 1).view(pred.shape[0], -1, *shape)

    out = torch.tanh(pred)
    return out


@MODEL_REGISTRY.register(name=f"{__name__}.Generator_no_unfold")
class Generator_no_unfold(Generator):
  def _get_inr_net_in_dim(self):
    # coords + cell
    in_dim = self.arch['out_channels'][-1] + 2 + 2
    return in_dim

  def _inr_net_forward(self, h, shape):
    coord = make_coord(shape=shape, flatten=True).cuda()
    coord = coord.unsqueeze(0).expand(h.shape[0], *coord.shape[-2:])
    cell = make_cell(shape).cuda()
    cell = cell.unsqueeze(0).expand(h.shape[0], *cell.shape[-2:])

    feat = h
    # feat = F.unfold(h, 3, padding=1).view(h.shape[0], h.shape[1] * 9, h.shape[2], h.shape[3])

    sampled_feat = F.grid_sample(
      feat, coord.flip(-1).unsqueeze(1), mode='bilinear', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

    cell = cell.clone()
    cell[:, :, 0] *= feat.shape[-2]
    cell[:, :, 1] *= feat.shape[-1]
    inp = torch.cat([sampled_feat, coord, cell], dim=-1)

    bs, q = inp.shape[:2]
    pred = self.inr_net(inp.view(bs * q, -1)).view(bs, q, -1)
    pred = pred.permute(0, 2, 1).view(pred.shape[0], -1, *shape)

    out = torch.tanh(pred)
    return out


@MODEL_REGISTRY.register(name=f"{__name__}.Generator_no_coord")
class Generator_no_coord(Generator):
  def _get_inr_net_in_dim(self):
    # coords + cell
    in_dim = self.arch['out_channels'][-1] * 9 + 2
    return in_dim

  def _inr_net_forward(self, h, shape):
    coord = make_coord(shape=shape, flatten=True).cuda()
    coord = coord.unsqueeze(0).expand(h.shape[0], *coord.shape[-2:])
    cell = make_cell(shape).cuda()
    cell = cell.unsqueeze(0).expand(h.shape[0], *cell.shape[-2:])

    feat = F.unfold(h, 3, padding=1).view(h.shape[0], h.shape[1] * 9, h.shape[2], h.shape[3])

    sampled_feat = F.grid_sample(
      feat, coord.flip(-1).unsqueeze(1), mode='bilinear', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

    cell = cell.clone()
    cell[:, :, 0] *= feat.shape[-2]
    cell[:, :, 1] *= feat.shape[-1]
    inp = torch.cat([sampled_feat, cell], dim=-1)
    # inp = torch.cat([sampled_feat, coord, cell], dim=-1)

    bs, q = inp.shape[:2]
    pred = self.inr_net(inp.view(bs * q, -1)).view(bs, q, -1)
    pred = pred.permute(0, 2, 1).view(pred.shape[0], -1, *shape)

    out = torch.tanh(pred)
    return out


@MODEL_REGISTRY.register(name=f"{__name__}.Generator_no_cell")
class Generator_no_cell(Generator):
  def _get_inr_net_in_dim(self):
    # coords + cell
    in_dim = self.arch['out_channels'][-1] * 9 + 2
    return in_dim

  def _inr_net_forward(self, h, shape):
    coord = make_coord(shape=shape, flatten=True).cuda()
    coord = coord.unsqueeze(0).expand(h.shape[0], *coord.shape[-2:])
    cell = make_cell(shape).cuda()
    cell = cell.unsqueeze(0).expand(h.shape[0], *cell.shape[-2:])

    feat = F.unfold(h, 3, padding=1).view(h.shape[0], h.shape[1] * 9, h.shape[2], h.shape[3])

    sampled_feat = F.grid_sample(
      feat, coord.flip(-1).unsqueeze(1), mode='bilinear', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

    cell = cell.clone()
    cell[:, :, 0] *= feat.shape[-2]
    cell[:, :, 1] *= feat.shape[-1]
    inp = torch.cat([sampled_feat, coord], dim=-1)
    # inp = torch.cat([sampled_feat, coord, cell], dim=-1)

    bs, q = inp.shape[:2]
    pred = self.inr_net(inp.view(bs * q, -1)).view(bs, q, -1)
    pred = pred.permute(0, 2, 1).view(pred.shape[0], -1, *shape)

    out = torch.tanh(pred)
    return out







