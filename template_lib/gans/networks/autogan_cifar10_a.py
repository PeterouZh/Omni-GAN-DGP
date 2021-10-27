import yaml
import json
import logging
from easydict import EasyDict
import functools
import torch
from torch import nn

from template_lib.utils import get_eval_attr, get_attr_kwargs
from template_lib.v2.config import update_config
from template_lib.d2.layers import build_d2layer
from template_lib.d2.utils import comm

from .pagan.layers import SNEmbedding

from .building_blocks import Cell, DisBlock, OptimizedDisBlock
from .build import DISCRIMINATOR_REGISTRY, GENERATOR_REGISTRY
from .autogan_cifar10_utils import MixedCell, MixedActCell


@GENERATOR_REGISTRY.register()
class AutoGANCIFAR10AGenerator(nn.Module):
    def __init__(self, args):
        super(AutoGANCIFAR10AGenerator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * args.gf_dim)
        self.cell1 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=0, short_cut=True)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, 'bilinear', num_skip_in=1, short_cut=True)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=2, short_cut=False)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim),
            nn.ReLU(),
            nn.Conv2d(args.gf_dim, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out, ))
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)

        return output


@DISCRIMINATOR_REGISTRY.register()
class AutoGANCIFAR10ADiscriminator(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(AutoGANCIFAR10ADiscriminator, self).__init__()

        self.ch                    = get_attr_kwargs(cfg, 'ch', default=128, **kwargs)
        self.d_spectral_norm       = get_attr_kwargs(cfg, 'd_spectral_norm', default=True, **kwargs)
        self.init_type             = get_attr_kwargs(cfg, 'init_type', default='xavier_uniform', **kwargs)
        self.cfg_act               = get_attr_kwargs(cfg, 'cfg_act', default=EasyDict(name='ReLU'), **kwargs)

        self.activation            = build_d2layer(cfg=self.cfg_act)

        self.block1 = OptimizedDisBlock(d_spectral_norm=self.d_spectral_norm,
                                        in_channels=3, out_channels=self.ch)
        self.block2 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=True)
        self.block3 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=False)
        self.block4 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=False)
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        self.model = model
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if self.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

        weights_init_func = functools.partial(
            self.weights_init, init_type=self.init_type)
        self.apply(weights_init_func)

    def forward(self, x, *args, **kwargs):
        h = x

        h = self.model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output

    @staticmethod
    def weights_init(m, init_type='orth'):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


@DISCRIMINATOR_REGISTRY.register()
class AutoGANCIFAR10ADiscriminatorCProj(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.ch                    = get_attr_kwargs(cfg, 'ch', default=128, **kwargs)
        self.d_spectral_norm       = get_attr_kwargs(cfg, 'd_spectral_norm', default=True, **kwargs)
        self.init_type             = get_attr_kwargs(cfg, 'init_type', default='xavier_uniform', **kwargs)
        self.cfg_act               = get_attr_kwargs(cfg, 'cfg_act', default=EasyDict(name='ReLU'), **kwargs)
        self.n_classes             = get_attr_kwargs(cfg, 'n_classes', **kwargs)
        self.num_D_SVs             = get_attr_kwargs(cfg, 'num_D_SVs', default=1, **kwargs)
        self.num_D_SV_itrs         = get_attr_kwargs(cfg, 'num_D_SV_itrs', default=1, **kwargs)
        self.SN_eps                = get_attr_kwargs(cfg, 'SN_eps', default=1e-6, **kwargs)

        self.activation            = build_d2layer(cfg=self.cfg_act)

        self.which_embedding = functools.partial(SNEmbedding,
                                                 num_svs=self.num_D_SVs, num_itrs=self.num_D_SV_itrs,
                                                 eps=self.SN_eps)
        self.embed = self.which_embedding(self.n_classes, self.ch)

        self.block1 = OptimizedDisBlock(d_spectral_norm=self.d_spectral_norm,
                                        in_channels=3, out_channels=self.ch)
        self.block2 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=True)
        self.block3 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=False)
        self.block4 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=False)
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        self.model = model
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if self.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

        weights_init_func = functools.partial(
            self.weights_init, init_type=self.init_type)
        self.apply(weights_init_func)

    def forward(self, x, y, *args, **kwargs):
        h = x

        h = self.model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        out = self.l5(h)

        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)

        return out

    @staticmethod
    def weights_init(m, init_type='orth'):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


@DISCRIMINATOR_REGISTRY.register()
class AutoGANCIFAR10ADiscriminatorCProj_v1(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.ch                    = get_attr_kwargs(cfg, 'ch', default=128, **kwargs)
        self.d_spectral_norm       = get_attr_kwargs(cfg, 'd_spectral_norm', default=True, **kwargs)
        self.init_type             = get_attr_kwargs(cfg, 'init_type', default='xavier_uniform', **kwargs)
        self.cfg_act               = get_attr_kwargs(cfg, 'cfg_act', default=EasyDict(name='ReLU'), **kwargs)
        self.n_classes             = get_attr_kwargs(cfg, 'n_classes', **kwargs)
        self.activation            = build_d2layer(cfg=self.cfg_act)

        self.embed = nn.Embedding(self.n_classes, self.ch)

        self.block1 = OptimizedDisBlock(d_spectral_norm=self.d_spectral_norm,
                                        in_channels=3, out_channels=self.ch)
        self.block2 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=True)
        self.block3 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=False)
        self.block4 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=False)
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        self.model = model
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if self.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

        weights_init_func = functools.partial(
            self.weights_init, init_type=self.init_type)
        self.apply(weights_init_func)

    def forward(self, x, y, *args, **kwargs):
        h = x

        h = self.model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        out = self.l5(h)

        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)

        return out

    @staticmethod
    def weights_init(m, init_type='orth'):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


@DISCRIMINATOR_REGISTRY.register()
class AutoGANCIFAR10ADiscriminatorCProjNoFC(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.ch                    = get_attr_kwargs(cfg, 'ch', default=128, **kwargs)
        self.d_spectral_norm       = get_attr_kwargs(cfg, 'd_spectral_norm', default=True, **kwargs)
        self.init_type             = get_attr_kwargs(cfg, 'init_type', default='xavier_uniform', **kwargs)
        self.cfg_act               = get_attr_kwargs(cfg, 'cfg_act', default=EasyDict(name='ReLU'), **kwargs)
        self.n_classes             = get_attr_kwargs(cfg, 'n_classes', **kwargs)
        self.num_D_SVs             = get_attr_kwargs(cfg, 'num_D_SVs', default=1, **kwargs)
        self.num_D_SV_itrs         = get_attr_kwargs(cfg, 'num_D_SV_itrs', default=1, **kwargs)
        self.SN_eps                = get_attr_kwargs(cfg, 'SN_eps', default=1e-6, **kwargs)

        self.activation            = build_d2layer(cfg=self.cfg_act)

        self.which_embedding = functools.partial(SNEmbedding,
                                                 num_svs=self.num_D_SVs, num_itrs=self.num_D_SV_itrs,
                                                 eps=self.SN_eps)
        self.embed = self.which_embedding(self.n_classes, self.ch)

        self.block1 = OptimizedDisBlock(d_spectral_norm=self.d_spectral_norm,
                                        in_channels=3, out_channels=self.ch)
        self.block2 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=True)
        self.block3 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=False)
        self.block4 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=False)
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        self.model = model
        # self.l5 = nn.Linear(self.ch, 1, bias=False)
        # if self.d_spectral_norm:
        #     self.l5 = nn.utils.spectral_norm(self.l5)

        weights_init_func = functools.partial(
            self.weights_init, init_type=self.init_type)
        self.apply(weights_init_func)

    def forward(self, x, y, *args, **kwargs):
        h = x

        h = self.model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        # out = self.l5(h)

        out = torch.sum(self.embed(y) * h, 1, keepdim=True)

        return out

    @staticmethod
    def weights_init(m, init_type='orth'):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


@DISCRIMINATOR_REGISTRY.register()
class AutoGANCIFAR10ADiscriminatorCProjMLP(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.ch                    = get_attr_kwargs(cfg, 'ch', default=128, **kwargs)
        self.d_spectral_norm       = get_attr_kwargs(cfg, 'd_spectral_norm', default=True, **kwargs)
        self.init_type             = get_attr_kwargs(cfg, 'init_type', default='xavier_uniform', **kwargs)
        self.cfg_act               = get_attr_kwargs(cfg, 'cfg_act', default=EasyDict(name='ReLU'), **kwargs)
        self.n_classes             = get_attr_kwargs(cfg, 'n_classes', **kwargs)
        self.num_D_SVs             = get_attr_kwargs(cfg, 'num_D_SVs', default=1, **kwargs)
        self.num_D_SV_itrs         = get_attr_kwargs(cfg, 'num_D_SV_itrs', default=1, **kwargs)
        self.SN_eps                = get_attr_kwargs(cfg, 'SN_eps', default=1e-6, **kwargs)
        self.emb_dim               = get_attr_kwargs(cfg, 'emb_dim', default=256, **kwargs)
        self.mlp_dim               = get_attr_kwargs(cfg, 'mlp_dim', default=256, **kwargs)
        self.num_layers_mlp        = get_attr_kwargs(cfg, 'num_layers_mlp', default=1, **kwargs)

        self.activation            = build_d2layer(cfg=self.cfg_act)

        self.which_embedding = functools.partial(SNEmbedding,
                                                 num_svs=self.num_D_SVs, num_itrs=self.num_D_SV_itrs,
                                                 eps=self.SN_eps)
        if self.num_layers_mlp == 0 and self.emb_dim != self.ch:
            logging.getLogger('tl').warning('emb_dim != ch')
            self.emb_dim = self.ch
        self.embedding = self.which_embedding(self.n_classes, self.emb_dim)
        mlp_layers = []

        mlp_layers.append(self.embedding)
        last_dim = self.emb_dim
        for idx in range(self.num_layers_mlp-1):
            if idx == 0:
                mlp = nn.Linear(self.emb_dim, self.mlp_dim)
                last_dim = self.mlp_dim
            else:
                mlp = nn.Linear(self.mlp_dim, self.mlp_dim)
            mlp = nn.utils.spectral_norm(mlp)
            mlp_layers.append(mlp)
            mlp_layers.append(nn.ReLU())
        if self.num_layers_mlp > 0:
            affine_layer = nn.Linear(last_dim, self.ch)
            affine_layer = nn.utils.spectral_norm(affine_layer)
            mlp_layers.append(affine_layer)
        self.embed = nn.Sequential(*mlp_layers)

        self.block1 = OptimizedDisBlock(d_spectral_norm=self.d_spectral_norm,
                                        in_channels=3, out_channels=self.ch)
        self.block2 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=True)
        self.block3 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=False)
        self.block4 = DisBlock(d_spectral_norm=self.d_spectral_norm,
                               in_channels=self.ch, out_channels=self.ch,
                               activation=self.activation, downsample=False)
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        self.model = model
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if self.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

        weights_init_func = functools.partial(
            self.weights_init, init_type=self.init_type)
        self.apply(weights_init_func)

    def forward(self, x, y, *args, **kwargs):
        h = x

        h = self.model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        out = self.l5(h)

        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)

        return out

    @staticmethod
    def weights_init(m, init_type='orth'):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)



@GENERATOR_REGISTRY.register()
class PathAwareAutoGANCIFAR10AGenerator(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(PathAwareAutoGANCIFAR10AGenerator, self).__init__()

        self.ch                 = get_attr_kwargs(cfg, 'ch', default=256, **kwargs)
        self.bottom_width       = get_attr_kwargs(cfg, 'bottom_width', default=4, **kwargs)
        self.latent_dim         = get_attr_kwargs(cfg, 'latent_dim', default=128, **kwargs)
        self.init_type          = get_attr_kwargs(cfg, 'init_type', default='xavier_uniform', **kwargs)

        self.num_branches = len(cfg.cfg_ops)
        self.num_layers = 6
        self.dim_z = self.latent_dim
        self.device = torch.device(f'cuda:{comm.get_rank()}')

        self.l1 = nn.Linear(self.latent_dim, (self.bottom_width ** 2) * self.ch)
        self.cell1 = MixedCell(cfg=cfg.cfg_mixedcell, in_channels=self.ch, out_channels=self.ch,
                               up_mode='nearest', num_skip_in=0, short_cut=True, cfg_ops=cfg.cfg_ops)
        self.cell2 = MixedCell(cfg=cfg.cfg_mixedcell, in_channels=self.ch, out_channels=self.ch,
                               up_mode='bilinear', num_skip_in=1, short_cut=True, cfg_ops=cfg.cfg_ops)
        self.cell3 = MixedCell(cfg=cfg.cfg_mixedcell, in_channels=self.ch, out_channels=self.ch,
                               up_mode='nearest', num_skip_in=2, short_cut=False, cfg_ops=cfg.cfg_ops)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(self.ch),
            nn.ReLU(),
            nn.Conv2d(self.ch, 3, 3, 1, 1),
            nn.Tanh()
        )

        weights_init_func = functools.partial(self.weights_init, init_type=self.init_type)
        self.apply(weights_init_func)

    def forward(self, z, batched_arcs, *args, **kwargs):
        batched_arcs = batched_arcs.to(self.device)
        z = z.to(self.device)

        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        sample_arc1 = batched_arcs[:, 0]
        sample_arc2 = batched_arcs[:, 1]
        h1_skip_out, h1 = self.cell1(h, sample_arc1=sample_arc1, sample_arc2=sample_arc2)

        sample_arc1 = batched_arcs[:, 2]
        sample_arc2 = batched_arcs[:, 3]
        h2_skip_out, h2 = self.cell2(h1, sample_arc1=sample_arc1, sample_arc2=sample_arc2, skip_ft=(h1_skip_out, ))

        sample_arc1 = batched_arcs[:, 4]
        sample_arc2 = batched_arcs[:, 5]
        _, h3 = self.cell3(h2, sample_arc1=sample_arc1, sample_arc2=sample_arc2, skip_ft=(h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)

        return output

    @staticmethod
    def weights_init(m, init_type='orth'):

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(init_type))
        elif (isinstance(m, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


@GENERATOR_REGISTRY.register()
class SearchActAutoGANCIFAR10AGenerator(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(SearchActAutoGANCIFAR10AGenerator, self).__init__()

        self.ch                 = get_attr_kwargs(cfg, 'ch', default=256, **kwargs)
        self.bottom_width       = get_attr_kwargs(cfg, 'bottom_width', default=4, **kwargs)
        self.latent_dim         = get_attr_kwargs(cfg, 'latent_dim', default=128, **kwargs)
        self.init_type          = get_attr_kwargs(cfg, 'init_type', default='xavier_uniform', **kwargs)
        self.cfg_ops            = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)

        self.num_branches = len(cfg.cfg_ops)
        self.num_layers = 6
        self.dim_z = self.latent_dim
        self.device = torch.device(f'cuda:{comm.get_rank()}')

        self.l1 = nn.Linear(self.latent_dim, (self.bottom_width ** 2) * self.ch)
        self.cell1 = MixedActCell(cfg=cfg.cfg_mixedcell, in_channels=self.ch, out_channels=self.ch,
                                  up_mode='nearest', num_skip_in=0, short_cut=True,
                                  cfg_ops=self.cfg_ops)
        self.cell2 = MixedActCell(cfg=cfg.cfg_mixedcell, in_channels=self.ch, out_channels=self.ch,
                                  up_mode='bilinear', num_skip_in=1, short_cut=True,
                                  cfg_ops=self.cfg_ops)
        self.cell3 = MixedActCell(cfg=cfg.cfg_mixedcell, in_channels=self.ch, out_channels=self.ch,
                                  up_mode='nearest', num_skip_in=2, short_cut=False,
                                  cfg_ops=self.cfg_ops)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(self.ch),
            nn.ReLU(),
            nn.Conv2d(self.ch, 3, 3, 1, 1),
            nn.Tanh()
        )

        weights_init_func = functools.partial(self.weights_init, init_type=self.init_type)
        self.apply(weights_init_func)

    def forward(self, z, batched_arcs, *args, **kwargs):
        batched_arcs = batched_arcs.to(self.device)
        z = z.to(self.device)

        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        sample_arc1 = batched_arcs[:, 0]
        sample_arc2 = batched_arcs[:, 1]
        h1_skip_out, h1 = self.cell1(h, sample_arc1=sample_arc1, sample_arc2=sample_arc2)

        sample_arc1 = batched_arcs[:, 2]
        sample_arc2 = batched_arcs[:, 3]
        h2_skip_out, h2 = self.cell2(h1, sample_arc1=sample_arc1, sample_arc2=sample_arc2, skip_ft=(h1_skip_out, ))

        sample_arc1 = batched_arcs[:, 4]
        sample_arc2 = batched_arcs[:, 5]
        _, h3 = self.cell3(h2, sample_arc1=sample_arc1, sample_arc2=sample_arc2, skip_ft=(h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)

        return output

    @staticmethod
    def weights_init(m, init_type='orth'):

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(init_type))
        elif (isinstance(m, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


@GENERATOR_REGISTRY.register()
class DenseGenerator(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(DenseGenerator, self).__init__()

        self.ch                 = get_attr_kwargs(cfg, 'ch', default=256, **kwargs)
        self.bottom_width       = get_attr_kwargs(cfg, 'bottom_width', default=4, **kwargs)
        self.dim_z              = get_attr_kwargs(cfg, 'dim_z', default=128, **kwargs)
        self.init_type          = get_attr_kwargs(cfg, 'init_type', default='xavier_uniform', **kwargs)
        self.cfg_upsample       = get_attr_kwargs(cfg, 'cfg_upsample', **kwargs)
        self.num_cells          = get_attr_kwargs(cfg, 'num_cells', **kwargs)
        self.cfg_cell           = get_attr_kwargs(cfg, 'cfg_cell', **kwargs)
        self.layer_op_idx       = get_attr_kwargs(cfg, 'layer_op_idx', default=None, **kwargs)
        self.cfg_ops            = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)
        self.cfg_out_bn         = get_attr_kwargs(cfg, 'cfg_out_bn', **kwargs)

        if self.layer_op_idx is not None:
            self.layer_op_idx = json.loads(self.layer_op_idx.replace(' ', ', ').replace('][', '], [').strip())
            assert len(self.layer_op_idx) % self.num_cells == 0
            num_edges_of_cell = len(self.layer_op_idx) // self.num_cells

        self.device = torch.device(f'cuda:{comm.get_rank()}')

        self.l1 = nn.Linear(self.dim_z, (self.bottom_width ** 2) * self.ch)
        self.upsample = build_d2layer(self.cfg_upsample)

        self.cells = nn.ModuleList()
        for i in range(self.num_cells):
            if self.layer_op_idx is not None:
                cell = build_d2layer(self.cfg_cell, in_channels=self.ch, out_channels=self.ch, cfg_ops=self.cfg_ops,
                                     cell_op_idx=self.layer_op_idx[i*num_edges_of_cell:(i+1)*num_edges_of_cell])
            else:
                cell = build_d2layer(self.cfg_cell, in_channels=self.ch, out_channels=self.ch, cfg_ops=self.cfg_ops)
            self.cells.append(cell)

        self.num_branches = len(self.cfg_ops)
        self.num_edges_of_cell = cell.num_edges
        self.num_layers = len(self.cells) * self.num_edges_of_cell

        out_bn = build_d2layer(self.cfg_out_bn, num_features=self.ch, **kwargs)
        self.to_rgb = nn.Sequential(
            out_bn,
            nn.ReLU(),
            nn.Conv2d(self.ch, 3, 3, 1, 1),
            nn.Tanh()
        )

        weights_init_func = functools.partial(self.weights_init, init_type=self.init_type)
        self.apply(weights_init_func)

    def forward(self, z, batched_arcs, *args, **kwargs):
        batched_arcs = batched_arcs.to(self.device)
        z = z.to(self.device)

        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)

        for idx, cell in enumerate(self.cells):
            h = self.upsample(h)
            h = cell(h, batched_arcs=batched_arcs[:, idx*self.num_edges_of_cell:(idx+1)*self.num_edges_of_cell])

        output = self.to_rgb(h)

        return output

    @staticmethod
    def weights_init(m, init_type='orth'):

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(init_type))
        elif (isinstance(m, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


