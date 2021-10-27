from torch import nn
import torch.nn.functional as F

from template_lib.utils import get_attr_kwargs
from template_lib.d2.layers import build_d2layer

UP_MODES = ['nearest', 'bilinear']
NORMS = ['in', 'bn']


class Cell(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode, ksize=3, num_skip_in=0, short_cut=False, norm=None):
        super(Cell, self).__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, ksize, padding=ksize//2)
        self.c2 = nn.Conv2d(out_channels, out_channels, ksize, padding=ksize//2)
        assert up_mode in UP_MODES
        self.up_mode = up_mode
        self.norm = norm
        if norm:
            assert norm in NORMS
            if norm == 'bn':
                self.n1 = nn.BatchNorm2d(in_channels)
                self.n2 = nn.BatchNorm2d(out_channels)
            elif norm == 'in':
                self.n1 = nn.InstanceNorm2d(in_channels)
                self.n2 = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError(norm)

        # inner shortcut
        self.c_sc = None
        if short_cut:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # cross scale skip
        self.skip_in_ops = None
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=1) for _ in range(num_skip_in)])

    def forward(self, x, skip_ft=None):
        residual = x

        # first conv
        if self.norm:
            residual = self.n1(residual)
        h = nn.ReLU()(residual)
        h = F.interpolate(h, scale_factor=2, mode=self.up_mode)
        _, _, ht, wt = h.size()
        h = self.c1(h)
        h_skip_out = h

        # second conv
        if self.skip_in_ops:
            assert len(self.skip_in_ops) == len(skip_ft)
            for ft, skip_in_op in zip(skip_ft, self.skip_in_ops):
                h += skip_in_op(F.interpolate(ft, size=(ht, wt), mode=self.up_mode))
        if self.norm:
            h = self.n2(h)
        h = nn.ReLU()(h)
        final_out = self.c2(h)

        # shortcut
        if self.c_sc:
            final_out += self.c_sc(F.interpolate(x, scale_factor=2, mode=self.up_mode))

        return h_skip_out, final_out



class MixedCell(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(MixedCell, self).__init__()

        in_channels                         = get_attr_kwargs(cfg, "in_channels", **kwargs)
        out_channels                        = get_attr_kwargs(cfg, "out_channels", **kwargs)
        up_mode                             = get_attr_kwargs(cfg, "up_mode", **kwargs)
        num_skip_in                         = get_attr_kwargs(cfg, "num_skip_in", default=0, **kwargs)
        short_cut                           = get_attr_kwargs(cfg, "short_cut", default=False, **kwargs)
        norm                                = get_attr_kwargs(cfg, "norm", default=None, **kwargs)
        cfg_ops                             = get_attr_kwargs(cfg, "cfg_ops", **kwargs)

        self.c1 = build_d2layer(cfg=cfg.cfg_mix_layer, in_channels=in_channels, out_channels=out_channels,
                                cfg_ops=cfg_ops)
        self.c2 = build_d2layer(cfg=cfg.cfg_mix_layer, in_channels=out_channels, out_channels=out_channels,
                                cfg_ops=cfg_ops)

        assert up_mode in UP_MODES
        self.up_mode = up_mode
        self.norm = norm
        if norm:
            assert norm in NORMS
            if norm == 'bn':
                self.n1 = nn.BatchNorm2d(in_channels)
                self.n2 = nn.BatchNorm2d(out_channels)
            elif norm == 'in':
                self.n1 = nn.InstanceNorm2d(in_channels)
                self.n2 = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError(norm)

        # inner shortcut
        self.c_sc = None
        if short_cut:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # cross scale skip
        self.skip_in_ops = None
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=1)
                                              for _ in range(num_skip_in)])
        pass

    def forward(self, x, sample_arc1, sample_arc2, skip_ft=None):
        residual = x

        # first conv
        if self.norm:
            residual = self.n1(residual)
        h = nn.ReLU()(residual)
        h = F.interpolate(h, scale_factor=2, mode=self.up_mode)
        _, _, ht, wt = h.size()
        h = self.c1(h, sample_arc1)
        h_skip_out = h

        # second conv
        if self.skip_in_ops:
            assert len(self.skip_in_ops) == len(skip_ft)
            for ft, skip_in_op in zip(skip_ft, self.skip_in_ops):
                h += skip_in_op(F.interpolate(ft, size=(ht, wt), mode=self.up_mode))
        if self.norm:
            h = self.n2(h)
        h = nn.ReLU()(h)
        final_out = self.c2(h, sample_arc2)

        # shortcut
        if self.c_sc:
            final_out += self.c_sc(F.interpolate(x, scale_factor=2, mode=self.up_mode))

        return h_skip_out, final_out


class MixedActCell(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(MixedActCell, self).__init__()

        in_channels                         = get_attr_kwargs(cfg, "in_channels", **kwargs)
        out_channels                        = get_attr_kwargs(cfg, "out_channels", **kwargs)
        ksize                               = get_attr_kwargs(cfg, "ksize", default=3, **kwargs)
        up_mode                             = get_attr_kwargs(cfg, "up_mode", **kwargs)
        num_skip_in                         = get_attr_kwargs(cfg, "num_skip_in", default=0, **kwargs)
        short_cut                           = get_attr_kwargs(cfg, "short_cut", default=False, **kwargs)
        norm                                = get_attr_kwargs(cfg, "norm", default=None, **kwargs)
        cfg_mix_layer                       = get_attr_kwargs(cfg, "cfg_mix_layer", **kwargs)
        cfg_ops                             = get_attr_kwargs(cfg, "cfg_ops", **kwargs)

        self.c1 = nn.Conv2d(in_channels, out_channels, ksize, padding=ksize // 2)
        self.c2 = nn.Conv2d(out_channels, out_channels, ksize, padding=ksize // 2)

        self.act1 = build_d2layer(cfg_mix_layer, cfg_ops=cfg_ops, num_parameters=in_channels, out_channels=in_channels)
        self.act2 = build_d2layer(cfg_mix_layer, cfg_ops=cfg_ops, num_parameters=out_channels, out_channels=out_channels)

        assert up_mode in UP_MODES
        self.up_mode = up_mode
        self.norm = norm
        if norm:
            assert norm in NORMS
            if norm == 'bn':
                self.n1 = nn.BatchNorm2d(in_channels)
                self.n2 = nn.BatchNorm2d(out_channels)
            elif norm == 'in':
                self.n1 = nn.InstanceNorm2d(in_channels)
                self.n2 = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError(norm)

        # inner shortcut
        self.c_sc = None
        if short_cut:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # cross scale skip
        self.skip_in_ops = None
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=1)
                                              for _ in range(num_skip_in)])
        pass

    def forward(self, x, sample_arc1, sample_arc2, skip_ft=None):
        residual = x

        # first conv
        if self.norm:
            residual = self.n1(residual)
        h = self.act1(residual, sample_arc1)
        h = F.interpolate(h, scale_factor=2, mode=self.up_mode)
        _, _, ht, wt = h.size()
        h = self.c1(h)
        h_skip_out = h

        # second conv
        if self.skip_in_ops:
            assert len(self.skip_in_ops) == len(skip_ft)
            for ft, skip_in_op in zip(skip_ft, self.skip_in_ops):
                h += skip_in_op(F.interpolate(ft, size=(ht, wt), mode=self.up_mode))
        if self.norm:
            h = self.n2(h)
        h = self.act2(h, sample_arc2)
        final_out = self.c2(h)

        # shortcut
        if self.c_sc:
            final_out += self.c_sc(F.interpolate(x, scale_factor=2, mode=self.up_mode))

        return h_skip_out, final_out


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(self, d_spectral_norm, in_channels, out_channels,
                 ksize=3, pad=1, activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()

        self.d_spectral_norm = d_spectral_norm
        self.activation = activation

        self.c1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c_sc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0)
        if self.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(self, d_spectral_norm, in_channels, out_channels,
                 hidden_channels=None, ksize=3, pad=1, activation=nn.ReLU(),
                 downsample=False):
        super(DisBlock, self).__init__()

        self.d_spectral_norm = d_spectral_norm
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample

        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        if self.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            if self.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)