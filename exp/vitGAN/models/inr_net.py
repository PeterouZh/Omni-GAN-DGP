import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.utils import get_dict_str, get_attr_kwargs
from template_lib.d2.models_v2 import MODEL_REGISTRY

from exp.omni_inr_GAN.comm.utils import make_coord, make_cell


@MODEL_REGISTRY.register(name_prefix=__name__)
class InrNet(nn.Module):

  def __init__(self,
               cfg,
               which_linear=nn.Linear,
               verbose=True,
               **kwargs):
    super().__init__()
    kwargs['tl_ret_kwargs'] = {}
    # fmt: off
    in_dim                     = get_attr_kwargs(cfg, 'in_dim', **kwargs)
    out_dim                    = get_attr_kwargs(cfg, 'out_dim', **kwargs)
    spectral_norm              = get_attr_kwargs(cfg, 'spectral_norm', default=False, **kwargs)
    self.hidden_list           = get_attr_kwargs(cfg, 'hidden_list', **kwargs)

    # fmt: on
    if verbose:
      logging.getLogger('tl').info(f"  {self.__class__.__name__} kwargs: {get_dict_str(kwargs['tl_ret_kwargs'])}")

    last_dim = in_dim * 9 + 2 + 2
    for idx, hidden in enumerate(self.hidden_list):
      name = f"fc_{idx}"
      fc = which_linear(last_dim, hidden)
      if spectral_norm:
        fc = nn.utils.spectral_norm(fc)
      self.add_module(name, fc)
      last_dim = hidden

      name = f"act_{idx}"
      act = nn.ReLU()
      self.add_module(name, act)

    self.out_fc = which_linear(last_dim, out_dim)
    if spectral_norm:
      self.out_fc = nn.utils.spectral_norm(self.out_fc)
    pass

  def forward(self, h, shape):
    """
    h: (b, c, h, w)
    """
    coord = make_coord(shape=shape, flatten=True).to(h.device)
    coord = coord.unsqueeze(0).expand(h.shape[0], *coord.shape[-2:])
    cell = make_cell(shape).to(h.device)
    cell = cell.unsqueeze(0).expand(h.shape[0], *cell.shape[-2:])

    feat = F.unfold(h, 3, padding=1).view(h.shape[0], h.shape[1] * 9, h.shape[2], h.shape[3])

    sampled_feat = F.grid_sample(
      feat, coord.flip(-1).unsqueeze(1), mode='bilinear', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

    cell = cell.clone()
    cell[:, :, 0] *= feat.shape[-2]
    cell[:, :, 1] *= feat.shape[-1]
    inp = torch.cat([sampled_feat, coord, cell], dim=-1)

    bs, q = inp.shape[:2]
    pred = self._forward_fc(inp.view(bs * q, -1)).view(bs, q, -1)
    pred = pred.permute(0, 2, 1).view(pred.shape[0], -1, *shape)

    out = pred
    return out

  def _forward_fc(self, x):
    """
    x: (b*n, d)
    """
    h = x.view(-1, x.shape[-1])
    for idx, _ in enumerate(self.hidden_list):
      name = f"fc_{idx}"
      h = getattr(self, name)(h)

      name = f"act_{idx}"
      h = getattr(self, name)(h)

    h = self.out_fc(h)
    return h
