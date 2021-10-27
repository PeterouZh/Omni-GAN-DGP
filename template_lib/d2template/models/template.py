from easydict import EasyDict
import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn

from template_lib.utils import get_attr_kwargs, update_config
from template_lib.d2template.models import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class GNNEncoder(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()

    cfg = self.update_cfg(cfg)

    ndim                            = get_attr_kwargs(cfg, 'ndim', default=250, **kwargs)

    pass

  @staticmethod
  def test_case():
    from template_lib.d2template.models import build_model

    cfg_str = """
                name: GNNEncoder
                update_cfg: true
              """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    model = build_model(cfg)

    model = model.cuda()

    return

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
                name: "GNNEncoder"
                ndim: 250
                gdim: 56
                num_gnn_layers: 2
                num_node_atts: 5
              """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg
