# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from fvcore.common.registry import Registry
#
# GAN_LOSS_REGISTRY = Registry("GAN_LOSS_REGISTRY")  # noqa F401 isort:skip
# GAN_LOSS_REGISTRY.__doc__ = """
#
# """
#
#
# def build_GAN_loss(cfg,  **kwargs):
#     """
#     """
#     name = cfg.name
#     return GAN_LOSS_REGISTRY.get(name)(cfg=cfg, **kwargs)
#

import logging
# from fvcore.common.registry import Registry

from template_lib.utils import register_modules
from template_lib.v2.utils.registry import Registry


REGISTRY = Registry("GAN_LOSS_REGISTRY")  # noqa F401 isort:skip
GAN_LOSS_REGISTRY = REGISTRY
REGISTRY.__doc__ = """

"""

def _build(cfg, **kwargs):
    logging.getLogger('tl').info(f"Building {cfg.name} ...")
    register_modules(register_modules=cfg.get('register_modules', {}))
    ret = REGISTRY.get(cfg.name)(cfg=cfg, **kwargs)
    # REGISTRY._obj_map.clear()
    return ret

def build_GAN_loss(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    return _build(cfg, **kwargs)

