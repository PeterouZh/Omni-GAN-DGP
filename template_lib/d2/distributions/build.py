# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from fvcore.common.registry import Registry
#
# D2DISTRIBUTIONS_REGISTRY = Registry("D2DISTRIBUTIONS_REGISTRY")  # noqa F401 isort:skip
# D2DISTRIBUTIONS_REGISTRY.__doc__ = """
#
# """
#
#
# def build_d2distributions(cfg, **kwargs):
#     """
#     Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
#     Note that it does not load any weights from ``cfg``.
#     """
#     return D2DISTRIBUTIONS_REGISTRY.get(cfg.name)(cfg, **kwargs)
#

import logging
from fvcore.common.registry import Registry

from template_lib.utils import register_modules


REGISTRY = Registry("DISTRIBUTIONS_REGISTRY")  # noqa F401 isort:skip
DISTRIBUTIONS_REGISTRY = REGISTRY
REGISTRY.__doc__ = """

"""

def _build(cfg, **kwargs):
    logging.getLogger('tl').info(f"Building {cfg.name} ...")
    register_modules(register_modules=cfg.get('register_modules', {}))
    ret = REGISTRY.get(cfg.name)(cfg=cfg, **kwargs)
    REGISTRY._obj_map.clear()
    return ret

def build_distributions(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    return _build(cfg, **kwargs)

