import logging
# from fvcore.common.registry import Registry

from template_lib.utils import register_modules
from template_lib.v2.utils.registry import Registry


REGISTRY = Registry("DATALOADER_REGISTRY")  # noqa F401 isort:skip
DATALOADER_REGISTRY = REGISTRY
REGISTRY.__doc__ = """

"""

def _build(cfg, **kwargs):
    logging.getLogger('tl').info(f"Building {cfg.name} ...")
    # REGISTRY._obj_map.clear()
    register_modules(register_modules=cfg.get('register_modules', {}))
    ret = REGISTRY.get(cfg.name)(cfg=cfg, **kwargs)
    # REGISTRY._obj_map.clear()
    return ret

def build_dataloader(cfg, kwargs_priority=False, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    return _build(cfg, kwargs_priority=kwargs_priority, **kwargs)

