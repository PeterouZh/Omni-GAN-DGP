import logging

from template_lib.utils import register_modules
from template_lib.v2.utils.registry import Registry, build_from_cfg

REGISTRY = Registry("MODEL_REGISTRY")  # noqa F401 isort:skip
MODEL_REGISTRY = REGISTRY
REGISTRY.__doc__ = """

"""

def _build(cfg, kwargs_priority, cfg_to_kwargs, **kwargs):
    cfg = cfg.clone()
    logging.getLogger('tl').info(f"Building {cfg.name} ...")
    register_modules(register_modules=cfg.pop('register_modules', []))
    print("")
    if not cfg_to_kwargs:
        ret = REGISTRY.get(cfg.name)(cfg=cfg, kwargs_priority=kwargs_priority, **kwargs)
    else:
        ret = build_from_cfg(cfg=cfg, registry=REGISTRY, kwargs_priority=kwargs_priority, default_args=kwargs)
    return ret

def build_model(cfg, kwargs_priority=False, cfg_to_kwargs=False, **kwargs):

    return _build(cfg, kwargs_priority=kwargs_priority, cfg_to_kwargs=cfg_to_kwargs, **kwargs)


