from fvcore.common.registry import Registry

OPS_REGISTRY = Registry("OPS_REGISTRY")  # noqa F401 isort:skip
OPS_REGISTRY.__doc__ = """
"""
def build_ops(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.name
    return OPS_REGISTRY.get(name)(cfg=cfg, **kwargs)


LAYER_REGISTRY = Registry("LAYER_REGISTRY")  # noqa F401 isort:skip
LAYER_REGISTRY.__doc__ = """
"""
def build_layer(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.name
    return LAYER_REGISTRY.get(name)(cfg=cfg, **kwargs)
