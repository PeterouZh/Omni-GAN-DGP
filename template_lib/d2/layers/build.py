from fvcore.common.registry import Registry


D2LAYER_REGISTRY = Registry("D2LAYER_REGISTRY")  # noqa F401 isort:skip
D2LAYER_REGISTRY.__doc__ = """
"""
def build_d2layer(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.name
    return D2LAYER_REGISTRY.get(name)(cfg=cfg, **kwargs)