from fvcore.common.registry import Registry

LAYER_REGISTRY = Registry("LAYER_REGISTRY")  # noqa F401 isort:skip
LAYER_REGISTRY.__doc__ = """

"""


def build_layer(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    return LAYER_REGISTRY.get(cfg.name)(cfg=cfg, **kwargs)
