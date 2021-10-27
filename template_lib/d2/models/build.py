from fvcore.common.registry import Registry

D2MODEL_REGISTRY = Registry("D2MODEL_REGISTRY")  # noqa F401 isort:skip
D2MODEL_REGISTRY.__doc__ = """

"""


def build_d2model(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.name
    return D2MODEL_REGISTRY.get(name)(cfg=cfg, **kwargs)
