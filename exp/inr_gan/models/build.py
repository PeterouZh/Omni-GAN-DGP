from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL_REGISTRY")  # noqa F401 isort:skip
MODEL_REGISTRY.__doc__ = """

"""


def build_model(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    return MODEL_REGISTRY.get(cfg.name)(cfg=cfg, **kwargs)
