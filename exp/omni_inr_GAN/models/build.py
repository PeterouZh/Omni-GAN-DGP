# from fvcore.common.registry import Registry
#
# MODEL_REGISTRY = Registry("MODEL_REGISTRY")  # noqa F401 isort:skip
# MODEL_REGISTRY.__doc__ = """
#
# """
#
#
# def build_model(cfg, **kwargs):
#     """
#     Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
#     Note that it does not load any weights from ``cfg``.
#     """
#     return MODEL_REGISTRY.get(cfg.name)(cfg=cfg, **kwargs)

from template_lib.d2.models_v2 import build_model, MODEL_REGISTRY