# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fvcore.common.registry import Registry

GAN_MODEL_REGISTRY = Registry("GAN_MODEL_REGISTRY")  # noqa F401 isort:skip
GAN_MODEL_REGISTRY.__doc__ = """

"""


def build_GAN_model(cfg,  **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.name
    return GAN_MODEL_REGISTRY.get(name)(cfg=cfg, **kwargs)

