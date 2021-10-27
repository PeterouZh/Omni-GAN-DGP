# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fvcore.common.registry import Registry

DISCRIMINATOR_REGISTRY = Registry("DISCRIMINATOR_REGISTRY")  # noqa F401 isort:skip
DISCRIMINATOR_REGISTRY.__doc__ = """

"""

GENERATOR_REGISTRY = Registry("GENERATOR_REGISTRY")  # noqa F401 isort:skip
GENERATOR_REGISTRY.__doc__ = """

"""


def build_discriminator(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.name
    return DISCRIMINATOR_REGISTRY.get(name)(cfg=cfg, **kwargs)


def build_generator(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.name
    return GENERATOR_REGISTRY.get(name)(cfg=cfg, **kwargs)

