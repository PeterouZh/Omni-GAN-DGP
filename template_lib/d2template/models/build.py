# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL_REGISTRY")  # noqa F401 isort:skip
MODEL_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.name
    return MODEL_REGISTRY.get(name)(cfg=cfg, **kwargs)
