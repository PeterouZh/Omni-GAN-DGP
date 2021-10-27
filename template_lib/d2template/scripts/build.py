# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.utils.registry import Registry

START_REGISTRY = Registry("START_REGISTRY")  # noqa F401 isort:skip
START_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_start(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    return START_REGISTRY.get(cfg.start.name)(cfg=cfg, **kwargs)
