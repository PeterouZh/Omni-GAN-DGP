# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.utils.registry import Registry

TRAINER_REGISTRY = Registry("TRAINER_REGISTRY")  # noqa F401 isort:skip
TRAINER_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_trainer(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    return TRAINER_REGISTRY.get(cfg.name)(cfg=cfg, **kwargs)
