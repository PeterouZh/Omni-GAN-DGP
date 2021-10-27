# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fvcore.common.registry import Registry

DATASET_MAPPER_REGISTRY = Registry("DATASET_MAPPER_REGISTRY")  # noqa F401 isort:skip
DATASET_MAPPER_REGISTRY.__doc__ = """

"""


def build_dataset_mapper(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.name
    if name.lower() == 'none':
        return None
    return DATASET_MAPPER_REGISTRY.get(name)(cfg=cfg, **kwargs)
