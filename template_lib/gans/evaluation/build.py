# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fvcore.common.registry import Registry

GAN_METRIC_REGISTRY = Registry("GAN_METRIC_REGISTRY")  # noqa F401 isort:skip
GAN_METRIC_REGISTRY.__doc__ = """

"""


def build_GAN_metric_dict(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    if cfg.GAN_metric.get('names') is None:
        return {}
    ret_dict = {}
    for name in cfg.GAN_metric.names:
        ret_dict.update({name: GAN_METRIC_REGISTRY.get(name)(cfg=cfg, **kwargs)})
    return ret_dict


def build_GAN_metric(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """

    metric = GAN_METRIC_REGISTRY.get(cfg.name)(cfg=cfg, **kwargs)
    return metric
