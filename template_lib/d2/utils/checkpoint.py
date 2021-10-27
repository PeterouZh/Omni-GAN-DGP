import os
import pprint
import logging
import sys
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import Checkpointer, PeriodicCheckpointer
import fvcore.common.checkpoint as fv_ckpt

from template_lib.utils import print_number_params


class DumpModule(nn.Module):
  def __init__(self, model_dict):
    super(DumpModule, self).__init__()

    for name, model in model_dict.items():
      if isinstance(model, DistributedDataParallel):
        model = model.module
      setattr(self, name, model)
    pass


class D2Checkpointer(object):

  def __init__(self, model_dict, optim_dict, ckptdir,
               period=1, max_to_keep=5, maxsize=sys.maxsize, state_dict=None, save_circle=False, verbose=True):

    self.period = 1
    self.max_to_keep = max_to_keep
    self.maxsize = maxsize
    self.save_circle = save_circle

    self.state_dict = state_dict if state_dict is not None else {'itr': 0, 'epoch': 0}

    os.makedirs(ckptdir, exist_ok=True)
    self.logger = logging.getLogger('fvcore')
    if len(self.logger.handlers) == 0:
      self.logger = setup_logger(output=ckptdir, name='fvcore')

    if verbose:
      for k, v in model_dict.items():
        self.logger.info(f"{k}:\n{v}")
    print_number_params(model_dict, logger=self.logger)

    self.checkpointer = self.get_d2_checkpointer(model_dict=model_dict, optim_dict=optim_dict, ckptdir=ckptdir)
    self.periodic_checkpointer = self.get_d2_periodic_checkpointer()

    pass

  @staticmethod
  def get_d2_checkpointer(model_dict, optim_dict, ckptdir):
    ckpt_model = DumpModule(model_dict)
    checkpointer = Checkpointer(ckpt_model, ckptdir, **optim_dict)
    return checkpointer

  def get_d2_periodic_checkpointer(self, ):
    """
    periodic_checkpointer.step(epoch, **{'first_epoch': epoch})
    periodic_checkpointer.save(name='best', **{'max_mIoU': max_mIoU})
    """
    periodic_checkpointer = PeriodicCheckpointer(
      self.checkpointer, period=self.period, max_iter=self.maxsize, max_to_keep=self.max_to_keep)
    return periodic_checkpointer

  def step(self, itr, **kwargs):
    if self.save_circle:
      itr = itr % self.max_to_keep
      saved_name = f"model_{itr:08d}"
      self.periodic_checkpointer.save(name=saved_name, **self.state_dict, **kwargs)
    else:
      self.periodic_checkpointer.step(itr, **self.state_dict, **kwargs)
    pass

  def save(self, name, **kwargs):
    self.periodic_checkpointer.save(name=name, **self.state_dict, **kwargs)

  def resume_or_load(self, path, resume=True):
    loaded_dict = self.checkpointer.resume_or_load(path=path, resume=resume)
    self.logger.info("Unloaded keys: \n" + pprint.pformat(list(loaded_dict.keys())))
    for k in self.state_dict.keys():
      if k in loaded_dict:
        self.state_dict[k] = loaded_dict[k]
    self.logger.info("Loaded state_dict: \n" + pprint.pformat(self.state_dict))
    return loaded_dict


class VisualModelCkpt(Checkpointer):
  def __init__(self, model):
    logger = logging.getLogger('fvcore')
    if len(logger.handlers) == 0:
      setup_logger(name='fvcore')
    super(VisualModelCkpt, self).__init__(model, save_to_disk=False)
    pass

  def _load_from_path(self, checkpoint):

    checkpoint_state_dict = checkpoint
    self._convert_ndarray_to_tensor(checkpoint_state_dict)

    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching.
    fv_ckpt._strip_prefix_if_present(checkpoint_state_dict, "module.")

    # work around https://github.com/pytorch/pytorch/issues/24139
    model_state_dict = self.model.state_dict()
    incorrect_shapes = []
    for k in list(checkpoint_state_dict.keys()):
      if k in model_state_dict:
        shape_model = tuple(model_state_dict[k].shape)
        shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
        if shape_model != shape_checkpoint:
          incorrect_shapes.append((k, shape_checkpoint, shape_model))
          checkpoint_state_dict.pop(k)
    # pyre-ignore
    incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)
    return fv_ckpt._IncompatibleKeys(
      missing_keys=incompatible.missing_keys,
      unexpected_keys=incompatible.unexpected_keys,
      incorrect_shapes=incorrect_shapes,
    )

  def load_from_path(self, path):
    self.logger.info("Loading checkpoint from {}".format(path))
    checkpoint = self._load_file(path)
    incompatible = self._load_from_path(checkpoint)
    if (incompatible is not None):  # handle some existing subclasses that returns None
      self._log_incompatible_keys(incompatible)
    return checkpoint