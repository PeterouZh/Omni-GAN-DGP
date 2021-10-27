import copy
import functools
import logging
import os

import tqdm
import collections
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import numpy as np

from fvcore.common.timer import Timer
from detectron2.structures import ImageList
from detectron2.utils import comm
from detectron2.utils.comm import get_world_size
from detectron2.utils.events import get_event_storage
from detectron2.checkpoint import Checkpointer, DetectionCheckpointer

from template_lib.utils import get_eval_attr, print_number_params, get_attr_kwargs, get_attr_eval

from .build import TRAINER_REGISTRY


from detectron2.data import (
  get_detection_dataset_dicts,
  DatasetFromList, DatasetMapper, MapDataset, samplers,
)
from template_lib.d2.data import build_dataset_mapper


def _trivial_batch_collator(batch):
  """
  A batch collator that does nothing.
  """
  return batch


def build_detection_test_loader(cfg, dataset_name, batch_size, mapper=None):

  dataset_dicts = get_detection_dataset_dicts(
    [dataset_name],
    filter_empty=False,
    proposal_files=[
      cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
    ]
    if cfg.MODEL.LOAD_PROPOSALS
    else None,
  )

  dataset = DatasetFromList(dataset_dicts)
  if mapper is None:
    mapper = DatasetMapper(cfg, False)
  dataset = MapDataset(dataset, mapper)

  sampler = samplers.InferenceSampler(len(dataset))
  # Always use 1 image per worker during inference since this is the
  # standard when reporting inference time in papers.
  batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)

  data_loader = torch.utils.data.DataLoader(
    dataset,
    num_workers=cfg.DATALOADER.NUM_WORKERS,
    batch_sampler=batch_sampler,
    collate_fn=_trivial_batch_collator,
  )
  return data_loader



class DumpModule(nn.Module):
  def __init__(self, model_dict):
    super(DumpModule, self).__init__()
    for name, model in model_dict.items():
      setattr(self, name, model)
    pass


@TRAINER_REGISTRY.register()
class BaseTrainer(nn.Module):

    def __init__(self, cfg, args, iter_every_epoch, **kwargs):
      super().__init__()

      # fmt: off
      self.cfg                           = cfg
      self.args                          = args
      self.iter_every_epoch              = iter_every_epoch
      # fmt: on

      self.timer = Timer()
      self.device = torch.device(f'cuda:{comm.get_rank()}')
      self.logger = logging.getLogger('tl')
      self.num_gpu = comm.get_world_size()
      self.distributed = comm.get_world_size() > 1

      # torch.cuda.set_device(self.device)
      # self.build_models(cfg=cfg)
      self.to(self.device)

    def build_models(self, **kwargs):
      self.models = {}
      self.optims = {}
      self.schedulers = {}

      self._print_number_params(self.models)

    def build_optimizer(self):

      optims_dict = self.optims

      return optims_dict

    def build_lr_scheduler(self):
      scheduler = {}
      if hasattr(self, 'schedulers'):
        scheduler = self.schedulers
      return scheduler

    def get_saved_model(self):
      models = {}
      for name, model in self.models.items():
        if isinstance(model, DistributedDataParallel):
          models[name] = model.module
        else:
          models[name] = model
      saved_model = DumpModule(models)
      return saved_model

    def train_func(self, data, iteration, pbar):
      """Perform architecture search by training a controller and shared_cnn.
      """
      if comm.is_main_process() and iteration % self.iter_every_epoch == 0:
        pbar.set_postfix_str(s="BaseTrainer ")

      images, labels = self._preprocess_image(data)
      images = images.tensor

      comm.synchronize()

    def _get_bs_per_worker(self, bs):
      num_workers = get_world_size()
      bs_per_worker = bs // num_workers
      return bs_per_worker

    def _get_tensor_of_main_processing(self, tensor):
      tensor_list = comm.all_gather(tensor)
      tensor = tensor_list[0].to(self.device)
      return tensor

    def _preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        labels = torch.LongTensor([x["label"] for x in batched_inputs]).to(self.device)
        images = ImageList.from_tensors(images)
        return images, labels

    def _get_ckpt_path(self, ckpt_dir, ckpt_epoch, iter_every_epoch):
      eval_iter = (ckpt_epoch) * iter_every_epoch - 1
      eval_ckpt = os.path.join(ckpt_dir, f'model_{eval_iter:07}.pth')
      self.logger.info(f'Load weights:\n{os.path.abspath(eval_ckpt)}')
      return eval_ckpt

    def _print_number_params(self, models_dict):
      print_number_params(models_dict=models_dict)

    def after_resume(self):
      pass

    def build_test_loader(self, cfg, dataset_name, batch_size, dataset_mapper):

      if dataset_mapper is not None:
        dataset_mapper = build_dataset_mapper(dataset_mapper)

      data_loader = build_detection_test_loader(
        cfg, dataset_name=dataset_name, batch_size=batch_size, mapper=dataset_mapper)
      return data_loader

    def load_model_weights(self, ckpt_dir, ckpt_epoch, ckpt_iter_every_epoch, ckpt_path=None):
      if ckpt_path is None:
        ckpt_path = self._get_ckpt_path(ckpt_dir, ckpt_epoch, ckpt_iter_every_epoch)

      model = self.get_saved_model()
      checkpointer = Checkpointer(model, save_to_disk=False)
      checkpointer.resume_or_load(ckpt_path, resume=False)
      pass