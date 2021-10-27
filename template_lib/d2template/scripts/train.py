import logging
import os
import weakref
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import tqdm

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import (
  MetadataCatalog,
  build_detection_test_loader,
  build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (CommonMetricPrinter, EventStorage, JSONWriter, TensorboardXWriter)

from template_lib.utils import detection2_utils, get_attr_eval
from template_lib.utils.detection2_utils import D2Utils
from template_lib.utils import modelarts_utils
from template_lib.utils import seed_utils
from template_lib.utils.modelarts_utils import prepare_dataset
from template_lib.d2.data import build_dataset_mapper

from ..trainer import build_trainer
from .build import START_REGISTRY, build_start

logger = logging.getLogger("detectron2")

@START_REGISTRY.register()
def train(cfg, args, myargs):
  dataset_name                                 = cfg.start.dataset_name
  IMS_PER_BATCH                                = cfg.start.IMS_PER_BATCH
  max_epoch                                    = cfg.start.max_epoch
  ASPECT_RATIO_GROUPING                        = cfg.start.ASPECT_RATIO_GROUPING
  NUM_WORKERS                                  = cfg.start.NUM_WORKERS
  checkpoint_period                            = cfg.start.checkpoint_period

  cfg.defrost()
  cfg.DATASETS.TRAIN                           = (dataset_name, )
  cfg.SOLVER.IMS_PER_BATCH                     = IMS_PER_BATCH
  cfg.DATALOADER.ASPECT_RATIO_GROUPING         = ASPECT_RATIO_GROUPING
  cfg.DATALOADER.NUM_WORKERS                   = NUM_WORKERS
  cfg.freeze()

  # build dataset
  mapper = build_dataset_mapper(cfg)
  data_loader = build_detection_train_loader(cfg, mapper=mapper)
  metadata = MetadataCatalog.get(dataset_name)
  num_images = metadata.get('num_images')
  iter_every_epoch = num_images // IMS_PER_BATCH
  max_iter = iter_every_epoch * max_epoch

  model = build_trainer(cfg, myargs=myargs, iter_every_epoch=iter_every_epoch)
  model.train()

  logger.info("Model:\n{}".format(model))

  # optimizer = build_optimizer(cfg, model)
  optims_dict = model.build_optimizer()
  # scheduler = build_lr_scheduler(cfg, optimizer)

  checkpointer = DetectionCheckpointer(model.get_saved_model(), cfg.OUTPUT_DIR, **optims_dict)
  start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume).get("iteration", -1) + 1)

  checkpoint_period = eval(checkpoint_period, dict(iter_every_epoch=iter_every_epoch))
  periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter)

  logger.info("Starting training from iteration {}".format(start_iter))
  modelarts_utils.modelarts_sync_results(args=myargs.args, myargs=myargs, join=True, end=False)
  with EventStorage(start_iter) as storage:
    pbar = zip(data_loader, range(start_iter, max_iter))
    if comm.is_main_process():
      pbar = tqdm.tqdm(pbar,
                       desc=f'train, {myargs.args.time_str_suffix}, '
                            f'iters {iter_every_epoch} * bs {IMS_PER_BATCH} = imgs {iter_every_epoch*IMS_PER_BATCH}',
                       file=myargs.stdout, initial=start_iter, total=max_iter)

    for data, iteration in pbar:
      comm.synchronize()
      iteration = iteration + 1
      storage.step()

      model.train_func(data, iteration - 1, pbar=pbar)

      periodic_checkpointer.step(iteration)
      pass
  modelarts_utils.modelarts_sync_results(args=myargs.args, myargs=myargs, join=True, end=True)
  comm.synchronize()







