import collections
import logging
import os
import weakref
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import tqdm

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import (
  MetadataCatalog,
  build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
  CommonMetricPrinter,
  EventStorage,
)

from template_lib.utils import detection2_utils, get_attr_eval, get_attr_kwargs
from template_lib.utils.detection2_utils import D2Utils
from template_lib.d2.data import build_dataset_mapper
from template_lib.d2template.trainer import build_trainer
from template_lib.d2template.scripts import build_start, START_REGISTRY

from template_lib.v2.config import update_parser_defaults_from_yaml
from template_lib.v2.config import get_dict_str, setup_logger_global_cfg_global_textlogger
from template_lib.v2.config import global_cfg
from template_lib.d2.utils import D2Utils

logger = logging.getLogger("detectron2")


@START_REGISTRY.register()
def do_train(cfg, args):
  # fmt: off
  run_func                                     = cfg.start.get('run_func', 'train_func')
  dataset_name                                 = cfg.start.dataset_name
  IMS_PER_BATCH                                = cfg.start.IMS_PER_BATCH * comm.get_world_size()
  NUM_WORKERS                                  = cfg.start.NUM_WORKERS
  dataset_mapper                               = cfg.start.dataset_mapper

  max_epoch                                    = cfg.start.max_epoch
  checkpoint_period                            = cfg.start.checkpoint_period

  resume_cfg                                   = get_attr_kwargs(cfg.start, 'resume_cfg', default=None)

  cfg.defrost()
  cfg.DATASETS.TRAIN                           = (dataset_name, )
  cfg.SOLVER.IMS_PER_BATCH                     = IMS_PER_BATCH
  cfg.DATALOADER.NUM_WORKERS                   = NUM_WORKERS
  cfg.freeze()
  # fmt: on

  # build dataset
  mapper = build_dataset_mapper(dataset_mapper)
  data_loader = build_detection_train_loader(cfg, mapper=mapper)
  metadata = MetadataCatalog.get(dataset_name)
  num_samples = metadata.get('num_samples')
  iter_every_epoch = num_samples // IMS_PER_BATCH
  max_iter = iter_every_epoch * max_epoch

  model = build_trainer(cfg=cfg, args=args, iter_every_epoch=iter_every_epoch,
                        batch_size=IMS_PER_BATCH, max_iter=max_iter, metadata=metadata, max_epoch=max_epoch,
                        data_loader=data_loader)
  model.train()

  # optimizer = build_optimizer(cfg, model)
  optims_dict = model.build_optimizer()
  scheduler = model.build_lr_scheduler()

  checkpointer = DetectionCheckpointer(model.get_saved_model(), cfg.OUTPUT_DIR, **optims_dict, **scheduler)
  if resume_cfg and resume_cfg.resume:
    resume_ckpt_dir = model._get_ckpt_path(ckpt_dir=resume_cfg.ckpt_dir, ckpt_epoch=resume_cfg.ckpt_epoch,
                                           iter_every_epoch=resume_cfg.iter_every_epoch)
    start_iter = (checkpointer.resume_or_load(resume_ckpt_dir).get("iteration", -1) + 1)
    if get_attr_kwargs(resume_cfg, 'finetune', default=False):
      start_iter = 0
    model.after_resume()
  else:
    start_iter = 0

  if run_func != 'train_func':
    eval(f'model.{run_func}()')
    exit(0)

  checkpoint_period = eval(checkpoint_period, dict(iter_every_epoch=iter_every_epoch))
  periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter)
  logger.info("Starting training from iteration {}".format(start_iter))
  # modelarts_utils.modelarts_sync_results(args=myargs.args, myargs=myargs, join=True, end=False)
  with EventStorage(start_iter) as storage:
    pbar = zip(data_loader, range(start_iter, max_iter))
    if comm.is_main_process():
      pbar = tqdm.tqdm(pbar,
                       desc=f'do_train, {args.tl_time_str}, '
                            f'iters {iter_every_epoch} * bs {IMS_PER_BATCH} = '
                            f'imgs {iter_every_epoch*IMS_PER_BATCH}',
                       initial=start_iter, total=max_iter)

    for data, iteration in pbar:
      comm.synchronize()
      iteration = iteration + 1
      storage.step()

      model.train_func(data, iteration - 1, pbar=pbar)

      periodic_checkpointer.step(iteration)
      pass
  # modelarts_utils.modelarts_sync_results(args=myargs.args, myargs=myargs, join=True, end=True)
  comm.synchronize()


def setup(args, config):
  """
  Create configs and perform basic setups.
  """
  from detectron2.config import CfgNode
  # detectron2 default cfg
  # cfg = get_cfg()
  cfg = CfgNode()
  cfg.SEED = -1
  cfg.CUDNN_BENCHMARK = False
  cfg.DATASETS = CfgNode()
  cfg.SOLVER = CfgNode()

  cfg.DATALOADER = CfgNode()
  cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
  cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
  cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

  cfg.MODEL = CfgNode()
  cfg.MODEL.KEYPOINT_ON = False
  cfg.MODEL.LOAD_PROPOSALS = False
  cfg.MODEL.WEIGHTS = ""

  if args.config_file:
    cfg.merge_from_file(args.config_file)

  cfg.OUTPUT_DIR = f'{args.tl_outdir}/detectron2'
  cfg.merge_from_list(args.opts)

  cfg = D2Utils.cfg_merge_from_easydict(cfg, config)

  cfg.freeze()
  default_setup(
    cfg, args
  )  # if you don't like any of the default setup, write your own setup code
  return cfg


def main(args):

  setup_logger_global_cfg_global_textlogger(args, tl_textdir=args.tl_textdir)

  cfg = setup(args, global_cfg)
  # seed_utils.set_random_seed(cfg.seed)

  build_start(cfg=cfg, args=args)
  return


def run():

  parser = default_argument_parser()
  update_parser_defaults_from_yaml(parser)
  args = parser.parse_args()

  logger = logging.getLogger('tl')
  logger.info(get_dict_str(vars(args)))

  launch(
    main,
    args.num_gpus,
    num_machines=args.num_machines,
    machine_rank=args.machine_rank,
    dist_url=args.dist_url,
    args=(args, ),
  )


if __name__ == "__main__":
  run()
  # from template_lib.examples import test_bash
  # test_bash.TestingUnit().test_resnet(gpu=os.environ['CUDA_VISIBLE_DEVICES'])
