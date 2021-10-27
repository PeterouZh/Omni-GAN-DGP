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

from template_lib.d2template.scripts.build import build_start

logger = logging.getLogger("detectron2")


def setup(args, config):
  """
  Create configs and perform basic setups.
  """
  cfg = get_cfg()
  # cfg.merge_from_file(args.config_file)
  cfg.merge_from_list(args.opts)

  cfg = detection2_utils.D2Utils.cfg_merge_from_easydict(cfg, config)

  cfg.freeze()
  default_setup(
    cfg, args
  )  # if you don't like any of the default setup, write your own setup code
  return cfg


def main(args, myargs):
  cfg = setup(args, myargs.config)
  myargs = D2Utils.setup_myargs_for_multiple_processing(myargs)
  # seed_utils.set_random_seed(cfg.seed)

  build_start(cfg=cfg, args=args, myargs=myargs)

  modelarts_utils.modelarts_sync_results(args=myargs.args, myargs=myargs, join=True, end=True)
  return


def run(argv_str=None):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  if hasattr(myargs.config, 'datasets'):
    prepare_dataset(myargs.config.datasets, cfg=myargs.config)

  args = default_argument_parser().parse_args(args=[])
  args = config2args(myargs.config.args, args)

  args.opts += ['OUTPUT_DIR', args1.outdir + '/detectron2']
  print("Command Line Args:", args)

  myargs = D2Utils.unset_myargs_for_multiple_processing(myargs, num_gpus=args.num_gpus)

  launch(
    main,
    args.num_gpus,
    num_machines=args.num_machines,
    machine_rank=args.machine_rank,
    dist_url=args.dist_url,
    args=(args, myargs),
  )


if __name__ == "__main__":
  run()
  # from template_lib.examples import test_bash
  # test_bash.TestingUnit().test_resnet(gpu=os.environ['CUDA_VISIBLE_DEVICES'])
