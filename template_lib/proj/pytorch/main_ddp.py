import argparse
import os
import random
import numpy as np

import torch
import torch.distributed as dist
import torch.utils.data as data_utils
import torchvision.transforms as tv_trans
from mmcv.runner import dist_utils

from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, global_cfg
from template_lib.modelarts import modelarts_utils
from template_lib.proj import pil_utils, torch_data_utils, ddp_utils


def setup_runtime(seed):
  # Setup CUDNN
  if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

  # Setup random seeds for reproducibility
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  pass


def build_parser():
  ## runtime arguments
  parser = argparse.ArgumentParser(description='Training configurations.')
  parser.add_argument('--launcher', default='pytorch', type=str, choices=['pytorch', ], help='Launcher')
  parser.add_argument('--local_rank', type=int, default=0)
  parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
  parser.add_argument('--num_workers', type=int, default=0)
  return parser


def main():
  parser = build_parser()
  args, _ = parser.parse_known_args()
  is_main_process = args.local_rank == 0

  update_parser_defaults_from_yaml(parser, is_main_process=is_main_process)

  if is_main_process:
    modelarts_utils.setup_tl_outdir_obs(global_cfg)
    modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)

  args = parser.parse_args()

  setup_runtime(seed=args.seed)

  distributed = ddp_utils.is_distributed()
  if distributed:
      dist_utils.init_dist(args.launcher, backend='nccl')
      # important: use different random seed for different process
      torch.manual_seed(args.seed + dist.get_rank())

  # dataset
  dataset = torch_data_utils.ImageListDataset(meta_file=global_cfg.image_list_file, )
  if distributed:
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
  else:
    sampler = None

  train_loader = data_utils.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=sampler,
    num_workers=args.num_workers,
    pin_memory=False)

  # test
  data_iter = iter(train_loader)
  data = next(data_iter)

  if is_main_process:
    modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
  if distributed:
    dist.barrier()
  pass


if __name__ == '__main__':
  main()
