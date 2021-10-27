import numpy as np
import os
import argparse
import torch
import torch.distributed as dist


def main():
  from template_lib.v2.config import update_parser_defaults_from_yaml

  parser = argparse.ArgumentParser()
  parser.add_argument("--local_rank", type=int, default=0)
  parser.add_argument("--run_func", type=str)

  update_parser_defaults_from_yaml(parser)

  args = parser.parse_args()

  n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
  args.distributed = n_gpu > 1

  if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # comm.synchronize()
    dist.barrier()

  # eval(args.run_func)()
  return args


def ddp_init():
  """
  use config_cfgnode
  """
  from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml
  from template_lib.v2.ddp import parser_local_rank

  rank = parser_local_rank()

  parser = update_parser_defaults_from_yaml(parser=None, append_local_rank=True,
                                            is_main_process=(rank==0))

  args = parser.parse_args()

  n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
  args.distributed = n_gpu > 1

  if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # comm.synchronize()
    dist.barrier()

  # eval(args.run_func)()
  return args


def gather_tensor_of_master(tensor):
  from template_lib.d2.utils import comm

  tensor_list = comm.all_gather(tensor)
  tensor = tensor_list[0].to(tensor.device)
  return tensor


def gather_tensor(data):
  from template_lib.d2.utils import comm

  data_list = comm.gather(data=data)
  if len(data_list) > 0:
    if isinstance(data, np.ndarray):
      data = np.concatenate(data_list, axis=0)
    else:
      data = torch.cat(data_list, dim=0).to(device=data.device)
  else:
    data = None
  return data

def parser_local_rank():
  parser = argparse.ArgumentParser()
  parser.add_argument("--local_rank", type=int, default=0)
  args, _ = parser.parse_known_args()
  return args.local_rank