import os
import argparse

import torch as th
import torch.multiprocessing as mp
import torch.distributed as dist


class Mod(th.nn.Module):
  def __init__(self):
    super().__init__()
    self.lin = th.nn.Conv2d(3, 5, 3, padding=1)
    self.bn = th.nn.BatchNorm2d(5)

  def forward(self, z):
    print('forward_before\t{}'.format(self.bn.running_mean))
    t = self.lin(z)
    return (self.bn(t)).sum()


def main_worker(rank, world_size, seed=0, broadcast_buffers=True, use_sync_bn=False):
  world_size = world_size

  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12345'

  # initialize the process group
  dist.init_process_group("nccl", rank=rank, world_size=world_size)

  mod = Mod().cuda(rank)

  if use_sync_bn:
    process_group = th.distributed.new_group(list(range(world_size)))
    mod = th.nn.SyncBatchNorm.convert_sync_batchnorm(mod, process_group)

  optim = th.optim.Adam(mod.parameters(), lr=1e-3)

  mod = th.nn.parallel.DistributedDataParallel(mod,
                                               device_ids=[rank], output_device=rank,
                                               broadcast_buffers=broadcast_buffers)

  if rank % 2 == 0:
    z1 = th.zeros(7, 3, 5, 5).cuda(rank)
  else:
    z1 = th.ones(7, 3, 5, 5).cuda(rank)

  out = mod(z1)

  dist.barrier()
  # if rank == 1:
  print('forward_after\t{}'.format(mod.module.bn.running_mean))

  # mod(z2) # <<---- The presence of this unused call causes an inplace error in backward() below if dec is a DDP module.

  loss = (out ** 2).mean()

  optim.zero_grad()
  loss.backward()
  optim.step()

  dist.barrier()
  print('backward_after\t{}'.format(mod.module.bn.running_mean))

  out = mod(z1)

  dist.barrier()
  print('forward_after\t{}'.format(mod.module.bn.running_mean))
  pass


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
  broadcast_buffers = False
  use_sync_bn = True

  mp.spawn(main_worker, nprocs=2, args=(2, 0, broadcast_buffers, use_sync_bn))
