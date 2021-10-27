from detectron2.utils import comm
from template_lib.v2.GAN.evaluation import build_GAN_metric
from template_lib.v2.ddp import ddp_init
from template_lib.v2.config_cfgnode import global_cfg
from template_lib.d2.data_v2 import build_dataloader


def run(args):
  from template_lib.d2.utils import set_ddp_seed
  set_ddp_seed(outdir=f"{global_cfg.tl_outdir}/d2")

  total_batch_size = global_cfg.build_dataloader.batch_size
  num_workers = comm.get_world_size()
  batch_size = total_batch_size // num_workers

  data_loader = build_dataloader(global_cfg.build_dataloader, kwargs_priority=True,
                                 batch_size=batch_size, distributed=args.distributed)

  FID_IS_torch = build_GAN_metric(global_cfg.GAN_metric)
  if global_cfg.tl_debug:
    num_images = 50
  else:
    num_images = float('inf')
  FID_IS_torch.calculate_fid_stat_of_dataloader(data_loader=data_loader, num_images=num_images,
                                                save_fid_stat=global_cfg.save_fid_stat)

  comm.synchronize()

  pass


if __name__ == '__main__':
  args = ddp_init()
  run(args)









