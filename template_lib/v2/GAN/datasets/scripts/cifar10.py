import torch.distributed as dist
from template_lib.d2.data_v2 import build_dataloader


if __name__ == '__main__':
  from template_lib.v2.ddp import ddp_init
  from template_lib.v2.config_cfgnode import global_cfg

  ddp_init()

  ddp_dataloader = build_dataloader(global_cfg)
  ddp_data_iter = iter(ddp_dataloader)
  ddp_data, ddp_label = next(ddp_data_iter)

  dataloader = build_dataloader(global_cfg, distributed=False, batch_size=global_cfg.batch_size*2,
                                kwargs_priority=True)
  data_iter = iter(dataloader)
  data, label = next(data_iter)

  pass
