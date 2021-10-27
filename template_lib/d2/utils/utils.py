from detectron2.engine import default_setup


def set_ddp_seed(seed=0, outdir=None, CUDNN_BENCHMARK=False):
  from template_lib.v2.config_cfgnode import TLCfgNode
  import argparse

  d2_cfg = TLCfgNode()
  d2_cfg.SEED = seed
  d2_cfg.OUTPUT_DIR = outdir
  d2_cfg.CUDNN_BENCHMARK = CUDNN_BENCHMARK

  args = argparse.Namespace()
  default_setup(d2_cfg, args)
  pass
