from easydict import EasyDict
import logging
import sys

from detectron2.config import CfgNode

try:
  from yacs.config import CfgNode as yacs_CfgNode
except:
  import traceback
  traceback.print_exc()

# from template_lib.utils import logging_utils


def _convert_dict_2_CfgNode(config):
  ret_cfg = CfgNode()
  for k in config:
    if isinstance(config[k], dict):
      setattr(ret_cfg, k, _convert_dict_2_CfgNode(config[k]))
      pass
    else:
      val = config[k]
      if isinstance(config[k], list):
        if len(config[k]) > 0 and isinstance(config[k][0], dict):
          temp = list(map(dict, config[k]))
          val = temp
          pass
      setattr(ret_cfg, k, val)
  return ret_cfg

def _allow_CfgNode_new_allowed(cfg):
  cfg.__dict__[CfgNode.NEW_ALLOWED] = True
  for k in cfg:
    if isinstance(cfg[k], (CfgNode, yacs_CfgNode)):
      cfg[k] = _allow_CfgNode_new_allowed((cfg[k]))
  return cfg


class D2Utils(object):
  """

  """
  @staticmethod
  def cfg_merge_from_easydict(cfg, config):
    config = EasyDict(config)
    config_cfg = _convert_dict_2_CfgNode(config)
    # cfg = CfgNode(cfg, new_allowed=True)
    cfg = _allow_CfgNode_new_allowed(cfg)
    cfg.merge_from_other_cfg(config_cfg)
    return cfg

  @staticmethod
  def unset_myargs_for_multiple_processing(myargs, num_gpus):
    from detectron2.utils import comm
    distributed = num_gpus > 1
    if distributed:
      myargs.writer = None
      myargs.logger = None
      sys.stdout = myargs.stdout
      sys.stderr = myargs.stderr
      myargs.stdout = None
      myargs.stderr = None
    return myargs

  # @staticmethod
  # def setup_myargs_for_multiple_processing(myargs):
  #   from detectron2.utils import comm
  #   distributed = comm.get_world_size() > 1
  #   if distributed and comm.is_main_process():
  #     # setup logging in the project
  #     logfile = myargs.args.logfile
  #     logging_utils.get_logger(
  #       filename=logfile, logger_names=['template_lib', 'tl'], stream=True)
  #     logger = logging.getLogger('tl')
  #     myargs.logger = logger
  #     myargs.stdout = sys.stdout
  #     myargs.stderr = sys.stderr
  #     logging_utils.redirect_print_to_logger(logger=logger)
  #   return myargs

  @staticmethod
  def create_cfg():
    """
    Create configs and perform basic setups.
    """
    from detectron2.config import CfgNode
    # detectron2 default cfg
    # cfg = get_cfg()
    cfg = CfgNode()
    cfg.OUTPUT_DIR = "./output"
    cfg.SEED = -1
    cfg.CUDNN_BENCHMARK = False
    cfg.DATASETS = CfgNode()
    cfg.SOLVER = CfgNode()

    cfg.DATALOADER = CfgNode()
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

    cfg.MODEL = CfgNode()
    cfg.MODEL.KEYPOINT_ON = False
    cfg.MODEL.LOAD_PROPOSALS = False
    cfg.MODEL.WEIGHTS = ""

    cfg.freeze()
    return cfg


