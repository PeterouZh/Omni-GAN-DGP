import pprint
import sys
import os
from collections import OrderedDict


class CheckpointTool(object):
  def __init__(self, ckptdir):
    self.ckptdir = ckptdir
    os.makedirs(ckptdir, exist_ok=True)
    pass

  def save_checkpoint(self, checkpoint_dict, is_best=True, filename='ckpt.tar'):
    """

    :param checkpoint_dict: dict
    :param is_best:
    :param filename:
    :return:
    """
    import torch, shutil
    filename = os.path.join(self.ckptdir, filename)
    state_dict = {}
    for key in checkpoint_dict:
      if hasattr(checkpoint_dict[key], 'state_dict'):
        state_dict[key] = getattr(checkpoint_dict[key], 'state_dict')()
      else:
        state_dict[key] = checkpoint_dict[key]

    torch.save(state_dict, filename)
    if is_best:
      shutil.copyfile(filename, filename + '.best')

  def load_checkpoint(self, checkpoint_dict, filename):
    """
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    :param filename:
    :return:
    """
    import torch
    if os.path.isfile(filename):
      state_dict = torch.load(filename)

      for key in checkpoint_dict:
        if hasattr(checkpoint_dict[key], 'state_dict'):
          checkpoint_dict[key].load_state_dict(state_dict.pop(key))

      return state_dict
    else:
      print("=> no checkpoint found at '{}'".format(filename))
      assert 0


# def get_tbwriter_logger_checkpoint(args, od, myargs, outdir, tbdir, logfile, ckptdir, **kwargs):
#   # save args to pickle
#   save_args_pickle_and_txt(args, os.path.join(outdir, 'args.pk'))
#
#   # logger
#   logger = get_logger(logfile, stream=True, propagate=False)
#   myargs.logger = logger
#   logger.info(pprint.pformat(od))
#
#   # config.json
#   if hasattr(args, 'config'):
#     logger.info('=> Load config.yaml.')
#     config_parser = YamlConfigParser(fname=args.config, saved_fname=od['configfile'])
#     config = config_parser.config_dotdict
#     myargs.config = config
#     config_str = pprint.pformat(myargs.config)
#     logger.info_msg(config_str)
#     config_str = config_str.strip().replace('\n', '  \n>')
#     pass
#
#   # tensorboard
#   tbtool = TensorBoardTool(dir_path=tbdir, tb_logdir=od['tb_logdir'])
#   writer = tbtool.run()
#   tbtool.add_text_args_and_od(args, od)
#   myargs.writer = writer
#
#   # checkpoint
#   checkpoint = CheckpointTool(ckptdir=ckptdir)
#   myargs.checkpoint = checkpoint
#   myargs.checkpoint_dict = OrderedDict()
#
#   if hasattr(args, 'config'):
#     myargs.writer.add_text('config', config_str, 0)
#     pass


def print_number_params(**model_dict):
  for label, model in model_dict.items():
    print('Number of params in {}:\t {}M'.format(
      label, sum([p.data.nelement() for p in model.parameters()]) / 1e6
    ))

