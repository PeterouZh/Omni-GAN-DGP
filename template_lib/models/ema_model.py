import logging

import torch


class EMA(object):
  """
  # Simple wrapper that applies EMA to a model. Could be better done in 1.0 using
  # the parameters() and buffers() module functions, but for now this works
  # with state_dicts using .copy_
  Usage:
    ema = ema_model.EMA(G, G_ema, decay=config['ema_decay'],
                        start_itr=config['ema_start'])
    ema.update(state_dict['itr'])
  """

  def __init__(self, source, target, decay=0.9999, start_itr=0):
    self.source = source
    self.target = target
    self.decay = decay
    # Optional parameter indicating what iteration to start the decay at
    self.start_itr = start_itr
    logger = logging.getLogger('tl')
    # fix bug after calling model.cuda()
    # self.source_dict = self.source.state_dict()
    # self.target_dict = self.target.state_dict()
    logger.info('Initializing EMA parameters to be source parameters.')
    # Initialize target's params to be source's
    with torch.no_grad():
      source_dict = self.source.state_dict()
      target_dict = self.target.state_dict()
      for key in self.source.state_dict():
        target_dict[key].data.copy_(source_dict[key].data)

  def update(self, itr=None):
    # If an iteration counter is provided and itr is less than the start itr,
    # peg the ema weights to the underlying weights.
    if itr is not None and itr < self.start_itr:
      decay = 0.0
    else:
      decay = self.decay
    with torch.no_grad():
      source_dict = self.source.state_dict()
      target_dict = self.target.state_dict()
      for key in source_dict:
        target_dict[key].data.copy_(target_dict[key].data * decay + source_dict[key].data * (1 - decay))
