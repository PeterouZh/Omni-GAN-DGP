import os
import random

import torch
import torch.nn.functional as F

from template_lib.utils import get_attr_kwargs

from .build import GAN_LOSS_REGISTRY


def _multilabel_categorical_crossentropy(y_true, y_pred, margin=0., gamma=1.):
  """
  y_true: positive=1, negative=0, ignore=-1

  """
  y_true = y_true.clamp(-1, 1)

  y_pred = y_pred + margin
  y_pred = y_pred * gamma

  y_pred[y_true == 1] = -1 * y_pred[y_true == 1]
  y_pred[y_true == -1] = -1e12

  y_pred_neg = y_pred.clone()
  y_pred_neg[y_true == 1] = -1e12

  y_pred_pos = y_pred.clone()
  y_pred_pos[y_true == 0] = -1e12

  zeros = torch.zeros_like(y_pred[..., :1])
  y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
  y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
  neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
  pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
  return neg_loss + pos_loss


@GAN_LOSS_REGISTRY.register(name=f"{__name__}.OmniLoss")
class OmniLoss(object):

  def __init__(self, cfg, **kwargs):
    # fmt: off
    self.default_label              = get_attr_kwargs(cfg, 'default_label', default=0, **kwargs)
    self.margin                     = get_attr_kwargs(cfg, 'margin', default=0., **kwargs)
    self.gamma                      = get_attr_kwargs(cfg, 'gamma', default=1., **kwargs)
    # fmt: on
    pass

  @staticmethod
  def get_one_hot(label_list, one_hot, b, filled_value=0):

    for label in label_list:
      if isinstance(label, int):
        label = torch.empty(b, dtype=torch.int64, device=one_hot.device).fill_(label)
      one_hot.scatter_(dim=1, index=label.view(-1, 1), value=filled_value)
    return one_hot

  def __call__(self,
               pred,
               positive=None,
               negative=None,
               default_label=None,
               margin=None,
               gamma=None,
               return_logits=False):
    default_label = self.default_label if default_label is None else default_label
    margin = self.margin if margin is None else margin
    gamma = self.gamma if gamma is None else gamma

    b, nc = pred.shape[:2]
    label_onehot = torch.empty(b, nc, dtype=torch.int64, device=pred.device).fill_(default_label)

    if positive is not None:
      label_onehot = OmniLoss.get_one_hot(label_list=positive, one_hot=label_onehot, b=b, filled_value=1)

    if negative is not None:
      label_onehot = OmniLoss.get_one_hot(label_list=negative, one_hot=label_onehot, b=b, filled_value=0)

    loss = _multilabel_categorical_crossentropy(
      y_true=label_onehot, y_pred=pred, margin=margin, gamma=gamma)
    loss_mean = loss.mean()

    if return_logits:
      logits_pos = pred.detach()[label_onehot == 1].mean().item()
      logits_neg = pred.detach()[label_onehot == 0].mean().item()
      return loss_mean, logits_pos, logits_neg
    else:
      return loss_mean












