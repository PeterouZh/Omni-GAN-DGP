import torch.nn.functional as F
import torch
import torch.nn as nn
from template_lib.d2.utils import comm
import collections

from template_lib.trainer.base_trainer import Trainer
from template_lib.utils import get_attr_kwargs

from .build import GAN_MODEL_REGISTRY


class CircleLoss(nn.Module):
  def __init__(self, m=0.25, gamma=80):
    super(CircleLoss, self).__init__()
    self.m = m
    self.gamma = gamma
    self.soft_plus = nn.Softplus()

  def forward(self, sp, sn):
    ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
    an = torch.clamp_min(sn.detach() + self.m, min=0.)

    delta_p = 1 - self.m
    delta_n = self.m

    logit_p = - ap * (sp - delta_p) * self.gamma
    logit_n = an * (sn - delta_n) * self.gamma

    loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

    return loss


@GAN_MODEL_REGISTRY.register()
class CircleLossCond(object):

  def __init__(self, cfg, **kwargs):

    self.myargs                 = kwargs['myargs']
    self.m                      = get_attr_kwargs(cfg, 'm', default=0.25, **kwargs)
    self.gamma                  = get_attr_kwargs(cfg, 'gamma', default=80, **kwargs)
    self.use_sigmoid            = get_attr_kwargs(cfg, 'use_sigmoid', default=False, **kwargs)
    self.D                      = kwargs['D']
    self.G                      = kwargs['G']
    self.D_optim                = kwargs['D_optim']
    self.G_optim                = kwargs['G_optim']
    self.n_critic               = get_attr_kwargs(cfg, 'n_critic', default=5, **kwargs)
    self.log_every              = getattr(cfg, 'log_every', 50)
    self.dummy                  = getattr(cfg, 'dummy', False)

    self.circle_loss = CircleLoss(m=self.m, gamma=self.gamma)

    self.device = torch.device(f'cuda:{comm.get_rank()}')
    pass

  def __call__(self, images, labels, z, iteration, ema=None, **kwargs):
    """

    :param images:
    :param labels:
    :param z: z.sample()
    :param iteration:
    :param kwargs:
    :return:
    """

    if self.dummy:
      return

    summary_d = collections.defaultdict(dict)

    real = images
    dy = labels
    gy = dy

    self.G.train()
    self.D.train()
    self.D.zero_grad()

    d_real = self.D(real, y=dy, **kwargs)

    z_sample = z.sample()
    z_sample = z_sample.to(self.device)
    fake = self.G(z_sample, y=gy, **kwargs)
    d_fake = self.D(fake.detach(), y=gy, **kwargs)

    r_logit_mean, f_logit_mean, d_loss = self.circle_loss_discriminator(r_logit=d_real, f_logit=d_fake)
    summary_d['d_logit_mean']['r_logit_mean'] = r_logit_mean.item()
    summary_d['d_logit_mean']['f_logit_mean'] = f_logit_mean.item()

    d_loss.backward()
    self.D_optim.step()
    summary_d['d_loss']['d_loss'] = d_loss.item()

    ############################
    # (2) Update G network
    ###########################
    if iteration % self.n_critic == 0:
      self.G.zero_grad()
      z_sample = z.sample()
      z_sample = z_sample.to(self.device)
      gy = dy

      fake = self.G(z_sample, y=gy, **kwargs)
      d_fake_g = self.D(fake, y=gy, **kwargs)

      G_f_logit_mean, g_loss = self.circle_loss_generator(r_logit=d_real, f_logit=d_fake_g)
      summary_d['d_logit_mean']['G_f_logit_mean'] = G_f_logit_mean.item()
      summary_d['g_loss']['g_loss'] = g_loss.item()

      g_loss.backward()
      self.G_optim.step()

      if ema is not None:
        ema.update(iteration)

    if iteration % self.log_every == 0:
      Trainer.summary_defaultdict2txtfig(default_dict=summary_d,
                                         prefix='CircleLossCond',
                                         step=iteration,
                                         textlogger=self.myargs.textlogger)

    comm.synchronize()
    return

  def circle_loss_discriminator(self, r_logit, f_logit):
    r_logit_mean = r_logit.mean()
    f_logit_mean = f_logit.mean()
    if self.use_sigmoid:
      r_logit = torch.sigmoid(r_logit)
      f_logit = torch.sigmoid(f_logit)
    D_loss = self.circle_loss(sp=r_logit, sn=f_logit)

    return r_logit_mean, f_logit_mean, D_loss

  def circle_loss_generator(self, r_logit, f_logit):
    f_logit_mean = f_logit.mean()
    if self.use_sigmoid:
      r_logit = torch.sigmoid(r_logit)
      f_logit = torch.sigmoid(f_logit)

    G_loss = - self.circle_loss(sp=r_logit.detach(), sn=f_logit)
    return f_logit_mean, G_loss
