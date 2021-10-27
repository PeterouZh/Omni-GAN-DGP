import torch
from template_lib.d2.utils import comm
import collections

from template_lib.trainer.base_trainer import Trainer
from template_lib.utils import get_attr_kwargs

from .build import GAN_MODEL_REGISTRY



@GAN_MODEL_REGISTRY.register()
class WGANGPCond(object):

  def __init__(self, cfg, **kwargs):

    self.myargs                 = kwargs['myargs']
    self.D                      = kwargs['D']
    self.G                      = kwargs['G']
    self.D_optim                = kwargs['D_optim']
    self.G_optim                = kwargs['G_optim']
    self.n_critic               = cfg.n_critic
    self.gp_lambda              = getattr(cfg, 'gp_lambda', 10.)
    self.child_grad_bound       = getattr(cfg, 'child_grad_bound', -1)
    self.log_every              = getattr(cfg, 'log_every', 50)
    self.dummy                  = getattr(cfg, 'dummy', False)

    self.device = torch.device(f'cuda:{comm.get_rank()}')
    pass

  def __call__(self, images, labels, z, iteration, **kwargs):
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

    d_real = self.D(real, dy)
    d_real_mean = d_real.mean()
    summary_d['d_logit_mean']['d_real_mean'] = d_real_mean.item()

    z_sample = z.sample()
    z_sample = z_sample.to(self.device)
    fake = self.G(z_sample, y=gy, **kwargs)
    d_fake = self.D(fake.detach(), gy, **kwargs)
    d_fake_mean = d_fake.mean()
    summary_d['d_logit_mean']['d_fake_mean'] = d_fake_mean.item()

    gp = self.wgan_gp_gradient_penalty_cond(
      x=real, G_z=fake, gy=gy, f=self.D, backward=False, gp_lambda=self.gp_lambda)
    summary_d['gp']['gp'] = gp.item()

    wd = d_real_mean - d_fake_mean
    summary_d['wd']['wd'] = wd.item()
    d_loss = -wd + gp

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
      d_fake_g = self.D(fake, gy, **kwargs)
      d_fake_g_mean = d_fake_g.mean()

      g_loss = -d_fake_g_mean
      g_loss.backward()
      summary_d['d_logit_mean']['d_fake_g_mean'] = d_fake_g_mean.item()
      summary_d['g_loss']['g_loss'] = g_loss.item()

      if self.child_grad_bound > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.child_grad_bound)
        summary_d['grad_norm']['grad_norm'] = grad_norm

      self.G_optim.step()

    if iteration % self.log_every == 0:
      Trainer.summary_defaultdict2txtfig(default_dict=summary_d,
                                         prefix='WGANGPCond',
                                         step=iteration,
                                         textlogger=self.myargs.textlogger)

    comm.synchronize()
    return

  @staticmethod
  def wgan_gp_gradient_penalty_cond(x, G_z, gy, f, backward=False, gp_lambda=10,
                                    return_gp=False):
    """
    gradient penalty for conditional discriminator
    :param x:
    :param G_z:
    :param gy: label for x * alpha + (1 - alpha) * G_z
    :param f:
    :return:
    """
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (G_z - x)

    # gradient penalty
    z.requires_grad_()
    o = f(z, gy)
    # o = torch.nn.parallel.data_parallel(f, (z, gy))
    g = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()
    if backward:
      gp_loss = gp * gp_lambda
      gp_loss.backward()
    else:
      gp_loss = gp * gp_lambda
    if return_gp:
      return gp_loss, gp
    else:
      return gp_loss
