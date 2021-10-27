import torch.autograd as autograd
from torch.autograd import grad
import torch.nn.functional as F
import torch
from template_lib.d2.utils import comm
import collections

from template_lib.trainer.base_trainer import Trainer
from template_lib.utils import get_attr_kwargs

from .build import GAN_MODEL_REGISTRY



@GAN_MODEL_REGISTRY.register()
class HingeGPRealLossCond(object):

  def __init__(self, cfg, **kwargs):

    self.myargs                 = kwargs['myargs']
    self.D                      = kwargs['D']
    self.G                      = kwargs['G']
    self.D_optim                = kwargs['D_optim']
    self.G_optim                = kwargs['G_optim']
    self.gp_every               = get_attr_kwargs(cfg, 'gp_every', default=10, **kwargs)
    self.n_critic               = get_attr_kwargs(cfg, 'n_critic', default=5, **kwargs)
    self.log_every              = getattr(cfg, 'log_every', 50)
    self.dummy                  = getattr(cfg, 'dummy', False)

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

    real.requires_grad_(True)

    d_real = self.D(real, y=dy, **kwargs)

    if self.gp_every:
      gp = self.compute_grad2(d_real, real, backward=True, gp_lambda=10)
      summary_d['gp']['gp'] = gp.item()

    z_sample = z.sample()
    z_sample = z_sample.to(self.device)
    fake = self.G(z_sample, y=gy, **kwargs)
    d_fake = self.D(fake.detach(), y=gy, **kwargs)

    r_logit_mean, f_logit_mean, d_loss = self.hinge_loss_discriminator(r_logit=d_real, f_logit=d_fake)
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

      G_f_logit_mean, g_loss = self.hinge_loss_generator(f_logit=d_fake_g)
      summary_d['d_logit_mean']['G_f_logit_mean'] = G_f_logit_mean.item()
      summary_d['g_loss']['g_loss'] = g_loss.item()

      g_loss.backward()
      self.G_optim.step()

      if ema is not None:
        ema.update(iteration)

    if iteration % self.log_every == 0:
      Trainer.summary_defaultdict2txtfig(default_dict=summary_d,
                                         prefix='HingeLossCond',
                                         step=iteration,
                                         textlogger=self.myargs.textlogger)

    comm.synchronize()
    return

  @staticmethod
  def hinge_loss_discriminator(r_logit, f_logit):
    r_logit_mean = r_logit.mean()
    f_logit_mean = f_logit.mean()

    loss_real = torch.mean(F.relu(1. - r_logit))
    loss_fake = torch.mean(F.relu(1. + f_logit))
    D_loss = loss_real + loss_fake
    return r_logit_mean, f_logit_mean, D_loss

  @staticmethod
  def hinge_loss_generator(f_logit):
    f_logit_mean = f_logit.mean()
    G_loss = - f_logit_mean
    return f_logit_mean, G_loss


  @staticmethod
  def compute_grad2(d_out, x_in, backward=False, gp_lambda=10.,
                    retain_graph=True, return_grad=False):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
      outputs=d_out.sum(), inputs=x_in,
      create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    reg_mean = reg.mean()
    reg_mean = gp_lambda * reg_mean
    if backward:
      reg_mean.backward(retain_graph=retain_graph)
    if return_grad:
      return grad_dout.detach(), reg_mean
    else:
      return reg_mean

