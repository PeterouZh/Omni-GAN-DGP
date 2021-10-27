import logging
import tqdm
import collections
from collections import OrderedDict
from easydict import EasyDict
import yaml
import functools
import numpy as np
import statistics

import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from template_lib.utils import get_attr_kwargs, get_ddp_attr, AverageMeter, get_prefix_abb
from template_lib.v2.config import update_config
from template_lib.d2.utils import comm
from template_lib.d2.layers import build_d2layer
from template_lib.d2.models.build import D2MODEL_REGISTRY
from template_lib.trainer.base_trainer import summary_defaultdict2txtfig


class _FairController(nn.Module):
  def __init__(self):
    super(_FairController, self).__init__()
    self.num_layers = 0
    self.num_branches = 0

  def get_fair_path(self, bs, **kwargs):
    """

    :param batch_imgs:
    :return: (bs x num_branches, num_layers)
    """
    arcs = []
    for l in range(self.num_layers):
      layer_arcs = torch.randperm(self.num_branches).view(-1, 1)
      arcs.append(layer_arcs)
    arcs = torch.cat(arcs, dim=1)
    batched_arcs = arcs.repeat(bs, 1)

    fair_arcs = arcs
    return batched_arcs, fair_arcs

  def fairnas_repeat_tensor(self, sample):
    repeat_arg = [1] * (sample.dim() + 1)
    repeat_arg[1] = self.num_branches
    sample = sample.unsqueeze(1).repeat(repeat_arg)
    sample = sample.view(-1, *sample.shape[2:])
    return sample


@D2MODEL_REGISTRY.register()
class FairController(_FairController):

  def __init__(self, cfg, **kwargs):
    super().__init__()
    self.num_layers                = get_attr_kwargs(cfg, 'num_layers', **kwargs)
    self.num_branches              = get_attr_kwargs(cfg, 'num_branches', **kwargs)


@D2MODEL_REGISTRY.register()
class ControllerRLAlpha(_FairController):

  def __init__(self, cfg, **kwargs):
    super(ControllerRLAlpha, self).__init__()

    self.FID_IS                  = kwargs['FID_IS']
    self.num_layers              = get_attr_kwargs(cfg, 'num_layers', **kwargs)
    self.num_branches            = get_attr_kwargs(cfg, 'num_branches', **kwargs)
    self.tanh_constant           = get_attr_kwargs(cfg, 'tanh_constant', default=1.5, **kwargs)
    self.temperature             = get_attr_kwargs(cfg, 'temperature', default=-1, **kwargs)
    self.num_aggregate           = get_attr_kwargs(cfg, 'num_aggregate', **kwargs)
    self.entropy_weight          = get_attr_kwargs(cfg, 'entropy_weight', default=0.0001, **kwargs)
    self.bl_dec                  = get_attr_kwargs(cfg, 'bl_dec', **kwargs)
    self.child_grad_bound        = get_attr_kwargs(cfg, 'child_grad_bound', **kwargs)
    self.log_every_iter          = get_attr_kwargs(cfg, 'log_every_iter', default=50, **kwargs)
    self.cfg_ops                 = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)

    self.device = torch.device(f'cuda:{comm.get_rank()}')
    self.baseline = None
    self.logger = logging.getLogger('tl')

    self.alpha = nn.ParameterList()
    for i in range(self.num_layers):
      self.alpha.append(nn.Parameter(1e-4 * torch.randn(1, self.num_branches)))

  def forward(self, ):
    '''
    '''

    entropys = []
    log_probs = []
    sampled_arcs = []

    self.op_dist = []
    for layer_id in range(self.num_layers):
      logit = self.alpha[layer_id]
      # if self.temperature > 0:
      #   logit /= self.temperature
      # if self.tanh_constant is not None:
      #   logit = self.tanh_constant * torch.tanh(logit)

      op_dist = Categorical(logits=logit)
      self.op_dist.append(op_dist)

      sampled_op = op_dist.sample()
      sampled_arcs.append(sampled_op.view(-1, 1))

      log_prob = op_dist.log_prob(sampled_op)
      log_probs.append(log_prob.view(-1, 1))
      entropy = op_dist.entropy()
      entropys.append(entropy.view(-1, 1))

      # inputs = self.w_emb(branch_id)

    self.sampled_arcs = torch.cat(sampled_arcs, dim=1)
    self.sample_entropy = torch.cat(entropys, dim=1)
    self.sample_log_prob = torch.cat(log_probs, dim=1)

    return self.sampled_arcs

  def get_sampled_arc(self, bs=1, *args, **kwargs):
    sampled_arc = []
    for layer_id in range(self.num_layers):
      logit = self.alpha[layer_id]
      op_dist = Categorical(logits=logit)

      sampled_op = op_dist.sample((bs, )).view(-1, 1)
      sampled_arc.append(sampled_op)

    sampled_arc = torch.cat(sampled_arc, dim=1)
    return sampled_arc

  def train_controller(self, G, z, y, controller, controller_optim, iteration, pbar):
    """

    :param controller: for ddp training
    :return:
    """
    if comm.is_main_process() and iteration % 1000 == 0:
      pbar.set_postfix_str("ControllerRLAlpha")

    meter_dict = {}

    G.eval()
    controller.train()

    controller.zero_grad()

    sampled_arcs = controller()
    sample_entropy = get_ddp_attr(controller, 'sample_entropy')
    sample_log_prob = get_ddp_attr(controller, 'sample_log_prob')

    z_samples = z.sample()
    bs = len(z_samples)
    batched_arcs = sampled_arcs.repeat(bs, 1)

    pool_list, logits_list = [], []
    for i in range(self.num_aggregate):
      z_samples = z.sample().to(self.device)
      y_samples = y.sample().to(self.device)
      with torch.set_grad_enabled(False):
        x = G(z=z_samples, y=y_samples, batched_arcs=batched_arcs)

      pool, logits = self.FID_IS.get_pool_and_logits(x)

      # pool_list.append(pool)
      logits_list.append(logits)

    # pool = np.concatenate(pool_list, 0)
    logits = np.concatenate(logits_list, 0)

    reward_g, _ = self.FID_IS.calculate_IS(logits)
    meter_dict['reward_g'] = reward_g

    # detach to make sure that gradients aren't backpropped through the reward
    reward = torch.tensor(reward_g).cuda()
    sample_entropy_mean = sample_entropy.mean()
    meter_dict['sample_entropy'] = sample_entropy_mean.item()
    reward += self.entropy_weight * sample_entropy_mean

    if self.baseline is None:
      baseline = torch.tensor(reward_g)
    else:
      baseline = self.baseline - (1 - self.bl_dec) * (self.baseline - reward)
      # detach to make sure that gradients are not backpropped through the baseline
      baseline = baseline.detach()

    sample_log_prob_mean = sample_log_prob.mean()
    meter_dict['sample_log_prob'] = sample_log_prob_mean.item()
    loss = -1 * sample_log_prob_mean * (reward - baseline)

    meter_dict['reward'] = reward.item()
    meter_dict['baseline'] = baseline.item()
    meter_dict['loss'] = loss.item()

    loss.backward(retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), self.child_grad_bound)
    meter_dict['grad_norm'] = grad_norm

    controller_optim.step()

    baseline_list = comm.all_gather(baseline)
    baseline_mean = sum(map(lambda v: v.item(), baseline_list)) / len(baseline_list)
    baseline.fill_(baseline_mean)
    self.baseline = baseline

    if iteration % self.log_every_iter == 0:
      self.print_distribution(iteration=iteration, print_interval=10)
      default_dicts = collections.defaultdict(dict)
      for meter_k, meter in meter_dict.items():
        if meter_k in ['reward', 'baseline']:
          default_dicts['reward_baseline'][meter_k] = meter
        else:
          default_dicts[meter_k][meter_k] = meter
      summary_defaultdict2txtfig(default_dict=default_dicts,
                                 prefix='train_controller',
                                 step=iteration,
                                 textlogger=self.myargs.textlogger)
    comm.synchronize()
    return

  def print_distribution(self, iteration, print_interval=float('inf')):
    # remove formats
    org_formatters = []
    for handler in self.logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("%(message)s"))

    default_dict = collections.defaultdict(dict)
    self.logger.info("####### distribution #######")
    searched_arc = []
    for layer_id, op_dist in enumerate(self.op_dist):
      prob = op_dist.probs
      max_op_id = prob.argmax().item()
      searched_arc.append(max_op_id)
      for op_id, op_name in enumerate(self.cfg_ops.keys()):
        op_prob = prob[0][op_id]
        default_dict[f'L{layer_id}'][get_prefix_abb(op_name)] = op_prob.item()

      if layer_id % print_interval == 0:
        self.logger.info(layer_id//print_interval)
      self.logger.info(prob)

    searched_arc = np.array(searched_arc)
    self.logger.info('\nsearched arcs: \n%s' % searched_arc)
    self.myargs.textlogger.logstr(iteration,
                                  searched_arc='\n' + np.array2string(searched_arc, threshold=np.inf))

    summary_defaultdict2txtfig(default_dict=default_dict, prefix='', step=iteration,
                               textlogger=self.myargs.textlogger)
    self.logger.info("#####################")

    # restore formats
    for handler, formatter in zip(self.logger.handlers, org_formatters):
      handler.setFormatter(formatter)


@D2MODEL_REGISTRY.register()
class ControllerProgressiveRLAlpha(_FairController):

  def __init__(self, cfg, **kwargs):
    super().__init__()

    self.FID_IS                  = kwargs['FID_IS']
    self.myargs                  = kwargs['myargs']
    self.num_layers              = get_attr_kwargs(cfg, 'num_layers', **kwargs)
    self.num_branches            = get_attr_kwargs(cfg, 'num_branches', **kwargs)
    self.num_stage               = get_attr_kwargs(cfg, 'num_stage', **kwargs)
    self.epochs_stage            = get_attr_kwargs(cfg, 'epochs_stage', **kwargs)
    self.iter_every_epoch        = get_attr_kwargs(cfg, 'iter_every_epoch', **kwargs)
    self.tanh_constant           = get_attr_kwargs(cfg, 'tanh_constant', default=1.5, **kwargs)
    self.temperature             = get_attr_kwargs(cfg, 'temperature', default=-1, **kwargs)
    self.num_aggregate           = get_attr_kwargs(cfg, 'num_aggregate', **kwargs)
    self.entropy_weight          = get_attr_kwargs(cfg, 'entropy_weight', default=0.0001, **kwargs)
    self.bl_dec                  = get_attr_kwargs(cfg, 'bl_dec', **kwargs)
    self.child_grad_bound        = get_attr_kwargs(cfg, 'child_grad_bound', **kwargs)
    self.log_every_iter          = get_attr_kwargs(cfg, 'log_every_iter', default=50, **kwargs)
    self.cfg_ops                 = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)

    self.device = torch.device(f'cuda:{comm.get_rank()}')
    self.baseline = None
    self.logger = logging.getLogger('tl')
    assert self.num_layers % self.num_stage == 0
    self.num_layers_per_stage = self.num_layers // self.num_stage
    assert self.num_stage == len(self.epochs_stage)

    self.alpha = nn.ParameterList()
    for i in range(self.num_layers):
      self.alpha.append(nn.Parameter(1e-4 * torch.randn(1, self.num_branches)))

  def _get_stage_index(self, iteration):
    cur_epoch = iteration // self.iter_every_epoch
    cur_stage = -1
    for i, cur_stage_epoch in enumerate(self.epochs_stage):
      if cur_epoch < cur_stage_epoch:
        cur_stage = i
        break

    if cur_stage == -1:
      cur_stage = len(self.epochs_stage) - 1
      start_idx = cur_stage * self.num_layers_per_stage
      end_idx = self.num_layers
    else:
      start_idx = cur_stage * self.num_layers_per_stage
      end_idx = (cur_stage + 1) * self.num_layers_per_stage
    return start_idx, end_idx

  def forward(self, iteration):
    '''
    '''

    entropys = []
    log_probs = []
    sampled_arcs = []

    start_idx, end_idx = self._get_stage_index(iteration)
    cur_layer_idx = list(range(start_idx, end_idx))
    self.op_dist = []
    for layer_id in range(self.num_layers):
      logit = self.alpha[layer_id]
      # if self.temperature > 0:
      #   logit /= self.temperature
      # if self.tanh_constant is not None:
      #   logit = self.tanh_constant * torch.tanh(logit)

      op_dist = Categorical(logits=logit)
      self.op_dist.append(op_dist)

      if layer_id in cur_layer_idx:
        sampled_op = op_dist.sample()
        log_prob = op_dist.log_prob(sampled_op)
        log_probs.append(log_prob.view(-1, 1))
        entropy = op_dist.entropy()
        entropys.append(entropy.view(-1, 1))
      elif layer_id < start_idx:
        sampled_op = logit.argmax()
      elif layer_id >= end_idx:
        sampled_op = op_dist.sample()
      sampled_arcs.append(sampled_op.view(-1, 1))

    self.sampled_arcs = torch.cat(sampled_arcs, dim=1)
    self.sample_entropy = torch.cat(entropys, dim=1)
    self.sample_log_prob = torch.cat(log_probs, dim=1)

    return self.sampled_arcs

  def train_controller(self, G, z, y, controller, controller_optim, iteration, pbar):
    """

    :param controller: for ddp training
    :return:
    """
    if comm.is_main_process() and iteration % 1000 == 0:
      pbar.set_postfix_str("ControllerRLAlpha")

    meter_dict = {}

    G.eval()
    controller.train()

    controller.zero_grad()

    sampled_arcs = controller(iteration)

    sample_entropy = get_ddp_attr(controller, 'sample_entropy')
    sample_log_prob = get_ddp_attr(controller, 'sample_log_prob')

    z_samples = z.sample()
    bs = len(z_samples)
    batched_arcs = sampled_arcs.repeat(bs, 1)

    pool_list, logits_list = [], []
    for i in range(self.num_aggregate):
      z_samples = z.sample().to(self.device)
      y_samples = y.sample().to(self.device)
      with torch.set_grad_enabled(False):
        x = G(z=z_samples, y=y_samples, batched_arcs=batched_arcs)

      pool, logits = self.FID_IS.get_pool_and_logits(x)

      # pool_list.append(pool)
      logits_list.append(logits)

    # pool = np.concatenate(pool_list, 0)
    logits = np.concatenate(logits_list, 0)

    reward_g, _ = self.FID_IS.calculate_IS(logits)
    meter_dict['reward_g'] = reward_g

    # detach to make sure that gradients aren't backpropped through the reward
    reward = torch.tensor(reward_g).cuda()
    sample_entropy_mean = sample_entropy.mean()
    meter_dict['sample_entropy'] = sample_entropy_mean.item()
    reward += self.entropy_weight * sample_entropy_mean

    if self.baseline is None:
      baseline = torch.tensor(reward_g)
    else:
      baseline = self.baseline - (1 - self.bl_dec) * (self.baseline - reward)
      # detach to make sure that gradients are not backpropped through the baseline
      baseline = baseline.detach()

    sample_log_prob_mean = sample_log_prob.mean()
    meter_dict['sample_log_prob'] = sample_log_prob_mean.item()
    loss = -1 * sample_log_prob_mean * (reward - baseline)

    meter_dict['reward'] = reward.item()
    meter_dict['baseline'] = baseline.item()
    meter_dict['loss'] = loss.item()

    loss.backward(retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), self.child_grad_bound)
    meter_dict['grad_norm'] = grad_norm

    controller_optim.step()

    baseline_list = comm.all_gather(baseline)
    baseline_mean = sum(map(lambda v: v.item(), baseline_list)) / len(baseline_list)
    baseline.fill_(baseline_mean)
    self.baseline = baseline

    if iteration % self.log_every_iter == 0:
      self.print_distribution(iteration=iteration, print_interval=10)
      self.logger.info('\nsampled arcs: \n%s' % sampled_arcs.view(self.num_stage, -1))
      self.myargs.textlogger.logstr(iteration,
                                    sampled_arcs='\n' + np.array2string(sampled_arcs.cpu().numpy(), threshold=np.inf))
      default_dicts = collections.defaultdict(dict)
      for meter_k, meter in meter_dict.items():
        if meter_k in ['reward', 'baseline']:
          default_dicts['reward_baseline'][meter_k] = meter
        else:
          default_dicts[meter_k][meter_k] = meter
      summary_defaultdict2txtfig(default_dict=default_dicts,
                                 prefix='train_controller',
                                 step=iteration,
                                 textlogger=self.myargs.textlogger)
    comm.synchronize()
    return

  def print_distribution(self, iteration, print_interval=float('inf')):
    # remove formats
    org_formatters = []
    for handler in self.logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("%(message)s"))

    default_dict = collections.defaultdict(dict)
    self.logger.info("####### distribution #######")
    searched_arc = []
    for layer_id, op_dist in enumerate(self.op_dist):
      prob = op_dist.probs
      max_op_id = prob.argmax().item()
      searched_arc.append(max_op_id)
      for op_id, op_name in enumerate(self.cfg_ops.keys()):
        op_prob = prob[0][op_id]
        default_dict[f'L{layer_id}'][get_prefix_abb(op_name)] = op_prob.item()

      if layer_id % print_interval == 0:
        self.logger.info(layer_id//print_interval)
      self.logger.info(prob)

    searched_arc = np.array(searched_arc)
    self.logger.info('\nsearched arcs: \n%s' % searched_arc.reshape((-1, print_interval)))
    self.myargs.textlogger.logstr(iteration,
                                  searched_arc='\n' + np.array2string(searched_arc, threshold=np.inf))

    summary_defaultdict2txtfig(default_dict=default_dict, prefix='', step=iteration,
                               textlogger=self.myargs.textlogger)
    self.logger.info("#####################")

    # restore formats
    for handler, formatter in zip(self.logger.handlers, org_formatters):
      handler.setFormatter(formatter)

  def get_fair_path(self, bs, iteration):
    """

    :param batch_imgs:
    :return: (bs x num_branches, num_layers)
    """
    arcs = []
    start_idx, end_idx = self._get_stage_index(iteration)
    fixed_layer_idx = list(range(0, start_idx))

    for l in range(self.num_layers):
      if l in fixed_layer_idx:
        layer_arcs = self.alpha[l].argmax().view(-1, 1).repeat(self.num_branches, 1).to(self.device)
      else:
        layer_arcs = torch.randperm(self.num_branches).view(-1, 1).to(self.device)
      arcs.append(layer_arcs)
    arcs = torch.cat(arcs, dim=1)
    batched_arcs = arcs.repeat(bs, 1)

    fair_arcs = arcs
    return batched_arcs, fair_arcs

  def get_sampled_arc(self, bs=1, iteration=0):
    sampled_arc = []
    start_idx, end_idx = self._get_stage_index(iteration)
    fixed_layer_idx = list(range(0, start_idx))

    for layer_id in range(self.num_layers):
      logit = self.alpha[layer_id]

      if layer_id in fixed_layer_idx:
        sampled_op = logit.argmax().view(-1, 1).to(self.device)
      else:
        op_dist = Categorical(logits=logit)
        sampled_op = op_dist.sample((bs,)).view(-1, 1).to(self.device)
      sampled_arc.append(sampled_op)

    sampled_arc = torch.cat(sampled_arc, dim=1)
    return sampled_arc

@D2MODEL_REGISTRY.register()
class CondControllerProgressiveRLAlpha(_FairController):

  def __init__(self, cfg, **kwargs):
    super().__init__()

    self.FID_IS                  = kwargs['FID_IS']
    self.myargs                  = kwargs['myargs']
    self.num_layers              = get_attr_kwargs(cfg, 'num_layers', **kwargs)
    self.num_branches            = get_attr_kwargs(cfg, 'num_branches', **kwargs)
    self.n_classes               = get_attr_kwargs(cfg, 'n_classes', **kwargs)
    self.num_stage               = get_attr_kwargs(cfg, 'num_stage', **kwargs)
    self.epochs_stage            = get_attr_kwargs(cfg, 'epochs_stage', **kwargs)
    self.iter_every_epoch        = get_attr_kwargs(cfg, 'iter_every_epoch', **kwargs)
    self.tanh_constant           = get_attr_kwargs(cfg, 'tanh_constant', default=1.5, **kwargs)
    self.temperature             = get_attr_kwargs(cfg, 'temperature', default=-1, **kwargs)
    self.num_aggregate           = get_attr_kwargs(cfg, 'num_aggregate', **kwargs)
    self.entropy_weight          = get_attr_kwargs(cfg, 'entropy_weight', default=0.0001, **kwargs)
    self.bl_dec                  = get_attr_kwargs(cfg, 'bl_dec', **kwargs)
    self.child_grad_bound        = get_attr_kwargs(cfg, 'child_grad_bound', **kwargs)
    self.log_every_iter          = get_attr_kwargs(cfg, 'log_every_iter', default=50, **kwargs)
    self.cfg_ops                 = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)
    self.progressive             = get_attr_kwargs(cfg, 'progressive', default=True, **kwargs)

    self.device = torch.device(f'cuda:{comm.get_rank()}')
    self.baseline = None
    self.logger = logging.getLogger('tl')
    assert self.num_layers % self.num_stage == 0
    self.num_layers_per_stage = self.num_layers // self.num_stage
    assert self.num_stage == len(self.epochs_stage)

    self.alpha = nn.ParameterList()
    for i in range(self.num_layers):
      self.alpha.append(nn.Parameter(1e-4 * torch.randn(self.n_classes, self.num_branches)))

  def _get_stage_index(self, iteration):
    if not self.progressive:
      return 0, self.num_layers
    cur_epoch = iteration // self.iter_every_epoch
    cur_stage = -1
    for i, cur_stage_epoch in enumerate(self.epochs_stage):
      if cur_epoch < cur_stage_epoch:
        cur_stage = i
        break

    if cur_stage == -1:
      cur_stage = len(self.epochs_stage) - 1
      start_idx = cur_stage * self.num_layers_per_stage
      end_idx = self.num_layers
    else:
      start_idx = cur_stage * self.num_layers_per_stage
      end_idx = (cur_stage + 1) * self.num_layers_per_stage
    return start_idx, end_idx

  def forward(self, iteration):
    '''
    '''

    entropys = []
    log_probs = []
    sampled_arcs = []

    start_idx, end_idx = self._get_stage_index(iteration)
    cur_layer_idx = list(range(start_idx, end_idx))
    self.op_dist = []
    for layer_id in range(self.num_layers):
      logit = self.alpha[layer_id]
      # if self.temperature > 0:
      #   logit /= self.temperature
      # if self.tanh_constant is not None:
      #   logit = self.tanh_constant * torch.tanh(logit)

      op_dist = Categorical(logits=logit)
      self.op_dist.append(op_dist)

      if layer_id in cur_layer_idx:
        sampled_op = op_dist.sample()
        log_prob = op_dist.log_prob(sampled_op)
        log_probs.append(log_prob.view(-1, 1))
        entropy = op_dist.entropy()
        entropys.append(entropy.view(-1, 1))
      elif layer_id < start_idx:
        sampled_op = logit.argmax(-1)
      elif layer_id >= end_idx:
        sampled_op = op_dist.sample()
      sampled_arcs.append(sampled_op.view(-1, 1))

    self.sampled_arcs = torch.cat(sampled_arcs, dim=1)
    self.sample_entropy = torch.cat(entropys, dim=1)
    self.sample_log_prob = torch.cat(log_probs, dim=1)

    return self.sampled_arcs

  def train_controller(self, G, z, y, controller, controller_optim, iteration, pbar):
    """

    :param controller: for ddp training
    :return:
    """
    if comm.is_main_process() and iteration % 1000 == 0:
      pbar.set_postfix_str("ControllerRLAlpha")

    meter_dict = {}

    G.eval()
    controller.train()

    controller.zero_grad()

    sampled_arcs = controller(iteration)

    sample_entropy = get_ddp_attr(controller, 'sample_entropy')
    sample_log_prob = get_ddp_attr(controller, 'sample_log_prob')

    pool_list, logits_list = [], []
    for i in range(self.num_aggregate):
      z_samples = z.sample().to(self.device)
      y_samples = y.sample().to(self.device)
      with torch.set_grad_enabled(False):
        batched_arcs = sampled_arcs[y_samples]
        x = G(z=z_samples, y=y_samples, batched_arcs=batched_arcs)

      pool, logits = self.FID_IS.get_pool_and_logits(x)

      # pool_list.append(pool)
      logits_list.append(logits)

    # pool = np.concatenate(pool_list, 0)
    logits = np.concatenate(logits_list, 0)

    reward_g, _ = self.FID_IS.calculate_IS(logits)
    meter_dict['reward_g'] = reward_g

    # detach to make sure that gradients aren't backpropped through the reward
    reward = torch.tensor(reward_g).cuda()
    sample_entropy_mean = sample_entropy.mean()
    meter_dict['sample_entropy'] = sample_entropy_mean.item()
    reward += self.entropy_weight * sample_entropy_mean

    if self.baseline is None:
      baseline = torch.tensor(reward_g)
    else:
      baseline = self.baseline - (1 - self.bl_dec) * (self.baseline - reward)
      # detach to make sure that gradients are not backpropped through the baseline
      baseline = baseline.detach()

    sample_log_prob_mean = sample_log_prob.mean()
    meter_dict['sample_log_prob'] = sample_log_prob_mean.item()
    loss = -1 * sample_log_prob_mean * (reward - baseline)

    meter_dict['reward'] = reward.item()
    meter_dict['baseline'] = baseline.item()
    meter_dict['loss'] = loss.item()

    loss.backward(retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), self.child_grad_bound)
    meter_dict['grad_norm'] = grad_norm

    controller_optim.step()

    baseline_list = comm.all_gather(baseline)
    baseline_mean = sum(map(lambda v: v.item(), baseline_list)) / len(baseline_list)
    baseline.fill_(baseline_mean)
    self.baseline = baseline

    if iteration % self.log_every_iter == 0:
      self.print_distribution(iteration=iteration, print_interval=10)
      if len(sampled_arcs) <= 10:
        self.logger.info('\nsampled arcs: \n%s' % sampled_arcs.cpu().numpy())
      self.myargs.textlogger.logstr(iteration,
                                    sampled_arcs='\n' + np.array2string(sampled_arcs.cpu().numpy(), threshold=np.inf))
      default_dicts = collections.defaultdict(dict)
      for meter_k, meter in meter_dict.items():
        if meter_k in ['reward', 'baseline']:
          default_dicts['reward_baseline'][meter_k] = meter
        else:
          default_dicts[meter_k][meter_k] = meter
      summary_defaultdict2txtfig(default_dict=default_dicts,
                                 prefix='train_controller',
                                 step=iteration,
                                 textlogger=self.myargs.textlogger)
    comm.synchronize()
    return

  def print_distribution(self, iteration, print_interval=float('inf')):
    # remove formats
    org_formatters = []
    for handler in self.logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("%(message)s"))

    self.logger.info("####### distribution #######")
    class_arcs = []
    for class_idx in range(self.n_classes):
      default_dict = collections.defaultdict(dict)
      searched_arc = []
      for layer_id, op_dist in enumerate(self.op_dist):
        prob = op_dist.probs
        max_op_id = prob[class_idx].argmax().item()
        searched_arc.append(max_op_id)
        for op_id, op_name in enumerate(self.cfg_ops.keys()):
          op_prob = prob[class_idx][op_id]
          default_dict[f'C{class_idx}L{layer_id}'][get_prefix_abb(op_name)] = op_prob.item()

      class_arcs.append(searched_arc)
      searched_arc = np.array(searched_arc)
      self.logger.info(f'Class {class_idx} searched arcs: {searched_arc}')
      # self.myargs.textlogger.logstr(iteration,
      #                               searched_arc='\n' + np.array2string(searched_arc, threshold=np.inf))

      summary_defaultdict2txtfig(default_dict=default_dict, prefix='', step=iteration,
                                 textlogger=self.myargs.textlogger)

    class_arcs = np.array(class_arcs)
    self.myargs.textlogger.logstr(iteration,
                                  searched_class_arc='\n' + np.array2string(class_arcs, threshold=np.inf))
    self.logger.info("#####################")
    # restore formats
    for handler, formatter in zip(self.logger.handlers, org_formatters):
      handler.setFormatter(formatter)

  def get_fair_path(self, labels, iteration, **kwargs):
    """
    :return: (bs x num_branches, num_layers)
    """
    bs = len(labels)
    arcs = []
    start_idx, end_idx = self._get_stage_index(iteration)
    fixed_layer_idx = list(range(0, start_idx))

    for l in range(self.num_layers):
      if l in fixed_layer_idx:
        layer_arcs = self.alpha[l].argmax(-1).view(-1, 1)[labels]
        layer_arcs = layer_arcs.repeat_interleave(self.num_branches, dim=0).to(self.device)
      else:
        layer_arcs = torch.randperm(self.num_branches).view(-1, 1).repeat(bs, 1).to(self.device)
      arcs.append(layer_arcs)
    batched_arcs = torch.cat(arcs, dim=1)

    fair_arcs = None
    return batched_arcs, fair_arcs

  def get_sampled_arc(self, iteration=0):
    sampled_arc = []
    start_idx, end_idx = self._get_stage_index(iteration)
    fixed_layer_idx = list(range(0, start_idx))

    for layer_id in range(self.num_layers):
      logit = self.alpha[layer_id]

      if layer_id in fixed_layer_idx:
        sampled_op = logit.argmax(-1).view(-1, 1).to(self.device)
      else:
        op_dist = Categorical(logits=logit)
        sampled_op = op_dist.sample().view(-1, 1).to(self.device)
      sampled_arc.append(sampled_op)

    sampled_arc = torch.cat(sampled_arc, dim=1)
    return sampled_arc




