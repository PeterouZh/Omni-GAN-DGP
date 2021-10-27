''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import pprint
import logging
import functools
import math
import numpy as np
from tqdm import tqdm, trange
import os

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses

from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, global_cfg
from template_lib.d2.models_v2 import build_model
from template_lib.modelarts import modelarts_utils
from template_lib.d2.utils.checkpoint import VisualModelCkpt
from template_lib.v2.logger import global_textlogger, summary_dict2txtfig


def run(config):
  logger = logging.getLogger('tl')
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # update config (see train.py for explanation)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  config = utils.update_config_roots(config)
  config['skip_init'] = True
  config['no_optim'] = True
  device = 'cuda'

  # Seed RNG
  utils.seed_rng(config['seed'])

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  if 'Generator' not in global_cfg:
    model = __import__(config['model'])
    # experiment_name = (config['experiment_name'] if config['experiment_name']
    #                    else utils.name_from_config(config))
    experiment_name = 'exp'
    G = model.Generator(**config).cuda()
  else:
    G = build_model(cfg=global_cfg.Generator)
    experiment_name = 'exp'
  print('Experiment name is %s' % experiment_name)
  utils.count_parameters(G)

  # Load weights
  print('Loading weights...')
  # Here is where we deal with the ema--load ema weights or load normal weights
  utils.load_weights(G if not (config['use_ema']) else None, None, state_dict,
                     global_cfg['weights_root'], experiment_name, config['load_weights'],
                     G if config['ema'] and config['use_ema'] else None,
                     strict=False, load_optim=False)

  ckpt = VisualModelCkpt(G)
  ckpt.load_from_path(path=global_cfg.pretrained_model)
  del ckpt
  # Update batch size setting used for G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'],
                             z_var=config['z_var'])
  G.cuda()
  if config['G_eval_mode']:
    print('Putting G in eval mode..')
    G.eval()
  else:
    print('G is in %s mode...' % ('training' if G.training else 'eval'))

  # Sample function
  sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
  if config['accumulate_stats']:
    print('Accumulating standing stats across %d accumulations...' % config['num_standing_accumulations'])
    utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
                                    config['num_standing_accumulations'])

  # Get Inception Score and FID
  logger.info(f"Loading inception file: {global_cfg['inception_file']}")
  get_inception_metrics = inception_utils.prepare_inception_metrics(
    global_cfg['inception_file'], config['parallel'], config['no_fid'])

  # Prepare a simple function get metrics that we use for trunc curves
  def get_metrics():
    sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
    IS_mean, IS_std, FID = get_inception_metrics(
      sample, config['num_inception_images'], num_splits=10, prints=False)
    # Prepare output string
    outstring = 'Using %s weights ' % ('ema' if config['use_ema'] else 'non-ema')
    outstring += 'in %s mode, \n' % ('eval' if config['G_eval_mode'] else 'training')
    outstring += 'with noise variance %3.3f, \n' % z_.var
    outstring += 'over %d images, \n' % config['num_inception_images']
    if config['accumulate_stats'] or not config['G_eval_mode']:
      outstring += 'with batch size %d, \n' % G_batch_size
    if config['accumulate_stats']:
      outstring += 'using %d standing stat accumulations, \n' % config['num_standing_accumulations']
    outstring += 'Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (
    state_dict['itr'], IS_mean, IS_std, FID)
    logger.info(outstring)
    return IS_mean, IS_std, FID

  if config['sample_inception_metrics']:
    print('Calculating Inception metrics...')
    print(global_cfg.tl_outdir)
    get_metrics()

  # Sample truncation curve stuff. This is basically the same as the inception metrics code
  if config['sample_trunc_curves']:
    start, step, end = [float(item) for item in config['sample_trunc_curves'].split('_')]
    print('Getting truncation values for variance in range (%3.3f:%3.3f:%3.3f)...' % (start, step, end))
    for var in tqdm(np.arange(start, end + step, step), desc=f"{global_cfg.tl_outdir}"):
      z_.var = var
      # Optionally comment this out if you want to run with standing stats
      # accumulated at one z variance setting
      if config['accumulate_stats']:
        utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])
      IS_mean, IS_std, FID = get_metrics()
      summary_d = {'FID': FID}
      summary_dict2txtfig(summary_d, prefix='eval', step=IS_mean, textlogger=global_textlogger,
                          save_fig_sec=5)

  pass


def main():
  # parse command line and run

  parser = utils.prepare_parser()
  parser = utils.add_sample_parser(parser)

  update_parser_defaults_from_yaml(parser)

  config = vars(parser.parse_args())
  pprint.pprint(config)
  run(config)


if __name__ == '__main__':
  main()