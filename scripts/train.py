""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""
import shutil
import os
import functools
import math
import copy
import logging
import pprint
import importlib

import numpy as np
from tqdm import tqdm, trange
import easydict

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, get_dict_str, global_cfg
from template_lib.v2.logger.textlogger import global_textlogger, summary_dict2txtfig
from template_lib.d2.models_v2 import build_model
from template_lib.d2.utils.checkpoint import VisualModelCkpt

# Import my stuff
import inception_utils
import utils
import losses
# import train_fns
from sync_batchnorm import patch_replication_callback


# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):

  logger = logging.getLogger('tl')

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  # By default, skip init if resuming training.
  if config['resume']:
    print('Skipping initialization for training resumption...')
    config['skip_init'] = True
  config = utils.update_config_roots(config)
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])

  # Prepare root folders if necessary
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = importlib.import_module(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)

  # Next, build the model
  if 'Generator' in global_cfg and 'register_modules' in global_cfg.Generator:
    G = build_model(cfg=global_cfg.Generator).to(device)
  else:
    G = model.Generator(**config).to(device)
  if 'Discriminator' in global_cfg and 'register_modules' in global_cfg.Discriminator:
    D = build_model(cfg=global_cfg.Discriminator).to(device)
  else:
    D = model.Discriminator(**config).to(device)
  
   # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    if 'Generator' in global_cfg and 'register_modules' in global_cfg.Generator:
      G_ema = build_model(cfg=global_cfg.Generator, kwargs_priority=True,
                          skip_init=True, no_optim=True, verbose=False).to(device)
    else:
      G_ema = model.Generator(**{**config, 'skip_init':True,
                                 'no_optim': True}).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None
  
  # FP16?
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    print('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D)
  logger.info(G)
  logger.info(D)
  logger.info('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config,
                'shown_images': 0}

  # If loading from a pre-trained model, load weights
  if global_cfg.tl_resume:
    print('Loading weights...')
    utils.load_weights(G, D, state_dict,
                       config['weights_root'], experiment_name, 
                       config['load_weights'] if config['load_weights'] else None,
                       G_ema if config['ema'] else None)
    state_dict_pop_config = (copy.deepcopy(state_dict))
    del state_dict_pop_config['config']
    logger.info(f'Loaded state_dict: \n{pprint.pformat(state_dict_pop_config)}')

  # If parallel, parallelize the GD module
  if config['parallel']:
    GD = nn.DataParallel(GD)
    if config['cross_replica']:
      patch_replication_callback(GD)

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                            experiment_name)
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils.MetricsLogger(test_metrics_fname, 
                                 reinitialize=(not config['resume']))
  print('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname, 
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
  # Write metadata
  utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  if not ('evaluation' in global_cfg and global_cfg.evaluation.eval):
    loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                        'start_itr': state_dict['itr']})

  # Prepare inception metrics: FID and IS
  get_inception_metrics = inception_utils.prepare_inception_metrics(global_cfg.inception_file,
                                                                    config['parallel'], config['no_fid'])

  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])
  # Prepare a fixed z & y to see individual sample evolution throghout training
  fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                       config['n_classes'], device=device,
                                       fp16=config['G_fp16'])  
  fixed_z.sample_()
  fixed_y.sample_()
  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN':
    import train_fns
    train = train_fns.GAN_training_function(G, D, GD, z_, y_, 
                                            ema, state_dict, config)
  elif config['which_train_fn'] in ['exp.omniGAN.train_fns', 'exp.biggan.train_fns']:
    train_fns = importlib.import_module(config['which_train_fn'])
    train = train_fns.GAN_training_function(G, D, GD, z_, y_,
                                            ema, state_dict, config)
  # Else, assume debugging and use the dummy train fn
  else:
    import train_fns
    train = train_fns.dummy_training_function()
  # Prepare Sample function for use with inception metrics
  sample = functools.partial(utils.sample,
                              G=(G_ema if config['ema'] and config['use_ema']
                                 else G),
                              z_=z_, y_=y_, config=config)

  if ('evaluation' in global_cfg and global_cfg.evaluation.eval):
    logger.info('Beginning evaluation ...')
    cfg = global_cfg.evaluation
    eval_ckpt = VisualModelCkpt(G_ema)
    eval_ckpt.load_from_path(path=cfg.G_ema_model)
    logger.info(f"Using inception file: {global_cfg.inception_file}")
    # logger.info(f'Loading G_ema from {cfg.G_ema_model}')
    # ret = G_ema.load_state_dict(torch.load(cfg.G_ema_model), strict=True)
    G_ema.eval()
    IS_mean, IS_std, FID = train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                                          get_inception_metrics, experiment_name, test_log)
    logger.info('PYTORCH Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (IS_mean, IS_std, FID))
    shutil.rmtree(config['base_root'], ignore_errors=True)
    exit(0)
    pass

  print('Beginning training at epoch %d...' % state_dict['epoch'])
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(state_dict['epoch'], config['num_epochs']):    
    # Which progressbar to use? TQDM or my own?
    if config['pbar'] == 'mine':
      pbar = utils.progress(loaders[0], desc=f"Epoch [{epoch}/{config['num_epochs']}], itr: ",
                            displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
    else:
      pbar = tqdm(loaders[0])
    for i, (x, y) in enumerate(pbar):
      # Increment the iteration counter
      state_dict['itr'] += 1
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      G.train()
      D.train()
      if config['ema']:
        G_ema.train()
      if config['D_fp16']:
        x, y = x.to(device).half(), y.to(device)
      else:
        x, y = x.to(device), y.to(device)

      metrics = train(x, y)

      state_dict['shown_images'] += D_batch_size
      train_log.log(itr=int(state_dict['itr']), **metrics)
      
      # Every sv_log_interval, log singular values
      if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
        train_log.log(itr=int(state_dict['itr']), 
                      **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

      # If using my progbar, print metrics.
      if config['pbar'] == 'mine':
          print(', '.join(['itr: %d' % state_dict['itr']] 
                           + ['%s : %+4.3f' % (key, metrics[key])
                           for key in metrics]), end=' ')

      # Save weights and copies as configured at specified interval
      if state_dict['itr'] == 1 or \
            config['save_every'] > 0 and not (state_dict['itr'] % config['save_every']):
        if config['G_eval_mode']:
          print('\nSwitchin G to eval mode...')
          G.eval()
          if config['ema']:
            G_ema.eval()
        train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                                  state_dict, config, experiment_name)

      # Test every specified interval
      if state_dict['itr'] == 1 or \
            config['test_every'] > 0 and not (state_dict['itr'] % config['test_every']) or \
            global_cfg.tl_debug:
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
        logger.info(f"===Evaluation=== Training mode: G: {G.training}, G_ema: {G_ema.training}")
        IS_mean, IS_std, FID = train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                                              get_inception_metrics, experiment_name, test_log)
        logger.info(f"===End evaluation=== Training mode: G: {G.training}, G_ema: {G_ema.training}")
        summary_d = {}
        if not math.isnan(IS_mean):
          summary_d['IS_mean'] = IS_mean
          summary_d['IS_std'] = IS_std
        if not math.isnan(FID):
          summary_d['FID'] = FID
        summary_dict2txtfig(summary_d, prefix='evaltorch', step=state_dict['shown_images'],
                            textlogger=global_textlogger)

    # Increment epoch counter at end of epoch
    state_dict['epoch'] += 1


def main():
  parser = utils.prepare_parser()
  update_parser_defaults_from_yaml(parser)

  config = easydict.EasyDict(vars(parser.parse_args()))

  logger = logging.getLogger('tl')

  config['base_root'] = f"{config['tl_outdir']}/biggan"
  logger.info('config: \n' + get_dict_str(config, use_pprint=False))

  global_cfg.merge_from_dict(config)

  run(config)
  pass

if __name__ == '__main__':
  main()