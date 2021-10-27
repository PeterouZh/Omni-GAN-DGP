''' Calculate Inception Moments
 This script iterates over the dataset and calculates the moments of the 
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.
 
 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. '''
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import inception_utils
from tqdm import tqdm, trange
from argparse import ArgumentParser
import easydict

from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, get_dict_str
from template_lib.modelarts import modelarts_utils


def prepare_parser():
  usage = 'Calculate and store inception metrics.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--dataset', type=str, default='I128_hdf5',
    help='Which Dataset to train on, out of I128, I256, C10, C100...'
         'Append _hdf5 to use the hdf5 version of the dataset. (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)') 
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=False,
    help='Augment with random crops and flips (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers (default: %(default)s)')
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='Shuffle the data? (default: %(default)s)') 
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use.')
  return parser

def run(config):
  logger = logging.getLogger('tl')
  # Get loader
  config['drop_last'] = False
  loaders = utils.get_data_loaders(**config)

  # Load inception net
  net = inception_utils.load_inception_net(parallel=config['parallel'])
  pool, logits, labels = [], [], []
  device = 'cuda'
  debug_num_batches = eval(config.debug_num_batches)
  for i, (x, y) in enumerate(tqdm(loaders[0])):
    x = x.to(device)
    with torch.no_grad():
      pool_val, logits_val = net(x)
      pool += [np.asarray(pool_val.cpu())]
      logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
      labels += [np.asarray(y.cpu())]
    if len(logits) * len(logits[0]) >= debug_num_batches:
      break

  pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
  # uncomment to save pool, logits, and labels to disk
  # print('Saving pool, logits, and labels to disk...')
  # np.savez(config['dataset']+'_inception_activations.npz',
  #           {'pool': pool, 'logits': logits, 'labels': labels})
  # Calculate inception metrics and report them
  logger.info('Calculating inception metrics...')
  IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
  logger.info('Training data from dataset %s has IS of %5.5f +/- %5.5f' % (config['dataset'], IS_mean, IS_std))
  # Prepare mu and sigma, save to disk. Remove "hdf5" by default 
  # (the FID code also knows to strip "hdf5")
  if not config.shuffle:
    logger.info('Calculating means and covariances...')
    mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
    logger.info('Saving calculated means and covariances to disk...')
    np.savez(config['saved_inception_file'], **{'mu' : mu, 'sigma' : sigma})
    logger.info(f"Saved to {config['saved_inception_file']}")

def main():
  # parse command line    
  parser = prepare_parser()
  update_parser_defaults_from_yaml(parser, use_cfg_as_args=True)
  config = easydict.EasyDict(vars(parser.parse_args()))
  logger = logging.getLogger('tl')
  logger.info('config: \n' + get_dict_str(config))

  modelarts_utils.setup_tl_outdir_obs(config)
  modelarts_utils.modelarts_sync_results_dir(config, join=True)

  modelarts_utils.prepare_dataset(config.get('modelarts_download', {}), global_cfg=config)
  run(config)
  modelarts_utils.prepare_dataset(config.get('modelarts_upload', {}), global_cfg=config, download=False)

  modelarts_utils.modelarts_sync_results_dir(config, join=True)

if __name__ == '__main__':    
    main()