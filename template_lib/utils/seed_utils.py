

def set_random_seed(manualSeed, cudnn=True):
  import torch
  import numpy as np
  import random
  random.seed(manualSeed)
  np.random.seed(manualSeed)

  torch.manual_seed(manualSeed)
  # if you are suing GPU
  torch.cuda.manual_seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)

  if cudnn:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
  else:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
  return