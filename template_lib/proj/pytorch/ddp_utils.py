import os


def is_distributed():
  n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
  distributed = n_gpu > 1
  return distributed


