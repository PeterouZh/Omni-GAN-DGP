import numpy as np


def np_savez(saved_file, *args, **kwargs):
  np.savez(saved_file, *args, **kwargs)


def np_load_dict(loaded_file, key):
  loaded_data = np.load(loaded_file, allow_pickle=True)
  data_dict = loaded_data[key][()]
  return data_dict


def get_(seed):
  # np.random.RandomState(seed).randn(w_avg_samples, G.z_dim)
  rand_state = np.random.RandomState(seed)
  return rand_state



