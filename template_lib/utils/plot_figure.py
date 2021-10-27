import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
# import cPickle as pickle
import pickle

_since_beginning = collections.defaultdict(dict)
_since_last_flush = collections.defaultdict(dict)

def plot(name, value, itr):
  _since_last_flush[name][itr] = value


def flush():

  for name, vals in _since_last_flush.items():
    _since_beginning[name].update(vals)

    x_vals = sorted(_since_beginning[name].keys())
    y_vals = [_since_beginning[name][x] for x in x_vals]

    plt.clf()
    plt.plot(x_vals, y_vals)
    plt.xlabel('iteration')
    plt.ylabel(name)
    # plt.savefig(name.replace(' ', '_') + '.jpg')

  _since_last_flush.clear()

  with open('log.pkl', 'wb') as f:
    pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  name = 'test'
  value = 1
  plot(name=name, value=value, itr=1)
  plot(name=name, value=value, itr=2)
  plot(name=name, value=2, itr=2)
  flush()
