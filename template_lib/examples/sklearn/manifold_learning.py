import os
import sys
import unittest
import argparse

from template_lib import utils


class TestingManifold(unittest.TestCase):

  def test_plot_compare_methods(self):
    """

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'

    from datetime import datetime
    TIME_STR = bool(int(os.getenv('TIME_STR', 0)))
    time_str = datetime.now().strftime("%Y%m%d-%H_%M_%S_%f")[:-3]
    outdir = outdir if not TIME_STR else (outdir + '_' + time_str)
    print(outdir)

    import collections, shutil
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    from collections import OrderedDict
    from functools import partial
    from time import time

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import NullFormatter

    from sklearn import manifold, datasets

    # Next line to silence pyflakes. This import is needed.
    Axes3D

    n_points = 1000
    X, color = datasets.make_s_curve(n_points, random_state=0)
    n_neighbors = 10
    n_components = 2

    # Create figure
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("Manifold Learning with %i points, %i neighbors"
                 % (1000, n_neighbors), fontsize=14)

    # Add 3d scatter plot
    ax = fig.add_subplot(251, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.view_init(4, -72)

    # Set-up manifold methods
    LLE = partial(manifold.LocallyLinearEmbedding,
                  n_neighbors, n_components, eigen_solver='auto')

    methods = OrderedDict()
    methods['LLE'] = LLE(method='standard')
    methods['LTSA'] = LLE(method='ltsa')
    methods['Hessian LLE'] = LLE(method='hessian')
    methods['Modified LLE'] = LLE(method='modified')
    methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
    methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
    methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                               n_neighbors=n_neighbors)
    methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                     random_state=0)

    # Plot results
    for i, (label, method) in enumerate(methods.items()):
      t0 = time()
      Y = method.fit_transform(X)
      t1 = time()
      print("%s: %.2g sec" % (label, t1 - t0))
      ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
      ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
      ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
      ax.xaxis.set_major_formatter(NullFormatter())
      ax.yaxis.set_major_formatter(NullFormatter())
      ax.axis('tight')

    plt.show()
    pass

  def test_plot_S_geodesic(self):
    """

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'

    from datetime import datetime
    TIME_STR = bool(int(os.getenv('TIME_STR', 0)))
    time_str = datetime.now().strftime("%Y%m%d-%H_%M_%S_%f")[:-3]
    outdir = outdir if not TIME_STR else (outdir + '_' + time_str)
    print(outdir)

    import collections, shutil
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    from collections import OrderedDict
    from functools import partial
    from time import time

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import NullFormatter

    from sklearn import manifold, datasets

    # Next line to silence pyflakes. This import is needed.
    Axes3D

    n_points = 1000
    X, color = datasets.make_s_curve(n_points, random_state=0)
    n_neighbors = 10
    n_components = 2

    # Create figure
    fig = plt.figure(figsize=(15, 8))
    plt.show()
    fig.suptitle("Manifold Learning with %i points, %i neighbors"
                 % (1000, n_neighbors), fontsize=14)

    # Add 3d scatter plot
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.view_init(4, -72)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.zaxis.set_major_formatter(NullFormatter())

    # Set-up manifold methods
    LLE = partial(manifold.LocallyLinearEmbedding,
                  n_neighbors, n_components, eigen_solver='auto')

    methods = OrderedDict()

    # Plot results
    label = 'Hessian LLE'
    method = LLE(method='hessian')
    t0 = time()
    Y = method.fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % (label, t1 - t0))
    ax = fig.add_subplot(122)
    ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)

    ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')

    plt.show()
    pass



