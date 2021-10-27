import numpy as np
import unittest
import os, sys


class Testing(unittest.TestCase):
  """

  """

  def test_demo(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=1,2,3,4;
        python -c "import test_train; \
        test_train.Testing().test_demo()"
    :return:
    """
    a = np.arange(25).reshape(5, 5)
    b = np.arange(5)
    c = np.arange(6).reshape(2, 3)

    np.einsum('ii', a)
    np.einsum(a, [0, 0])
    np.trace(a)

    np.einsum('ii->i', a)
    np.einsum(a, [0, 0], [0])
    np.diag(a)

    np.einsum('ij,j', a, b)
    np.einsum(a, [0, 1], b, [1])
    np.dot(a, b)
    np.einsum('...j,j', a, b)

    np.einsum('ji', c)
    np.einsum(c, [1, 0])
    c.T

    np.einsum('..., ...', 3, c)
    np.einsum(',ij', 3, C)
    np.einsum(3, [Ellipsis], c, [Ellipsis])
    np.multiply(3, c)

    np.einsum('i,i', b, b)
    np.einsum(b, [0], b, [0])
    np.inner(b, b)

    np.einsum('i,j', np.arange(2) + 1, b)
    np.einsum(np.arange(2) + 1, [0], b, [1])
    np.outer(np.arange(2) + 1, b)

    np.einsum('i...->...', a)
    np.einsum(a, [0, Ellipsis], [Ellipsis])
    np.sum(a, axis=0)

    a = np.arange(60.).reshape(3, 4, 5)
    b = np.arange(24.).reshape(4, 3, 2)
    np.einsum('ijk,jil->kl', a, b)
    np.einsum(a, [0, 1, 2], b, [1, 0, 3], [2, 3])
    np.tensordot(a, b, axes=([1, 0], [0, 1]))

    a = np.arange(6).reshape((3, 2))
    b = np.arange(12).reshape((4, 3))
    np.einsum('ki,jk->ij', a, b)
    np.einsum('ki,...k->i...', a, b)
    np.einsum('k...,jk', a, b)

    # since version 1.10.0
    a = np.zeros((3, 3))
    np.einsum('ii->i', a)[:] = 1
    a
