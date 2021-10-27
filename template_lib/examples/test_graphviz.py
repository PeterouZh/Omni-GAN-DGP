import os
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils


class TestingGraphviz(unittest.TestCase):

  def test_hello(self):
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
    import shutil
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    from graphviz import Digraph

    filename = os.path.join(outdir, 'hello')
    g = Digraph('G', filename=filename, format='png')

    g.edge('Hello', 'World')

    g.view()

  def test_process(self):
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
    import shutil
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    from graphviz import Digraph

    filename = os.path.join(outdir, 'hello')

    from graphviz import Graph

    g = Graph('G', filename=filename, format='png')

    g.edge('run', 'intr')
    g.edge('intr', 'runbl')
    g.edge('runbl', 'run')
    g.edge('run', 'kernel')
    g.edge('kernel', 'zombie')
    g.edge('kernel', 'sleep')
    g.edge('kernel', 'runmem')
    g.edge('sleep', 'swap')
    g.edge('swap', 'runswap')
    g.edge('runswap', 'new')
    g.edge('runswap', 'runmem')
    g.edge('new', 'runmem')
    g.edge('sleep', 'runmem')

    g.view()

  def test_fsm(self):
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
    import shutil
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    from graphviz import Digraph

    filename = os.path.join(outdir, 'hello')

    from graphviz import Digraph

    f = Digraph('finite_state_machine', filename=filename, format='png')
    f.attr(rankdir='LR', size='8,5')

    f.attr('node', shape='doublecircle')
    f.node('LR_0')
    f.node('LR_3')
    f.node('LR_4')
    f.node('LR_8')

    f.attr('node', shape='circle')
    f.edge('LR_0', 'LR_2', label='SS(B)')
    f.edge('LR_0', 'LR_1', label='SS(S)')
    f.edge('LR_1', 'LR_3', label='S($end)')
    f.edge('LR_2', 'LR_6', label='SS(b)')
    f.edge('LR_2', 'LR_5', label='SS(a)')
    f.edge('LR_2', 'LR_4', label='S(A)')
    f.edge('LR_5', 'LR_7', label='S(b)')
    f.edge('LR_5', 'LR_5', label='S(a)')
    f.edge('LR_6', 'LR_6', label='S(b)')
    f.edge('LR_6', 'LR_5', label='S(a)')
    f.edge('LR_7', 'LR_8', label='S(b)')
    f.edge('LR_7', 'LR_5', label='S(a)')
    f.edge('LR_8', 'LR_6', label='S(b)')
    f.edge('LR_8', 'LR_5', label='S(a)')

    f.view()

  def test_cluster(self):
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
    import shutil
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    from graphviz import Digraph

    filename = os.path.join(outdir, 'hello')

    from graphviz import Digraph

    g = Digraph('G', filename=filename, format='png')

    # NOTE: the subgraph name needs to begin with 'cluster' (all lowercase)
    #       so that Graphviz recognizes it as a special cluster subgraph

    with g.subgraph(name='cluster_0') as c:
      c.attr(style='filled', color='lightgrey')
      c.node_attr.update(style='filled', color='white')
      c.edges([('a0', 'a1'), ('a1', 'a2'), ('a2', 'a3')])
      c.attr(label='process #1')

    with g.subgraph(name='cluster_1') as c:
      c.attr(color='blue')
      c.node_attr['style'] = 'filled'
      c.edges([('b0', 'b1'), ('b1', 'b2'), ('b2', 'b3')])
      c.attr(label='process #2')

    g.edge('start', 'a0')
    g.edge('start', 'b0')
    g.edge('a1', 'b3')
    g.edge('b2', 'a3')
    g.edge('a3', 'a0')
    g.edge('a3', 'end')
    g.edge('b3', 'end')

    g.node('start', shape='Mdiamond')
    g.node('end', shape='Msquare')

    g.view()

  def test_rank_same(self):
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
    import shutil
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    from graphviz import Digraph

    filename = os.path.join(outdir, 'hello')

    d = Digraph('G', filename=filename, format='png')

    with d.subgraph() as s:
      s.attr(rank='same')
      s.node('A')
      s.node('X')

    d.node('C')

    with d.subgraph() as s:
      s.attr(rank='same')
      s.node('B')
      s.node('D')
      s.node('Y')

    d.edges(['AB', 'AC', 'CD', 'XY'])

    d.view()
