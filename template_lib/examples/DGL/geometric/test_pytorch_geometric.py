import os
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils


class TestingGeometric(unittest.TestCase):

  def test_data_transforms(self):
    """
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32
    proxychains python -c "from template_lib.examples.DGL.geometric.test_pytorch_geometric import TestingGeometric;\
      TestingGeometric().test_data_transforms()"

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

    import torch_geometric.transforms as T
    from torch_geometric.datasets import ShapeNet
    from template_lib.d2.data.build_points_toy import plot_points

    dataset = ShapeNet(root='datasets/shapenet', categories=['Airplane'])
    idx = -1
    plot_points(dataset[idx].pos)
    plot_points(dataset[idx].x)

    dataset = ShapeNet(root='datasets/shapenet', categories=['Airplane'],
                       pre_transform=T.KNNGraph(k=6))

    dataset = ShapeNet(root='datasets/shapenet', categories=['Airplane'],
                       pre_transform=T.KNNGraph(k=6),
                       transform=T.RandomTranslate(0.01))
    pass

  def test_learning_methods_on_graphs(self):
    """
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32
    proxychains python -c "from template_lib.examples.DGL.geometric.test_pytorch_geometric import TestingGeometric;\
      TestingGeometric().test_learning_methods_on_graphs()"

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

    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root='datasets/cora', name='Cora')

    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class Net(torch.nn.Module):
      def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

      def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
      optimizer.zero_grad()
      out = model(data)
      loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))
    pass

  def test_visualize_cora(self):
    """
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32
    proxychains python -c "from template_lib.examples.DGL.geometric.test_pytorch_geometric import TestingGeometric;\
      TestingGeometric().test_learning_methods_on_graphs()"

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

    import networkx as nx
    import torch
    import numpy as np
    import pandas as pd
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils.convert import to_networkx

    dataset1 = Planetoid(root='datasets/cora', name='Cora')

    cora = dataset1[0]

    coragraph = to_networkx(cora)

    node_labels = cora.y[list(coragraph.nodes)].numpy()

    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(14, 12))
    nx.draw(coragraph, cmap=plt.get_cmap('Set1'), node_color=node_labels, node_size=75, linewidths=6)
    plt.show()
    pass

  def test_visualize_KarateClub(self):
    """
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32
    proxychains python -c "from template_lib.examples.DGL.geometric.test_pytorch_geometric import TestingGeometric;\
      TestingGeometric().test_learning_methods_on_graphs()"

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

    from torch_geometric.datasets import KarateClub
    dataset = KarateClub()

    for i in dataset[0]:
      print(i)
    # this torch.geometric.datasets object comprises of edge(edge information for each node), x(nodes) and y(labels for each node)

    edge, x, y = dataset[0]
    numpyx = x[1].numpy()
    numpyy = y[1].numpy()
    numpyedge = edge[1].numpy()

    import networkx as nx

    g = nx.Graph(numpyx)

    name, edgeinfo = edge

    src = edgeinfo[0].numpy()
    dst = edgeinfo[1].numpy()
    edgelist = zip(src, dst)

    for i, j in edgelist:
      g.add_edge(i, j)

    nx.draw_networkx(g)
    pass

  def test_MessagePassing(self):
    """
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32
    proxychains python -c "from template_lib.examples.DGL.geometric.test_pytorch_geometric import TestingGeometric;\
      TestingGeometric().test_learning_methods_on_graphs()"

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

    import torch
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, degree

    class GCNConv(MessagePassing):
      def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

      def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-6: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

      def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

      def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 6: Return new node embeddings.
        return aggr_out

    from torch_geometric.datasets import KarateClub
    dataset = KarateClub()
    x = dataset[0].x
    edge_index = dataset[0].edge_index
    conv = GCNConv(34, 64)
    x = conv(x, edge_index)
    pass







