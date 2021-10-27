import torch
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
  def __init__(self):
    super(ToyModel, self).__init__()
    self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
    self.relu = torch.nn.ReLU()
    self.net2 = torch.nn.Linear(10, 5).to('cuda:1')

  def forward(self, x):
    x = self.relu(self.net1(x.to('cuda:0')))
    return self.net2(x.to('cuda:1'))


model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))
labels = torch.randn(20, 5).to('cuda:1')
loss_fn(outputs, labels).backward()
optimizer.step()


from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000


class ModelParallelResNet50(ResNet):
  def __init__(self, *args, **kwargs):
    super(ModelParallelResNet50, self).__init__(
      Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

    self.seq1 = nn.Sequential(
      self.conv1,
      self.bn1,
      self.relu,
      self.maxpool,

      self.layer1,
      self.layer2
    ).to('cuda:0')

    self.seq2 = nn.Sequential(
      self.layer3,
      self.layer4,
      self.avgpool,
    ).to('cuda:1')

    self.fc.to('cuda:1')

  def forward(self, x):
    x = self.seq2(self.seq1(x).to('cuda:1'))
    return self.fc(x.view(x.size(0), -1))


import torchvision.models as models

num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


def train(model):
  model.train(True)
  loss_fn = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001)

  one_hot_indices = torch.LongTensor(batch_size) \
    .random_(0, num_classes) \
    .view(batch_size, 1)

  for _ in range(num_batches):
    # generate random inputs and labels
    inputs = torch.randn(batch_size, 3, image_w, image_h)
    labels = torch.zeros(batch_size, num_classes) \
      .scatter_(1, one_hot_indices, 1)

    # run forward pass
    optimizer.zero_grad()
    outputs = model(inputs.to('cuda:0'))

    # run backward pass
    labels = labels.to(outputs.device)
    loss_fn(outputs, labels).backward()
    optimizer.step()


import matplotlib.pyplot as plt

plt.switch_backend('Agg')
import numpy as np
import timeit

num_repeat = 10

stmt = "train(model)"

setup = "model = ModelParallelResNet50()"
# globals arg is only available in Python 3. In Python 2, use the following
# import __builtin__
# __builtin__.__dict__.update(locals())
mp_run_times = timeit.repeat(
  stmt, setup, number=1, repeat=num_repeat, globals=globals())
mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

setup = "import torchvision.models as models;" + \
        "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
rn_run_times = timeit.repeat(
  stmt, setup, number=1, repeat=num_repeat, globals=globals())
rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)


def plot(means, stds, labels, fig_name):
  fig, ax = plt.subplots()
  ax.bar(np.arange(len(means)), means, yerr=stds,
         align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
  ax.set_ylabel('ResNet50 Execution Time (Second)')
  ax.set_xticks(np.arange(len(means)))
  ax.set_xticklabels(labels)
  ax.yaxis.grid(True)
  plt.tight_layout()
  plt.savefig(fig_name)
  plt.close(fig)


plot([mp_mean, rn_mean],
     [mp_std, rn_std],
     ['Model Parallel', 'Single GPU'],
     'mp_vs_rn.png')



class PipelineParallelResNet50(ModelParallelResNet50):
  def __init__(self, split_size=20, *args, **kwargs):
    super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
    self.split_size = split_size

  def forward(self, x):
    splits = iter(x.split(self.split_size, dim=0))
    s_next = next(splits)
    s_prev = self.seq1(s_next).to('cuda:1')
    ret = []

    for s_next in splits:
      # A. s_prev runs on cuda:1
      s_prev = self.seq2(s_prev)
      ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

      # B. s_next runs on cuda:0, which can run concurrently with A
      s_prev = self.seq1(s_next).to('cuda:1')

    s_prev = self.seq2(s_prev)
    ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

    return torch.cat(ret)


setup = "model = PipelineParallelResNet50()"
pp_run_times = timeit.repeat(
  stmt, setup, number=1, repeat=num_repeat, globals=globals())
pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

plot([mp_mean, rn_mean, pp_mean],
     [mp_std, rn_std, pp_std],
     ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
     'mp_vs_rn_vs_pp.png')


means = []
stds = []
split_sizes = [1, 3, 5, 8, 10, 12, 20, 40, 60]

for split_size in split_sizes:
  setup = "model = PipelineParallelResNet50(split_size=%d)" % split_size
  pp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
  means.append(np.mean(pp_run_times))
  stds.append(np.std(pp_run_times))

fig, ax = plt.subplots()
ax.plot(split_sizes, means)
ax.errorbar(split_sizes, means, yerr=stds, ecolor='red', fmt='ro')
ax.set_ylabel('ResNet50 Execution Time (Second)')
ax.set_xlabel('Pipeline Split Size')
ax.set_xticks(split_sizes)
ax.yaxis.grid(True)
plt.tight_layout()
plt.savefig("split_size_tradeoff.png")
plt.close(fig)




