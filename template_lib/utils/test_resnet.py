import os
import random
import time
# from multiprocessing import Process
from torch.multiprocessing import Process

# from multiprocessing import set_start_method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass


class TorchResnetWorker(Process):
  def run(self):
    bs, gpu, determine_bs, q = self._args
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    import torch
    import torch.nn.functional as F
    import torchvision
    net = torchvision.models.resnet152().cuda()
    net = torch.nn.DataParallel(net).cuda()

    if determine_bs:
      self.determine_bs(net, q)
    else:
      self.train(net, bs)

  def train(self, net, bs):
    try:
      import torch
      import torch.nn.functional as F
      rbs = bs
      print(bs)
      while True:
        t = random.random()
        time.sleep(t)

        x = torch.rand(rbs, 3, 224, 224).cuda()
        y = net(x)

        tensor = torch.randint(0, 1000, (rbs,))  # tensor([0, 1, 2, 0, 1])
        one_hot = F.one_hot(tensor, num_classes=1000).float().cuda()
        loss = (y - one_hot).mean()
        loss.backward()

        t = random.random()
        time.sleep(t)
        rbs = random.randint(1, bs)
    except RuntimeError:
      torch.cuda.empty_cache()
      pass

  def determine_bs(self, net, q):
    import torch
    import torch.nn.functional as F
    bs = 0
    try:
      while True:
        bs += 1
        print('%s' % bs)
        x = torch.rand(bs, 3, 224, 224).cuda()
        y = net(x)

        tensor = torch.randint(0, 1000, (bs,))  # tensor([0, 1, 2, 0, 1])
        one_hot = F.one_hot(tensor, num_classes=1000).float().cuda()
        loss = (y - one_hot).mean()
        loss.backward()
    except RuntimeError:
      torch.cuda.empty_cache()
      q.put(bs - 1)
      # import traceback
      # print(traceback.format_exc())

def run(localrank, bs, gpu, is_determine_bs, q):

  os.environ['CUDA_VISIBLE_DEVICES'] = gpu
  import torch
  import torch.nn.functional as F
  import torchvision
  net = torchvision.models.resnet152().cuda()
  net = torch.nn.DataParallel(net).cuda()

  if is_determine_bs:
    determine_bs(net, q)
  else:
    train(net, bs)

def train(net, bs):
  try:
    import torch
    import torch.nn.functional as F
    rbs = bs
    print(bs)
    while True:
      # t = random.random()
      # time.sleep(t)

      x = torch.rand(rbs, 3, 224, 224).cuda()
      y = net(x)

      tensor = torch.randint(0, 1000, (rbs,))  # tensor([0, 1, 2, 0, 1])
      one_hot = F.one_hot(tensor, num_classes=1000).float().cuda()
      loss = (y - one_hot).mean()
      loss.backward()

      # t = random.random()
      # time.sleep(t)
      rbs = random.randint(1, bs)
  except RuntimeError:
    torch.cuda.empty_cache()
    pass

def determine_bs(net, q):
  import torch
  import torch.nn.functional as F
  bs = 0
  try:
    while True:
      bs += 1
      print('%s' % bs)
      x = torch.rand(bs, 3, 224, 224).cuda()
      y = net(x)

      tensor = torch.randint(0, 1000, (bs,))  # tensor([0, 1, 2, 0, 1])
      one_hot = F.one_hot(tensor, num_classes=1000).float().cuda()
      loss = (y - one_hot).mean()
      loss.backward()
  except:
    torch.cuda.empty_cache()
    import traceback
    print(traceback.format_exc())
    os.makedirs('results', exist_ok=True)
    with open('results/max_bs.txt', 'w') as f:
      f.write(str(bs - 1))
    # q.put(bs - 1)
