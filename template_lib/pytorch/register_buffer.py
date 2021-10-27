import torch
import torch.nn as nn
from torch.autograd import Variable

class Simple(nn.Module):
    '''A simple example'''

    def __init__(self):
        super(Simple, self).__init__()
        self.counter = 0

    def forward(self, x):
        print(self.counter)
        self.counter += 1
        print(self.counter)
        return x

if __name__ == '__main__':
    x = Variable(torch.randn(10, 10))
    net = Simple()
    # data_parallel = False  # True
    data_parallel = True
    if data_parallel:
        net = torch.nn.DataParallel(net)
    net = net.cuda()
    for i in range(10):
        print('iteration: {}'.format(i))
        y = net(x)