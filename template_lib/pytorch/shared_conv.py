import torch
import torch.nn as nn

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv_weight = nn.Parameter(torch.randn(16 ,3 ,5 ,5))

  def forward(self, x):
    x1 = nn.functional.conv2d(x, self.conv_weight, bias=None,
                              stride=1, padding=2, dilation=1, groups=1)
    x2 = nn.functional.conv2d(x, self.conv_weight.transpose(2 ,3), bias=None,
                              stride=1, padding=2, dilation=1, groups=1)

    return x1 + x2

def main():
  x = torch.rand((2, 3, 128, 128))
  net = Net()
  x = net(x)
  pass

if __name__ == '__main__':
  main()