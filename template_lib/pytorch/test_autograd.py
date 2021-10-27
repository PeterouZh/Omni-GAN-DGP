import torch
import torch.nn as nn

torch.cuda.manual_seed_all(0)
torch.manual_seed(0)


def gradient_penalty(netD, x):
  """Functional Gradient Calculation"""
  output = netD(x)
  gradients = torch.autograd.grad(
    outputs=output, inputs=x,
    grad_outputs=x.new_ones(output.size()),
    create_graph=True, retain_graph=True)[0].mean()
  gradients.backward()
  return gradients


net = nn.Linear(4, 1).cuda()
multigpu_net = nn.DataParallel(net, [0, 1])

x = torch.ones(2, 4, requires_grad=True).cuda()

# print("Single GPU Functional")
# net.zero_grad()
# loss = gradient_penalty(net, x)
# loss.backward()
# print("Loss:", loss.item())
# print("Grad:", [p.grad for p in net.parameters() if p.grad is not None])

print("\nMulti-GPU Functional")
multigpu_net.zero_grad()
loss = gradient_penalty(multigpu_net, x)
# loss.backward()
print("Loss:", loss.item())
print("Grad:", [p.grad for p in net.parameters() if p.grad is not None])

print("\nMulti-GPU Inline")
multigpu_net.zero_grad()
output = multigpu_net(x)

# Compute grad inline
loss = torch.autograd.grad(outputs=output, inputs=x,
                           grad_outputs=x.new_ones(output.size()),
                           create_graph=True, retain_graph=True)[0].mean()
loss.backward()
print("Loss:", loss.item())
print("Grad:", [p.grad for p in net.parameters() if p.grad is not None])
