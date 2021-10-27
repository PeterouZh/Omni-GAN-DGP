import torch


def ortho(model, strength=1e-4, blacklist=[]):
  """
  # Apply modified ortho reg to a model
  # This function is an optimized version that directly computes the gradient,
  # instead of computing and then differentiating the loss.
  :param model:
  :param strength:
  :param blacklist:
  :return:
  """
  with torch.no_grad():
    for param in model.parameters():
      # Only apply this to parameters with at least 2 axes, and not in the blacklist
      if len(param.shape) < 2 or any([param is item for item in blacklist]):
        continue
      w = param.view(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t())
              * (1. - torch.eye(w.shape[0], device=w.device)), w))
      param.grad.data += strength * grad.view(param.shape)