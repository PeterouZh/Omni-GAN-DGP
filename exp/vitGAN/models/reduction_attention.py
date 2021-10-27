from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.models_v2 import MODEL_REGISTRY
from template_lib.utils import get_attr_kwargs, get_dict_str


@MODEL_REGISTRY.register(name_prefix=__name__)
class ReductionAttention(nn.Module):
  def __init__(self,
               dim,
               num_heads=8,
               qkv_bias=True,
               qk_scale=None,
               attn_drop=0.,
               proj_drop=0.,
               sr_ratio=1,
               **kwargs):
    super().__init__()
    assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

    self.dim = dim
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = qk_scale or head_dim ** -0.5

    self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
    self.kv_fc = nn.Linear(dim, dim * 2, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    # self.proj_fc = nn.Linear(dim, dim)
    # self.proj_drop = nn.Dropout(proj_drop)
    pass

    self.sr_ratio = sr_ratio
    if sr_ratio > 1:
      self.conv_down = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
      self.conv_down_norm = nn.LayerNorm(dim)

  def forward(self, x, H, W):
    B, N, C = x.shape
    # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    q = rearrange(self.q_fc(x), "b n (h d) -> b h n d", h=self.num_heads)

    if self.sr_ratio > 1:
      x_ = rearrange(x, "b (h w) d -> b d h w", h=H, w=W)
      # x_ = x.permute(0, 2, 1).reshape(B, C, H, W)

      x_ = self.conv_down(x_)
      x_ = rearrange(x_, "b d h w -> b (h w) d")
      # x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
      x_ = self.conv_down_norm(x_)

      x_ = self.kv_fc(x_)
      kv = rearrange(x_, "b n (kv h d) -> kv b h n d", kv=2, h=self.num_heads)
      # kv = x_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    else:
      x_ = self.kv_fc(x)
      kv = rearrange(x_, "b n (kv h d) -> kv b h n d", kv=2, h=self.num_heads)
      # kv = x_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = attn @ v
    x = rearrange(x, "b h n d -> b n (h d)")
    # x = x.transpose(1, 2).reshape(B, N, C)

    # x = self.proj_fc(x)
    # x = self.proj_drop(x)
    return x

