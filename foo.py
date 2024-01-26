import os
import torchaudio
import torch

from einops import rearrange

B, Q, T = 1, 3, 5

x = torch.arange(Q * T).view(1, Q, T)

print(x)

print("flattened")

x = rearrange(x, "B Q T -> B T Q")
x = rearrange(x, "B T Q -> B (T Q)")

print(x)

print("unflattened")

x = rearrange(x, "B (T Q) -> B T Q", T=T)
x = rearrange(x, "B T Q -> B Q T")

print(x)