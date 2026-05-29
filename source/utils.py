
from typing import Optional
import torch

def set_seed(seed: int = 42, device: Optional[torch.device] = None):
  g = torch.Generator(device=device)    # Creates random number generator
  g.manual_seed(seed)                   # Set a seed
  torch.manual_seed(seed)               # Set PyTorch global generator
  return g

def read_csv(path: str, skip_header=True, dtype=torch.float32):
  with open(path, "r", encoding="utf-8") as f:
      lines = f.read().strip().splitlines()
  if skip_header:
      lines = lines[1:]
  rows = [[float(x) for x in line.split(",") if x.strip() != ""] for line in lines]
  return torch.tensor(rows, dtype=dtype).squeeze()
