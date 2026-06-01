
from typing import Optional
import torch

import pandas as pd

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


def summarize_histories(histories: dict) -> pd.DataFrame:

    variable_map = {
        # states
        "N": ("state", 0),
        "D": ("state", 1),
        "E": ("state", 2),
        "P": ("state", 3),

        # controls
        "eta": ("control", 0),
        "delta": ("control", 1),
        "gamma": ("control", 2),

        # endogenous
        "q": ("endog", 0),
        "ec": ("endog", 1),
        "ed": ("endog", 2),
        "Ub": ("endog", 3),
        "Us": ("endog", 4),

        # events
        "A": ("arrival", None),
        "B": ("departure", None),

        # costs
        "c": ("cost", None),

        # episode objective
        "J": ("J_episode", None),
    }

    summary = {}

    for policy_name, hist in histories.items():

        summary[policy_name] = {}

        for var_name, (group, idx) in variable_map.items():

            x = hist[group]

            if idx is not None:
                x = x[..., idx]

            summary[policy_name][f"{var_name}_mean"] = x.mean().item()
            
            if x.numel() > 1:
                summary[policy_name][f"{var_name}_std"] = x.std().item()

    return pd.DataFrame(summary)