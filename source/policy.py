
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from torch import Tensor
  from source.system import SystemParams

import torch.nn as nn
import torch
import math

# ----- Policy -----------------------------------------------------------------
# Define neural network called NeuralPolicy that inherits from nn.Module
class NeuralPolicy(nn.Module):
  """
  pi_theta(S_t) -> (eta_t, delta_t, gamma_t) in [0,1]x[-1,1]x[0,1]
  """

  # Input size: 2 + 4 (dimension of t + dimension of S_t)
  # Number of hidden layers: 3 (h1, h2, h3)
  # Neurons per layer: 64 (h{i1}, ..., h{i64})
  # Activation function: ReLU

  def __init__(self,
      input_dim: int = 6,
      output_dim: int = 3,
      units: list[int] = [16, 16, 16],
      activations: list[str] = ["relu", "relu", "relu"],
      output_activations: list[str] = ["sigmoid", "tanh", "sigmoid"],
      output_names: list[str] = ["eta", "delta", "gamma"]
    ):

    super().__init__()

    act_map = {
      "relu": nn.ReLU,
      "tanh": nn.Tanh,
      "elu": nn.ELU,
      "gelu": nn.GELU,
      "sigmoid": nn.Sigmoid,
      "softplus": nn.Softplus
    }

    self.out_acts = nn.ModuleList([act_map[a.lower()]() for a in output_activations])
    self.output_names = output_names

    layers = []                                       # To accumulate the backbone layers (the core network body)
    in_dim = input_dim

    for out_dim, act_name in zip(units, activations):
      act = act_map.get(act_name.lower(), nn.ReLU)
      layers += [nn.Linear(in_dim, out_dim), act()]
      in_dim = out_dim


    self.backbone = nn.Sequential(*layers)      # Packs the layer list into an nn.Sequential
    self.head = nn.Linear(in_dim, output_dim)   # Projects 3 logits: (eta, delta, gamma)
    self.reset_parameters()                     # Initializes weights/biases using a custom method


  def reset_parameters(self):
    for m in self.modules():               # Iterates over all modules
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Applies Xavier uniform initialization to weights
        nn.init.zeros_(m.bias)             # Sets all biases to zero


  def forward(self, S: Tensor):
    h = self.backbone(S)
    logits = self.head(h)              # (batch, output_dim)

    outputs = []
    for i, act in enumerate(self.out_acts):
        outputs.append(act(logits[:, i]))

    u = torch.stack(outputs, dim=-1)
    out_dict = {name: u[:, i] for i, name in enumerate(self.output_names)}

    return u, out_dict
  


def normalize_state(S: torch.Tensor, params: SystemParams):
    N, D, E, P = S.unbind(dim=-1)

    Nn = N / params.N_max
    Dn = (D / (params.D_cap + 1e-6)).clamp(0., 1.)
    En = E / params.E_max

    Pn = ((P - params.P_min) / (params.P_max - params.P_min)).clamp(0., 1.)

    return torch.stack([Nn, Dn, En, Pn], dim=-1)



def build_policy_input(t: float, S: torch.Tensor, params: SystemParams):

    t_tensor = torch.full((S.shape[0],), t, device=S.device, dtype=S.dtype)

    phase = 2 * math.pi * t_tensor / params.T

    time_features = torch.stack([
        torch.sin(phase),
        torch.cos(phase)
    ], dim=-1)

    S_norm = normalize_state(S, params)

    policy_input = torch.cat([time_features, S_norm], dim=-1)

    return policy_input


# ----- Heuristic Policy -----------------------------------------------------------------
class HeuristicPolicy(nn.Module):
  """
  EV-first price-threshold heuristic policy.

  eta:
    power allocation to EV demand

  delta:
    ESS charge/discharge control
    delta > 0: charge battery
    delta < 0: discharge battery

  gamma:
    grid purchase interpolation
    gamma = 0: minimum feasible grid purchase
    gamma = 1: maximum feasible grid purchase
  """

  def __init__(
      self,
      low_price: float = 0.17,
      high_price: float = 0.25,
      output_names: list[str] = ["eta", "delta", "gamma"]
    ):
    super().__init__()

    if low_price >= high_price:
      raise ValueError("low_price must be smaller than high_price.")

    self.low_price = low_price
    self.high_price = high_price
    self.output_names = output_names


  def forward(self, S: torch.Tensor):
    batch = S.shape[0]
    device = S.device
    dtype = S.dtype

    # policy input = [sin_t, cos_t, Nn, Dn, En, Pn]
    price = S[:, 5]

    # Always prioritize EV charging
    eta = torch.ones(batch, device=device, dtype=dtype)

    # Neutral defaults
    delta = torch.zeros(batch, device=device, dtype=dtype)
    gamma = torch.full((batch,), 0.5, device=device, dtype=dtype)

    low_price_mask = price <= self.low_price
    high_price_mask = price >= self.high_price

    # Low price: charge ESS and allow maximum feasible grid purchase
    delta[low_price_mask] = 1.0
    gamma[low_price_mask] = 1.0

    # High price: discharge ESS and use minimum feasible grid purchase
    delta[high_price_mask] = -1.0
    gamma[high_price_mask] = 0.0

    u = torch.stack([eta, delta, gamma], dim=-1)

    out_dict = {
      name: u[:, i] for i, name in enumerate(self.output_names)
    }

    return u, out_dict