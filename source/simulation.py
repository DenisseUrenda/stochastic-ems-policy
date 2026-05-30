from __future__ import annotations
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
   from source.system import SystemParams
   from source.policy import NeuralPolicy

import torch

from source.system import system_dynamics
from source.utils import set_seed
from source.policy import build_policy_input
from source.stochastic_processes import arrivals_departures, lambda_intensity, innovations


# ----- Rollout of the entire trajectory --------------------
def rollout_trajectory(
    policy: NeuralPolicy,
    S0: torch.Tensor,                  # (batch, 4)
    T: float,                    # horizon in hours
    params: SystemParams,
    next_price_fn: Callable[[torch.Tensor], torch.Tensor],
    lambda_cost: float = 0.0,
    alpha_grid: float = 0.0,
    stochastic: bool = True,
    seed: int = 123,
    device: Optional[torch.device] = None,
):
    """
    Simulates a full trajectory with time steps of size params.dt.
    Returns tensors containing the history of states, endogenous and actions.
    """
    device = device or torch.device("cpu")
    g = set_seed(seed, device=device)

    policy = policy.to(device)
    S = S0.to(device)

    steps = int(T / params.dt)
    batch = S.size(0)          # Batch size (S has shape (batch, 4))

    # Buffers to store the history
    S_hist = torch.zeros(steps + 1, batch, 4, device=device)
    endog_hist = torch.zeros(steps + 1, batch, 5, device=device)
    u_hist = torch.zeros(steps    , batch, 3, device=device)
    t_hist = torch.zeros(steps + 1, device=device)

   # Store event variables
    A_hist = torch.zeros(steps, batch, device=device, dtype=S.dtype)
    B_hist = torch.zeros(steps, batch, device=device, dtype=S.dtype)
    e_hist = torch.zeros(steps, batch, device=device, dtype=S.dtype)


    S_hist[0] = S
    t0 = 0.0
    c_hist = []

    for k in range(steps):
      t_k = t0 + k * params.dt
      t_hist[k] = t_k

      input = build_policy_input(t_k, S, params)

      # Compute policy using the nn: u_k = pi_theta(S_k)
      u_k, _ = policy(input)    # Forward pass through the entire network to obtain u_k of shape (batch, 3)
      u_hist[k] = u_k


      # Arrivals and departures
      A_t, B_t = arrivals_departures(
          t=t_k,
          S_t=S,
          dt=params.dt,
          phi=params.phi,
          stochastic=stochastic,
          lambda_fn=lambda_intensity,
          rng=g,
          mu=10.77, sigma=3.32, scale=40.0
        )

      # SARIMA innovations
      e_t = innovations(t=torch.tensor(t_k).expand(S.shape[0]), stochastic=stochastic)

      A_hist[k] = A_t
      B_hist[k] = B_t
      e_hist[k] = e_t

      # System dynamics: S_{k+1}
      S, c, endog = system_dynamics(
          S_t=S,
          u_t=u_k,
          t=t_k,
          params=params,
          next_price_fn=next_price_fn,
          A_t_override=A_t.detach(),
          B_t_override=B_t.detach(),
          e_t_override=e_t.detach(),
          rng=g,
      )


      U_b = endog[:, 3]   # Grid usage ------------
      c = c + alpha_grid * U_b.pow(2)

      S_hist[k + 1] = S
      endog_hist[k + 1] = endog
      c_hist.append(c)


    c_hist = torch.stack(c_hist, dim=0)
    c_pos = torch.clamp(c_hist, min=0.0)
    c_total = c_hist + lambda_cost * c_pos

    J_per_traj = c_total.sum(dim=0)
    J_episode  = J_per_traj.mean()

    t_hist[-1] = t0 + steps * params.dt

    return t_hist, S_hist, endog_hist, u_hist, A_hist, B_hist, c_hist, J_episode



def sample_initial_state(batch_size, params, device=None, seed=None):

    device = device or torch.device("cpu")

    if seed is not None:
        torch.manual_seed(seed)

    N0 = torch.randint(0, 3, (batch_size, 1), dtype=torch.float32, device=device)
    D0 = N0 * params.d_avg
    E0 = torch.rand(batch_size, 1, device=device) * params.E_max
    P0 = 0.01 + torch.rand(batch_size, 1, device=device) * (0.02 - 0.01)

    return torch.cat([N0, D0, E0, P0], dim=1)