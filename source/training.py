
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from source.policy import NeuralPolicy
    from source.system import SystemParams

import torch
from torch import optim

import time
import os

from source.stochastic_processes import make_price_function
from source.simulation import rollout_trajectory

def train_step(
    policy: NeuralPolicy,
    optimizer: optim.Optimizer,
    S0_single: torch.Tensor,
    params: SystemParams,
    endog_history: torch.Tensor,
    future_endog: torch.Tensor,
    lambda_cost: float = 0.0,
    alpha_grid: float = 0.0,
    batch_size: int = 10,
    stochastic: bool = True,
    seed: int = 123,
    device: torch.device | None = None,
):

    device = device or torch.device("cpu")
    policy.train()
    optimizer.zero_grad()

    next_price_fn = make_price_function(endog_history, future_endog, params)

    S0 = S0_single.to(device).repeat(batch_size, 1)

    # Trajectory
    history = rollout_trajectory(
        policy=policy,
        S0=S0,
        params=params,
        next_price_fn=next_price_fn,
        lambda_cost=lambda_cost,
        alpha_grid=alpha_grid,
        stochastic=stochastic,
        seed=seed,
        device=device,
    )

    J = history["J_episode"]

    J.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    return float(J.item())



def train_net(
    policy: NeuralPolicy,
    optimizer: optim.Optimizer,
    device: torch.device,
    params: SystemParams,
    S0_single: torch.Tensor,
    endog_history: torch.Tensor,
    future_endog: torch.Tensor,
    epochs: int = 100,
    lambda_cost: float = 0.0,
    alpha_grid: float = 0.0,
    batch_size: int = 64,
    seed_base: int = 123,
    verbose: bool = True,
    checkpoint_dir: str | None = None
):

  pred_J_eval = None
  best_J = float("inf")
  best_state = None
  t0 = time.time()

  J_train_hist = []
  J_eval_hist = []

  print_each = max(1, epochs // 20)

  for epoch in range(1, epochs + 1):

      J_train = train_step(
          policy=policy,
          optimizer=optimizer,
          S0_single=S0_single,
          params=params,
          endog_history=endog_history,
          future_endog=future_endog,
          lambda_cost=lambda_cost,
          alpha_grid=alpha_grid,
          batch_size=batch_size,
          stochastic=True,
          seed=seed_base + epoch,
          device=device,
      )

      J_train_hist.append(J_train)

      if J_train < best_J:
          best_J = J_train
          best_epoch = epoch
          best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

      if epoch % print_each == 0:
          with torch.no_grad():
            next_price_fn_eval = make_price_function(endog_history, future_endog, params)
            history = rollout_trajectory(
                policy=policy,
                S0=S0_single.repeat(batch_size, 1),
                lambda_cost=lambda_cost,
                alpha_grid=alpha_grid,
                params=params,
                next_price_fn=next_price_fn_eval,
                stochastic=False,
                seed=seed_base,
                device=device,
            )
          J_eval = history["J_episode"]
          J_eval_hist.append(J_eval.item())

          if verbose:
            print(f"[{epoch:03d}] J_train={J_train:.4f} | J_eval(det)={J_eval.item():.4f}")

  last_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

  dt = time.time() - t0
  print(f"\nTraining ready in {dt:.1f}s. Epoch: {best_epoch}. Best J={best_J:.4f}")

  if checkpoint_dir is not None:
      os.makedirs(checkpoint_dir, exist_ok=True)

      torch.save(
          {"state_dict": best_state, "best_J": best_J, "params": vars(params)},
          os.path.join(checkpoint_dir, "policy_best.pt")
      )

      torch.save(
          {"state_dict": last_state, "last_J": J_train, "params": vars(params)},
          os.path.join(checkpoint_dir, "policy_last.pt")
      )

      print("\nSaved policies:")
      print(f" - Best policy:  {os.path.join(checkpoint_dir, 'policy_best.pt')}  (epoch {best_epoch}, J={best_J:.4f})")
      print(f" - Last policy:  {os.path.join(checkpoint_dir, 'policy_last.pt')}  (epoch {epoch}, J={J_train:.4f})")

  return {
      "best_state": best_state,
      "best_J": best_J,
      "last_state": last_state,
      "J_train_hist": J_train_hist,
      "J_eval_hist": J_eval_hist,
      "policy": policy
  }
