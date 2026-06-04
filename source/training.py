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
    lambda_demand: float = 0.0,
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
        lambda_demand=lambda_demand,
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
    lambda_demand: float = 0.0,
    alpha_grid: float = 0.0,
    batch_size: int = 64,
    seed_base: int = 123,
    eval_seed: int = 9999,
    verbose: bool = True,
    checkpoint_dir: str | None = None,
):
    best_epoch = None
    best_J = float("inf")
    best_state = None

    t0 = time.time()

    J_train_hist = []
    J_stoch_eval_hist = []
    J_det_eval_hist = []

    training_steps = []
    evaluation_steps = []

    print_each = max(1, epochs // 20)

    for epoch in range(1, epochs + 1):

        epoch_seed = seed_base + epoch

        J_train = train_step(
            policy=policy,
            optimizer=optimizer,
            S0_single=S0_single,
            params=params,
            endog_history=endog_history,
            future_endog=future_endog,
            lambda_cost=lambda_cost,
            lambda_demand=lambda_demand,
            alpha_grid=alpha_grid,
            batch_size=batch_size,
            stochastic=True,
            seed=epoch_seed,
            device=device,
        )

        J_train_hist.append(J_train)
        training_steps.append(epoch)

        if epoch % print_each == 0:
            with torch.no_grad():
                next_price_fn_eval = make_price_function(
                    endog_history,
                    future_endog,
                    params,
                )

                S0_eval = S0_single.to(device).repeat(batch_size, 1)

                hist_stoch = rollout_trajectory(
                    policy=policy,
                    S0=S0_eval,
                    params=params,
                    next_price_fn=next_price_fn_eval,
                    lambda_cost=lambda_cost,
                    lambda_demand=lambda_demand,
                    alpha_grid=alpha_grid,
                    stochastic=True,
                    seed=eval_seed,
                    device=device,
                )

                hist_det = rollout_trajectory(
                    policy=policy,
                    S0=S0_eval,
                    params=params,
                    next_price_fn=next_price_fn_eval,
                    lambda_cost=lambda_cost,
                    lambda_demand=lambda_demand,
                    alpha_grid=alpha_grid,
                    stochastic=False,
                    seed=eval_seed,
                    device=device,
                )

            J_stoch_eval = hist_stoch["J_episode"].item()
            J_det_eval = hist_det["J_episode"].item()

            J_stoch_eval_hist.append(J_stoch_eval)
            J_det_eval_hist.append(J_det_eval)
            evaluation_steps.append(epoch)

            if J_stoch_eval < best_J:
                best_J = J_stoch_eval
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in policy.state_dict().items()
                }

            if verbose:
                print(
                    f"[{epoch:03d}] "
                    f"J_train={J_train:.4f} | "
                    f"J_stoch_eval_fixed={J_stoch_eval:.4f} | "
                    f"J_det_eval={J_det_eval:.4f}"
                )

    last_state = {
        k: v.detach().cpu().clone()
        for k, v in policy.state_dict().items()
    }

    dt = time.time() - t0

    print(
        f"\nTraining ready in {dt:.1f}s. "
        f"Best epoch: {best_epoch}. "
        f"Best J_stoch_eval={best_J:.4f}"
    )

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save(
            {
                "state_dict": best_state,
                "best_J_stoch_eval": best_J,
                "best_epoch": best_epoch,
                "eval_seed": eval_seed,
                "params": vars(params),
            },
            os.path.join(checkpoint_dir, "policy_best.pt"),
        )

        torch.save(
            {
                "state_dict": last_state,
                "last_J_train": J_train,
                "params": vars(params),
            },
            os.path.join(checkpoint_dir, "policy_last.pt"),
        )

        print("\nSaved policies:")
        print(
            f" - Best policy: {os.path.join(checkpoint_dir, 'policy_best.pt')} "
            f"(epoch {best_epoch}, J_stoch_eval={best_J:.4f})"
        )
        print(
            f" - Last policy: {os.path.join(checkpoint_dir, 'policy_last.pt')} "
            f"(epoch {epoch}, J_train={J_train:.4f})"
        )

    return {
        "best_state": best_state,
        "best_J": best_J,
        "best_epoch": best_epoch,
        "last_state": last_state,
        "J_train_hist": J_train_hist,
        "J_stoch_eval_hist": J_stoch_eval_hist,
        "J_det_eval_hist": J_det_eval_hist,
        "training_steps": training_steps,
        "evaluation_steps": evaluation_steps,
        "eval_seed": eval_seed,
        "policy": policy,
    }