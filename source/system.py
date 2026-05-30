
from dataclasses import dataclass
import torch

from typing import Callable, Optional

# ----- System parameters -------------------------------------------------
@dataclass                # Creates simple container-style classes
class SystemParams:
  dt: float = 0.5        # Time increment (Δt)
  T:  int = 24           # Number of time steps

  E_max: float = 300.0   # Battery capacity
  N_max: float = 30.0    # Parking-lot capacity
  q_max: float = 11.0    # Maximum charging power per EV  # --------------- New line
  d_avg: float = 25.0    # Average EV demand
  phi:   float = 0.10    # Departure probability
  rho:   float = 0.01    # (>0) make-up factor for grid buy/sell
  kappa: float = 1.1     # (>1) tariff multiplier for EV charging

  D_cap: float = 2.0 * d_avg * N_max  # Maximum cumulative demand
  P_min: float = 0.00                 # Minimum energy price
  P_max: float = 0.20                 # Maximum energy price





# ----- Selling price, buying price, and tariff ---------------------------------------
def compute_prices(P_t: torch.Tensor, rho: float, kappa: float):
    """
    Price components from the market price P_t:
      P^b_t = P_t + rho,   P^s_t = max(0, P_t - rho),   P^EV_t = kappa * P^b_t
    Args:
        P_t: tensor (...) market price at time t
        rho: ρ > 0 (grid markup)
        kappa: κ > 1 (EV tariff factor)
    Returns:
        Pb_t, Ps_t, PEV_t  (same shape as P_t)
    """
    Pb_t  = P_t + rho
    Ps_t  = torch.clamp(P_t - rho, min=0.0)
    PEV_t = kappa * Pb_t
    return Pb_t, Ps_t, PEV_t



# ----- Interval cost c_t -------------------------------------------------
def interval_cost(Pb_t, Ps_t, PEV_t, U_b, U_s, q_t, dt):
    """
    Compute instant cost c_t
    """
    return Pb_t * U_b - Ps_t * U_s - PEV_t * q_t * dt



# ----- System dynamics f(S,u,xi) -----

def system_dynamics(
  S_t: torch.Tensor,
  u_t: torch.Tensor,
  t: torch.Tensor | float,
  params: SystemParams,
  A_t_override: torch.Tensor, # Arrivals
  B_t_override: torch.Tensor, # Departures
  e_t_override: torch.Tensor, # Innovations for SARIMA process
  next_price_fn: Callable[[torch.Tensor], torch.Tensor],
  rng: Optional[torch.Generator] = None,
  ) -> torch.Tensor:
  """
  S_{t + delta t} = f(S_t, u_t) with capacity N_max and admitted arrivals tilde A_t.
  State:  S_t = [N_t, D_t, E_t, P_t]
  Control: u_t = [eta_t, delta_t, gamma_t]
  """
  assert S_t.ndim == 2 and S_t.size(-1) == 4  # Shape check for S_t: (batch, 4)
  assert u_t.ndim == 2 and u_t.size(-1) == 3  # Shape check for u_t: (batch, 3)

  N_t, D_t, E_t, P_t = S_t.unbind(dim=-1)  # Unpack columns along the last (-1) dimension: 4 tensors of shape (batch,)
  eta, delta, gamma  = u_t.unbind(dim=-1)  # Unpack columns along the last (-1) dimension: 3 tensors of shape (batch,)


  tT = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=S_t.dtype, device=S_t.device)
  if tT.ndim == 0:
    tT = tT.expand_as(N_t)


  # (1) Sampling arrivals and departures: A_t and B_t
  A_t = A_t_override.to(S_t.device, S_t.dtype).reshape(-1).detach()
  B_t = B_t_override.to(S_t.device, S_t.dtype).reshape(-1).detach()


  # (2) Admitted arrivals and occupancy (capacity constraint)
  headroom = torch.clamp(params.N_max - (N_t - B_t), min=0.0)
  A_adm = torch.minimum(A_t, headroom)
  N_next = torch.clamp(N_t + A_adm - B_t, 0.0, params.N_max)


  # (3) Pending demand
  # delivered in the step: q_t = eta * D_t / dt
  # loss due to departures: b_t * B_t, where b_t = D_t / max(N_t, epsilon)
  q_t = eta * D_t / params.dt
  q_t = torch.minimum(q_t, N_t * params.q_max)
  b_t = torch.where(N_t > 0.0, D_t / torch.clamp(N_t, min=1e-6), torch.zeros_like(D_t))
  D_next = D_t + params.d_avg * A_adm - q_t * params.dt - b_t * B_t
  D_next = torch.clamp(D_next, min=0.0)


  # (4) Stored energy (asymmetric: charge/discharge)
  delta_plus  = torch.clamp(delta, min=0.0)
  delta_minus = torch.clamp(-delta, min=0.0)
  e_charge = delta_plus * (params.E_max - E_t)
  e_discharge = delta_minus * E_t
  E_next = torch.clamp(E_t + e_charge - e_discharge, 0.0, params.E_max)


  # (5) Energy price
  mu_P = next_price_fn()
  P_next = torch.exp(torch.log(mu_P) + e_t_override) * 0.001 # In USD/kWh


  # (6) Energy purchased from and sold to the grid
  Ub_min = torch.clamp(q_t * params.dt + e_charge - e_discharge, min=0.0)
  Ub_max = q_t * params.dt + e_charge
  U_b = Ub_min + gamma * (Ub_max - Ub_min)
  U_s = U_b + e_discharge - q_t * params.dt - e_charge


  # (7) Interval cost
  Pb_t, Ps_t, PEV_t = compute_prices(P_t, rho=params.rho, kappa=params.kappa)

  c_t = interval_cost(
      Pb_t=Pb_t,
      Ps_t=Ps_t,
      PEV_t=PEV_t,
      U_b=U_b,
      U_s=U_s,
      q_t=q_t,
      dt=params.dt
  )

  S_next = torch.stack([N_next, D_next, E_next, P_next], dim=-1)
  endog = torch.stack([q_t, e_charge, e_discharge, U_b, U_s], dim=-1)
  return S_next, c_t, endog
