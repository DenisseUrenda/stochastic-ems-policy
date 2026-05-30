from __future__ import annotations
from typing import Tuple, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from source.system import SystemParams

import torch
import math

from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAProcess:
  """
  Implements a fixed-parameter SARIMA process for electricity prices.

  `SARIMAProcess` receives historical prices in USD/MWh and internally applies
  a log transformation before updating the SARIMA state using the specified
  orders and parameters. The method `forecast_next_step()` returns the
  one-step-ahead predicted price (without innovation) converted back to the
  original scale (USD/MWh) via the exponential transformation.
  """
  def __init__(
      self,
      endog_hist: torch.Tensor,
      future_endog: torch.Tensor,
      order: Tuple[int, int, int] = (1,1,1),
      seasonal_order: Tuple[int, int, int, int] = (1,1,1,24),
      params_dict_sarima:
        dict = {
          "ar.L1": 0.9358, "ma.L1": -0.9998,
          "ar.S.L24": 0.1986, "ma.S.L24": -0.9135,
          "sigma2": 0.0174
          },
      t: float = 0.0,
      dt: float = 0.5
  ):
    assert endog_hist.ndim == 1 and torch.all(endog_hist > 0)
    self.device, self.dtype = endog_hist.device, endog_hist.dtype

    log_endog_hist = torch.log(endog_hist.detach().cpu()).tolist()
    self.future_endog = future_endog

    spec = SARIMAX(
        endog=log_endog_hist,
        order=order,
        seasonal_order=seasonal_order,
        enforce_invertibility=True,
        enforce_stationarity=False,
    )
    names = spec.param_names
    params_vec = [float(params_dict_sarima[name]) for name in names]
    self.model = spec.filter(params_vec)

    self.t = t
    self.dt = dt
    self.P_current = endog_hist[-1]
    self.P_next_hour = None


  def _interpolate(self, x1, x2, tau):
    return torch.exp ((1 - tau) * torch.log(x1) + tau * torch.log(x2))


  def _update_sarima(self, P_obs: torch.Tensor):
    self.P_current = P_obs
    self.model = self.model.append(endog=[torch.log(self.P_current).cpu().numpy()])


  def forecast_next_hour(self) -> torch.Tensor:
    log_mean = self.model.forecast(1)[0]
    self.P_next_hour = torch.exp(torch.tensor(log_mean))# + eps)
    return self.P_next_hour


  def forecast_next_step(self) -> torch.Tensor:

    self.t += self.dt
    tau = float(self.t - math.floor(self.t))

    if math.isclose(tau, 0.0): # t is integer
      P_forecasted = self.P_next_hour
    elif math.isclose(tau, self.dt): # t is integer + dt
      P_obs = self.future_endog[int(math.floor(self.t))]
      self._update_sarima(P_obs)
      self.P_next_hour = self.forecast_next_hour()
      P_forecasted = self._interpolate(self.P_current, self.P_next_hour, tau)
    else:
      P_forecasted = self._interpolate(self.P_current, self.P_next_hour, tau)

    return P_forecasted


# ----- Intensity function for arrivals (A_t) ------------------------------
# Defines normal_pdf because PyTorch does not provide a normal_pdf function
def normal_pdf(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
  z = (x - mu) / sigma
  return torch.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


# Defines the intensity function lambda(t) for the NHPP
def lambda_intensity(t: torch.Tensor | float, mu: float = 10.77, sigma: float = 3.32, scale: float = 40.0) -> torch.Tensor:
  tT = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
  return scale * normal_pdf(tT, mu, sigma)


def arrivals_departures(
    t: torch.Tensor | float,
    S_t: torch.Tensor,
    dt: float,
    phi: float,
    stochastic: bool = True,
    lambda_fn: Callable[..., torch.Tensor] = lambda_intensity,
    rng: Optional[torch.Generator] = None,
    idx_n: int = 0,
    **lambda_kwargds
    ) -> torch.Tensor:
  """
  Returns (A_t, B_t) with shape (batch,):
    - A_t ~ Poisson( λ(t) * dt )
    - B_t ~ Binomial( N_t, phi )
  Rules:
    - If t is a scalar → λ(t) is shared across the batch (but Poisson samples are independent).
    - If t is (batch,) → λ(t) is applied elementwise.
    - If N_t = 0 → B_t = 0 (forced).
    - phi may be a scalar or a (batch,) tensor.
  """
  device, dtype = S_t.device, S_t.dtype
  batch = S_t.shape[0]

  # ------- arrivals ---------------------
  lam = lambda_fn(t, **lambda_kwargds)
  if not torch.is_tensor(lam):
      lam = torch.tensor(lam, device=device, dtype=dtype)
  else:
      lam = lam.to(device=device, dtype=dtype)

  if lam.ndim == 0:
      lam = lam.expand(batch)

  rate = torch.clamp(lam * dt, min=0.0)
  if stochastic:
    A_t = torch.poisson(rate, generator=rng).to(dtype)
  else: A_t = rate

  # -------- departures ----------------------
  N_t = S_t[:, idx_n].to(device=device, dtype=dtype).clamp(min=0.0).round()
  B_t = torch.zeros_like(N_t, dtype=dtype)

  mask = N_t > 0
  if mask.any():
    if stochastic:
      probs = torch.full_like(N_t[mask], phi)
      B_t[mask] = torch.binomial(N_t[mask], probs, generator=rng).to(dtype)
    else:
      B_t[mask] = (phi * N_t[mask])

  return A_t, B_t



# ----- Defines innovation for SARIMA process as Student T ----------------------
def innovations(
    t: torch.Tensor,
    df: float = 10.8,
    loc: float = 0.0,
    scale: float = math.sqrt(0.0174),
    stochastic: bool = True,
    rng: Optional[torch.Generator] = None
  ) -> torch.Tensor:

  n = t.shape[0] if t.ndim > 0 else 1

  if stochastic:
    dist = torch.distributions.StudentT(df=df, loc=loc, scale=scale)
    return dist.sample((n,))
  
  else:
    return torch.tensor(loc).expand(n)



# ----- Price function ------------------------------------------------------
def make_price_function(
    endog_history: torch.Tensor, 
    future_endog: torch.Tensor, 
    params: SystemParams
    ):
    sarima = SARIMAProcess(
        endog_hist=endog_history, 
        future_endog=future_endog, 
        dt=params.dt
    )

    def f() -> torch.Tensor:
        return sarima.forecast_next_step()

    return f
