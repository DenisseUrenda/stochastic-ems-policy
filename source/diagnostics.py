
def check_energy_balance(endog_hist, dt):

    q_hist  = endog_hist[..., 0]
    ec_hist = endog_hist[..., 1]
    ed_hist = endog_hist[..., 2]
    Ub_hist = endog_hist[..., 3]
    Us_hist = endog_hist[..., 4]

    lhs = Ub_hist + ed_hist
    rhs = q_hist * dt + ec_hist + Us_hist

    residual = lhs - rhs

    return {
        "max_abs_error": residual.abs().max().item(),
        "mean_abs_error": residual.abs().mean().item(),
        "mean_Ub": Ub_hist.mean().item(),
        "max_Ub": Ub_hist.max().item(),
    }