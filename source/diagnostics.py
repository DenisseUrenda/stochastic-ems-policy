
def check_energy_balance(history, dt):

    endog = history["endog"]

    q  = endog[..., 0]
    ec = endog[..., 1]
    ed = endog[..., 2]
    Ub = endog[..., 3]
    Us = endog[..., 4]

    lhs = Ub + ed
    rhs = q * dt + ec + Us

    residual = lhs - rhs

    return {
        "max_abs_error": residual.abs().max().item(),
        "mean_abs_error": residual.abs().mean().item(),
        "mean_Ub": Ub.mean().item(),
        "max_Ub": Ub.max().item(),
    }