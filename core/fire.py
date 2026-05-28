from __future__ import annotations

import numpy as np
import pandas as pd


def run_fire_simulation(
    portfolio_value: float,
    annual_spending: float,
    years: int,
    annual_return_mean: float,
    annual_return_std: float,
    inflation_rate: float,
    n_simulations: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo FIRE / Safe Withdrawal Rate simulation.

    Returns:
      success_rate       – fraction of simulations where portfolio survives
      safe_swr           – max annual withdrawal rate for ≥95% success
      percentiles        – DataFrame(Ano, p5, p25, p50, p75, p95)
      depletion_years    – list of year indices when failed simulations depleted
      n_simulations      – total simulations run
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(annual_return_mean, annual_return_std,
                         (n_simulations, years))

    values = np.zeros((n_simulations, years + 1))
    values[:, 0] = portfolio_value

    for yr in range(years):
        infl = (1 + inflation_rate) ** yr
        after = np.maximum(0.0, values[:, yr] - annual_spending * infl)
        values[:, yr + 1] = after * (1 + returns[:, yr])

    survived = values[:, -1] > 0
    success_rate = float(survived.mean())

    pcts = np.percentile(values, [5, 25, 50, 75, 95], axis=0)
    percentiles = pd.DataFrame({
        "Ano": np.arange(years + 1),
        "p5": pcts[0], "p25": pcts[1], "p50": pcts[2],
        "p75": pcts[3], "p95": pcts[4],
    })

    # Binary-search for highest SWR with ≥95% survival
    lo, hi, safe_swr = 0.001, 0.20, None
    for _ in range(30):
        mid = (lo + hi) / 2
        w0 = portfolio_value * mid
        v = np.zeros((n_simulations, years + 1))
        v[:, 0] = portfolio_value
        for yr in range(years):
            infl = (1 + inflation_rate) ** yr
            v[:, yr + 1] = np.maximum(0, v[:, yr] - w0 * infl) * (1 + returns[:, yr])
        if float((v[:, -1] > 0).mean()) >= 0.95:
            safe_swr = mid
            lo = mid
        else:
            hi = mid

    # Depletion year for failed simulations
    failed = ~survived
    depletion_years: list[int] = []
    if failed.any():
        first_zero = np.argmax(values[failed] <= 0, axis=1)
        depletion_years = first_zero.tolist()

    return {
        "success_rate": success_rate,
        "safe_swr": safe_swr,
        "percentiles": percentiles,
        "depletion_years": depletion_years,
        "n_simulations": n_simulations,
    }
