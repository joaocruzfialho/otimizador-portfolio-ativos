from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def _prep(closes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Annualized mean returns, cov matrix, tickers list."""
    rets = closes.pct_change().dropna()
    tickers = list(rets.columns)
    mean = rets.mean().values * TRADING_DAYS
    cov = rets.cov().values * TRADING_DAYS
    cov += np.eye(len(tickers)) * 1e-8   # regularize against singular matrices
    return mean, cov, tickers


def _perf(w: np.ndarray, mean: np.ndarray, cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    r = float(w @ mean)
    v = float(np.sqrt(max(0.0, w @ cov @ w)))
    s = (r - rf) / v if v > 0 else 0.0
    return r, v, s


def _solve(obj, n: int, extra_cons: list, wmin: float, wmax: float) -> np.ndarray:
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}] + extra_cons
    bds = [(wmin, wmax)] * n
    best_fun, best_w = np.inf, None
    x0_list = [np.ones(n) / n] + [np.eye(n)[i] for i in range(min(n, 4))]
    for x0 in x0_list:
        res = minimize(obj, x0, method="SLSQP", bounds=bds, constraints=cons,
                       options={"ftol": 1e-9, "maxiter": 500, "disp": False})
        if res.success and res.fun < best_fun:
            best_fun, best_w = res.fun, res.x
    return best_w if best_w is not None else np.ones(n) / n


def compute_risk_metrics(
    closes: pd.DataFrame,
    target_pct: dict[str, float],
    rf_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per-asset stats, correlation matrix, portfolio-level metrics."""
    rets = closes.pct_change().dropna(how="all")
    if rets.empty or len(rets) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    mean_ann = rets.mean() * TRADING_DAYS
    vol_ann = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = (mean_ann - rf_rate) / vol_ann.replace(0, np.nan)
    corr = rets.corr()

    per_asset = pd.DataFrame({
        "Ticker": rets.columns.tolist(),
        "Retorno Anual.": mean_ann.values,
        "Volatilidade Anual.": vol_ann.values,
        "Sharpe": sharpe.values,
    })

    tickers = list(rets.columns)
    raw_w = np.array([float(target_pct.get(t, 0)) for t in tickers])
    if raw_w.sum() > 0:
        w = raw_w / raw_w.sum()
        mean_v = mean_ann.values
        cov_v = rets.cov().values * TRADING_DAYS
        port_ret = float(w @ mean_v)
        port_vol = float(np.sqrt(max(0.0, w @ cov_v @ w)))
        port_sharpe = (port_ret - rf_rate) / port_vol if port_vol > 0 else float("nan")
        port = {"mean_ret_ann": port_ret, "vol_ann": port_vol,
                "sharpe": port_sharpe, "n_obs": int(len(rets))}
    else:
        port = {"n_obs": int(len(rets))}

    return per_asset, corr, port


def compute_drawdown(
    closes: pd.DataFrame,
    weights: dict[str, float],
) -> dict:
    """
    Portfolio drawdown series from normalised historical closes.

    Returns: drawdown (Series), port_value (Series), max_drawdown (float),
             max_drawdown_date, periods (DataFrame of distinct drawdown episodes).
    """
    if closes.empty:
        return {}

    tickers = list(closes.columns)
    w = np.array([float(weights.get(t, 0)) for t in tickers], dtype=float)
    if w.sum() == 0:
        w = np.ones(len(tickers)) / len(tickers)
    else:
        w = w / w.sum()

    normed = closes / closes.iloc[0]
    port = (normed * w).sum(axis=1)
    rolling_max = port.cummax()
    drawdown = (port - rolling_max) / rolling_max

    max_dd = float(drawdown.min())
    max_dd_date = drawdown.idxmin()

    periods, in_dd = [], False
    peak_dt = trough_dt = None
    trough_val = 0.0
    for dt, val in drawdown.items():
        if not in_dd and val < -0.001:
            in_dd, peak_dt, trough_dt, trough_val = True, dt, dt, val
        elif in_dd:
            if val < trough_val:
                trough_dt, trough_val = dt, val
            if val >= -0.001:
                in_dd = False
                periods.append({
                    "Início": peak_dt,
                    "Mínimo": trough_dt,
                    "Recuperação": dt,
                    "Queda Máx. (%)": round(trough_val * 100, 2),
                    "Dias até Mínimo": (trough_dt - peak_dt).days,
                    "Duração Total (dias)": (dt - peak_dt).days,
                })

    return {
        "drawdown": drawdown,
        "port_value": port,
        "max_drawdown": max_dd,
        "max_drawdown_date": max_dd_date,
        "periods": pd.DataFrame(periods) if periods else pd.DataFrame(),
    }


def markowitz_analysis(
    closes: pd.DataFrame,
    target_pct: dict[str, float],
    rf_rate: float,
    w_min: float = 0.0,
    w_max: float = 1.0,
    n_frontier: int = 40,
) -> dict:
    """
    Returns dict with:
      tickers, cur_w, mv_w, ms_w,
      cur_perf, mv_perf, ms_perf  → each (ret, vol, sharpe)
      frontier   → DataFrame(Retorno, Volatilidade, Sharpe)
      random     → DataFrame(Volatilidade, Retorno, Sharpe)  [Monte Carlo]
    """
    if len(closes.columns) < 2 or len(closes) < 30:
        return {}

    mean, cov, tickers = _prep(closes)
    n = len(tickers)

    raw_w = np.array([float(target_pct.get(t, 0)) for t in tickers])
    cur_w = raw_w / raw_w.sum() if raw_w.sum() > 0 else np.ones(n) / n

    mv_w = _solve(lambda w: _perf(w, mean, cov, 0)[1], n, [], w_min, w_max)
    ms_w = _solve(lambda w: -_perf(w, mean, cov, rf_rate)[2], n, [], w_min, w_max)

    # Efficient frontier
    ret_lo = float(mv_w @ mean)
    ret_hi = float(np.max(mean))
    frontier_rows = []
    for r_target in np.linspace(ret_lo, ret_hi, n_frontier):
        extra = [{"type": "eq", "fun": lambda w, r=r_target: float(w @ mean) - r}]
        w_f = _solve(lambda w: _perf(w, mean, cov, rf_rate)[1], n, extra, w_min, w_max)
        r, v, s = _perf(w_f, mean, cov, rf_rate)
        frontier_rows.append({"Retorno": r, "Volatilidade": v, "Sharpe": s})

    # Monte Carlo cloud
    np.random.seed(42)
    rw = np.random.dirichlet(np.ones(n), size=min(2000, 200 * n))
    rw = np.clip(rw, w_min, w_max)
    rw /= rw.sum(axis=1, keepdims=True)
    rv = np.sqrt(np.maximum(0, np.diag(rw @ cov @ rw.T)))
    rr = rw @ mean
    rs = np.where(rv > 0, (rr - rf_rate) / rv, np.nan)

    return {
        "tickers": tickers,
        "cur_w": dict(zip(tickers, cur_w)),
        "mv_w": dict(zip(tickers, mv_w)),
        "ms_w": dict(zip(tickers, ms_w)),
        "cur_perf": _perf(cur_w, mean, cov, rf_rate),
        "mv_perf": _perf(mv_w, mean, cov, rf_rate),
        "ms_perf": _perf(ms_w, mean, cov, rf_rate),
        "frontier": pd.DataFrame(frontier_rows),
        "random": pd.DataFrame({"Volatilidade": rv, "Retorno": rr, "Sharpe": rs}),
    }
