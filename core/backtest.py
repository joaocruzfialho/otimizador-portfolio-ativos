from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Well-known model portfolios (US-listed ETF proxies; long history on yfinance)
MODEL_PORTFOLIOS: dict[str, dict[str, float]] = {
    "60/40 Clássico": {
        "SPY": 0.60,   # US equities
        "TLT": 0.40,   # US long-term bonds
    },
    "All-Weather (Dalio)": {
        "VTI": 0.30,
        "TLT": 0.40,
        "IEF": 0.15,
        "GLD": 0.075,
        "DBC": 0.075,
    },
    "Permanent Portfolio": {
        "VTI": 0.25,
        "TLT": 0.25,
        "GLD": 0.25,
        "SHY": 0.25,
    },
    "Golden Butterfly": {
        "VTI": 0.20,
        "IJS": 0.20,   # small-cap value
        "TLT": 0.20,
        "SHY": 0.20,
        "GLD": 0.20,
    },
}


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set:
    if freq == "never":
        return set()
    freq_map_new = {"M": "ME", "Q": "QE", "A": "YE"}
    freq_map_old = {"M": "M", "Q": "Q", "A": "A"}
    try:
        dr = pd.date_range(index[0], index[-1], freq=freq_map_new[freq])
    except Exception:
        dr = pd.date_range(index[0], index[-1], freq=freq_map_old[freq])
    result = set()
    for d in dr:
        valid = index[index <= d]
        if len(valid) > 0:
            result.add(valid[-1])
    return result


def _metrics(series: pd.Series, rf_rate: float) -> dict:
    dr = series.pct_change().dropna()
    n_days = (series.index[-1] - series.index[0]).days
    n_years = max(n_days / 365.25, 1 / 365.25)
    cagr = float((series.iloc[-1] / series.iloc[0]) ** (1 / n_years) - 1)
    vol = float(dr.std() * np.sqrt(TRADING_DAYS))
    sharpe = (cagr - rf_rate) / vol if vol > 0 else float("nan")
    peak = series.expanding().max()
    max_dd = float(((series - peak) / peak).min())
    return {
        "CAGR": cagr,
        "Volatilidade": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Retorno Total": float(series.iloc[-1] / series.iloc[0] - 1),
        "Valor Final (€)": float(series.iloc[-1]),
    }


def run_backtest(
    closes: pd.DataFrame,
    target_weights: dict[str, float],
    initial_capital: float,
    rebalance_freq: str,
    rf_rate: float,
) -> dict:
    """
    Simula rebalanceamento periódico (com vendas) vs. buy-and-hold.

    rebalance_freq: 'M' (mensal), 'Q' (trimestral), 'A' (anual), 'never'
    """
    tickers = [t for t in target_weights if t in closes.columns]
    if not tickers:
        return {}

    cls = closes[tickers].ffill().dropna()
    if len(cls) < 5:
        return {}

    raw_w = np.array([float(target_weights[t]) for t in tickers])
    w = raw_w / raw_w.sum()
    weights = dict(zip(tickers, w))

    reb_dates = _rebalance_dates(cls.index, rebalance_freq)

    first_prices = cls.iloc[0]
    shares = {t: initial_capital * weights[t] / float(first_prices[t]) for t in tickers}
    bh_shares = dict(shares)

    port_vals: list[float] = []
    bh_vals: list[float] = []
    reb_events: list[dict] = []

    for i in range(len(cls)):
        row = cls.iloc[i]
        date = cls.index[i]
        pv = sum(shares[t] * float(row[t]) for t in tickers)
        bv = sum(bh_shares[t] * float(row[t]) for t in tickers)
        port_vals.append(pv)
        bh_vals.append(bv)
        if date in reb_dates and i < len(cls) - 1:
            reb_events.append({"Data": date, "Valor (€)": pv})
            for t in tickers:
                shares[t] = pv * weights[t] / float(row[t])

    port = pd.Series(port_vals, index=cls.index)
    bh = pd.Series(bh_vals, index=cls.index)

    return {
        "port": port,
        "bh": bh,
        "port_metrics": _metrics(port, rf_rate),
        "bh_metrics": _metrics(bh, rf_rate),
        "reb_events": pd.DataFrame(reb_events) if reb_events else pd.DataFrame(),
        "n_reb": len(reb_events),
        "missing_tickers": [t for t in target_weights if t not in closes.columns],
        "tickers_used": tickers,
    }


def compare_model_portfolios(
    model_closes: pd.DataFrame,
    initial_capital: float,
    rf_rate: float,
    user_bh_series: "pd.Series | None" = None,
) -> pd.DataFrame:
    """
    Run buy-and-hold backtest for each MODEL_PORTFOLIO and return a comparison table.
    model_closes: DataFrame with all needed tickers as columns.
    user_bh_series: optional Series of user portfolio value (same index) for comparison.
    Returns DataFrame with one row per portfolio.
    """
    rows = []

    if user_bh_series is not None and len(user_bh_series) > 1:
        m = _metrics(user_bh_series, rf_rate)
        rows.append({
            "Portfólio": "O Meu Portfólio",
            "CAGR (%)": m["CAGR"] * 100,
            "Volatilidade (%)": m["Volatilidade"] * 100,
            "Sharpe": m["Sharpe"],
            "Max Drawdown (%)": m["Max Drawdown"] * 100,
            "Retorno Total (%)": m["Retorno Total"] * 100,
            "Valor Final (€)": m["Valor Final (€)"],
            "_series": user_bh_series,
        })

    for name, weights in MODEL_PORTFOLIOS.items():
        r = run_backtest(model_closes, weights, initial_capital, "never", rf_rate)
        if not r or not r.get("tickers_used"):
            continue
        m = r["bh_metrics"]
        rows.append({
            "Portfólio": name,
            "CAGR (%)": m["CAGR"] * 100,
            "Volatilidade (%)": m["Volatilidade"] * 100,
            "Sharpe": m["Sharpe"],
            "Max Drawdown (%)": m["Max Drawdown"] * 100,
            "Retorno Total (%)": m["Retorno Total"] * 100,
            "Valor Final (€)": m["Valor Final (€)"],
            "_series": r["bh"],
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()
