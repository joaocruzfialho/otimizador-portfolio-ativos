from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dividends(ticker: str) -> pd.Series:
    """Dividend history as a float Series with tz-naive DatetimeIndex."""
    try:
        divs = yf.Ticker(ticker).dividends
        if divs is None or len(divs) == 0:
            return pd.Series(dtype=float, name=ticker)
        divs.index = pd.to_datetime(divs.index).tz_localize(None)
        return divs.sort_index()
    except Exception:
        return pd.Series(dtype=float, name=ticker)


def asset_dividend_metrics(
    ticker: str,
    quantity: float,
    avg_buy_price: float,
    current_price_eur: float,
) -> dict:
    """Dividend metrics for a single asset."""
    divs = fetch_dividends(ticker)
    result: dict = {
        "annual_income_eur": 0.0,
        "yield_current": 0.0,
        "yield_on_cost": 0.0,
        "dgr_3y": None,
        "dgr_5y": None,
        "history_df": pd.DataFrame(),
        "calendar_df": pd.DataFrame(),
    }
    if divs.empty or quantity <= 0:
        return result

    now = pd.Timestamp.now()
    last_12m = divs[divs.index >= now - pd.DateOffset(months=12)]
    annual_dps = float(last_12m.sum()) if not last_12m.empty else 0.0
    result["annual_income_eur"] = annual_dps * quantity
    if current_price_eur > 0:
        result["yield_current"] = annual_dps / current_price_eur
    if avg_buy_price > 0:
        result["yield_on_cost"] = annual_dps / avg_buy_price

    # Annual aggregated history
    yearly = divs.groupby(divs.index.year).sum()
    if not yearly.empty:
        result["history_df"] = pd.DataFrame({
            "Ano": yearly.index.astype(int),
            "Div/Ação (€)": yearly.values.astype(float),
            "Rendimento Total (€)": yearly.values.astype(float) * quantity,
        })

    # DGR (annualised)
    def _cagr(n: int) -> float | None:
        if len(yearly) < n + 1:
            return None
        end, start = float(yearly.iloc[-1]), float(yearly.iloc[-(n + 1)])
        return (end / start) ** (1 / n) - 1 if start > 0 and end > 0 else None

    result["dgr_3y"] = _cagr(3)
    result["dgr_5y"] = _cagr(5)

    # Recent calendar (last 24 months)
    recent = divs[divs.index >= now - pd.DateOffset(months=24)]
    if not recent.empty:
        cal = recent.reset_index()
        cal.columns = ["Data", "Div/Ação (€)"]
        cal["Rendimento (€)"] = cal["Div/Ação (€)"] * quantity
        cal["Mês"] = cal["Data"].dt.to_period("M").astype(str)
        result["calendar_df"] = (
            cal[["Mês", "Data", "Div/Ação (€)", "Rendimento (€)"]]
            .sort_values("Data", ascending=False)
            .reset_index(drop=True)
        )

    return result


def portfolio_dividend_report(
    tickers: list[str],
    quantities: dict[str, float],
    avg_prices: dict[str, float],
    prices_eur: dict[str, float],
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Returns: (per_asset_df, total_annual_income_eur, calendar_df)
    """
    rows, calendars = [], []
    for t in tickers:
        m = asset_dividend_metrics(
            t,
            float(quantities.get(t, 0)),
            float(avg_prices.get(t, 0)),
            float(prices_eur.get(t, 0)),
        )
        rows.append({
            "Ticker": t,
            "Rendimento Anual (€)": m["annual_income_eur"],
            "Yield Atual (%)": m["yield_current"] * 100,
            "Yield on Cost (%)": m["yield_on_cost"] * 100,
            "DGR 3A (%)": m["dgr_3y"] * 100 if m["dgr_3y"] is not None else float("nan"),
            "DGR 5A (%)": m["dgr_5y"] * 100 if m["dgr_5y"] is not None else float("nan"),
        })
        if not m["calendar_df"].empty:
            cal = m["calendar_df"].copy()
            cal.insert(0, "Ticker", t)
            calendars.append(cal)

    per_asset = pd.DataFrame(rows)
    total = float(per_asset["Rendimento Anual (€)"].sum())
    calendar_all = (
        pd.concat(calendars, ignore_index=True).sort_values("Data", ascending=False)
        if calendars else pd.DataFrame()
    )
    return per_asset, total, calendar_all
