from __future__ import annotations

import pandas as pd


def compute_unrealized_gains(
    tickers: list[str],
    quantities: dict[str, float],
    avg_buy_prices: dict[str, float],
    prices_eur: dict[str, float],
    tax_rate: float = 0.28,
) -> tuple[pd.DataFrame, float, float]:
    """
    Per-asset unrealized gain/loss report.
    Positions with avg_buy_price = 0 are skipped (no cost basis recorded).

    Returns: (df, total_unrealized_gain_eur, estimated_tax_eur)
    """
    rows = []
    for t in tickers:
        qty = float(quantities.get(t, 0))
        avg = float(avg_buy_prices.get(t, 0))
        price = float(prices_eur.get(t, 0))
        if qty <= 0 or price <= 0 or avg <= 0:
            continue
        cost = avg * qty
        value = price * qty
        gain = value - cost
        gain_pct = gain / cost * 100
        rows.append({
            "Ticker": t,
            "Qtd.": qty,
            "Preço Médio (€)": avg,
            "Preço Atual (€)": price,
            "Custo Base (€)": cost,
            "Valor Atual (€)": value,
            "Mais-Valia (€)": gain,
            "Mais-Valia (%)": gain_pct,
            "Imposto Estimado (€)": max(0.0, gain) * tax_rate,
        })

    if not rows:
        return pd.DataFrame(), 0.0, 0.0

    df = pd.DataFrame(rows)
    return df, float(df["Mais-Valia (€)"].sum()), float(df["Imposto Estimado (€)"].sum())
