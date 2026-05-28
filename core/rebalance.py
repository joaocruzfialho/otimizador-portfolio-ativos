from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_no_sell(
    current_values: list[float],
    target_weights: list[float],
    money_to_invest: float,
) -> np.ndarray:
    """Distribui money_to_invest pelos ativos sem vender (X_i >= 0)."""
    n = len(current_values)
    current = np.asarray(current_values, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    target = target / target.sum()
    total_after = current.sum() + money_to_invest
    buy = np.zeros(n)
    fixed = np.zeros(n, dtype=bool)
    for _ in range(n + 1):
        active = ~fixed
        if not active.any():
            break
        fixed_value = current[fixed].sum()
        active_pool = total_after - fixed_value
        active_target_sum = target[active].sum()
        if active_target_sum <= 0 or active_pool <= 0:
            break
        ideal = np.where(active, target / active_target_sum * active_pool, current)
        over = active & (ideal < current - 1e-9)
        if not over.any():
            buy = np.where(active, ideal - current, 0.0)
            break
        fixed = fixed | over
    return np.maximum(buy, 0.0)


def compute_rebalance_result(
    portfolio_df: pd.DataFrame,
    money_to_invest: float,
    prices: dict[str, float],
    currencies: dict[str, str],
    fx: dict[str, float],
) -> pd.DataFrame:
    df = portfolio_df.copy().reset_index(drop=True)
    df["Moeda"] = df["Ticker"].map(currencies)
    df["Preço Local"] = df["Ticker"].map(prices).astype(float)
    df["FX→EUR"] = df["Moeda"].map(fx).astype(float)
    df["Preço Atual (€)"] = df["Preço Local"] * df["FX→EUR"]
    df["Valor Atual (€)"] = df["Quantidade Detida"] * df["Preço Atual (€)"]

    buy = rebalance_no_sell(
        df["Valor Atual (€)"].tolist(),
        df["Percentagem Alvo (%)"].tolist(),
        money_to_invest,
    )
    df["Investir (€)"] = buy
    df["Qtd a Comprar"] = df["Investir (€)"] / df["Preço Atual (€)"]
    df["Valor Final (€)"] = df["Valor Atual (€)"] + df["Investir (€)"]

    total_cur = df["Valor Atual (€)"].sum()
    total_fin = df["Valor Final (€)"].sum()
    df["Alocação Atual (%)"] = df["Valor Atual (€)"] / total_cur * 100 if total_cur > 0 else 0.0
    df["Alocação Final (%)"] = df["Valor Final (€)"] / total_fin * 100 if total_fin > 0 else 0.0
    df["Desvio Atual (pp)"] = df["Alocação Atual (%)"] - df["Percentagem Alvo (%)"]
    df["Desvio Final (pp)"] = df["Alocação Final (%)"] - df["Percentagem Alvo (%)"]
    return df


def parse_scenarios(text: str) -> list[float]:
    out = []
    for piece in text.replace(";", ",").split(","):
        piece = piece.strip().replace(" ", "").replace("€", "").replace("_", "")
        if not piece:
            continue
        try:
            v = float(piece)
            if v >= 0:
                out.append(v)
        except ValueError:
            continue
    seen: set[float] = set()
    result = []
    for v in out:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def health_score(deviations_pp: list[float]) -> int:
    if not deviations_pp:
        return 100
    return max(0, min(100, int(round(100 - max(abs(d) for d in deviations_pp) / 25.0 * 100))))


def rebalance_urgency(deviations_pp: list[float], threshold: float) -> tuple[str, str]:
    if not deviations_pp:
        return "ok", "Sem desvios — verificar trimestralmente."
    m = max(abs(d) for d in deviations_pp)
    if m < threshold / 2:
        return "ok", "Desvios baixos — verificar trimestralmente."
    if m < threshold:
        return "ok", "Próximo do limiar — verificar mensalmente."
    if m < threshold * 2:
        return "warning", "Rebalanceamento recomendado nas próximas semanas."
    return "error", "Rebalanceamento urgente — desvio significativo."
