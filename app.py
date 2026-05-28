"""
Otimizador de Portfólio de Ativos com Alertas de Rebalanceamento Inteligente

App Streamlit que:
- Recebe um portfólio (Tickers, Percentagens Alvo, Quantidades Detidas)
- Obtém preços atuais via yfinance
- Calcula a distribuição ideal de um valor a investir
- Garante que NÃO sugere vendas — apenas compras corretivas

Autor: João Fialho
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ------------------------------------------------------------------
# Algoritmo de rebalanceamento (apenas compras)
# ------------------------------------------------------------------

def rebalance_no_sell(
    current_values: list[float],
    target_weights: list[float],
    money_to_invest: float,
) -> np.ndarray:
    """
    Distribui `money_to_invest` pelos ativos para aproximar a alocação alvo,
    sem vender nada (X_i >= 0 para todo i).

    Algoritmo iterativo:
      1. Calcula o valor ideal de cada ativo após investimento.
      2. Ativos cujo valor atual já excede o ideal são "fixados" (compra = 0).
      3. Redistribui o dinheiro restante pelos ativos não-fixados,
         proporcionalmente aos respetivos pesos alvo renormalizados.
      4. Repete até nenhum ativo novo ficar sobre-alocado.

    Parameters
    ----------
    current_values : list[float]
        Valor atual em EUR de cada ativo (quantidade * preço).
    target_weights : list[float]
        Pesos alvo (em %) — devem somar 100.
    money_to_invest : float
        Montante (EUR) disponível para investir.

    Returns
    -------
    np.ndarray
        Array de compras em EUR (>= 0) na mesma ordem dos inputs.
    """
    n = len(current_values)
    current = np.asarray(current_values, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    target = target / target.sum()  # normaliza para somar 1

    total_after = current.sum() + money_to_invest
    buy = np.zeros(n)
    fixed = np.zeros(n, dtype=bool)

    # Limite de segurança (n iterações no pior caso)
    for _ in range(n + 1):
        active = ~fixed
        if not active.any():
            break

        fixed_value = current[fixed].sum()
        active_pool = total_after - fixed_value
        active_target_sum = target[active].sum()

        if active_target_sum <= 0 or active_pool <= 0:
            break

        # Valor ideal por ativo (somente para ativos ativos)
        ideal = np.where(
            active,
            target / active_target_sum * active_pool,
            current,
        )

        # Novos sobre-alocados: ideal < atual (com tolerância)
        over = active & (ideal < current - 1e-9)

        if not over.any():
            # Nenhum novo sobre-alocado — finalizar
            buy = np.where(active, ideal - current, 0.0)
            break

        fixed = fixed | over

    # Segurança: sem valores negativos por erros de vírgula flutuante
    return np.maximum(buy, 0.0)


# ------------------------------------------------------------------
# Obtenção de preços via yfinance
# ------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(tickers: tuple[str, ...]) -> dict[str, float | None]:
    """Obtém preço atual de cada ticker. Cache de 5 minutos."""
    prices: dict[str, float | None] = {}
    for ticker in tickers:
        if not ticker or not ticker.strip():
            prices[ticker] = None
            continue
        try:
            t = yf.Ticker(ticker)
            price: float | None = None
            try:
                fast = t.fast_info
                price = float(fast["last_price"]) if "last_price" in fast else None
            except Exception:
                price = None
            if price is None:
                hist = t.history(period="5d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            prices[ticker] = price
        except Exception:
            prices[ticker] = None
    return prices


# ------------------------------------------------------------------
# UI Streamlit
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Otimizador de Portfólio",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Otimizador de Portfólio de Ativos")
st.caption(
    "Rebalanceamento inteligente sem vendas — apenas **compras corretivas**."
)

# Portfólio por defeito (exemplo: ETFs comuns + cripto)
DEFAULT_PORTFOLIO = pd.DataFrame(
    {
        "Ticker": ["VWCE.DE", "AGGH.MI", "SGLD.MI", "BTC-EUR"],
        "Percentagem Alvo (%)": [60.0, 25.0, 10.0, 5.0],
        "Quantidade Detida": [10.0, 50.0, 5.0, 0.05],
    }
)

# --- Sidebar: parâmetros ---
with st.sidebar:
    st.header("Parâmetros")
    money_to_invest = st.number_input(
        "Valor a Investir (€)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        format="%.2f",
        help="Montante em euros disponível para compras corretivas.",
    )
    deviation_alert = st.slider(
        "Limiar de Alerta (pp)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help=(
            "Desvio absoluto (pontos percentuais) entre alocação atual e alvo "
            "acima do qual é gerado um alerta."
        ),
    )
    st.divider()
    st.caption(
        "Os tickers seguem a convenção do Yahoo Finance "
        "(ex.: `VWCE.DE`, `AAPL`, `BTC-EUR`)."
    )

# --- 1. Definir Portfólio ---
st.subheader("1. Defina o seu Portfólio")
st.markdown(
    "Edite os tickers, percentagens alvo e quantidades detidas. "
    "Pode adicionar ou remover linhas."
)

edited_df = st.data_editor(
    DEFAULT_PORTFOLIO,
    num_rows="dynamic",
    use_container_width=True,
    key="portfolio_editor",
    column_config={
        "Ticker": st.column_config.TextColumn(
            "Ticker", help="Símbolo Yahoo Finance", required=True
        ),
        "Percentagem Alvo (%)": st.column_config.NumberColumn(
            "Percentagem Alvo (%)",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            format="%.2f",
        ),
        "Quantidade Detida": st.column_config.NumberColumn(
            "Quantidade Detida",
            min_value=0.0,
            step=0.0001,
            format="%.4f",
        ),
    },
)

# Limpar linhas vazias / inválidas
edited_df = edited_df.dropna(subset=["Ticker"]).copy()
edited_df["Ticker"] = edited_df["Ticker"].astype(str).str.strip()
edited_df = edited_df[edited_df["Ticker"] != ""]

# Validar soma das percentagens
total_target = float(edited_df["Percentagem Alvo (%)"].sum())
sum_is_valid = abs(total_target - 100.0) < 0.01

col_v1, col_v2 = st.columns([1, 3])
col_v1.metric("Soma Alvo (%)", f"{total_target:.2f}%")
if sum_is_valid:
    col_v2.success("✓ Soma das percentagens alvo correta (100%).")
else:
    col_v2.error(
        f"⚠️ A soma é {total_target:.2f}% — deve ser exatamente 100%."
    )

# --- 2. Calcular ---
st.subheader("2. Calcular Rebalanceamento")

calc = st.button(
    "🔄 Calcular distribuição ótima",
    type="primary",
    disabled=not sum_is_valid or edited_df.empty,
)

if calc:
    tickers = tuple(edited_df["Ticker"].tolist())

    with st.spinner("A obter preços atuais (Yahoo Finance)..."):
        prices = fetch_prices(tickers)

    missing = [t for t, p in prices.items() if p is None]
    if missing:
        st.error(
            "Não foi possível obter preço para os seguintes tickers: "
            + ", ".join(f"`{t}`" for t in missing)
            + ". Verifique a grafia (convenção Yahoo Finance)."
        )
        st.stop()

    df = edited_df.reset_index(drop=True).copy()
    df["Preço Atual (€)"] = df["Ticker"].map(prices).astype(float)
    df["Valor Atual (€)"] = df["Quantidade Detida"] * df["Preço Atual (€)"]

    current_values = df["Valor Atual (€)"].tolist()
    target_pcts = df["Percentagem Alvo (%)"].tolist()

    buy_amounts = rebalance_no_sell(current_values, target_pcts, money_to_invest)

    df["Investir (€)"] = buy_amounts
    df["Qtd a Comprar"] = df["Investir (€)"] / df["Preço Atual (€)"]
    df["Valor Final (€)"] = df["Valor Atual (€)"] + df["Investir (€)"]

    total_current = df["Valor Atual (€)"].sum()
    total_final = df["Valor Final (€)"].sum()

    if total_current > 0:
        df["Alocação Atual (%)"] = df["Valor Atual (€)"] / total_current * 100
    else:
        df["Alocação Atual (%)"] = 0.0

    if total_final > 0:
        df["Alocação Final (%)"] = df["Valor Final (€)"] / total_final * 100
    else:
        df["Alocação Final (%)"] = 0.0

    df["Desvio Atual (pp)"] = df["Alocação Atual (%)"] - df["Percentagem Alvo (%)"]
    df["Desvio Final (pp)"] = df["Alocação Final (%)"] - df["Percentagem Alvo (%)"]

    # --- 3. Resumo ---
    st.subheader("3. Resumo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Valor Atual Total", f"€{total_current:,.2f}")
    m2.metric("Investimento", f"€{money_to_invest:,.2f}")
    m3.metric("Valor Final Total", f"€{total_final:,.2f}")
    invested = float(df["Investir (€)"].sum())
    m4.metric(
        "Total Distribuído",
        f"€{invested:,.2f}",
        delta=f"{(invested - money_to_invest):+.2f} € vs. input",
    )

    # --- 4. Tabela detalhada ---
    st.subheader("4. Distribuição Recomendada")
    display_cols = [
        "Ticker",
        "Preço Atual (€)",
        "Quantidade Detida",
        "Valor Atual (€)",
        "Alocação Atual (%)",
        "Percentagem Alvo (%)",
        "Desvio Atual (pp)",
        "Investir (€)",
        "Qtd a Comprar",
        "Valor Final (€)",
        "Alocação Final (%)",
        "Desvio Final (pp)",
    ]
    st.dataframe(
        df[display_cols].style.format(
            {
                "Preço Atual (€)": "{:.2f}",
                "Quantidade Detida": "{:.4f}",
                "Valor Atual (€)": "{:.2f}",
                "Alocação Atual (%)": "{:.2f}",
                "Percentagem Alvo (%)": "{:.2f}",
                "Desvio Atual (pp)": "{:+.2f}",
                "Investir (€)": "{:.2f}",
                "Qtd a Comprar": "{:.4f}",
                "Valor Final (€)": "{:.2f}",
                "Alocação Final (%)": "{:.2f}",
                "Desvio Final (pp)": "{:+.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --- 5. Comparação visual ---
    st.subheader("5. Alocação: Atual vs. Alvo vs. Final")
    chart_df = df.set_index("Ticker")[
        ["Alocação Atual (%)", "Percentagem Alvo (%)", "Alocação Final (%)"]
    ]
    st.bar_chart(chart_df)

    # --- 6. Alertas Inteligentes ---
    st.subheader("6. Alertas Inteligentes")
    alerts: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        dev = row["Desvio Atual (pp)"]
        if abs(dev) >= deviation_alert:
            if dev > 0:
                alerts.append(
                    (
                        "warning",
                        f"🔴 **{row['Ticker']}** está **sobre-alocado** em "
                        f"{dev:+.2f} pp (atual {row['Alocação Atual (%)']:.2f}% "
                        f"vs. alvo {row['Percentagem Alvo (%)']:.2f}%). "
                        f"Como não vendemos, só normaliza com novas entradas "
                        f"noutros ativos.",
                    )
                )
            else:
                alerts.append(
                    (
                        "info",
                        f"🔵 **{row['Ticker']}** está **sub-alocado** em "
                        f"{dev:+.2f} pp — recebe €{row['Investir (€)']:.2f} "
                        f"({row['Qtd a Comprar']:.4f} unidades).",
                    )
                )

    # Alertas pós-rebalanceamento
    max_post_deviation = float(df["Desvio Final (pp)"].abs().max())
    if max_post_deviation < 0.5:
        st.success(
            f"✅ Após o investimento, todos os ativos ficam dentro de "
            f"±0.5 pp do alvo. Rebalanceamento perfeito."
        )
    elif max_post_deviation < deviation_alert:
        st.success(
            f"✅ Após o investimento, o desvio máximo é "
            f"{max_post_deviation:.2f} pp — dentro do limiar de alerta "
            f"({deviation_alert:.1f} pp)."
        )
    else:
        st.warning(
            f"⚠️ Desvio máximo pós-investimento: {max_post_deviation:.2f} pp. "
            f"Considere aumentar o montante a investir para um rebalanceamento "
            f"mais completo."
        )

    if not alerts:
        st.info("Nenhum desvio acima do limiar definido.")
    else:
        for level, msg in alerts:
            if level == "warning":
                st.warning(msg)
            else:
                st.info(msg)

else:
    st.info(
        "Defina o portfólio acima e clique em **Calcular distribuição ótima** "
        "para ver a recomendação."
    )
