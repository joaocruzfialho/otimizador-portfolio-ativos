"""
Otimizador de Portfólio de Ativos com Alertas de Rebalanceamento Inteligente

App Streamlit que:
- Recebe um portfólio (Tickers, Percentagens Alvo, Quantidades Detidas)
- Obtém preços atuais via yfinance e converte para EUR
- Calcula a distribuição ideal de um valor a investir
- Garante que NÃO sugere vendas — apenas compras corretivas
- Gera alertas de desvio, concentração e urgência de rebalanceamento
- Persiste o portfólio entre sessões (JSON local + import/export)

Autor: João Fialho
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

__version__ = "0.2.0"

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"

DEFAULT_PORTFOLIO = pd.DataFrame(
    {
        "Ticker": ["VWCE.DE", "AGGH.MI", "SGLD.MI", "BTC-EUR"],
        "Percentagem Alvo (%)": [60.0, 25.0, 10.0, 5.0],
        "Quantidade Detida": [10.0, 50.0, 5.0, 0.05],
    }
)

EDITOR_COLS = ["Ticker", "Percentagem Alvo (%)", "Quantidade Detida"]


# ------------------------------------------------------------------
# Persistência
# ------------------------------------------------------------------

def load_portfolio() -> pd.DataFrame:
    """Carrega portfólio guardado em JSON, ou devolve o default."""
    if PORTFOLIO_FILE.exists():
        try:
            data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
            rows = data.get("portfolio", [])
            if rows:
                df = pd.DataFrame(
                    [
                        {
                            "Ticker": r["ticker"],
                            "Percentagem Alvo (%)": float(r["target_pct"]),
                            "Quantidade Detida": float(r["quantity"]),
                        }
                        for r in rows
                    ]
                )
                return df[EDITOR_COLS]
        except Exception:
            pass
    return DEFAULT_PORTFOLIO.copy()


def save_portfolio(df: pd.DataFrame) -> None:
    """Guarda portfólio em data/portfolio.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "ticker": str(row["Ticker"]),
            "target_pct": float(row["Percentagem Alvo (%)"]),
            "quantity": float(row["Quantidade Detida"]),
        }
        for _, row in df.iterrows()
        if str(row["Ticker"]).strip()
    ]
    payload = {
        "version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "portfolio": rows,
    }
    PORTFOLIO_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def df_to_export_json(df: pd.DataFrame) -> str:
    payload = {
        "version": 1,
        "portfolio": [
            {
                "ticker": str(row["Ticker"]),
                "target_pct": float(row["Percentagem Alvo (%)"]),
                "quantity": float(row["Quantidade Detida"]),
            }
            for _, row in df.iterrows()
            if str(row["Ticker"]).strip()
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


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
    sem vender nada (X_i >= 0). Ver README para detalhes do algoritmo.
    """
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

        ideal = np.where(
            active,
            target / active_target_sum * active_pool,
            current,
        )
        over = active & (ideal < current - 1e-9)

        if not over.any():
            buy = np.where(active, ideal - current, 0.0)
            break

        fixed = fixed | over

    return np.maximum(buy, 0.0)


# ------------------------------------------------------------------
# Preços e câmbio
# ------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_and_currency(ticker: str) -> tuple[float | None, str | None]:
    """Devolve (preço na moeda nativa, moeda) do ticker."""
    if not ticker or not ticker.strip():
        return None, None
    try:
        t = yf.Ticker(ticker)
        price: float | None = None
        currency: str | None = None
        try:
            fast = t.fast_info
            if "last_price" in fast:
                price = float(fast["last_price"])
            cur = fast.get("currency")
            if cur:
                currency = str(cur).upper()
        except Exception:
            pass
        if price is None:
            hist = t.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        if currency is None:
            try:
                info = t.info
                cur = info.get("currency")
                currency = str(cur).upper() if cur else "EUR"
            except Exception:
                currency = "EUR"
        return price, currency
    except Exception:
        return None, None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fx_to_eur(currency: str) -> float | None:
    """FX da `currency` para EUR (1 unidade da moeda em EUR)."""
    if not currency or currency.upper() == "EUR":
        return 1.0
    pair = f"{currency.upper()}EUR=X"
    try:
        t = yf.Ticker(pair)
        try:
            rate = float(t.fast_info["last_price"])
            if rate > 0:
                return rate
        except Exception:
            pass
        hist = t.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        return None
    return None


# ------------------------------------------------------------------
# Métricas de saúde
# ------------------------------------------------------------------

def health_score(deviations_pp: list[float]) -> int:
    """0-100 baseado no desvio máximo absoluto (25pp ⇒ 0; 0pp ⇒ 100)."""
    if not deviations_pp:
        return 100
    max_abs = max(abs(d) for d in deviations_pp)
    return max(0, min(100, int(round(100 - (max_abs / 25.0) * 100))))


def rebalance_urgency(deviations_pp: list[float], threshold_pp: float) -> tuple[str, str]:
    """Devolve (nível, sugestão) baseado no desvio máximo vs. limiar."""
    if not deviations_pp:
        return "ok", "Sem desvios — verificar trimestralmente."
    max_abs = max(abs(d) for d in deviations_pp)
    if max_abs < threshold_pp / 2:
        return "ok", "Desvios baixos — verificar trimestralmente."
    if max_abs < threshold_pp:
        return "ok", "Próximo do limiar — verificar mensalmente."
    if max_abs < threshold_pp * 2:
        return "warning", "Rebalanceamento recomendado nas próximas semanas."
    return "error", "Rebalanceamento urgente — desvio significativo."


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Otimizador de Portfólio",
    page_icon="📊",
    layout="wide",
)

# --- Session state ---
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = load_portfolio()
if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "editor_nonce" not in st.session_state:
    st.session_state.editor_nonce = 0
if "last_money" not in st.session_state:
    st.session_state.last_money = 1000.0

st.title("📊 Otimizador de Portfólio de Ativos")
st.caption(
    "Rebalanceamento inteligente sem vendas — apenas **compras corretivas**. "
    "Conversão automática para EUR."
)

# --- Sidebar ---
with st.sidebar:
    st.header("Parâmetros")
    money_to_invest = st.number_input(
        "Valor a Investir (€)",
        min_value=0.0,
        value=st.session_state.last_money,
        step=100.0,
        format="%.2f",
        help="Montante em euros disponível para compras corretivas.",
    )
    deviation_alert = st.slider(
        "Limiar de Desvio (pp)",
        min_value=1.0, max_value=20.0, value=5.0, step=0.5,
        help="Desvio absoluto entre alocação atual e alvo acima do qual há alerta.",
    )
    concentration_alert = st.slider(
        "Limiar de Concentração (%)",
        min_value=20.0, max_value=80.0, value=40.0, step=5.0,
        help="Avisa se algum ativo individual passar deste limiar do portfólio.",
    )

    st.divider()
    st.subheader("Persistência")
    if PORTFOLIO_FILE.exists():
        try:
            updated_at = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8")).get(
                "updated_at", "?"
            )
            st.caption(f"Último guardado: `{updated_at}`")
        except Exception:
            pass

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📥 Recarregar", use_container_width=True, help="Recarrega o JSON guardado"):
            st.session_state.portfolio_df = load_portfolio()
            st.session_state.result_df = None
            st.session_state.editor_nonce += 1
            st.rerun()
    with col_b:
        st.download_button(
            "📤 Exportar",
            df_to_export_json(st.session_state.portfolio_df),
            file_name="portfolio.json",
            mime="application/json",
            use_container_width=True,
        )

    uploaded = st.file_uploader("Importar JSON", type="json", label_visibility="collapsed")
    if uploaded:
        try:
            data = json.loads(uploaded.read().decode("utf-8"))
            rows = data.get("portfolio", [])
            new_df = pd.DataFrame(
                [
                    {
                        "Ticker": r["ticker"],
                        "Percentagem Alvo (%)": float(r["target_pct"]),
                        "Quantidade Detida": float(r["quantity"]),
                    }
                    for r in rows
                ]
            )
            if not new_df.empty:
                st.session_state.portfolio_df = new_df[EDITOR_COLS]
                st.session_state.result_df = None
                st.session_state.editor_nonce += 1
                st.success(f"✓ {len(new_df)} ativos importados.")
                st.rerun()
        except Exception as e:
            st.error(f"Erro a importar: {e}")

    st.divider()
    st.caption(
        f"v{__version__} · tickers no formato Yahoo Finance "
        "(`VWCE.DE`, `AAPL`, `BTC-EUR`, ...)"
    )

# --- 1. Portfólio ---
st.subheader("1. Defina o seu Portfólio")
st.markdown(
    "Edite os tickers, percentagens alvo e quantidades detidas. "
    "Pode adicionar ou remover linhas."
)

edited_df = st.data_editor(
    st.session_state.portfolio_df,
    num_rows="dynamic",
    use_container_width=True,
    key=f"portfolio_editor_{st.session_state.editor_nonce}",
    column_config={
        "Ticker": st.column_config.TextColumn(
            "Ticker", help="Símbolo Yahoo Finance", required=True
        ),
        "Percentagem Alvo (%)": st.column_config.NumberColumn(
            "Percentagem Alvo (%)",
            min_value=0.0, max_value=100.0, step=0.5, format="%.2f",
        ),
        "Quantidade Detida": st.column_config.NumberColumn(
            "Quantidade Detida",
            min_value=0.0, step=0.0001, format="%.4f",
        ),
    },
)

# Limpeza
edited_df = edited_df.dropna(subset=["Ticker"]).copy()
edited_df["Ticker"] = edited_df["Ticker"].astype(str).str.strip()
edited_df = edited_df[edited_df["Ticker"] != ""].reset_index(drop=True)

# Validação
total_target = float(edited_df["Percentagem Alvo (%)"].sum())
sum_is_valid = abs(total_target - 100.0) < 0.01

col_v1, col_v2 = st.columns([1, 3])
col_v1.metric("Soma Alvo (%)", f"{total_target:.2f}%")
if sum_is_valid:
    col_v2.success("✓ Soma das percentagens alvo correta (100%).")
else:
    col_v2.error(f"⚠️ A soma é {total_target:.2f}% — deve ser exatamente 100%.")

# --- 2. Calcular ---
st.subheader("2. Calcular Rebalanceamento")

calc = st.button(
    "🔄 Calcular distribuição ótima",
    type="primary",
    disabled=not sum_is_valid or edited_df.empty,
)

if calc:
    st.session_state.last_money = money_to_invest
    tickers = edited_df["Ticker"].tolist()

    with st.spinner("A obter preços e taxas de câmbio..."):
        price_data: dict[str, tuple[float | None, str | None]] = {
            t: fetch_price_and_currency(t) for t in tickers
        }
        currencies = {t: c for t, (_, c) in price_data.items()}
        unique_currencies = {c for c in currencies.values() if c}
        fx = {c: fetch_fx_to_eur(c) for c in unique_currencies}

    missing_price = [t for t, (p, _) in price_data.items() if p is None]
    missing_fx = [c for c, r in fx.items() if r is None]

    if missing_price:
        st.error(
            "Sem preço para: " + ", ".join(f"`{t}`" for t in missing_price)
            + ". Verifica a grafia do ticker (Yahoo Finance)."
        )
        st.stop()
    if missing_fx:
        st.error(
            "Sem taxa de câmbio para: " + ", ".join(f"`{c}`" for c in missing_fx)
            + "."
        )
        st.stop()

    df = edited_df.copy()
    df["Moeda"] = df["Ticker"].map(currencies)
    df["Preço Local"] = df["Ticker"].map({t: p for t, (p, _) in price_data.items()})
    df["FX→EUR"] = df["Moeda"].map(fx)
    df["Preço Atual (€)"] = df["Preço Local"] * df["FX→EUR"]
    df["Valor Atual (€)"] = df["Quantidade Detida"] * df["Preço Atual (€)"]

    buy_amounts = rebalance_no_sell(
        df["Valor Atual (€)"].tolist(),
        df["Percentagem Alvo (%)"].tolist(),
        money_to_invest,
    )

    df["Investir (€)"] = buy_amounts
    df["Qtd a Comprar"] = df["Investir (€)"] / df["Preço Atual (€)"]
    df["Valor Final (€)"] = df["Valor Atual (€)"] + df["Investir (€)"]

    total_current = df["Valor Atual (€)"].sum()
    total_final = df["Valor Final (€)"].sum()

    df["Alocação Atual (%)"] = (
        df["Valor Atual (€)"] / total_current * 100 if total_current > 0 else 0.0
    )
    df["Alocação Final (%)"] = (
        df["Valor Final (€)"] / total_final * 100 if total_final > 0 else 0.0
    )
    df["Desvio Atual (pp)"] = df["Alocação Atual (%)"] - df["Percentagem Alvo (%)"]
    df["Desvio Final (pp)"] = df["Alocação Final (%)"] - df["Percentagem Alvo (%)"]

    st.session_state.portfolio_df = edited_df[EDITOR_COLS].copy()
    st.session_state.result_df = df
    save_portfolio(edited_df)
    st.rerun()

# --- Mostrar resultados (se existirem) ---
if st.session_state.result_df is not None:
    df = st.session_state.result_df
    total_current = float(df["Valor Atual (€)"].sum())
    total_final = float(df["Valor Final (€)"].sum())
    invested = float(df["Investir (€)"].sum())

    # 3. Resumo
    st.subheader("3. Resumo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Valor Atual Total", f"€{total_current:,.2f}")
    m2.metric("Investimento", f"€{st.session_state.last_money:,.2f}")
    m3.metric("Valor Final Total", f"€{total_final:,.2f}")

    score_now = health_score(df["Desvio Atual (pp)"].tolist())
    score_after = health_score(df["Desvio Final (pp)"].tolist())
    m4.metric(
        "Saúde Pós-Investimento",
        f"{score_after}/100",
        delta=f"{score_after - score_now:+d} vs. atual ({score_now}/100)",
    )

    if abs(invested - st.session_state.last_money) > 0.01:
        st.caption(
            f"⚠️ Total distribuído: €{invested:,.2f} "
            f"(diferença vs. input: {invested - st.session_state.last_money:+.2f} €)"
        )

    # 4. Distribuição
    st.subheader("4. Distribuição Recomendada")
    display_cols = [
        "Ticker", "Moeda", "Preço Atual (€)", "Quantidade Detida", "Valor Atual (€)",
        "Alocação Atual (%)", "Percentagem Alvo (%)", "Desvio Atual (pp)",
        "Investir (€)", "Qtd a Comprar", "Valor Final (€)",
        "Alocação Final (%)", "Desvio Final (pp)",
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

    # 5. Alocação visual
    st.subheader("5. Alocação Visual")

    def _donut(values_col: str, title: str):
        fig = px.pie(
            df, values=values_col, names="Ticker", hole=0.55, title=title,
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center"),
            margin=dict(l=10, r=10, t=40, b=10),
            height=380,
            title_x=0.5,
        )
        return fig

    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(_donut("Alocação Atual (%)", "Atual"), use_container_width=True)
    c2.plotly_chart(_donut("Percentagem Alvo (%)", "Alvo"), use_container_width=True)
    c3.plotly_chart(_donut("Alocação Final (%)", "Final (Pós-Investimento)"), use_container_width=True)

    st.bar_chart(
        df.set_index("Ticker")[
            ["Alocação Atual (%)", "Percentagem Alvo (%)", "Alocação Final (%)"]
        ]
    )

    # 6. Alertas Inteligentes
    st.subheader("6. Alertas Inteligentes")

    level, msg = rebalance_urgency(df["Desvio Atual (pp)"].tolist(), deviation_alert)
    if level == "ok":
        st.success(f"🟢 **Urgência:** {msg}")
    elif level == "warning":
        st.warning(f"🟡 **Urgência:** {msg}")
    else:
        st.error(f"🔴 **Urgência:** {msg}")

    max_post = float(df["Desvio Final (pp)"].abs().max())
    if max_post < 0.5:
        st.success(
            f"✅ Após o investimento, todos os ativos ficam dentro de ±0.5 pp do alvo."
        )
    elif max_post < deviation_alert:
        st.success(
            f"✅ Desvio máximo pós-investimento: {max_post:.2f} pp "
            f"(dentro do limiar de {deviation_alert:.1f} pp)."
        )
    else:
        st.warning(
            f"⚠️ Desvio máximo pós-investimento: {max_post:.2f} pp — "
            f"considera aumentar o montante para um rebalanceamento mais completo."
        )

    # Concentração
    over_concentration = df[df["Alocação Final (%)"] > concentration_alert]
    if not over_concentration.empty:
        for _, row in over_concentration.iterrows():
            st.warning(
                f"🟠 **{row['Ticker']}** representa {row['Alocação Final (%)']:.1f}% do "
                f"portfólio final (limiar: {concentration_alert:.0f}%) — risco de concentração."
            )

    # Desvios individuais
    individual_alerts = []
    for _, row in df.iterrows():
        dev = row["Desvio Atual (pp)"]
        if abs(dev) >= deviation_alert:
            if dev > 0:
                individual_alerts.append(
                    (
                        "warning",
                        f"🔴 **{row['Ticker']}** sobre-alocado em {dev:+.2f} pp "
                        f"(atual {row['Alocação Atual (%)']:.2f}% vs. alvo "
                        f"{row['Percentagem Alvo (%)']:.2f}%) — só normaliza com novas entradas noutros ativos.",
                    )
                )
            else:
                individual_alerts.append(
                    (
                        "info",
                        f"🔵 **{row['Ticker']}** sub-alocado em {dev:+.2f} pp — recebe "
                        f"€{row['Investir (€)']:.2f} ({row['Qtd a Comprar']:.4f} unidades).",
                    )
                )

    if not individual_alerts:
        st.info("Nenhum ativo individual com desvio acima do limiar.")
    else:
        for lvl, m in individual_alerts:
            (st.warning if lvl == "warning" else st.info)(m)

    # 7. Aplicar Rebalanceamento
    st.subheader("7. Aplicar Rebalanceamento")
    st.markdown(
        "Quando executares as compras na tua corretora, clica aqui para somar "
        "as `Qtd a Comprar` às quantidades detidas (e guardar). "
        "Depois disso podes recalcular a qualquer momento."
    )

    if st.button("✅ Aplicar quantidades", type="secondary"):
        new_df = pd.DataFrame(
            {
                "Ticker": df["Ticker"].values,
                "Percentagem Alvo (%)": df["Percentagem Alvo (%)"].values,
                "Quantidade Detida": (
                    df["Quantidade Detida"] + df["Qtd a Comprar"]
                ).values,
            }
        )
        st.session_state.portfolio_df = new_df
        st.session_state.result_df = None
        st.session_state.editor_nonce += 1
        save_portfolio(new_df)
        st.success(
            "✓ Quantidades atualizadas e guardadas. Recalcule quando estiver pronto."
        )
        st.rerun()

else:
    st.info(
        "Defina o portfólio acima e clique em **Calcular distribuição ótima** "
        "para ver a recomendação."
    )

# --- Footer ---
st.divider()
st.caption(
    f"Otimizador de Portfólio · v{__version__} · "
    f"[GitHub](https://github.com/joaocruzfialho/otimizador-portfolio-ativos)"
)
