"""
Otimizador de Portfólio de Ativos com Alertas de Rebalanceamento Inteligente

v0.3.0 — Histórico de snapshots, Comparação de Cenários, Análise de Risco

App Streamlit que:
- Recebe um portfólio (Tickers, Percentagens Alvo, Quantidades Detidas)
- Obtém preços atuais via yfinance e converte para EUR
- Calcula a distribuição ideal de um valor a investir, SEM vender
- Persiste portfólio e histórico de snapshots
- Compara múltiplos cenários de investimento lado a lado
- Análise de risco: volatilidade, correlação e métricas de portfólio

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

__version__ = "0.3.0"

# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
HISTORY_FILE = DATA_DIR / "history.json"
MAX_SNAPSHOTS = 500
TRADING_DAYS = 252

DEFAULT_PORTFOLIO = pd.DataFrame(
    {
        "Ticker": ["VWCE.DE", "AGGH.MI", "SGLD.MI", "BTC-EUR"],
        "Percentagem Alvo (%)": [60.0, 25.0, 10.0, 5.0],
        "Quantidade Detida": [10.0, 50.0, 5.0, 0.05],
    }
)
EDITOR_COLS = ["Ticker", "Percentagem Alvo (%)", "Quantidade Detida"]

PERIOD_OPTIONS = {
    "1 mês": "1mo",
    "3 meses": "3mo",
    "6 meses": "6mo",
    "1 ano": "1y",
    "2 anos": "2y",
    "5 anos": "5y",
}


# ----------------------------------------------------------------------
# Persistência: portfólio
# ----------------------------------------------------------------------

def load_portfolio() -> pd.DataFrame:
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


# ----------------------------------------------------------------------
# Persistência: histórico de snapshots
# ----------------------------------------------------------------------

def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return data.get("snapshots", [])
        except Exception:
            pass
    return []


def write_history(snapshots: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(
        json.dumps(
            {"version": 1, "snapshots": snapshots[-MAX_SNAPSHOTS:]},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def append_snapshot(snap: dict) -> None:
    snapshots = load_history()
    snapshots.append(snap)
    write_history(snapshots)


def build_snapshot(
    result_df: pd.DataFrame,
    *,
    use_final: bool,
    money_invested: float = 0.0,
    tag: str = "manual",
) -> dict:
    """Constrói snapshot a partir de result_df (estado Atual ou Final pós-aplicar)."""
    val_col = "Valor Final (€)" if use_final else "Valor Atual (€)"
    alloc_col = "Alocação Final (%)" if use_final else "Alocação Atual (%)"

    assets = []
    for _, row in result_df.iterrows():
        qty = float(row["Quantidade Detida"])
        if use_final:
            qty += float(row["Qtd a Comprar"])
        assets.append({
            "ticker": str(row["Ticker"]),
            "qty": qty,
            "price_eur": float(row["Preço Atual (€)"]),
            "value_eur": float(row[val_col]),
            "target_pct": float(row["Percentagem Alvo (%)"]),
            "allocation_pct": float(row[alloc_col]),
        })
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "money_invested": float(money_invested),
        "total_value_eur": float(result_df[val_col].sum()),
        "assets": assets,
    }


def history_to_evolution_dfs(snapshots: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (df_total, df_alloc) prontos para gráficos."""
    if not snapshots:
        return pd.DataFrame(), pd.DataFrame()
    rows_total, rows_alloc = [], []
    for snap in snapshots:
        ts = snap["ts"]
        rows_total.append({"ts": ts, "Valor Total (€)": snap["total_value_eur"]})
        for a in snap["assets"]:
            rows_alloc.append({
                "ts": ts,
                "Ticker": a["ticker"],
                "Valor (€)": a["value_eur"],
                "Alocação (%)": a["allocation_pct"],
            })
    df_total = pd.DataFrame(rows_total)
    df_alloc = pd.DataFrame(rows_alloc)
    df_total["ts"] = pd.to_datetime(df_total["ts"])
    df_alloc["ts"] = pd.to_datetime(df_alloc["ts"])
    return df_total.sort_values("ts"), df_alloc.sort_values("ts")


# ----------------------------------------------------------------------
# Algoritmo de rebalanceamento (apenas compras)
# ----------------------------------------------------------------------

def rebalance_no_sell(
    current_values: list[float],
    target_weights: list[float],
    money_to_invest: float,
) -> np.ndarray:
    """
    Distribui `money_to_invest` pelos ativos para aproximar a alocação alvo,
    sem vender nada (X_i >= 0).
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
        ideal = np.where(active, target / active_target_sum * active_pool, current)
        over = active & (ideal < current - 1e-9)
        if not over.any():
            buy = np.where(active, ideal - current, 0.0)
            break
        fixed = fixed | over
    return np.maximum(buy, 0.0)


# ----------------------------------------------------------------------
# Preços, FX e históricos
# ----------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_and_currency(ticker: str) -> tuple[float | None, str | None]:
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


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_historical_closes(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    """Preços de fecho ajustados na moeda nativa, colunas=tickers."""
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            list(tickers),
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        try:
            closes = raw["Close"]
            if isinstance(closes, pd.Series):
                closes = closes.to_frame(name=tickers[0])
        except KeyError:
            return pd.DataFrame()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        closes = pd.DataFrame({tickers[0]: raw["Close"]})
    return closes.dropna(how="all")


# ----------------------------------------------------------------------
# Métricas: saúde, urgência, risco
# ----------------------------------------------------------------------

def health_score(deviations_pp: list[float]) -> int:
    if not deviations_pp:
        return 100
    max_abs = max(abs(d) for d in deviations_pp)
    return max(0, min(100, int(round(100 - (max_abs / 25.0) * 100))))


def rebalance_urgency(deviations_pp: list[float], threshold_pp: float) -> tuple[str, str]:
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


def compute_risk_metrics(
    closes: pd.DataFrame,
    target_pct: dict[str, float],
    rf_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Devolve (per_asset_df, correlation_matrix, portfolio_metrics).

    - per_asset_df: Ticker, Retorno Anual., Volatilidade Anual., Sharpe
    - correlation_matrix: DataFrame quadrada [tickers x tickers]
    - portfolio_metrics: dict {mean_ret_ann, vol_ann, sharpe, n_obs}
    """
    returns = closes.pct_change().dropna(how="all")
    if returns.empty or len(returns) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    mean_ret = returns.mean() * TRADING_DAYS
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    safe_vol = vol.replace(0, np.nan)
    sharpe = (mean_ret - rf_rate) / safe_vol
    corr = returns.corr()

    per_asset_df = pd.DataFrame({
        "Ticker": returns.columns.tolist(),
        "Retorno Anual.": mean_ret.values,
        "Volatilidade Anual.": vol.values,
        "Sharpe": sharpe.values,
    })

    tickers_in_data = list(returns.columns)
    weights = np.array([float(target_pct.get(t, 0.0)) for t in tickers_in_data])
    if weights.sum() <= 0:
        return per_asset_df, corr, {"n_obs": int(len(returns))}
    weights = weights / weights.sum()

    port_ret = float(np.dot(weights, mean_ret.values))
    cov_matrix = returns.cov().values * TRADING_DAYS
    port_var = float(weights @ cov_matrix @ weights)
    port_vol = float(np.sqrt(max(0.0, port_var)))
    port_sharpe = (port_ret - rf_rate) / port_vol if port_vol > 0 else float("nan")

    return per_asset_df, corr, {
        "mean_ret_ann": port_ret,
        "vol_ann": port_vol,
        "sharpe": port_sharpe,
        "n_obs": int(len(returns)),
    }


# ----------------------------------------------------------------------
# Helpers: fetch + rebalance
# ----------------------------------------------------------------------

def fetch_prices_and_fx(tickers: list[str]) -> tuple[dict, dict, dict, list[str], list[str]]:
    """Devolve (prices, currencies, fx, missing_price, missing_fx)."""
    price_data = {t: fetch_price_and_currency(t) for t in tickers}
    prices = {t: p for t, (p, _) in price_data.items()}
    currencies = {t: c for t, (_, c) in price_data.items()}
    unique_currencies = {c for c in currencies.values() if c}
    fx = {c: fetch_fx_to_eur(c) for c in unique_currencies}
    missing_price = [t for t, p in prices.items() if p is None]
    missing_fx = [c for c, r in fx.items() if r is None]
    return prices, currencies, fx, missing_price, missing_fx


def compute_rebalance_result(
    portfolio_df: pd.DataFrame,
    money_to_invest: float,
    prices: dict,
    currencies: dict,
    fx: dict,
) -> pd.DataFrame:
    df = portfolio_df.copy().reset_index(drop=True)
    df["Moeda"] = df["Ticker"].map(currencies)
    df["Preço Local"] = df["Ticker"].map(prices).astype(float)
    df["FX→EUR"] = df["Moeda"].map(fx).astype(float)
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
    return df


def parse_scenarios(text: str) -> list[float]:
    """'500, 1000, 2000' -> [500.0, 1000.0, 2000.0]; ignora inválidos."""
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
    # Preserva ordem mas remove duplicados
    seen = set()
    result = []
    for v in out:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


# ======================================================================
# Streamlit UI
# ======================================================================

st.set_page_config(
    page_title="Otimizador de Portfólio",
    page_icon="📊",
    layout="wide",
)

# --- Session state defaults ---
_defaults = {
    "portfolio_df": None,
    "result_df": None,
    "editor_nonce": 0,
    "last_money": 1000.0,
    "scenarios_input": "500, 1000, 2000, 5000",
    "scenarios_result": None,
    "risk_data": None,
    "risk_period_label": "1 ano",
    "rf_rate_pct": 2.0,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if st.session_state.portfolio_df is None:
    st.session_state.portfolio_df = load_portfolio()


# --- Sidebar ---
with st.sidebar:
    st.header("Parâmetros")
    money_to_invest = st.number_input(
        "Valor a Investir (€)",
        min_value=0.0,
        value=float(st.session_state.last_money),
        step=100.0,
        format="%.2f",
    )
    deviation_alert = st.slider(
        "Limiar de Desvio (pp)",
        min_value=1.0, max_value=20.0, value=5.0, step=0.5,
    )
    concentration_alert = st.slider(
        "Limiar de Concentração (%)",
        min_value=20.0, max_value=80.0, value=40.0, step=5.0,
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
        if st.button("📥 Recarregar", use_container_width=True):
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
    st.caption(f"v{__version__} · tickers em formato Yahoo Finance")


# --- Title ---
st.title("📊 Otimizador de Portfólio de Ativos")
st.caption(
    "Rebalanceamento sem vendas · cenários · histórico · análise de risco."
)


# --- Portfolio editor (sobre as tabs) ---
st.subheader("Defina o seu Portfólio")
st.caption(
    "Edita tickers, percentagens alvo e quantidades. Estes dados alimentam todas as tabs."
)

edited_df = st.data_editor(
    st.session_state.portfolio_df,
    num_rows="dynamic",
    use_container_width=True,
    key=f"portfolio_editor_{st.session_state.editor_nonce}",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", required=True),
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
edited_df = edited_df.dropna(subset=["Ticker"]).copy()
edited_df["Ticker"] = edited_df["Ticker"].astype(str).str.strip()
edited_df = edited_df[edited_df["Ticker"] != ""].reset_index(drop=True)

total_target = float(edited_df["Percentagem Alvo (%)"].sum())
sum_is_valid = abs(total_target - 100.0) < 0.01

col_v1, col_v2 = st.columns([1, 3])
col_v1.metric("Soma Alvo", f"{total_target:.2f}%")
if sum_is_valid:
    col_v2.success("✓ Soma das percentagens alvo correta (100%).")
else:
    col_v2.error(f"⚠️ A soma é {total_target:.2f}% — deve ser exatamente 100%.")


# --- Tabs ---
tab_reb, tab_cen, tab_his, tab_ris = st.tabs(
    ["💼 Rebalancear", "🎯 Cenários", "📈 Histórico", "⚖️ Risco"]
)


# ======================================================================
# Tab 1: Rebalancear
# ======================================================================
with tab_reb:
    st.markdown(f"### Calcular Rebalanceamento (€{money_to_invest:,.2f})")

    calc = st.button(
        "🔄 Calcular distribuição ótima",
        type="primary",
        disabled=not sum_is_valid or edited_df.empty,
        key="btn_calc_reb",
    )

    if calc:
        st.session_state.last_money = money_to_invest
        tickers = edited_df["Ticker"].tolist()
        with st.spinner("A obter preços e taxas de câmbio..."):
            prices, currencies, fx, missing_p, missing_fx = fetch_prices_and_fx(tickers)
        if missing_p:
            st.error("Sem preço para: " + ", ".join(f"`{t}`" for t in missing_p))
            st.stop()
        if missing_fx:
            st.error("Sem taxa de câmbio para: " + ", ".join(f"`{c}`" for c in missing_fx))
            st.stop()
        df_res = compute_rebalance_result(
            edited_df, money_to_invest, prices, currencies, fx
        )
        st.session_state.portfolio_df = edited_df[EDITOR_COLS].copy()
        st.session_state.result_df = df_res
        save_portfolio(edited_df)
        st.rerun()

    if st.session_state.result_df is not None:
        df = st.session_state.result_df
        total_current = float(df["Valor Atual (€)"].sum())
        total_final = float(df["Valor Final (€)"].sum())
        invested = float(df["Investir (€)"].sum())

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

        st.markdown("#### Distribuição Recomendada")
        display_cols = [
            "Ticker", "Moeda", "Preço Atual (€)", "Quantidade Detida", "Valor Atual (€)",
            "Alocação Atual (%)", "Percentagem Alvo (%)", "Desvio Atual (pp)",
            "Investir (€)", "Qtd a Comprar", "Valor Final (€)",
            "Alocação Final (%)", "Desvio Final (pp)",
        ]
        st.dataframe(
            df[display_cols].style.format({
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
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### Alocação Visual")

        def _donut(values_col: str, title: str):
            fig = px.pie(df, values=values_col, names="Ticker", hole=0.55, title=title)
            fig.update_traces(textposition="inside", textinfo="percent")
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center"),
                margin=dict(l=10, r=10, t=40, b=10),
                height=380, title_x=0.5,
            )
            return fig

        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(_donut("Alocação Atual (%)", "Atual"), use_container_width=True)
        c2.plotly_chart(_donut("Percentagem Alvo (%)", "Alvo"), use_container_width=True)
        c3.plotly_chart(_donut("Alocação Final (%)", "Final"), use_container_width=True)

        st.bar_chart(
            df.set_index("Ticker")[
                ["Alocação Atual (%)", "Percentagem Alvo (%)", "Alocação Final (%)"]
            ]
        )

        st.markdown("#### Alertas Inteligentes")
        level, msg = rebalance_urgency(df["Desvio Atual (pp)"].tolist(), deviation_alert)
        if level == "ok":
            st.success(f"🟢 **Urgência:** {msg}")
        elif level == "warning":
            st.warning(f"🟡 **Urgência:** {msg}")
        else:
            st.error(f"🔴 **Urgência:** {msg}")

        max_post = float(df["Desvio Final (pp)"].abs().max())
        if max_post < 0.5:
            st.success("✅ Após o investimento, todos os ativos ficam dentro de ±0.5 pp do alvo.")
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

        over_conc = df[df["Alocação Final (%)"] > concentration_alert]
        if not over_conc.empty:
            for _, row in over_conc.iterrows():
                st.warning(
                    f"🟠 **{row['Ticker']}** representa {row['Alocação Final (%)']:.1f}% do "
                    f"portfólio (limiar: {concentration_alert:.0f}%) — risco de concentração."
                )

        individual = []
        for _, row in df.iterrows():
            dev = row["Desvio Atual (pp)"]
            if abs(dev) >= deviation_alert:
                if dev > 0:
                    individual.append((
                        "warning",
                        f"🔴 **{row['Ticker']}** sobre-alocado em {dev:+.2f} pp — "
                        f"só normaliza com entradas noutros ativos.",
                    ))
                else:
                    individual.append((
                        "info",
                        f"🔵 **{row['Ticker']}** sub-alocado em {dev:+.2f} pp — "
                        f"recebe €{row['Investir (€)']:.2f} ({row['Qtd a Comprar']:.4f} unidades).",
                    ))
        if not individual:
            st.info("Nenhum ativo individual com desvio acima do limiar.")
        else:
            for lvl, m in individual:
                (st.warning if lvl == "warning" else st.info)(m)

        st.markdown("#### Aplicar Rebalanceamento")
        st.caption(
            "Quando executares as compras na corretora, clica para somar as `Qtd a Comprar` "
            "às quantidades detidas. Um snapshot é automaticamente gravado no histórico."
        )
        if st.button(
            "✅ Aplicar quantidades + guardar snapshot",
            type="secondary",
            key="btn_apply",
        ):
            snap = build_snapshot(
                df, use_final=True,
                money_invested=st.session_state.last_money,
                tag="apply",
            )
            append_snapshot(snap)
            new_df = pd.DataFrame({
                "Ticker": df["Ticker"].values,
                "Percentagem Alvo (%)": df["Percentagem Alvo (%)"].values,
                "Quantidade Detida": (df["Quantidade Detida"] + df["Qtd a Comprar"]).values,
            })
            st.session_state.portfolio_df = new_df
            st.session_state.result_df = None
            st.session_state.editor_nonce += 1
            save_portfolio(new_df)
            st.success(
                "✓ Quantidades atualizadas, portfólio guardado, snapshot no histórico."
            )
            st.rerun()
    else:
        st.info("Clica em **Calcular distribuição ótima** para gerar a recomendação.")


# ======================================================================
# Tab 2: Cenários
# ======================================================================
with tab_cen:
    st.markdown("### Comparar Cenários de Investimento")
    st.caption(
        "Indica vários valores em € separados por vírgula. Para cada um é calculado "
        "o rebalanceamento sem vendas usando o portfólio definido acima."
    )

    scenarios_input = st.text_input(
        "Valores a investir (€, separados por vírgula)",
        value=st.session_state.scenarios_input,
        key="scenarios_input_widget",
        help="Ex.: 500, 1000, 2000, 5000",
    )

    parsed = parse_scenarios(scenarios_input)
    if parsed:
        st.caption(
            f"✓ Cenários detetados: {', '.join(f'€{v:,.0f}' for v in parsed[:6])}"
            + (" (limitado a 6)" if len(parsed) > 6 else "")
        )

    compare = st.button(
        "🔍 Comparar cenários",
        type="primary",
        disabled=not sum_is_valid or edited_df.empty or not parsed,
        key="btn_compare",
    )

    if compare:
        st.session_state.scenarios_input = scenarios_input
        tickers = edited_df["Ticker"].tolist()
        with st.spinner("A obter preços e taxas de câmbio..."):
            prices, currencies, fx, missing_p, missing_fx = fetch_prices_and_fx(tickers)
        if missing_p:
            st.error("Sem preço para: " + ", ".join(f"`{t}`" for t in missing_p))
            st.stop()
        if missing_fx:
            st.error("Sem taxa de câmbio para: " + ", ".join(f"`{c}`" for c in missing_fx))
            st.stop()

        scenarios_results = {}
        for amount in parsed[:6]:
            scenarios_results[amount] = compute_rebalance_result(
                edited_df, amount, prices, currencies, fx
            )
        st.session_state.scenarios_result = scenarios_results
        st.session_state.portfolio_df = edited_df[EDITOR_COLS].copy()
        save_portfolio(edited_df)
        st.rerun()

    if st.session_state.scenarios_result:
        results = st.session_state.scenarios_result
        amounts = list(results.keys())

        st.markdown("#### Investimento por Ativo")
        first_df = results[amounts[0]]
        compare_df = first_df[["Ticker", "Percentagem Alvo (%)", "Alocação Atual (%)"]].copy()
        for amt in amounts:
            r = results[amt]
            compare_df[f"€{amt:,.0f}"] = r["Investir (€)"].values

        format_map: dict[str, str] = {
            "Percentagem Alvo (%)": "{:.2f}",
            "Alocação Atual (%)": "{:.2f}",
        }
        for amt in amounts:
            format_map[f"€{amt:,.0f}"] = "€{:.2f}"

        st.dataframe(
            compare_df.style.format(format_map),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### Resumo por Cenário")
        cols = st.columns(len(amounts))
        for col, amt in zip(cols, amounts):
            r = results[amt]
            with col:
                st.markdown(f"##### €{amt:,.0f}")
                score = health_score(r["Desvio Final (pp)"].tolist())
                max_dev = float(r["Desvio Final (pp)"].abs().max())
                total_final = float(r["Valor Final (€)"].sum())
                st.metric("Saúde Final", f"{score}/100")
                st.metric("Desvio Máx.", f"{max_dev:.2f} pp")
                st.metric("Valor Final", f"€{total_final:,.0f}")

        st.markdown("#### Score de Saúde vs. Montante Investido")
        score_df = pd.DataFrame({
            "Montante (€)": amounts,
            "Score": [health_score(results[a]["Desvio Final (pp)"].tolist()) for a in amounts],
            "Desvio Máx (pp)": [
                float(results[a]["Desvio Final (pp)"].abs().max()) for a in amounts
            ],
        })
        fig_score = px.line(
            score_df, x="Montante (€)", y="Score", markers=True,
            title="Score de Saúde Pós-Investimento por Cenário",
        )
        fig_score.update_layout(yaxis=dict(range=[0, 105]), height=320, title_x=0.5)
        st.plotly_chart(fig_score, use_container_width=True)

        st.caption(
            "À medida que aumentas o investimento, mais facilmente o portfólio "
            "atinge a alocação alvo sem precisar de vender."
        )
    else:
        st.info("Define valores e clica em **Comparar cenários** para ver a análise.")


# ======================================================================
# Tab 3: Histórico
# ======================================================================
with tab_his:
    st.markdown("### Histórico de Snapshots")
    snapshots = load_history()

    col_top_a, col_top_b = st.columns([3, 1])
    col_top_a.metric("Snapshots guardados", len(snapshots))
    with col_top_b:
        if snapshots and st.button("🗑️ Limpar histórico", type="secondary"):
            write_history([])
            st.rerun()

    if st.session_state.result_df is not None:
        if st.button("📸 Tirar snapshot do estado atual"):
            snap = build_snapshot(
                st.session_state.result_df, use_final=False, tag="manual"
            )
            append_snapshot(snap)
            st.success("✓ Snapshot guardado.")
            st.rerun()
    else:
        st.caption(
            "Para tirar um snapshot manual, primeiro calcula o rebalanceamento na "
            "tab **Rebalancear** (assim temos preços frescos)."
        )

    if not snapshots:
        st.info(
            "Ainda não há snapshots. Eles são automaticamente gravados quando clicas "
            "em **Aplicar quantidades** na tab Rebalancear."
        )
    else:
        df_total, df_alloc = history_to_evolution_dfs(snapshots)

        st.markdown("#### Evolução do Valor Total")
        fig_total = px.line(
            df_total, x="ts", y="Valor Total (€)", markers=True,
            title="Valor Total do Portfólio ao Longo do Tempo",
        )
        fig_total.update_layout(height=350, title_x=0.5)
        st.plotly_chart(fig_total, use_container_width=True)

        st.markdown("#### Evolução por Ativo (€)")
        fig_area = px.area(
            df_alloc, x="ts", y="Valor (€)", color="Ticker",
            title="Valor por Ativo (Stacked)",
        )
        fig_area.update_layout(height=400, title_x=0.5)
        st.plotly_chart(fig_area, use_container_width=True)

        st.markdown("#### Evolução da Alocação (%)")
        fig_alloc = px.line(
            df_alloc, x="ts", y="Alocação (%)", color="Ticker", markers=True,
            title="Percentagem Alocada por Ativo",
        )
        fig_alloc.update_layout(height=350, title_x=0.5)
        st.plotly_chart(fig_alloc, use_container_width=True)

        st.markdown("#### Lista de Snapshots")
        list_df = pd.DataFrame([
            {
                "Data": pd.to_datetime(s["ts"]),
                "Tipo": s.get("tag", "?"),
                "Valor Total (€)": s["total_value_eur"],
                "Investido (€)": s.get("money_invested", 0.0),
                "# Ativos": len(s.get("assets", [])),
            }
            for s in snapshots
        ]).sort_values("Data", ascending=False)
        st.dataframe(
            list_df.style.format({
                "Valor Total (€)": "€{:,.2f}",
                "Investido (€)": "€{:,.2f}",
            }),
            use_container_width=True, hide_index=True,
        )


# ======================================================================
# Tab 4: Risco
# ======================================================================
with tab_ris:
    st.markdown("### Análise de Risco do Portfólio")
    st.caption(
        "Volatilidade anualizada, correlações entre ativos e métricas a nível do portfólio. "
        "Cálculos em **moeda nativa** — para ativos não-EUR, há risco cambial adicional não capturado."
    )

    col_r1, col_r2, col_r3 = st.columns([2, 2, 1])
    period_label = col_r1.selectbox(
        "Período",
        list(PERIOD_OPTIONS.keys()),
        index=list(PERIOD_OPTIONS.keys()).index(st.session_state.risk_period_label),
    )
    rf_pct = col_r2.number_input(
        "Taxa sem risco (% ao ano)",
        min_value=0.0, max_value=15.0,
        value=float(st.session_state.rf_rate_pct),
        step=0.25,
        help="Usada no rácio de Sharpe.",
    )
    analyse = col_r3.button(
        "📊 Analisar",
        type="primary",
        disabled=edited_df.empty,
    )

    if analyse:
        st.session_state.risk_period_label = period_label
        st.session_state.rf_rate_pct = rf_pct
        tickers = tuple(edited_df["Ticker"].tolist())
        period = PERIOD_OPTIONS[period_label]
        with st.spinner(f"A obter preços históricos ({period_label})..."):
            closes = fetch_historical_closes(tickers, period)
        if closes.empty:
            st.error("Sem dados históricos para os tickers indicados.")
            st.stop()

        target_pct = dict(zip(edited_df["Ticker"], edited_df["Percentagem Alvo (%)"]))
        per_asset, corr, port = compute_risk_metrics(closes, target_pct, rf_pct / 100.0)
        st.session_state.risk_data = {
            "per_asset": per_asset,
            "corr": corr,
            "portfolio": port,
            "period_label": period_label,
            "rf_pct": rf_pct,
            "missing": [t for t in tickers if t not in closes.columns],
        }
        st.rerun()

    if st.session_state.risk_data:
        rd = st.session_state.risk_data
        per_asset = rd["per_asset"]
        corr = rd["corr"]
        port = rd["portfolio"]

        if rd.get("missing"):
            st.warning(
                "Sem histórico para: " + ", ".join(f"`{t}`" for t in rd["missing"])
                + " — excluídos da análise."
            )

        st.markdown("#### Métricas a Nível do Portfólio")
        st.caption(
            f"Pesos = alvo · período = {rd['period_label']} · "
            f"taxa sem risco = {rd['rf_pct']:.2f}% · "
            f"{port.get('n_obs', 0)} observações diárias."
        )
        if port and "mean_ret_ann" in port:
            pm1, pm2, pm3 = st.columns(3)
            pm1.metric("Retorno Anualizado", f"{port['mean_ret_ann']*100:+.2f}%")
            pm2.metric("Volatilidade Anualizada", f"{port['vol_ann']*100:.2f}%")
            sharpe = port["sharpe"]
            pm3.metric("Sharpe", f"{sharpe:.2f}" if np.isfinite(sharpe) else "—")

        st.markdown("#### Por Ativo")
        st.dataframe(
            per_asset.style.format({
                "Retorno Anual.": "{:+.2%}",
                "Volatilidade Anual.": "{:.2%}",
                "Sharpe": "{:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### Volatilidade Anualizada por Ativo")
        vol_df = per_asset[["Ticker", "Volatilidade Anual."]].copy()
        vol_df["Volatilidade Anual."] = vol_df["Volatilidade Anual."] * 100
        fig_vol = px.bar(
            vol_df, x="Ticker", y="Volatilidade Anual.",
            text=vol_df["Volatilidade Anual."].apply(lambda v: f"{v:.1f}%"),
            title="Volatilidade Anualizada (%)",
        )
        fig_vol.update_layout(height=350, title_x=0.5, yaxis_title="% ao ano")
        fig_vol.update_traces(textposition="outside")
        st.plotly_chart(fig_vol, use_container_width=True)

        if not corr.empty and len(corr) > 1:
            st.markdown("#### Matriz de Correlação")
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                origin="lower",
                zmin=-1, zmax=1,
                title="Correlação entre Retornos Diários",
            )
            fig_corr.update_layout(
                height=max(320, 70 * len(corr)),
                title_x=0.5,
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption(
                "Correlações próximas de 0 indicam boa diversificação; valores próximos de 1 "
                "indicam que os ativos se movem em conjunto (diversificação fraca)."
            )
        elif len(corr) == 1:
            st.info("Necessários pelo menos 2 ativos para calcular correlações.")
    else:
        st.info("Clica em **Analisar** para calcular volatilidade, retornos e correlações.")


# --- Footer ---
st.divider()
st.caption(
    f"Otimizador de Portfólio · v{__version__} · "
    f"[GitHub](https://github.com/joaocruzfialho/otimizador-portfolio-ativos)"
)
