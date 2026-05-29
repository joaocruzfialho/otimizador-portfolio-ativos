"""
Otimizador de Portfólio de Ativos — v0.9.0

Tabs: Rebalancear · Cenários · Histórico · Risco · Otimizar · Backtest
      Dividendos · Fiscalidade · FIRE · Gerir
Multi-utilizador com autenticação Supabase e planos Free/Pro.

Autor: João Fialho
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.alerts import SmtpConfig, build_alert_html, send_alert_email
from core.auth import FREE_PLAN_LIMIT, get_email, get_plan, get_user_id, is_pro, logout, show_auth_page
from core.backtest import MODEL_PORTFOLIOS, compare_model_portfolios, run_backtest
from core.db import (
    ASSET_TYPES, delete_asset, get_asset, get_tickers,
    load_portfolio_full, upsert_asset,
)
from core.dividends import portfolio_dividend_report
from core.fire import run_fire_simulation
from core.history import (
    append_snapshot, build_snapshot, df_to_export_json,
    history_to_evolution_dfs, load_history, load_portfolio,
    save_portfolio, write_history, PORTFOLIO_FILE, EDITOR_COLS,
)
from core.prices import fetch_fx_to_eur, fetch_historical_closes, fetch_price_and_currency, fetch_ter
from core.rebalance import (
    compute_rebalance_result, health_score, parse_scenarios,
    rebalance_urgency,
)
from core.risk import compute_drawdown, compute_risk_metrics, markowitz_analysis
from core.tax import compute_unrealized_gains

__version__ = "0.9.0"


@st.cache_data(ttl=None, show_spinner=False)
def _cached_full(user_id: str) -> pd.DataFrame:
    """Per-user cache of load_portfolio_full(). Cleared only on writes."""
    return load_portfolio_full(user_id)


@st.cache_data(ttl=None, show_spinner=False)
def _cached_history(user_id: str) -> list[dict]:
    """Per-user cache of load_history(). Cleared only on writes."""
    return load_history(user_id)


@st.cache_data(ttl=None, show_spinner=False)
def _cached_history(user_id: str) -> list[dict]:
    """Per-user cache of load_history(). Cleared only on writes."""
    return load_history(user_id)


PERIOD_OPTIONS = {
    "1 mês": "1mo", "3 meses": "3mo", "6 meses": "6mo",
    "1 ano": "1y", "2 anos": "2y", "3 anos": "3y", "5 anos": "5y",
}
FREQ_OPTIONS = {
    "Mensal": "M", "Trimestral": "Q", "Anual": "A", "Nunca (Buy & Hold)": "never",
}

_LIGHT_CSS = """
<style>
/* === TEMA CLARO === */

/* Fundo principal */
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main                { background-color: #f5f7fa !important; }
section[data-testid="stSidebar"] + section   { background-color: #f5f7fa !important; }

/* Header/toolbar do Streamlit */
header, header[data-testid="stHeader"],
[data-testid="stHeader"],
[data-testid="stToolbar"]                    { background: #ffffff !important;
                                               border-bottom: 1px solid #dde1eb !important; }
[data-testid="stDecoration"]                 { background: #1f77b4 !important; }

/* Sidebar */
[data-testid="stSidebar"]                    { background-color: #eaecf4 !important; }
[data-testid="stSidebarContent"],
[data-testid="stSidebar"] *                  { color: #2c3e50 !important; }

/* Texto geral */
.stMarkdown *, p, h1, h2, h3, h4, h5, h6,
.stText, [data-testid="stText"],
[data-testid="stCaptionContainer"] *         { color: #2c3e50 !important; }
small, .stCaption, [class*="caption"]        { color: #5e6b7a !important; }
code                                         { color: #c7254e !important;
                                               background: #f0f0f0 !important; }

/* Métricas */
[data-testid="stMetricValue"]               { color: #1a2535 !important; }
[data-testid="stMetricLabel"]               { color: #5e6b7a !important; }

/* Botões */
.stButton > button,
.stDownloadButton > button                  { background-color: #1f77b4 !important;
                                              color: #ffffff !important;
                                              border: 1px solid #1a66a0 !important; }
.stButton > button[kind="secondary"]        { background-color: #e8ecf4 !important;
                                              color: #2c3e50 !important;
                                              border: 1px solid #c0c8d8 !important; }
[data-testid="stFileUploader"] button       { background-color: #e8ecf4 !important;
                                              color: #2c3e50 !important; }

/* Inputs BaseWeb */
[data-baseweb="input"],
[data-baseweb="base-input"],
[data-baseweb="input"] > div               { background-color: #ffffff !important; }
[data-baseweb="input"] input,
[data-baseweb="base-input"] input,
input                                       { background-color: #ffffff !important;
                                              color: #2c3e50 !important; }
[data-testid="stNumberInput"] button        { background-color: #e8ecf4 !important;
                                              color: #2c3e50 !important; }

/* Selectbox */
[data-baseweb="select"] > div,
[data-baseweb="select"] [data-baseweb="input"]  { background-color: #ffffff !important;
                                                  color: #2c3e50 !important; }
[data-baseweb="popover"],
[data-baseweb="menu"]                           { background-color: #ffffff !important; }
[role="option"]                                 { background-color: #ffffff !important;
                                                  color: #2c3e50 !important; }
[role="option"]:hover                           { background-color: #e8ecf4 !important; }

/* Labels de widgets */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *             { color: #2c3e50 !important; }

/* File uploader */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] > div > div  { background-color: #eaecf4 !important; }
[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"] {
                                              background-color: #eaecf4 !important;
                                              border: 2px dashed #b0b8cc !important; }
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] *     { color: #2c3e50 !important; }
[data-testid="stFileUploaderDropzone"] button {
                                              background-color: #ffffff !important;
                                              color: #1f77b4 !important;
                                              border: 1px solid #1f77b4 !important; }

/* Tabs */
[data-baseweb="tab-list"]                   { background: transparent !important;
                                              border-bottom: 2px solid #dde1eb !important; }
[data-baseweb="tab"]                        { color: #5e6b7a !important;
                                              background: transparent !important; }
[aria-selected="true"][data-baseweb="tab"]  { color: #1f77b4 !important; }

/* Alertas */
[data-testid="stAlert"] > div              { color: #2c3e50 !important; }

/* Expanders */
[data-testid="stExpander"],
[data-testid="stExpander"] > details       { background-color: #eaecf4 !important; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *       { color: #2c3e50 !important; }

/* Dataframe / DataEditor: wrapper branco para minimizar contraste */
[data-testid="stDataFrame"],
[data-testid="stDataEditor"]               { background-color: #ffffff !important;
                                             border-radius: 6px !important;
                                             overflow: hidden !important; }

/* Slider */
[data-testid="stSlider"] *                 { color: #2c3e50 !important; }

/* Radio / Checkbox */
[data-baseweb="radio"] *,
[data-baseweb="checkbox"] *                { color: #2c3e50 !important; }

/* Dividers */
hr                                         { border-color: #c8d0de !important; }

/* Plotly charts background */
.js-plotly-plot .plotly .bg               { fill: #f5f7fa !important; }
</style>
"""


# ──────────────────────────────────────────────
# Helpers shared across tabs
# ──────────────────────────────────────────────

def fetch_prices_and_fx(
    tickers: list[str],
) -> tuple[dict, dict, dict, list[str], list[str]]:
    price_data = {t: fetch_price_and_currency(t) for t in tickers}
    prices = {t: p for t, (p, _) in price_data.items()}
    currencies = {t: c for t, (_, c) in price_data.items()}
    unique_cur = {c for c in currencies.values() if c}
    fx = {c: fetch_fx_to_eur(c) for c in unique_cur}
    missing_p = [t for t, p in prices.items() if p is None]
    missing_fx = [c for c, r in fx.items() if r is None]
    return prices, currencies, fx, missing_p, missing_fx


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────

st.set_page_config(page_title="Otimizador de Portfólio", page_icon="📊", layout="wide")

# Inject light theme CSS if needed (dark is the default in config.toml)
if st.session_state.get("theme") == "Claro":
    st.markdown(_LIGHT_CSS, unsafe_allow_html=True)

# ── Auth gate ────────────────────────────────────────────────────────────
if not show_auth_page():
    st.stop()

user_id: str = get_user_id()  # type: ignore[assignment]

_defaults: dict = {
    "portfolio_df": None,
    "theme": "Escuro",
    "result_df": None,
    "editor_nonce": 0,
    "last_money": 1000.0,
    "scenarios_input": "500, 1000, 2000, 5000",
    "scenarios_result": None,
    "risk_data": None,
    "risk_period": "1 ano",
    "rf_rate_pct": 2.0,
    "markowitz_result": None,
    "opt_period": "2 anos",
    "opt_wmin_pct": 0.0,
    "opt_wmax_pct": 100.0,
    "backtest_result": None,
    "bkt_period": "3 anos",
    "bkt_capital": 10000.0,
    "bkt_freq": "Trimestral",
    "model_cmp_result": None,
    "div_data": None,
    "tax_data": None,
    "fire_result": None,
    "fire_portfolio_value": 50000.0,
    "fire_spending": 20000.0,
    "fire_years": 30,
    "fire_ret_mean": 7.0,
    "fire_ret_std": 15.0,
    "fire_inflation": 2.5,
    "del_pending": None,
    # email (never persisted to disk)
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_user": "",
    "smtp_pass": "",
    "smtp_to": "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if st.session_state.portfolio_df is None:
    _full = _cached_full(user_id)
    if not _full.empty:
        st.session_state.portfolio_df = _full[EDITOR_COLS].reset_index(drop=True)
    else:
        st.session_state.portfolio_df = load_portfolio(user_id)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    # User info + logout
    _plan = get_plan()
    _badge = "⭐ Pro" if _plan == "pro" else "Gratuito"
    _email = get_email() or ""
    st.caption(f"**{_email}** · {_badge}")
    if st.button("Sair", use_container_width=True, key="btn_logout"):
        logout()

    st.divider()

    _theme_choice = st.radio(
        "Tema", ["Escuro", "Claro"], horizontal=True,
        index=0 if st.session_state.theme == "Escuro" else 1,
        key="theme_radio",
    )
    if _theme_choice != st.session_state.theme:
        st.session_state.theme = _theme_choice
        st.rerun()

    st.header("Parâmetros")
    money_to_invest = st.number_input(
        "Valor a Investir (€)", min_value=0.0,
        value=float(st.session_state.last_money), step=100.0, format="%.2f",
    )
    deviation_alert = st.slider("Limiar de Desvio (pp)", 1.0, 20.0, 5.0, 0.5)
    concentration_alert = st.slider("Limiar de Concentração (%)", 20.0, 80.0, 40.0, 5.0)

    st.divider()
    st.subheader("Persistência")
    if PORTFOLIO_FILE.exists():
        try:
            updated_at = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8")).get("updated_at", "?")
            st.caption(f"Último guardado: `{updated_at}`")
        except Exception:
            pass
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📥 Recarregar", use_container_width=True):
            st.session_state.portfolio_df = load_portfolio(user_id)
            st.session_state.result_df = None
            st.session_state.editor_nonce += 1
            st.rerun()
    with col_b:
        st.download_button(
            "📤 Exportar", df_to_export_json(st.session_state.portfolio_df),
            file_name="portfolio.json", mime="application/json", use_container_width=True,
        )
    uploaded = st.file_uploader("Importar JSON", type="json", label_visibility="collapsed")
    if uploaded:
        try:
            rows = json.loads(uploaded.read().decode("utf-8")).get("portfolio", [])
            new_df = pd.DataFrame([{
                "Ticker": r["ticker"],
                "Percentagem Alvo (%)": float(r["target_pct"]),
                "Quantidade Detida": float(r["quantity"]),
            } for r in rows])
            if not new_df.empty:
                st.session_state.portfolio_df = new_df[EDITOR_COLS]
                st.session_state.result_df = None
                st.session_state.editor_nonce += 1
                st.success(f"✓ {len(new_df)} ativos importados.")
                st.rerun()
        except Exception as e:
            st.error(f"Erro ao importar: {e}")

    # Email alerts
    st.divider()
    with st.expander("📧 Alertas por Email"):
        st.caption("Config. guardada apenas na sessão atual (nunca escrita em disco).")
        st.session_state.smtp_host = st.text_input("SMTP Host", value=st.session_state.smtp_host)
        st.session_state.smtp_port = int(st.number_input(
            "Porta", min_value=1, max_value=65535,
            value=int(st.session_state.smtp_port), step=1,
        ))
        st.session_state.smtp_user = st.text_input("Utilizador SMTP", value=st.session_state.smtp_user)
        st.session_state.smtp_pass = st.text_input(
            "Password / App Password", value=st.session_state.smtp_pass, type="password",
        )
        st.session_state.smtp_to = st.text_input("Enviar para (email)", value=st.session_state.smtp_to)
        st.caption("Gmail: usa uma *App Password* (not the account password).")

        if st.button("📨 Enviar alerta agora", use_container_width=True):
            df_r = st.session_state.result_df
            if df_r is None:
                st.warning("Calcula o rebalanceamento primeiro (tab Rebalancear).")
            elif not st.session_state.smtp_to:
                st.warning("Preenche o email de destino.")
            else:
                deviating = df_r[df_r["Desvio Atual (pp)"].abs() >= deviation_alert]
                if deviating.empty:
                    st.info("Nenhum desvio acima do limiar — alerta não enviado.")
                else:
                    cfg = SmtpConfig(
                        host=st.session_state.smtp_host,
                        port=st.session_state.smtp_port,
                        user=st.session_state.smtp_user,
                        password=st.session_state.smtp_pass,
                        from_addr=st.session_state.smtp_user,
                        to_addr=st.session_state.smtp_to,
                    )
                    total = float(df_r["Valor Atual (€)"].sum())
                    html = build_alert_html(df_r, deviation_alert, total)
                    ok, err = send_alert_email(cfg, "⚠️ Alerta de Rebalanceamento", html)
                    if ok:
                        st.success("✓ Email enviado.")
                    else:
                        st.error(f"Erro: {err}")

    st.divider()
    st.caption(f"v{__version__} · tickers em formato Yahoo Finance")


# ──────────────────────────────────────────────
# Title + portfolio editor (above all tabs)
# ──────────────────────────────────────────────

st.title("📊 Otimizador de Portfólio de Ativos")
st.caption("Rebalanceamento sem vendas · cenários · histórico · risco · Markowitz · backtest.")

st.subheader("Portfólio")
edited_df = st.data_editor(
    st.session_state.portfolio_df, num_rows="dynamic", use_container_width=True,
    key=f"portfolio_editor_{st.session_state.editor_nonce}",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", required=True),
        "Percentagem Alvo (%)": st.column_config.NumberColumn(
            "Percentagem Alvo (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.2f"),
        "Quantidade Detida": st.column_config.NumberColumn(
            "Quantidade Detida", min_value=0.0, step=0.0001, format="%.4f"),
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


# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────

tab_reb, tab_cen, tab_his, tab_ris, tab_opt, tab_bkt, tab_div, tab_tax, tab_fire, tab_mgr = st.tabs([
    "💼 Rebalancear", "🎯 Cenários", "📈 Histórico",
    "⚖️ Risco", "🔬 Otimizar", "🔄 Backtest",
    "💰 Dividendos", "🧾 Fiscalidade", "🔥 FIRE", "⚙️ Gerir",
])


# ══════════════════════════════════════════════
# Tab 1: Rebalancear
# ══════════════════════════════════════════════

with tab_reb:
    st.markdown(f"### Calcular Rebalanceamento (€{money_to_invest:,.2f})")

    if st.button("🔄 Calcular distribuição ótima", type="primary",
                 disabled=not sum_is_valid or edited_df.empty, key="btn_reb"):
        st.session_state.last_money = money_to_invest
        with st.spinner("A obter preços e taxas de câmbio..."):
            prices, currencies, fx, missing_p, missing_fx = fetch_prices_and_fx(edited_df["Ticker"].tolist())
        if missing_p:
            st.error("Sem preço para: " + ", ".join(f"`{t}`" for t in missing_p)); st.stop()
        if missing_fx:
            st.error("Sem câmbio para: " + ", ".join(f"`{c}`" for c in missing_fx)); st.stop()
        st.session_state.result_df = compute_rebalance_result(
            edited_df, money_to_invest, prices, currencies, fx)
        st.session_state.portfolio_df = edited_df[EDITOR_COLS].copy()
        save_portfolio(edited_df, user_id)
        _cached_full.clear()
        st.rerun()

    df = st.session_state.result_df
    if df is not None:
        total_cur = float(df["Valor Atual (€)"].sum())
        total_fin = float(df["Valor Final (€)"].sum())
        invested = float(df["Investir (€)"].sum())
        score_now = health_score(df["Desvio Atual (pp)"].tolist())
        score_after = health_score(df["Desvio Final (pp)"].tolist())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Valor Atual", f"€{total_cur:,.2f}")
        m2.metric("Investimento", f"€{st.session_state.last_money:,.2f}")
        m3.metric("Valor Final", f"€{total_fin:,.2f}")
        m4.metric("Saúde Final", f"{score_after}/100",
                  delta=f"{score_after - score_now:+d} vs. atual ({score_now}/100)")

        if abs(invested - st.session_state.last_money) > 0.01:
            st.caption(f"⚠️ Total distribuído: €{invested:,.2f} "
                       f"(diferença: {invested - st.session_state.last_money:+.2f} €)")

        st.markdown("#### Distribuição Recomendada")
        display_cols = [
            "Ticker", "Moeda", "Preço Atual (€)", "Quantidade Detida", "Valor Atual (€)",
            "Alocação Atual (%)", "Percentagem Alvo (%)", "Desvio Atual (pp)",
            "Investir (€)", "Qtd a Comprar", "Valor Final (€)",
            "Alocação Final (%)", "Desvio Final (pp)",
        ]
        st.dataframe(df[display_cols].style.format({
            "Preço Atual (€)": "{:.2f}", "Quantidade Detida": "{:.4f}",
            "Valor Atual (€)": "{:.2f}", "Alocação Atual (%)": "{:.2f}",
            "Percentagem Alvo (%)": "{:.2f}", "Desvio Atual (pp)": "{:+.2f}",
            "Investir (€)": "{:.2f}", "Qtd a Comprar": "{:.4f}",
            "Valor Final (€)": "{:.2f}", "Alocação Final (%)": "{:.2f}",
            "Desvio Final (pp)": "{:+.2f}",
        }), use_container_width=True, hide_index=True)

        st.markdown("#### Alocação Visual")

        def _donut(col: str, title: str):
            fig = px.pie(df, values=col, names="Ticker", hole=0.55, title=title)
            fig.update_traces(textposition="inside", textinfo="percent")
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center"),
                margin=dict(l=10, r=10, t=40, b=10), height=380, title_x=0.5,
            )
            return fig

        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(_donut("Alocação Atual (%)", "Atual"), use_container_width=True)
        c2.plotly_chart(_donut("Percentagem Alvo (%)", "Alvo"), use_container_width=True)
        c3.plotly_chart(_donut("Alocação Final (%)", "Final"), use_container_width=True)

        st.markdown("#### Atual vs. Alvo — Comparação por Ativo")
        st.caption("Cor das barras Atual: 🟢 desvio < 3 pp · 🟠 desvio 3–6 pp · 🔴 desvio > 6 pp")

        _ALERT_PP = 3.0

        def _dev_color(dev: float) -> str:
            a = abs(dev)
            if a < _ALERT_PP:
                return "#27ae60"
            if a < _ALERT_PP * 2:
                return "#e67e22"
            return "#e74c3c"

        _colors_atual = [_dev_color(float(r["Desvio Atual (pp)"])) for _, r in df.iterrows()]
        _x_max = max(
            float(df["Percentagem Alvo (%)"].max()),
            float(df["Alocação Atual (%)"].max()),
        ) * 1.3

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            y=df["Ticker"], x=df["Alocação Atual (%)"],
            name="Atual", orientation="h",
            marker_color=_colors_atual,
            customdata=df["Desvio Atual (pp)"],
            text=[f"{v:.1f}%" for v in df["Alocação Atual (%)"]],
            textposition="outside",
            hovertemplate="%{y}<br>Atual: %{x:.2f}%<br>Desvio: %{customdata:+.2f} pp<extra></extra>",
        ))
        fig_cmp.add_trace(go.Bar(
            y=df["Ticker"], x=df["Percentagem Alvo (%)"],
            name="Alvo", orientation="h",
            marker_color="rgba(100,149,237,0.6)",
            text=[f"{v:.1f}%" for v in df["Percentagem Alvo (%)"]],
            textposition="outside",
            hovertemplate="%{y}<br>Alvo: %{x:.2f}%<extra></extra>",
        ))
        fig_cmp.update_layout(
            barmode="group",
            title="Alocação Atual vs. Percentagem Alvo por Ativo",
            title_x=0.5,
            xaxis=dict(title="Percentagem (%)", range=[0, _x_max]),
            yaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=max(300, 80 * len(df)),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        _need_attn = df[df["Desvio Atual (pp)"].abs() > _ALERT_PP]
        if not _need_attn.empty:
            st.markdown(f"**⚠️ Ativos com desvio > {_ALERT_PP:.0f} pp:**")
            for _, _row in _need_attn.iterrows():
                _dev = float(_row["Desvio Atual (pp)"])
                _icon = "🔴" if _dev > 0 else "🔵"
                _word = "sobre-alocado" if _dev > 0 else "sub-alocado"
                st.warning(
                    f"{_icon} **{_row['Ticker']}** {_word} em **{_dev:+.2f} pp** "
                    f"· atual {_row['Alocação Atual (%)']:.1f}% · alvo {_row['Percentagem Alvo (%)']:.1f}%"
                )
        else:
            st.success(f"✅ Todos os ativos dentro do limiar de ±{_ALERT_PP:.0f} pp.")

        st.markdown("#### Alertas")
        level, msg = rebalance_urgency(df["Desvio Atual (pp)"].tolist(), deviation_alert)
        {"ok": st.success, "warning": st.warning, "error": st.error}[level](
            f"{'🟢' if level=='ok' else '🟡' if level=='warning' else '🔴'} **Urgência:** {msg}")

        max_post = float(df["Desvio Final (pp)"].abs().max())
        if max_post < 0.5:
            st.success("✅ Após o investimento, todos os ativos ficam dentro de ±0.5 pp do alvo.")
        elif max_post < deviation_alert:
            st.success(f"✅ Desvio máximo pós-investimento: {max_post:.2f} pp (< limiar {deviation_alert:.1f} pp).")
        else:
            st.warning(f"⚠️ Desvio máximo pós-investimento: {max_post:.2f} pp — considera aumentar o montante.")

        for _, row in df[df["Alocação Final (%)"] > concentration_alert].iterrows():
            st.warning(f"🟠 **{row['Ticker']}** representa {row['Alocação Final (%)']:.1f}% "
                       f"(limiar: {concentration_alert:.0f}%) — risco de concentração.")

        for _, row in df.iterrows():
            dev = row["Desvio Atual (pp)"]
            if abs(dev) >= deviation_alert:
                if dev > 0:
                    st.warning(f"🔴 **{row['Ticker']}** sobre-alocado em {dev:+.2f} pp — "
                                f"só normaliza com entradas noutros ativos.")
                else:
                    st.info(f"🔵 **{row['Ticker']}** sub-alocado em {dev:+.2f} pp — "
                             f"recebe €{row['Investir (€)']:.2f} ({row['Qtd a Comprar']:.4f} unid.).")

        st.markdown("#### Aplicar Rebalanceamento")
        st.caption("Clica depois de executares as compras na corretora. "
                   "Grava snapshot no histórico automaticamente.")
        if st.button("✅ Aplicar quantidades + guardar snapshot",
                     type="secondary", key="btn_apply"):
            append_snapshot(build_snapshot(df, use_final=True,
                                           money_invested=st.session_state.last_money,
                                           tag="apply"), user_id)
            new_df = pd.DataFrame({
                "Ticker": df["Ticker"].values,
                "Percentagem Alvo (%)": df["Percentagem Alvo (%)"].values,
                "Quantidade Detida": (df["Quantidade Detida"] + df["Qtd a Comprar"]).values,
            })
            st.session_state.portfolio_df = new_df
            st.session_state.result_df = None
            st.session_state.editor_nonce += 1
            save_portfolio(new_df, user_id)
            _cached_full.clear()
            _cached_history.clear()
            st.success("✓ Quantidades atualizadas, portfólio guardado, snapshot no histórico.")
            st.rerun()
    else:
        st.info("Clica em **Calcular distribuição ótima** para gerar a recomendação.")


# ══════════════════════════════════════════════
# Tab 2: Cenários
# ══════════════════════════════════════════════

with tab_cen:
    st.markdown("### Comparar Cenários de Investimento")
    st.caption("Até 6 valores em € separados por vírgula — calcula o rebalanceamento para cada um.")

    scenarios_input = st.text_input(
        "Valores a investir (€, separados por vírgula)",
        value=st.session_state.scenarios_input,
        key="scenarios_input_widget",
    )
    parsed = parse_scenarios(scenarios_input)
    if parsed:
        st.caption(f"✓ Cenários: {', '.join(f'€{v:,.0f}' for v in parsed[:6])}"
                   + (" (limitado a 6)" if len(parsed) > 6 else ""))

    if st.button("🔍 Comparar cenários", type="primary",
                 disabled=not sum_is_valid or edited_df.empty or not parsed, key="btn_cen"):
        st.session_state.scenarios_input = scenarios_input
        with st.spinner("A obter preços..."):
            prices, currencies, fx, missing_p, missing_fx = fetch_prices_and_fx(edited_df["Ticker"].tolist())
        if missing_p:
            st.error("Sem preço para: " + ", ".join(f"`{t}`" for t in missing_p)); st.stop()
        st.session_state.scenarios_result = {
            amt: compute_rebalance_result(edited_df, amt, prices, currencies, fx)
            for amt in parsed[:6]
        }
        st.session_state.portfolio_df = edited_df[EDITOR_COLS].copy()
        save_portfolio(edited_df, user_id)
        _cached_full.clear()
        st.rerun()

    res = st.session_state.scenarios_result
    if res:
        amounts = list(res.keys())
        first = res[amounts[0]]

        st.markdown("#### Investimento por Ativo (€)")
        cmp_df = first[["Ticker", "Percentagem Alvo (%)", "Alocação Atual (%)"]].copy()
        fmt: dict[str, str] = {"Percentagem Alvo (%)": "{:.2f}", "Alocação Atual (%)": "{:.2f}"}
        for amt in amounts:
            col = f"€{amt:,.0f}"
            cmp_df[col] = res[amt]["Investir (€)"].values
            fmt[col] = "€{:.2f}"
        st.dataframe(cmp_df.style.format(fmt), use_container_width=True, hide_index=True)

        st.markdown("#### Resumo por Cenário")
        cols = st.columns(len(amounts))
        for col, amt in zip(cols, amounts):
            r = res[amt]
            with col:
                st.markdown(f"##### €{amt:,.0f}")
                st.metric("Saúde Final", f"{health_score(r['Desvio Final (pp)'].tolist())}/100")
                st.metric("Desvio Máx.", f"{float(r['Desvio Final (pp)'].abs().max()):.2f} pp")
                st.metric("Valor Final", f"€{float(r['Valor Final (€)'].sum()):,.0f}")

        st.markdown("#### Score de Saúde vs. Montante")
        fig = px.line(
            pd.DataFrame({
                "Montante (€)": amounts,
                "Score": [health_score(res[a]["Desvio Final (pp)"].tolist()) for a in amounts],
            }),
            x="Montante (€)", y="Score", markers=True,
            title="Score de Saúde Pós-Investimento por Cenário",
        )
        fig.update_layout(yaxis=dict(range=[0, 105]), height=320, title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Define valores e clica em **Comparar cenários**.")


# ══════════════════════════════════════════════
# Tab 3: Histórico
# ══════════════════════════════════════════════

with tab_his:
    st.markdown("### Histórico de Snapshots")
    snapshots = _cached_history(user_id)
    col_top_a, col_top_b = st.columns([3, 1])
    col_top_a.metric("Snapshots guardados", len(snapshots))
    with col_top_b:
        if snapshots and st.button("🗑️ Limpar histórico", type="secondary"):
            _cached_history.clear()
            write_history([], user_id)
            st.rerun()

    if st.session_state.result_df is not None:
        if st.button("📸 Tirar snapshot do estado atual"):
            append_snapshot(build_snapshot(st.session_state.result_df, use_final=False, tag="manual"), user_id)
            _cached_history.clear()
            st.success("✓ Snapshot guardado.")
            st.rerun()
    else:
        st.caption("Calcula o rebalanceamento na tab **Rebalancear** para poder tirar snapshot manual.")

    if not snapshots:
        st.info("Ainda não há snapshots. São gravados automaticamente em **Aplicar quantidades**.")
    else:
        df_total, df_alloc = history_to_evolution_dfs(snapshots)

        st.markdown("#### Evolução do Valor Total")
        fig_t = px.line(df_total, x="ts", y="Valor Total (€)", markers=True,
                        title="Valor Total do Portfólio ao Longo do Tempo")
        fig_t.update_layout(height=350, title_x=0.5)
        st.plotly_chart(fig_t, use_container_width=True)

        st.markdown("#### Valor por Ativo (Stacked)")
        fig_a = px.area(df_alloc, x="ts", y="Valor (€)", color="Ticker",
                        title="Valor por Ativo (Stacked Area)")
        fig_a.update_layout(height=400, title_x=0.5)
        st.plotly_chart(fig_a, use_container_width=True)

        st.markdown("#### Alocação (%) por Ativo")
        fig_p = px.line(df_alloc, x="ts", y="Alocação (%)", color="Ticker", markers=True,
                        title="Percentagem Alocada por Ativo")
        fig_p.update_layout(height=350, title_x=0.5)
        st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("#### Lista de Snapshots")
        list_df = pd.DataFrame([{
            "Data": pd.to_datetime(s["ts"]),
            "Tipo": s.get("tag", "?"),
            "Valor Total (€)": s["total_value_eur"],
            "Investido (€)": s.get("money_invested", 0.0),
            "# Ativos": len(s.get("assets", [])),
        } for s in snapshots]).sort_values("Data", ascending=False)
        st.dataframe(list_df.style.format(
            {"Valor Total (€)": "€{:,.2f}", "Investido (€)": "€{:,.2f}"}),
            use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# Tab 4: Risco
# ══════════════════════════════════════════════

with tab_ris:
    st.markdown("### Análise de Risco")
    st.caption("Volatilidade, correlações e métricas de portfólio (moeda nativa — risco cambial não incluído).")

    col_r1, col_r2, col_r3 = st.columns([2, 2, 1])
    risk_period = col_r1.selectbox("Período", list(PERIOD_OPTIONS.keys()),
                                   index=list(PERIOD_OPTIONS.keys()).index(st.session_state.risk_period),
                                   key="sel_risk_period")
    rf_pct = col_r2.number_input("Taxa sem risco (%/ano)", 0.0, 15.0,
                                 float(st.session_state.rf_rate_pct), 0.25, key="ni_rf_risk")
    if col_r3.button("📊 Analisar", type="primary", disabled=edited_df.empty):
        st.session_state.risk_period = risk_period
        st.session_state.rf_rate_pct = rf_pct
        with st.spinner("A obter histórico..."):
            closes = fetch_historical_closes(tuple(edited_df["Ticker"].tolist()),
                                             PERIOD_OPTIONS[risk_period])
        if closes.empty:
            st.error("Sem dados históricos."); st.stop()
        target_pct = dict(zip(edited_df["Ticker"], edited_df["Percentagem Alvo (%)"]))
        per_asset, corr, port = compute_risk_metrics(closes, target_pct, rf_pct / 100)
        dd = compute_drawdown(closes, target_pct)
        st.session_state.risk_data = {
            "per_asset": per_asset, "corr": corr, "portfolio": port,
            "period": risk_period, "rf_pct": rf_pct,
            "missing": [t for t in edited_df["Ticker"] if t not in closes.columns],
            "drawdown": dd,
            "closes": closes,
            "target_pct": target_pct,
        }
        st.rerun()

    rd = st.session_state.risk_data
    if rd:
        if rd.get("missing"):
            st.warning("Sem histórico para: " + ", ".join(f"`{t}`" for t in rd["missing"]))
        per_asset = rd["per_asset"]; corr = rd["corr"]; port = rd["portfolio"]

        st.markdown("#### Métricas do Portfólio")
        st.caption(f"Período: {rd['period']} · taxa s/risco: {rd['rf_pct']:.2f}% · "
                   f"{port.get('n_obs', 0)} obs.")
        if "mean_ret_ann" in port:
            pm1, pm2, pm3 = st.columns(3)
            pm1.metric("Retorno Anual.", f"{port['mean_ret_ann']*100:+.2f}%")
            pm2.metric("Volatilidade Anual.", f"{port['vol_ann']*100:.2f}%")
            sh = port["sharpe"]
            pm3.metric("Sharpe", f"{sh:.2f}" if np.isfinite(sh) else "—")

        st.markdown("#### Por Ativo")
        st.dataframe(per_asset.style.format({
            "Retorno Anual.": "{:+.2%}", "Volatilidade Anual.": "{:.2%}", "Sharpe": "{:.2f}",
        }), use_container_width=True, hide_index=True)

        vol_df = per_asset[["Ticker", "Volatilidade Anual."]].copy()
        vol_df["Volatilidade Anual."] = vol_df["Volatilidade Anual."] * 100
        fig_v = px.bar(vol_df, x="Ticker", y="Volatilidade Anual.",
                       text=vol_df["Volatilidade Anual."].apply(lambda v: f"{v:.1f}%"),
                       title="Volatilidade Anualizada (%)")
        fig_v.update_traces(textposition="outside")
        fig_v.update_layout(height=350, title_x=0.5, yaxis_title="% ao ano")
        st.plotly_chart(fig_v, use_container_width=True)

        if not corr.empty and len(corr) > 1:
            st.markdown("#### Matriz de Correlação")
            fig_c = px.imshow(corr, text_auto=".2f", aspect="auto",
                              color_continuous_scale="RdBu_r", origin="lower",
                              zmin=-1, zmax=1, title="Correlação entre Retornos Diários")
            fig_c.update_layout(height=max(320, 70 * len(corr)), title_x=0.5)
            st.plotly_chart(fig_c, use_container_width=True)
            st.caption("Valores próximos de 0 → boa diversificação; próximos de 1 → ativos movem-se juntos.")

        # ── Drawdown ────────────────────────────────────
        dd = rd.get("drawdown", {})
        if dd:
            st.markdown("#### Drawdown — Quedas e Recuperações")
            dd_ser: pd.Series = dd["drawdown"]
            port_val: pd.Series = dd["port_value"]

            dm1, dm2, dm3 = st.columns(3)
            dm1.metric("Max Drawdown", f"{dd['max_drawdown']*100:.2f}%")
            dm2.metric("Data do Mínimo", str(dd["max_drawdown_date"])[:10])
            if not dd["periods"].empty:
                dm3.metric("Nº de Períodos", len(dd["periods"]))

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd_ser.index, y=dd_ser.values * 100,
                fill="tozeroy", fillcolor="rgba(231,76,60,0.25)",
                line=dict(color="#e74c3c", width=1.5),
                name="Drawdown (%)",
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>",
            ))
            fig_dd.update_layout(
                title="Underwater Chart — Afastamento do Máximo Histórico",
                title_x=0.5, height=320,
                yaxis=dict(title="Drawdown (%)", tickformat=".1f"),
                xaxis_title="", margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            if not dd["periods"].empty:
                with st.expander(f"📋 {len(dd['periods'])} períodos de drawdown"):
                    st.dataframe(dd["periods"].style.format({
                        "Queda Máx. (%)": "{:.2f}",
                        "Dias até Mínimo": "{:d}",
                        "Duração Total (dias)": "{:d}",
                    }), use_container_width=True, hide_index=True)

        # ── TER / Fee Drag ──────────────────────────────
        st.markdown("#### Custos — TER e Arrastamento de Comissões")
        ter_tickers = edited_df["Ticker"].tolist()
        ter_weights = dict(zip(edited_df["Ticker"], edited_df["Percentagem Alvo (%)"]))
        total_w = sum(ter_weights.values())

        ter_rows = []
        for t in ter_tickers:
            ter_val = fetch_ter(t)
            w_pct = ter_weights.get(t, 0) / total_w * 100 if total_w > 0 else 0
            ter_rows.append({
                "Ticker": t,
                "Peso (%)": w_pct,
                "TER (%)": ter_val * 100 if ter_val else None,
                "Contribuição (pp)": (ter_val * w_pct / 100 * 100) if ter_val else None,
            })

        ter_df = pd.DataFrame(ter_rows)
        ter_known = ter_df.dropna(subset=["TER (%)"])
        weighted_ter = float(
            (ter_known["TER (%)"] * ter_known["Peso (%)"] / 100).sum()
        ) if not ter_known.empty else 0.0

        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("TER Ponderado", f"{weighted_ter:.3f}%/ano")
        cap_base = float(total_cur if 'total_cur' in dir() else 10000.0)
        tc2.metric("Custo Anual Estimado",
                   f"€{cap_base * weighted_ter / 100:,.2f}")
        drag_30y = cap_base * ((1.07 ** 30) - (1.07 - weighted_ter / 100) ** 30)
        tc3.metric("Arrastamento em 30 anos (hipot.)",
                   f"€{drag_30y:,.0f}",
                   help="Diferença acumulada vs. portfólio idêntico sem custos, assumindo 7%/ano")

        if not ter_df.empty:
            st.dataframe(ter_df.style.format({
                "Peso (%)": "{:.1f}",
                "TER (%)": lambda v: f"{v:.3f}" if pd.notna(v) else "N/D",
                "Contribuição (pp)": lambda v: f"{v:.4f}" if pd.notna(v) else "N/D",
            }), use_container_width=True, hide_index=True)
            if ter_df["TER (%)"].isna().any():
                st.caption("N/D = TER não disponível via yfinance para esse ticker.")

        # Fee drag comparison chart (10/20/30 years)
        if weighted_ter > 0:
            years_range = list(range(0, 31))
            base_rate = 0.07
            fig_fee = go.Figure()
            fig_fee.add_trace(go.Scatter(
                x=years_range,
                y=[cap_base * (base_rate ** yr if yr == 0 else (1 + base_rate) ** yr) for yr in years_range],
                name="Sem custos (7%/ano)", line=dict(color="#27ae60", dash="dot"),
            ))
            fig_fee.add_trace(go.Scatter(
                x=years_range,
                y=[cap_base * (1 + base_rate - weighted_ter / 100) ** yr for yr in years_range],
                name=f"Com TER {weighted_ter:.3f}%", line=dict(color="#e74c3c"),
                fill="tonexty", fillcolor="rgba(231,76,60,0.10)",
            ))
            fig_fee.update_layout(
                title="Impacto dos Custos no Crescimento do Capital",
                title_x=0.5, height=320,
                yaxis=dict(title="Valor (€)", tickformat="€,.0f"),
                xaxis_title="Anos",
                legend=dict(orientation="h", y=1.02, x=0),
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_fee, use_container_width=True)
            st.caption("Hipótese: capital inicial = valor actual do portfólio; retorno bruto fixo de 7%/ano.")
    else:
        st.info("Clica em **Analisar** para calcular as métricas de risco.")


# ══════════════════════════════════════════════
# Tab 5: Otimizar (Markowitz)
# ══════════════════════════════════════════════

with tab_opt:
    st.markdown("### Otimização de Portfólio — Markowitz")
    st.caption(
        "Encontra os pesos que **minimizam a volatilidade** ou **maximizam o Sharpe**, "
        "dentro dos limites definidos. Mostra a fronteira eficiente e uma nuvem Monte Carlo."
    )

    col_o1, col_o2, col_o3, col_o4 = st.columns(4)
    opt_period = col_o1.selectbox("Período", list(PERIOD_OPTIONS.keys()),
                                  index=list(PERIOD_OPTIONS.keys()).index(st.session_state.opt_period),
                                  key="sel_opt_period")
    opt_rf = col_o2.number_input("Taxa sem risco (%/ano)", 0.0, 15.0,
                                 float(st.session_state.rf_rate_pct), 0.25, key="ni_rf_opt")
    opt_wmin = col_o3.number_input("Peso mín./ativo (%)", 0.0, 50.0,
                                   float(st.session_state.opt_wmin_pct), 1.0, key="ni_wmin")
    opt_wmax = col_o4.number_input("Peso máx./ativo (%)", 10.0, 100.0,
                                   float(st.session_state.opt_wmax_pct), 5.0, key="ni_wmax")

    if st.button("🔬 Otimizar", type="primary", disabled=edited_df.empty or not sum_is_valid):
        st.session_state.opt_period = opt_period
        st.session_state.opt_wmin_pct = opt_wmin
        st.session_state.opt_wmax_pct = opt_wmax
        tickers_t = tuple(edited_df["Ticker"].tolist())
        with st.spinner("A calcular fronteira eficiente..."):
            closes_opt = fetch_historical_closes(tickers_t, PERIOD_OPTIONS[opt_period])
        if closes_opt.empty:
            st.error("Sem dados históricos para os tickers."); st.stop()
        target_pct_opt = dict(zip(edited_df["Ticker"], edited_df["Percentagem Alvo (%)"]))
        with st.spinner("A resolver otimizações (pode levar alguns segundos)..."):
            mk = markowitz_analysis(
                closes_opt, target_pct_opt, opt_rf / 100,
                w_min=opt_wmin / 100, w_max=opt_wmax / 100,
            )
        if not mk:
            st.error("Dados insuficientes para otimização (mínimo 2 ativos, 30 observações)."); st.stop()
        st.session_state.markowitz_result = mk
        st.rerun()

    mk = st.session_state.markowitz_result
    if mk:
        tickers = mk["tickers"]

        # Comparison table
        st.markdown("#### Comparação de Portfólios")
        st.caption("Retorno e volatilidade anualizados na moeda nativa de cada ativo.")
        rows = [
            {"Portfólio": "Atual (alvos definidos)", **{t: f"{mk['cur_w'][t]*100:.1f}%" for t in tickers},
             "Retorno": mk["cur_perf"][0], "Volatilidade": mk["cur_perf"][1], "Sharpe": mk["cur_perf"][2]},
            {"Portfólio": "Mínima Volatilidade", **{t: f"{mk['mv_w'][t]*100:.1f}%" for t in tickers},
             "Retorno": mk["mv_perf"][0], "Volatilidade": mk["mv_perf"][1], "Sharpe": mk["mv_perf"][2]},
            {"Portfólio": "Máximo Sharpe", **{t: f"{mk['ms_w'][t]*100:.1f}%" for t in tickers},
             "Retorno": mk["ms_perf"][0], "Volatilidade": mk["ms_perf"][1], "Sharpe": mk["ms_perf"][2]},
        ]
        cmp_df = pd.DataFrame(rows).set_index("Portfólio")
        st.dataframe(cmp_df.style.format({
            "Retorno": "{:+.2%}", "Volatilidade": "{:.2%}", "Sharpe": "{:.2f}",
        }), use_container_width=True)

        # Efficient frontier plot
        st.markdown("#### Fronteira Eficiente")
        rand_df = mk["random"].dropna()
        front_df = mk["frontier"].dropna()

        fig_ef = px.scatter(rand_df, x="Volatilidade", y="Retorno", color="Sharpe",
                            color_continuous_scale="Viridis", opacity=0.35,
                            labels={"Volatilidade": "Volatilidade Anual.", "Retorno": "Retorno Anual."},
                            title="Fronteira Eficiente + Nuvem Monte Carlo")
        fig_ef.add_scatter(x=front_df["Volatilidade"], y=front_df["Retorno"],
                           mode="lines", line=dict(color="white", width=2.5),
                           name="Fronteira Eficiente")
        for label, w_key, perf_key, color, symbol in [
            ("Atual", "cur_w", "cur_perf", "cyan", "circle"),
            ("Mín. Vol.", "mv_w", "mv_perf", "lime", "star"),
            ("Máx. Sharpe", "ms_w", "ms_perf", "red", "diamond"),
        ]:
            r, v, _ = mk[perf_key]
            fig_ef.add_scatter(x=[v], y=[r], mode="markers+text",
                               marker=dict(size=14, color=color, symbol=symbol),
                               text=[label], textposition="top center",
                               name=label, showlegend=True)
        fig_ef.update_layout(height=520, title_x=0.5,
                             xaxis_tickformat=".1%", yaxis_tickformat=".1%")
        st.plotly_chart(fig_ef, use_container_width=True)

        # Weight charts
        st.markdown("#### Pesos por Portfólio")
        weight_data = pd.DataFrame({
            "Ticker": tickers,
            "Atual (%)": [mk["cur_w"][t] * 100 for t in tickers],
            "Mín. Volatilidade (%)": [mk["mv_w"][t] * 100 for t in tickers],
            "Máx. Sharpe (%)": [mk["ms_w"][t] * 100 for t in tickers],
        })
        fig_w = px.bar(weight_data.melt("Ticker", var_name="Portfólio", value_name="Peso (%)"),
                       x="Ticker", y="Peso (%)", color="Portfólio", barmode="group",
                       title="Distribuição de Pesos")
        fig_w.update_layout(height=380, title_x=0.5)
        st.plotly_chart(fig_w, use_container_width=True)

        # Apply buttons
        st.markdown("#### Aplicar Sugestão ao Portfólio")
        col_apply_mv, col_apply_ms = st.columns(2)
        with col_apply_mv:
            if st.button("📌 Aplicar pesos de Mínima Volatilidade", use_container_width=True):
                new_df = pd.DataFrame({
                    "Ticker": tickers,
                    "Percentagem Alvo (%)": [round(mk["mv_w"][t] * 100, 2) for t in tickers],
                    "Quantidade Detida": [
                        float(edited_df.loc[edited_df["Ticker"] == t, "Quantidade Detida"].iloc[0])
                        if t in edited_df["Ticker"].values else 0.0
                        for t in tickers
                    ],
                })
                st.session_state.portfolio_df = new_df
                st.session_state.result_df = None
                st.session_state.editor_nonce += 1
                save_portfolio(new_df, user_id)
                _cached_full.clear()
                st.success("✓ Pesos de Mínima Volatilidade aplicados.")
                st.rerun()
        with col_apply_ms:
            if st.button("📌 Aplicar pesos de Máximo Sharpe", use_container_width=True):
                new_df = pd.DataFrame({
                    "Ticker": tickers,
                    "Percentagem Alvo (%)": [round(mk["ms_w"][t] * 100, 2) for t in tickers],
                    "Quantidade Detida": [
                        float(edited_df.loc[edited_df["Ticker"] == t, "Quantidade Detida"].iloc[0])
                        if t in edited_df["Ticker"].values else 0.0
                        for t in tickers
                    ],
                })
                st.session_state.portfolio_df = new_df
                st.session_state.result_df = None
                st.session_state.editor_nonce += 1
                save_portfolio(new_df, user_id)
                _cached_full.clear()
                st.success("✓ Pesos de Máximo Sharpe aplicados.")
                st.rerun()

        st.caption(
            "⚠️ Os pesos sugeridos são baseados em dados históricos e não garantem resultados futuros. "
            "Usa como referência, não como conselho financeiro."
        )
    else:
        st.info("Clica em **Otimizar** para calcular os portfólios ótimos.")


# ══════════════════════════════════════════════
# Tab 6: Backtest
# ══════════════════════════════════════════════

with tab_bkt:
    st.markdown("### Backtest — Rebalanceamento vs. Buy & Hold")
    st.caption(
        "Simula a evolução de um investimento inicial com rebalanceamento periódico "
        "(permite vendas) vs. sem rebalanceamento. Usa preços históricos reais."
    )

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    bkt_period = col_b1.selectbox("Período histórico", list(PERIOD_OPTIONS.keys()),
                                  index=list(PERIOD_OPTIONS.keys()).index(st.session_state.bkt_period),
                                  key="sel_bkt_period")
    bkt_capital = col_b2.number_input("Capital inicial (€)", 1000.0, 1_000_000.0,
                                      float(st.session_state.bkt_capital), 1000.0,
                                      format="%.0f", key="ni_bkt_cap")
    bkt_freq = col_b3.selectbox("Frequência de Rebalanceamento", list(FREQ_OPTIONS.keys()),
                                index=list(FREQ_OPTIONS.keys()).index(st.session_state.bkt_freq),
                                key="sel_bkt_freq")
    bkt_rf = col_b4.number_input("Taxa sem risco (%/ano)", 0.0, 15.0,
                                 float(st.session_state.rf_rate_pct), 0.25, key="ni_rf_bkt")

    if st.button("▶️ Executar Backtest", type="primary", disabled=edited_df.empty or not sum_is_valid):
        st.session_state.bkt_period = bkt_period
        st.session_state.bkt_capital = bkt_capital
        st.session_state.bkt_freq = bkt_freq
        tickers_b = tuple(edited_df["Ticker"].tolist())
        with st.spinner("A obter preços históricos..."):
            closes_b = fetch_historical_closes(tickers_b, PERIOD_OPTIONS[bkt_period])
        if closes_b.empty:
            st.error("Sem dados históricos."); st.stop()
        target_w = dict(zip(edited_df["Ticker"], edited_df["Percentagem Alvo (%)"]))
        with st.spinner("A executar simulação..."):
            result = run_backtest(
                closes_b, target_w, bkt_capital,
                rebalance_freq=FREQ_OPTIONS[bkt_freq],
                rf_rate=bkt_rf / 100,
            )
        if not result:
            st.error("Backtest falhou — verifica se os tickers têm dados no período selecionado."); st.stop()
        st.session_state.backtest_result = result
        st.rerun()

    br = st.session_state.backtest_result
    if br:
        if br.get("missing_tickers"):
            st.warning("Tickers sem dados históricos (excluídos): "
                       + ", ".join(f"`{t}`" for t in br["missing_tickers"]))
        if br.get("tickers_used"):
            st.caption(f"Tickers usados: {', '.join(br['tickers_used'])} · "
                       f"Rebalanceamentos: {br['n_reb']}")

        # Cumulative value chart
        st.markdown("#### Evolução do Capital")
        chart_df = pd.DataFrame({
            "Data": br["port"].index,
            f"Rebalanceamento ({st.session_state.bkt_freq})": br["port"].values,
            "Buy & Hold": br["bh"].values,
        }).set_index("Data")
        st.line_chart(chart_df)

        # Metrics comparison
        st.markdown("#### Comparação de Métricas")
        pm = br["port_metrics"]
        bm = br["bh_metrics"]
        metrics_df = pd.DataFrame({
            "Métrica": ["CAGR", "Volatilidade Anual.", "Sharpe", "Max Drawdown",
                        "Retorno Total", "Valor Final (€)"],
            f"Rebalanceamento ({st.session_state.bkt_freq})": [
                pm["CAGR"], pm["Volatilidade"], pm["Sharpe"],
                pm["Max Drawdown"], pm["Retorno Total"], pm["Valor Final (€)"],
            ],
            "Buy & Hold": [
                bm["CAGR"], bm["Volatilidade"], bm["Sharpe"],
                bm["Max Drawdown"], bm["Retorno Total"], bm["Valor Final (€)"],
            ],
        }).set_index("Métrica")

        fmt_map = {
            "CAGR": "{:+.2%}", "Volatilidade Anual.": "{:.2%}", "Sharpe": "{:.2f}",
            "Max Drawdown": "{:.2%}", "Retorno Total": "{:+.2%}", "Valor Final (€)": "€{:,.2f}",
        }
        styled = metrics_df.copy()
        for col in styled.columns:
            styled[col] = [fmt_map[idx].format(v) for idx, v in zip(styled.index, metrics_df[col])]
        st.dataframe(styled, use_container_width=True)

        # Rebalancing events
        if not br["reb_events"].empty:
            with st.expander(f"📋 {br['n_reb']} eventos de rebalanceamento"):
                st.dataframe(
                    br["reb_events"].style.format({"Valor (€)": "€{:,.2f}"}),
                    use_container_width=True, hide_index=True,
                )

        st.caption(
            "⚠️ O backtest usa rebalanceamento clássico (com vendas) como comparação teórica. "
            "Não inclui custos de transação, spreads ou impostos."
        )
    else:
        st.info("Define os parâmetros e clica em **Executar Backtest**.")

    # ── Comparação com portfólios modelo ────────
    st.divider()
    st.markdown("#### Comparação com Portfólios Modelo")
    st.caption("Compara o teu portfólio (Buy & Hold) com carteiras de referência clássicas usando ETFs norte-americanos como proxy.")

    col_mc1, col_mc2, col_mc3 = st.columns([2, 2, 1])
    mc_period = col_mc1.selectbox("Período", list(PERIOD_OPTIONS.keys()),
                                   index=list(PERIOD_OPTIONS.keys()).index("3 anos"),
                                   key="sel_mc_period")
    mc_capital = col_mc2.number_input("Capital inicial (€)", 1000.0, 1_000_000.0,
                                       float(st.session_state.bkt_capital), 1000.0,
                                       format="%.0f", key="ni_mc_cap")
    if col_mc3.button("📊 Comparar", type="primary", disabled=edited_df.empty or not sum_is_valid):
        model_tickers = set()
        for w in MODEL_PORTFOLIOS.values():
            model_tickers.update(w.keys())
        with st.spinner("A obter dados dos portfólios modelo..."):
            model_closes = fetch_historical_closes(tuple(sorted(model_tickers)),
                                                   PERIOD_OPTIONS[mc_period])
            user_closes_mc = fetch_historical_closes(tuple(edited_df["Ticker"].tolist()),
                                                     PERIOD_OPTIONS[mc_period])
        user_weights_mc = dict(zip(edited_df["Ticker"], edited_df["Percentagem Alvo (%)"]))
        user_bh = None
        if not user_closes_mc.empty:
            r_user = run_backtest(user_closes_mc, user_weights_mc, mc_capital, "never",
                                  float(st.session_state.rf_rate_pct) / 100)
            if r_user:
                user_bh = r_user["bh"]
        cmp_df = compare_model_portfolios(model_closes, mc_capital,
                                          float(st.session_state.rf_rate_pct) / 100,
                                          user_bh_series=user_bh)
        st.session_state.model_cmp_result = {"df": cmp_df, "period": mc_period}
        st.rerun()

    mc = st.session_state.model_cmp_result
    if mc and not mc["df"].empty:
        cdf = mc["df"]
        display_cols = ["Portfólio", "CAGR (%)", "Volatilidade (%)", "Sharpe",
                        "Max Drawdown (%)", "Retorno Total (%)", "Valor Final (€)"]
        show_df = cdf[[c for c in display_cols if c in cdf.columns]]
        st.dataframe(show_df.style.format({
            "CAGR (%)": "{:+.2f}%",
            "Volatilidade (%)": "{:.2f}%",
            "Sharpe": "{:.2f}",
            "Max Drawdown (%)": "{:.2f}%",
            "Retorno Total (%)": "{:+.2f}%",
            "Valor Final (€)": "€{:,.2f}",
        }), use_container_width=True, hide_index=True)

        # Evolution chart
        fig_mc = go.Figure()
        for _, row in cdf.iterrows():
            ser = row.get("_series")
            if ser is not None and len(ser) > 1:
                fig_mc.add_trace(go.Scatter(
                    x=ser.index, y=ser.values,
                    name=row["Portfólio"],
                    mode="lines",
                    line=dict(width=2.5 if row["Portfólio"] == "O Meu Portfólio" else 1.5),
                ))
        fig_mc.update_layout(
            title=f"Evolução do Capital — {mc['period']}",
            title_x=0.5, height=400,
            yaxis=dict(title="Valor (€)", tickformat="€,.0f"),
            xaxis_title="",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        st.caption("Portfólios modelo usam ETFs norte-americanos (SPY, TLT, VTI, GLD, etc.) como proxy — sem custos de câmbio nem TER.")
    elif mc:
        st.warning("Sem dados suficientes para comparar no período escolhido.")


# ══════════════════════════════════════════════
# Tab 7: Dividendos
# ══════════════════════════════════════════════

with tab_div:
    st.markdown("### Dashboard de Dividendos")
    st.caption("Rendimento gerado pelos dividendos dos ativos, yield on cost e calendário de pagamentos.")

    full_for_div = _cached_full(user_id)
    if st.button("💰 Calcular Dividendos", type="primary",
                 disabled=edited_df.empty, key="btn_div"):
        with st.spinner("A obter histórico de dividendos..."):
            prices_d, currencies_d, fx_d, _, _ = fetch_prices_and_fx(edited_df["Ticker"].tolist())
        prices_eur_d = {}
        for t in edited_df["Ticker"]:
            p = prices_d.get(t)
            cur = currencies_d.get(t, "EUR")
            rate = fx_d.get(cur, 1.0) or 1.0
            prices_eur_d[t] = float(p * rate) if p else 0.0

        qtys_d = dict(zip(edited_df["Ticker"], edited_df["Quantidade Detida"]))
        avgs_d = {} if full_for_div.empty else dict(
            zip(full_for_div["Ticker"], full_for_div["Preço Médio (€)"])
        )
        per_asset_div, total_div, calendar_div = portfolio_dividend_report(
            edited_df["Ticker"].tolist(), qtys_d, avgs_d, prices_eur_d,
        )
        st.session_state.div_data = {
            "per_asset": per_asset_div,
            "total": total_div,
            "calendar": calendar_div,
        }
        st.rerun()

    dv = st.session_state.div_data
    if dv:
        total_div_val = dv["total"]
        dv1, dv2, dv3 = st.columns(3)
        dv1.metric("Rendimento Anual Total", f"€{total_div_val:,.2f}")
        dv2.metric("Rendimento Mensal Médio", f"€{total_div_val/12:,.2f}")
        dv3.metric("Ativos com Dividendos",
                   int((dv["per_asset"]["Rendimento Anual (€)"] > 0).sum()))

        st.markdown("#### Por Ativo")
        pa = dv["per_asset"]
        st.dataframe(pa.style.format({
            "Rendimento Anual (€)": "€{:.2f}",
            "Yield Atual (%)": "{:.2f}%",
            "Yield on Cost (%)": "{:.2f}%",
            "DGR 3A (%)": lambda v: f"{v:.2f}%" if pd.notna(v) else "N/D",
            "DGR 5A (%)": lambda v: f"{v:.2f}%" if pd.notna(v) else "N/D",
        }), use_container_width=True, hide_index=True)

        # Income bar chart
        if not pa.empty and pa["Rendimento Anual (€)"].sum() > 0:
            fig_div = px.bar(
                pa[pa["Rendimento Anual (€)"] > 0],
                x="Ticker", y="Rendimento Anual (€)",
                text=pa[pa["Rendimento Anual (€)"] > 0]["Rendimento Anual (€)"].apply(
                    lambda v: f"€{v:.2f}"),
                title="Rendimento Anual por Ativo (€)",
                color="Ticker",
            )
            fig_div.update_traces(textposition="outside")
            fig_div.update_layout(height=350, title_x=0.5, showlegend=False,
                                   yaxis_title="€/ano")
            st.plotly_chart(fig_div, use_container_width=True)

        # Calendar
        if not dv["calendar"].empty:
            st.markdown("#### Calendário de Pagamentos (últimos 24 meses)")
            st.dataframe(dv["calendar"].style.format({
                "Div/Ação (€)": "{:.4f}",
                "Rendimento (€)": "€{:.2f}",
            }), use_container_width=True, hide_index=True)
        else:
            st.info("Sem histórico de pagamentos nos últimos 24 meses.")
    else:
        st.info("Clica em **Calcular Dividendos** para analisar o rendimento do portfólio.")


# ══════════════════════════════════════════════
# Tab 8: Fiscalidade
# ══════════════════════════════════════════════

with tab_tax:
    st.markdown("### Mais-Valias Latentes e Tax-Loss Harvesting")
    st.caption(
        "Análise fiscal com base no preço médio de compra registado. "
        "Os ativos sem preço médio (= 0) são excluídos."
    )

    full_for_tax = _cached_full(user_id)
    tx1, tx2 = st.columns([3, 1])
    tax_rate_pct = tx1.slider("Taxa de imposto sobre mais-valias (%)", 10.0, 50.0, 28.0, 1.0,
                               key="tax_rate_slider")
    loss_thresh_pct = tx2.number_input("Limiar menos-valia (%)", -50.0, 0.0, -10.0, 1.0,
                                        key="tax_loss_thresh")

    if st.button("🧾 Calcular Mais-Valias", type="primary",
                 disabled=edited_df.empty, key="btn_tax"):
        with st.spinner("A obter preços actuais..."):
            prices_t, currencies_t, fx_t, _, _ = fetch_prices_and_fx(edited_df["Ticker"].tolist())
        prices_eur_t = {}
        for t in edited_df["Ticker"]:
            p = prices_t.get(t)
            cur = currencies_t.get(t, "EUR")
            rate = fx_t.get(cur, 1.0) or 1.0
            prices_eur_t[t] = float(p * rate) if p else 0.0

        qtys_t = dict(zip(edited_df["Ticker"], edited_df["Quantidade Detida"]))
        avgs_t = {} if full_for_tax.empty else dict(
            zip(full_for_tax["Ticker"], full_for_tax["Preço Médio (€)"])
        )
        tax_df, total_gain, total_tax = compute_unrealized_gains(
            edited_df["Ticker"].tolist(), qtys_t, avgs_t, prices_eur_t,
            tax_rate=tax_rate_pct / 100,
        )
        st.session_state.tax_data = {
            "df": tax_df, "total_gain": total_gain, "total_tax": total_tax,
        }
        st.rerun()

    td = st.session_state.tax_data
    if td:
        tax_df = td["df"]
        if tax_df.empty:
            st.warning("Nenhum ativo com preço médio de compra registado. "
                       "Regista os preços médios na tab **⚙️ Gerir**.")
        else:
            tg1, tg2, tg3 = st.columns(3)
            tg1.metric("Mais-Valia Total", f"€{td['total_gain']:,.2f}",
                       delta=f"{'ganho' if td['total_gain'] >= 0 else 'perda'}")
            tg2.metric("Imposto Estimado (se vendido tudo)", f"€{td['total_tax']:,.2f}")
            n_losses = int((tax_df["Mais-Valia (€)"] < 0).sum())
            tg3.metric("Posições com Menos-Valia", n_losses)

            st.markdown("#### Detalhe por Ativo")
            st.dataframe(tax_df.style.format({
                "Qtd.": "{:.4f}",
                "Preço Médio (€)": "€{:.4f}",
                "Preço Atual (€)": "€{:.4f}",
                "Custo Base (€)": "€{:.2f}",
                "Valor Atual (€)": "€{:.2f}",
                "Mais-Valia (€)": "€{:+.2f}",
                "Mais-Valia (%)": "{:+.2f}%",
                "Imposto Estimado (€)": "€{:.2f}",
            }).apply(
                lambda col: ["background-color: #2d5016" if v >= 0 else "background-color: #5a1a1a"
                             for v in tax_df["Mais-Valia (€)"]]
                if col.name == "Mais-Valia (€)" else [""] * len(col),
                axis=0,
            ), use_container_width=True, hide_index=True)

            # Tax-loss harvesting candidates
            candidates = tax_df[tax_df["Mais-Valia (%)"] < loss_thresh_pct]
            if not candidates.empty:
                st.markdown(f"#### 🎯 Candidatos a Tax-Loss Harvesting (< {loss_thresh_pct:.0f}%)")
                for _, row in candidates.iterrows():
                    st.warning(
                        f"🔴 **{row['Ticker']}** — menos-valia de **€{row['Mais-Valia (€)']:+.2f}** "
                        f"({row['Mais-Valia (%)']:+.2f}%) · pode reduzir imposto em €{abs(row['Mais-Valia (€)']) * tax_rate_pct / 100:.2f}"
                    )
            else:
                st.success(f"✅ Nenhum ativo com menos-valia abaixo de {loss_thresh_pct:.0f}%.")

            # Waterfall chart
            fig_tax = go.Figure(go.Bar(
                x=tax_df["Ticker"],
                y=tax_df["Mais-Valia (€)"],
                marker_color=["#27ae60" if v >= 0 else "#e74c3c"
                               for v in tax_df["Mais-Valia (€)"]],
                text=[f"€{v:+.0f}" for v in tax_df["Mais-Valia (€)"]],
                textposition="outside",
            ))
            fig_tax.update_layout(
                title="Mais-Valias Latentes por Ativo (€)",
                title_x=0.5, height=350,
                yaxis_title="€", yaxis_tickformat="€,.0f",
                xaxis_title="",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_tax, use_container_width=True)
    else:
        st.info("Clica em **Calcular Mais-Valias** para analisar a exposição fiscal.")


# ══════════════════════════════════════════════
# Tab 9: FIRE
# ══════════════════════════════════════════════

with tab_fire:
    st.markdown("### Calculadora FIRE — Monte Carlo de Retirada")
    st.caption(
        "Simula N=5000 cenários de retirada para estimar a probabilidade de o portfólio "
        "sobreviver ao horizonte de reforma. Retornos gerados por distribuição normal."
    )

    fc1, fc2 = st.columns(2)
    with fc1:
        fire_pv = st.number_input(
            "Capital Actual (€)", 1000.0, 10_000_000.0,
            float(st.session_state.fire_portfolio_value), 1000.0,
            format="%.0f", key="ni_fire_pv",
        )
        fire_spend = st.number_input(
            "Despesa Anual (€)", 1000.0, 1_000_000.0,
            float(st.session_state.fire_spending), 500.0,
            format="%.0f", key="ni_fire_spend",
        )
        fire_years = st.slider("Horizonte (anos)", 5, 60,
                               int(st.session_state.fire_years), 1, key="sl_fire_years")
    with fc2:
        fire_ret = st.number_input(
            "Retorno Médio Esperado (%/ano)", 0.0, 30.0,
            float(st.session_state.fire_ret_mean), 0.5,
            format="%.1f", key="ni_fire_ret",
        )
        fire_std = st.number_input(
            "Volatilidade (%/ano)", 1.0, 50.0,
            float(st.session_state.fire_ret_std), 0.5,
            format="%.1f", key="ni_fire_std",
        )
        fire_infl = st.number_input(
            "Inflação (%/ano)", 0.0, 15.0,
            float(st.session_state.fire_inflation), 0.1,
            format="%.1f", key="ni_fire_infl",
        )

    fire_swr_input = fire_pv * 0.04
    st.caption(f"Regra dos 4% → €{fire_swr_input:,.0f}/ano · "
               f"Rácio actual: {fire_spend/fire_pv*100:.2f}% do capital")

    if st.button("🔥 Simular", type="primary", key="btn_fire"):
        st.session_state.fire_portfolio_value = fire_pv
        st.session_state.fire_spending = fire_spend
        st.session_state.fire_years = fire_years
        st.session_state.fire_ret_mean = fire_ret
        st.session_state.fire_ret_std = fire_std
        st.session_state.fire_inflation = fire_infl
        with st.spinner("A correr 5 000 simulações Monte Carlo..."):
            fire_res = run_fire_simulation(
                portfolio_value=fire_pv,
                annual_spending=fire_spend,
                years=fire_years,
                annual_return_mean=fire_ret / 100,
                annual_return_std=fire_std / 100,
                inflation_rate=fire_infl / 100,
            )
        st.session_state.fire_result = fire_res
        st.rerun()

    fr = st.session_state.fire_result
    if fr:
        sr = fr["success_rate"]
        safe = fr["safe_swr"]

        # Main metrics
        fm1, fm2, fm3 = st.columns(3)
        color = "normal" if sr >= 0.90 else "inverse"
        fm1.metric("Probabilidade de Sucesso",
                   f"{sr*100:.1f}%",
                   delta="seguro" if sr >= 0.95 else ("atenção" if sr >= 0.80 else "risco"),
                   delta_color=("normal" if sr >= 0.95 else ("off" if sr >= 0.80 else "inverse")))
        fm2.metric("Taxa Máx. Segura (95% sucesso)",
                   f"{safe*100:.2f}%" if safe else "N/D",
                   delta=f"€{fire_pv*safe:,.0f}/ano" if safe else "")
        fm3.metric("Regra dos 4%",
                   f"{'✅' if fire_spend <= fire_pv*0.04 else '⚠️'} {fire_spend/fire_pv*100:.2f}% do capital")

        if sr < 0.80:
            st.error("🔴 Probabilidade de sucesso abaixo de 80% — considera reduzir a despesa ou aumentar o capital.")
        elif sr < 0.95:
            st.warning("🟡 Probabilidade entre 80–95% — margem de segurança reduzida.")
        else:
            st.success("🟢 Probabilidade ≥ 95% — plano robusto com os parâmetros actuais.")

        # Fan chart
        pcts = fr["percentiles"]
        fig_fire = go.Figure()
        fig_fire.add_trace(go.Scatter(
            x=pcts["Ano"], y=pcts["p95"], name="P95",
            line=dict(width=0), showlegend=False,
        ))
        fig_fire.add_trace(go.Scatter(
            x=pcts["Ano"], y=pcts["p75"], name="P25–P75",
            fill="tonexty", fillcolor="rgba(39,174,96,0.15)",
            line=dict(width=0),
        ))
        fig_fire.add_trace(go.Scatter(
            x=pcts["Ano"], y=pcts["p25"], name="P5–P25",
            fill="tonexty", fillcolor="rgba(39,174,96,0.25)",
            line=dict(width=0),
        ))
        fig_fire.add_trace(go.Scatter(
            x=pcts["Ano"], y=pcts["p5"], name="P5",
            fill="tonexty", fillcolor="rgba(231,76,60,0.15)",
            line=dict(color="#e74c3c", width=1, dash="dot"), showlegend=False,
        ))
        fig_fire.add_trace(go.Scatter(
            x=pcts["Ano"], y=pcts["p50"], name="Mediana (P50)",
            line=dict(color="#27ae60", width=2.5),
        ))
        fig_fire.add_hline(y=0, line=dict(color="white", dash="dash", width=1))
        fig_fire.update_layout(
            title=f"Evolução do Portfólio — {fire_years} anos · 5 000 simulações",
            title_x=0.5, height=420,
            yaxis=dict(title="Valor (€)", tickformat="€,.0f"),
            xaxis_title="Ano",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig_fire, use_container_width=True)

        # Depletion histogram
        if fr["depletion_years"]:
            depl_df = pd.DataFrame({"Ano de falência": fr["depletion_years"]})
            fig_depl = px.histogram(
                depl_df, x="Ano de falência", nbins=fire_years,
                title="Distribuição do Ano de Falência (simulações falhadas)",
                color_discrete_sequence=["#e74c3c"],
            )
            fig_depl.update_layout(height=300, title_x=0.5,
                                    yaxis_title="Nº de simulações")
            st.plotly_chart(fig_depl, use_container_width=True)
            st.caption(f"{len(fr['depletion_years'])} de {fr['n_simulations']} simulações esgotaram o capital.")
    else:
        st.info("Define os parâmetros e clica em **Simular** para correr a análise FIRE.")


# ══════════════════════════════════════════════
# Tab 10: Gerir
# ══════════════════════════════════════════════

with tab_mgr:
    st.markdown("### Gestão de Ativos")
    st.caption("Adiciona, edita ou elimina ativos. As alterações reflectem-se imediatamente nas outras tabs.")

    full_df = _cached_full(user_id)
    if not full_df.empty:
        st.markdown("#### Portfólio Atual")
        st.dataframe(full_df.style.format({
            "Percentagem Alvo (%)": "{:.2f}",
            "Quantidade Detida": "{:.4f}",
            "Preço Médio (€)": "{:.4f}",
        }), use_container_width=True, hide_index=True)
    else:
        st.info("Portfólio vazio. Adiciona o primeiro ativo abaixo.")

    st.divider()
    st.markdown("#### Adicionar / Editar Ativo")

    tickers_list = list(_cached_full(user_id)["Ticker"]) if not full_df.empty else []
    mode = st.radio("Modo", ["Adicionar novo", "Editar existente"],
                    horizontal=True, key="mgr_mode")

    _empty: dict = {
        "ticker": "", "name": "", "asset_type": ASSET_TYPES[0],
        "exchange": "", "target_pct": 0.0, "quantity": 0.0, "avg_buy_price": 0.0,
    }
    prefill = _empty.copy()

    if mode == "Editar existente":
        if tickers_list:
            sel = st.selectbox("Seleciona o Ativo", tickers_list, key="mgr_edit_sel")
            asset_data = get_asset(sel, user_id)
            if asset_data:
                prefill = asset_data
        else:
            st.info("Não há ativos registados. Muda para **Adicionar novo**.")

    with st.form("mgr_form"):
        fc1, fc2 = st.columns(2)
        f_ticker = fc1.text_input(
            "Ticker *", value=prefill["ticker"],
            disabled=(mode == "Editar existente"),
            placeholder="ex: VWCE.DE",
        )
        f_name = fc2.text_input(
            "Nome", value=prefill["name"],
            placeholder="ex: Vanguard FTSE All-World",
        )
        fc3, fc4 = st.columns(2)
        type_idx = ASSET_TYPES.index(prefill["asset_type"]) if prefill["asset_type"] in ASSET_TYPES else 0
        f_type = fc3.selectbox("Tipo", ASSET_TYPES, index=type_idx)
        f_exchange = fc4.text_input(
            "Bolsa", value=prefill["exchange"],
            placeholder="ex: XETRA",
        )
        fc5, fc6, fc7 = st.columns(3)
        f_target = fc5.number_input(
            "Alvo (%)", 0.0, 100.0,
            float(prefill["target_pct"]), 0.5, format="%.2f", key="mgr_f_target",
        )
        f_qty = fc6.number_input(
            "Quantidade Detida", 0.0,
            value=float(prefill["quantity"]), step=0.0001, format="%.4f", key="mgr_f_qty",
        )
        f_avg = fc7.number_input(
            "Preço Médio (€)", 0.0,
            value=float(prefill["avg_buy_price"]), step=0.01, format="%.4f", key="mgr_f_avg",
        )
        submitted = st.form_submit_button("💾 Guardar", type="primary", use_container_width=True)

    if submitted:
        t_save = (prefill["ticker"] if mode == "Editar existente"
                  else f_ticker.strip())
        if not t_save:
            st.error("O Ticker é obrigatório.")
        elif (mode == "Adicionar novo"
              and not is_pro()
              and len(tickers_list) >= FREE_PLAN_LIMIT):
            st.warning(
                f"O plano **Gratuito** está limitado a {FREE_PLAN_LIMIT} ativos. "
                "Actualiza para **Pro** para adicionar mais."
            )
        else:
            upsert_asset(t_save, f_name.strip(), f_type, f_exchange.strip(),
                         f_target, f_qty, f_avg, user_id=user_id)
            _cached_full.clear()
            st.session_state.portfolio_df = load_portfolio(user_id)
            st.session_state.result_df = None
            st.session_state.editor_nonce += 1
            st.success(f"✓ **{t_save}** guardado com sucesso.")
            st.rerun()

    st.divider()
    st.markdown("#### Eliminar Ativo")

    if tickers_list:
        del_ticker = st.selectbox("Ativo a Eliminar", tickers_list, key="mgr_del_sel")
        if st.button("🗑️ Iniciar eliminação", key="mgr_del_btn"):
            st.session_state.del_pending = del_ticker

        if st.session_state.get("del_pending"):
            st.warning(
                f"⚠️ Confirmas a eliminação de **{st.session_state.del_pending}**? "
                "Esta acção não pode ser desfeita."
            )
            col_y, col_n = st.columns(2)
            if col_y.button("✅ Confirmar", type="primary", key="mgr_del_yes"):
                delete_asset(st.session_state.del_pending, user_id)
                _cached_full.clear()
                del st.session_state["del_pending"]
                st.session_state.portfolio_df = load_portfolio(user_id)
                st.session_state.result_df = None
                st.session_state.editor_nonce += 1
                st.success("✓ Ativo eliminado.")
                st.rerun()
            if col_n.button("❌ Cancelar", key="mgr_del_no"):
                del st.session_state["del_pending"]
                st.rerun()
    else:
        st.info("Sem ativos para eliminar.")


# Footer
st.divider()
st.caption(
    f"Otimizador de Portfólio · v{__version__} · "
    f"[GitHub](https://github.com/joaocruzfialho/otimizador-portfolio-ativos)"
)
