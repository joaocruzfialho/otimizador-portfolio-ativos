"""
Otimizador de Portfólio de Ativos — v0.4.0

Tabs: Rebalancear · Cenários · Histórico · Risco · Otimizar · Backtest
Alertas por email configuráveis na sidebar.

Autor: João Fialho
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.alerts import SmtpConfig, build_alert_html, send_alert_email
from core.backtest import run_backtest
from core.history import (
    append_snapshot, build_snapshot, df_to_export_json,
    history_to_evolution_dfs, load_history, load_portfolio,
    save_portfolio, write_history, PORTFOLIO_FILE, EDITOR_COLS,
)
from core.prices import fetch_fx_to_eur, fetch_historical_closes, fetch_price_and_currency
from core.rebalance import (
    compute_rebalance_result, health_score, parse_scenarios,
    rebalance_urgency,
)
from core.risk import compute_risk_metrics, markowitz_analysis

__version__ = "0.4.0"

PERIOD_OPTIONS = {
    "1 mês": "1mo", "3 meses": "3mo", "6 meses": "6mo",
    "1 ano": "1y", "2 anos": "2y", "3 anos": "3y", "5 anos": "5y",
}
FREQ_OPTIONS = {
    "Mensal": "M", "Trimestral": "Q", "Anual": "A", "Nunca (Buy & Hold)": "never",
}


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

_defaults: dict = {
    "portfolio_df": None,
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
    st.session_state.portfolio_df = load_portfolio()


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
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
            st.session_state.portfolio_df = load_portfolio()
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

tab_reb, tab_cen, tab_his, tab_ris, tab_opt, tab_bkt = st.tabs([
    "💼 Rebalancear", "🎯 Cenários", "📈 Histórico",
    "⚖️ Risco", "🔬 Otimizar", "🔄 Backtest",
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
        save_portfolio(edited_df)
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

        st.bar_chart(df.set_index("Ticker")[
            ["Alocação Atual (%)", "Percentagem Alvo (%)", "Alocação Final (%)"]])

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
                                           tag="apply"))
            new_df = pd.DataFrame({
                "Ticker": df["Ticker"].values,
                "Percentagem Alvo (%)": df["Percentagem Alvo (%)"].values,
                "Quantidade Detida": (df["Quantidade Detida"] + df["Qtd a Comprar"]).values,
            })
            st.session_state.portfolio_df = new_df
            st.session_state.result_df = None
            st.session_state.editor_nonce += 1
            save_portfolio(new_df)
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
        save_portfolio(edited_df)
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
    snapshots = load_history()
    col_top_a, col_top_b = st.columns([3, 1])
    col_top_a.metric("Snapshots guardados", len(snapshots))
    with col_top_b:
        if snapshots and st.button("🗑️ Limpar histórico", type="secondary"):
            write_history([])
            st.rerun()

    if st.session_state.result_df is not None:
        if st.button("📸 Tirar snapshot do estado atual"):
            append_snapshot(build_snapshot(st.session_state.result_df, use_final=False, tag="manual"))
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
        st.session_state.risk_data = {
            "per_asset": per_asset, "corr": corr, "portfolio": port,
            "period": risk_period, "rf_pct": rf_pct,
            "missing": [t for t in edited_df["Ticker"] if t not in closes.columns],
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
                save_portfolio(new_df)
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
                save_portfolio(new_df)
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


# Footer
st.divider()
st.caption(
    f"Otimizador de Portfólio · v{__version__} · "
    f"[GitHub](https://github.com/joaocruzfialho/otimizador-portfolio-ativos)"
)
