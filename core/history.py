from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
HISTORY_FILE = DATA_DIR / "history.json"
MAX_SNAPSHOTS = 500

EDITOR_COLS = ["Ticker", "Percentagem Alvo (%)", "Quantidade Detida"]

DEFAULT_PORTFOLIO = pd.DataFrame({
    "Ticker": ["VWCE.DE", "AGGH.MI", "SGLD.MI", "BTC-EUR"],
    "Percentagem Alvo (%)": [60.0, 25.0, 10.0, 5.0],
    "Quantidade Detida": [10.0, 50.0, 5.0, 0.05],
})


def load_portfolio() -> pd.DataFrame:
    if PORTFOLIO_FILE.exists():
        try:
            data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
            rows = data.get("portfolio", [])
            if rows:
                df = pd.DataFrame([{
                    "Ticker": r["ticker"],
                    "Percentagem Alvo (%)": float(r["target_pct"]),
                    "Quantidade Detida": float(r["quantity"]),
                } for r in rows])
                return df[EDITOR_COLS]
        except Exception:
            pass
    return DEFAULT_PORTFOLIO.copy()


def save_portfolio(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{
        "ticker": str(row["Ticker"]),
        "target_pct": float(row["Percentagem Alvo (%)"]),
        "quantity": float(row["Quantidade Detida"]),
    } for _, row in df.iterrows() if str(row["Ticker"]).strip()]
    PORTFOLIO_FILE.write_text(json.dumps({
        "version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "portfolio": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def df_to_export_json(df: pd.DataFrame) -> str:
    return json.dumps({
        "version": 1,
        "portfolio": [{
            "ticker": str(row["Ticker"]),
            "target_pct": float(row["Percentagem Alvo (%)"]),
            "quantity": float(row["Quantidade Detida"]),
        } for _, row in df.iterrows() if str(row["Ticker"]).strip()],
    }, indent=2, ensure_ascii=False)


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8")).get("snapshots", [])
        except Exception:
            pass
    return []


def write_history(snapshots: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(
        {"version": 1, "snapshots": snapshots[-MAX_SNAPSHOTS:]},
        indent=2, ensure_ascii=False,
    ), encoding="utf-8")


def append_snapshot(snap: dict) -> None:
    snaps = load_history()
    snaps.append(snap)
    write_history(snaps)


def build_snapshot(result_df: pd.DataFrame, *, use_final: bool,
                   money_invested: float = 0.0, tag: str = "manual") -> dict:
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
    df_t = pd.DataFrame(rows_total)
    df_a = pd.DataFrame(rows_alloc)
    df_t["ts"] = pd.to_datetime(df_t["ts"])
    df_a["ts"] = pd.to_datetime(df_a["ts"])
    return df_t.sort_values("ts"), df_a.sort_values("ts")
