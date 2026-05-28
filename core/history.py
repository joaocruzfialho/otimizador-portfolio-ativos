from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
HISTORY_FILE = DATA_DIR / "history.json"
MAX_SNAPSHOTS = 500

EDITOR_COLS = ["Ticker", "Percentagem Alvo (%)", "Quantidade Detida"]

# ─── Supabase client (lazy singleton) ──────────────────────────────

_sb_client = None

def _sb():
    """Returns a Supabase client if env vars are set, else None."""
    global _sb_client
    if _sb_client is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if url and key:
            try:
                from supabase import create_client
                _sb_client = create_client(url, key)
            except Exception:
                pass
    return _sb_client


# ─── Portfolio persistence ──────────────────────────────────────────

def load_portfolio() -> pd.DataFrame:
    from core.db import load_portfolio as _db_load
    return _db_load()


def save_portfolio(df: pd.DataFrame) -> None:
    from core.db import save_portfolio_from_editor as _db_save
    _db_save(df)
    # Keep local JSON backup for sidebar "último guardado" display
    assets = [{
        "ticker": str(row["Ticker"]),
        "target_pct": float(row["Percentagem Alvo (%)"]),
        "quantity": float(row["Quantidade Detida"]),
    } for _, row in df.iterrows() if str(row["Ticker"]).strip()]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_FILE.write_text(json.dumps({
        "version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "portfolio": assets,
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


# ─── Snapshot persistence ───────────────────────────────────────────

def load_history() -> list[dict]:
    sb = _sb()
    if sb:
        try:
            resp = (
                sb.table("otimizador_snapshots")
                .select("ts,tag,money_invested,total_value_eur,assets")
                .order("ts", desc=False)
                .limit(MAX_SNAPSHOTS)
                .execute()
            )
            if resp.data:
                return [{
                    "ts": r["ts"],
                    "tag": r.get("tag", "manual"),
                    "money_invested": float(r.get("money_invested", 0)),
                    "total_value_eur": float(r.get("total_value_eur", 0)),
                    "assets": r.get("assets", []),
                } for r in resp.data]
        except Exception:
            pass
    # fallback: local file
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8")).get("snapshots", [])
        except Exception:
            pass
    return []


def write_history(snapshots: list[dict]) -> None:
    """Replace entire history (used for 'clear history')."""
    sb = _sb()
    if sb:
        try:
            # Delete all rows first, then insert if any
            sb.table("otimizador_snapshots").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            if snapshots:
                rows = [{
                    "ts": s["ts"],
                    "tag": s.get("tag", "manual"),
                    "money_invested": float(s.get("money_invested", 0)),
                    "total_value_eur": float(s["total_value_eur"]),
                    "assets": s.get("assets", []),
                } for s in snapshots[-MAX_SNAPSHOTS:]]
                sb.table("otimizador_snapshots").insert(rows).execute()
        except Exception:
            pass

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(
        {"version": 1, "snapshots": snapshots[-MAX_SNAPSHOTS:]},
        indent=2, ensure_ascii=False,
    ), encoding="utf-8")


def append_snapshot(snap: dict) -> None:
    sb = _sb()
    if sb:
        try:
            sb.table("otimizador_snapshots").insert({
                "ts": snap["ts"],
                "tag": snap.get("tag", "manual"),
                "money_invested": float(snap.get("money_invested", 0)),
                "total_value_eur": float(snap["total_value_eur"]),
                "assets": snap.get("assets", []),
            }).execute()
        except Exception:
            pass
    # local backup
    snaps = []
    if HISTORY_FILE.exists():
        try:
            snaps = json.loads(HISTORY_FILE.read_text(encoding="utf-8")).get("snapshots", [])
        except Exception:
            pass
    snaps.append(snap)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(
        {"version": 1, "snapshots": snaps[-MAX_SNAPSHOTS:]},
        indent=2, ensure_ascii=False,
    ), encoding="utf-8")


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
