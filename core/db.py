from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
SQLITE_FILE = DATA_DIR / "portfolio.db"

ASSET_TYPES = ["ETF", "Ação", "Obrigação", "Commodity", "Crypto", "Outro"]
EDITOR_COLS = ["Ticker", "Percentagem Alvo (%)", "Quantidade Detida"]
FULL_COLS = [
    "Ticker", "Nome", "Tipo", "Bolsa",
    "Percentagem Alvo (%)", "Quantidade Detida", "Preço Médio (€)",
]

_DEFAULT = [
    ("VWCE.DE", "Vanguard FTSE All-World",     "ETF",       "XETRA",         60.0, 10.0,  0.0),
    ("AGGH.MI", "iShares Core Global Agg Bond", "ETF",       "Milan",         25.0, 50.0,  0.0),
    ("SGLD.MI", "Invesco Physical Gold",        "Commodity", "Milan",         10.0,  5.0,  0.0),
    ("BTC-EUR", "Bitcoin",                      "Crypto",    "Yahoo Finance",  5.0,  0.05, 0.0),
]

# ── Supabase singleton ─────────────────────────────────

_sb_client = None


def _sb():
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


# ── SQLite ─────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(SQLITE_FILE)
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS assets (
            ticker      TEXT PRIMARY KEY,
            name        TEXT NOT NULL DEFAULT '',
            asset_type  TEXT NOT NULL DEFAULT 'ETF',
            exchange    TEXT NOT NULL DEFAULT ''
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS target_allocations (
            ticker      TEXT PRIMARY KEY,
            target_pct  REAL NOT NULL DEFAULT 0,
            FOREIGN KEY (ticker) REFERENCES assets(ticker) ON DELETE CASCADE
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS user_positions (
            ticker         TEXT PRIMARY KEY,
            quantity       REAL NOT NULL DEFAULT 0,
            avg_buy_price  REAL,
            FOREIGN KEY (ticker) REFERENCES assets(ticker) ON DELETE CASCADE
        )""")
        if c.execute("SELECT COUNT(*) FROM assets").fetchone()[0] == 0:
            for ticker, name, atype, exchange, tgt, qty, avg in _DEFAULT:
                c.execute("INSERT OR IGNORE INTO assets VALUES (?,?,?,?)",
                          (ticker, name, atype, exchange))
                c.execute("INSERT OR IGNORE INTO target_allocations VALUES (?,?)",
                          (ticker, tgt))
                c.execute("INSERT OR IGNORE INTO user_positions VALUES (?,?,?)",
                          (ticker, qty, avg or None))


# ── Supabase helpers ───────────────────────────────────

def _sb_load_all(sb) -> list[dict] | None:
    """Returns list (possibly empty) on success, None on error."""
    try:
        ar = sb.table("otimizador_assets").select("*").order("ticker").execute()
        if not ar.data:
            return []
        alloc = {
            r["ticker"]: float(r["target_pct"])
            for r in (sb.table("otimizador_allocations").select("*").execute().data or [])
        }
        pos = {
            r["ticker"]: (float(r["quantity"]), r.get("avg_buy_price"))
            for r in (sb.table("otimizador_positions").select("*").execute().data or [])
        }
        result = []
        for a in ar.data:
            t = a["ticker"]
            qty, avg = pos.get(t, (0.0, None))
            result.append({
                "ticker": t,
                "name": a.get("name", ""),
                "asset_type": a.get("asset_type", "ETF"),
                "exchange": a.get("exchange", ""),
                "target_pct": alloc.get(t, 0.0),
                "quantity": qty,
                "avg_buy_price": float(avg) if avg is not None else 0.0,
            })
        return result
    except Exception:
        return None


def _sb_seed_from_jsonb(sb) -> None:
    """Seed normalized tables from legacy JSONB portfolio (one-time migration)."""
    try:
        resp = sb.table("otimizador_portfolio").select("assets").eq("id", 1).execute()
        if resp.data and resp.data[0].get("assets"):
            for a in resp.data[0]["assets"]:
                _sb_upsert_one(sb, a["ticker"], "", "ETF", "",
                               float(a["target_pct"]), float(a["quantity"]), 0.0)
    except Exception:
        pass


def _sb_upsert_one(sb, ticker, name, asset_type, exchange,
                   target_pct, quantity, avg_buy_price) -> None:
    sb.table("otimizador_assets").upsert(
        {"ticker": ticker, "name": name, "asset_type": asset_type, "exchange": exchange}
    ).execute()
    sb.table("otimizador_allocations").upsert(
        {"ticker": ticker, "target_pct": target_pct}
    ).execute()
    sb.table("otimizador_positions").upsert(
        {"ticker": ticker, "quantity": quantity, "avg_buy_price": avg_buy_price or None}
    ).execute()


def _sb_delete_one(sb, ticker: str) -> None:
    # ON DELETE CASCADE handles allocations + positions at the DB level
    sb.table("otimizador_assets").delete().eq("ticker", ticker).execute()


# ── Public API ─────────────────────────────────────────

def load_portfolio_full() -> pd.DataFrame:
    """Return all assets with full metadata (7 columns)."""
    sb = _sb()
    if sb:
        rows = _sb_load_all(sb)
        if rows is not None:
            if len(rows) == 0:
                _sb_seed_from_jsonb(sb)
                rows = _sb_load_all(sb) or []
            if rows:
                return pd.DataFrame([{
                    "Ticker": r["ticker"],
                    "Nome": r["name"],
                    "Tipo": r["asset_type"],
                    "Bolsa": r["exchange"],
                    "Percentagem Alvo (%)": r["target_pct"],
                    "Quantidade Detida": r["quantity"],
                    "Preço Médio (€)": r["avg_buy_price"],
                } for r in rows])[FULL_COLS]

    # Fallback: SQLite
    with _conn() as c:
        data = c.execute("""
            SELECT a.ticker, a.name, a.asset_type, a.exchange,
                   COALESCE(al.target_pct, 0), COALESCE(p.quantity, 0),
                   COALESCE(p.avg_buy_price, 0)
            FROM assets a
            LEFT JOIN target_allocations al ON al.ticker = a.ticker
            LEFT JOIN user_positions p ON p.ticker = a.ticker
            ORDER BY a.ticker
        """).fetchall()
    if not data:
        return pd.DataFrame(columns=FULL_COLS)
    return pd.DataFrame(data, columns=FULL_COLS)


def load_portfolio() -> pd.DataFrame:
    """Return EDITOR_COLS DataFrame (Ticker / Percentagem Alvo (%) / Quantidade Detida)."""
    full = load_portfolio_full()
    if full.empty:
        return pd.DataFrame([
            {"Ticker": t, "Percentagem Alvo (%)": tgt, "Quantidade Detida": qty}
            for t, _, _, _, tgt, qty, _ in _DEFAULT
        ])
    return full[EDITOR_COLS].reset_index(drop=True)


def get_tickers() -> list[str]:
    sb = _sb()
    if sb:
        try:
            resp = sb.table("otimizador_assets").select("ticker").order("ticker").execute()
            if resp.data:
                return [r["ticker"] for r in resp.data]
        except Exception:
            pass
    with _conn() as c:
        return [r[0] for r in
                c.execute("SELECT ticker FROM assets ORDER BY ticker").fetchall()]


def get_asset(ticker: str) -> dict | None:
    sb = _sb()
    if sb:
        try:
            ar = sb.table("otimizador_assets").select("*").eq("ticker", ticker).execute()
            if not ar.data:
                return None
            a = ar.data[0]
            al = (sb.table("otimizador_allocations")
                  .select("target_pct").eq("ticker", ticker).execute())
            po = (sb.table("otimizador_positions")
                  .select("quantity,avg_buy_price").eq("ticker", ticker).execute())
            return {
                "ticker": a["ticker"],
                "name": a.get("name", ""),
                "asset_type": a.get("asset_type", "ETF"),
                "exchange": a.get("exchange", ""),
                "target_pct": float(al.data[0]["target_pct"]) if al.data else 0.0,
                "quantity": float(po.data[0]["quantity"]) if po.data else 0.0,
                "avg_buy_price": float(po.data[0]["avg_buy_price"] or 0) if po.data else 0.0,
            }
        except Exception:
            pass
    with _conn() as c:
        row = c.execute("""
            SELECT a.ticker, a.name, a.asset_type, a.exchange,
                   COALESCE(al.target_pct, 0), COALESCE(p.quantity, 0),
                   COALESCE(p.avg_buy_price, 0)
            FROM assets a
            LEFT JOIN target_allocations al ON al.ticker = a.ticker
            LEFT JOIN user_positions p ON p.ticker = a.ticker
            WHERE a.ticker = ?
        """, (ticker,)).fetchone()
    if not row:
        return None
    return {
        "ticker": row[0], "name": row[1], "asset_type": row[2], "exchange": row[3],
        "target_pct": row[4], "quantity": row[5], "avg_buy_price": row[6],
    }


def upsert_asset(
    ticker: str, name: str = "", asset_type: str = "ETF",
    exchange: str = "", target_pct: float = 0.0,
    quantity: float = 0.0, avg_buy_price: float = 0.0,
) -> None:
    """Insert or update a single asset across all 3 tables."""
    ticker = ticker.strip()
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO assets VALUES (?,?,?,?)",
                  (ticker, name, asset_type, exchange))
        c.execute("INSERT OR REPLACE INTO target_allocations VALUES (?,?)",
                  (ticker, target_pct))
        c.execute("INSERT OR REPLACE INTO user_positions VALUES (?,?,?)",
                  (ticker, quantity, avg_buy_price or None))
    sb = _sb()
    if sb:
        try:
            _sb_upsert_one(sb, ticker, name, asset_type, exchange,
                           target_pct, quantity, avg_buy_price)
        except Exception:
            pass


def delete_asset(ticker: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM assets WHERE ticker = ?", (ticker,))
    sb = _sb()
    if sb:
        try:
            _sb_delete_one(sb, ticker)
        except Exception:
            pass


def save_portfolio_from_editor(df: pd.DataFrame) -> None:
    """Bulk-save from editor df (preserves name/type/exchange/avg_buy_price metadata)."""
    full = load_portfolio_full()
    meta = {} if full.empty else {row["Ticker"]: row for _, row in full.iterrows()}

    new_tickers: set[str] = set()
    for _, row in df.iterrows():
        t = str(row["Ticker"]).strip()
        if not t:
            continue
        new_tickers.add(t)
        ex = meta.get(t)
        upsert_asset(
            ticker=t,
            name=ex["Nome"] if ex is not None else "",
            asset_type=ex["Tipo"] if ex is not None else "ETF",
            exchange=ex["Bolsa"] if ex is not None else "",
            target_pct=float(row["Percentagem Alvo (%)"]),
            quantity=float(row["Quantidade Detida"]),
            avg_buy_price=float(ex["Preço Médio (€)"]) if ex is not None else 0.0,
        )
    for t in set(meta.keys()) - new_tickers:
        delete_asset(t)


def df_to_export_json(df: pd.DataFrame) -> str:
    return json.dumps({
        "version": 1,
        "portfolio": [{
            "ticker": str(row["Ticker"]),
            "target_pct": float(row["Percentagem Alvo (%)"]),
            "quantity": float(row["Quantidade Detida"]),
        } for _, row in df.iterrows() if str(row["Ticker"]).strip()],
    }, indent=2, ensure_ascii=False)


# Initialize SQLite on first import
try:
    init_db()
except Exception:
    pass
