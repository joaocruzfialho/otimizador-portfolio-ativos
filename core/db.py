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

# Dev user (local without Supabase)
_DEV_USER_ID = "00000000-0000-0000-0000-000000000001"

_DEFAULT = [
    ("VWCE.DE", "Vanguard FTSE All-World",      "ETF",       "XETRA",         60.0, 10.0,  0.0),
    ("AGGH.MI", "iShares Core Global Agg Bond",  "ETF",       "Milan",         25.0, 50.0,  0.0),
    ("SGLD.MI", "Invesco Physical Gold",         "Commodity", "Milan",         10.0,  5.0,  0.0),
    ("BTC-EUR", "Bitcoin",                       "Crypto",    "Yahoo Finance",  5.0,  0.05, 0.0),
]


# ── Supabase singleton ────────────────────────────────────────────────────

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


# ── SQLite (local dev / offline fallback) ─────────────────────────────────

def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(SQLITE_FILE)
    c.execute("PRAGMA foreign_keys = ON")
    return c


def _sqlite_needs_migration(c: sqlite3.Connection) -> bool:
    """Check if the SQLite schema predates multi-user (missing user_id column)."""
    try:
        cols = {row[1] for row in c.execute("PRAGMA table_info(assets)").fetchall()}
        return "user_id" not in cols
    except Exception:
        return False


def init_db() -> None:
    with _conn() as c:
        # Migrate old single-user schema by dropping tables (ephemeral cache)
        if _sqlite_needs_migration(c):
            c.execute("DROP TABLE IF EXISTS user_positions")
            c.execute("DROP TABLE IF EXISTS target_allocations")
            c.execute("DROP TABLE IF EXISTS assets")

        c.execute("""CREATE TABLE IF NOT EXISTS assets (
            user_id     TEXT NOT NULL DEFAULT '',
            ticker      TEXT NOT NULL,
            name        TEXT NOT NULL DEFAULT '',
            asset_type  TEXT NOT NULL DEFAULT 'ETF',
            exchange    TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (user_id, ticker)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS target_allocations (
            user_id    TEXT NOT NULL DEFAULT '',
            ticker     TEXT NOT NULL,
            target_pct REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, ticker),
            FOREIGN KEY (user_id, ticker) REFERENCES assets(user_id, ticker) ON DELETE CASCADE
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS user_positions (
            user_id        TEXT NOT NULL DEFAULT '',
            ticker         TEXT NOT NULL,
            quantity       REAL NOT NULL DEFAULT 0,
            avg_buy_price  REAL,
            PRIMARY KEY (user_id, ticker),
            FOREIGN KEY (user_id, ticker) REFERENCES assets(user_id, ticker) ON DELETE CASCADE
        )""")
        # Seed defaults for dev user if empty
        if c.execute("SELECT COUNT(*) FROM assets WHERE user_id=?",
                     (_DEV_USER_ID,)).fetchone()[0] == 0:
            for ticker, name, atype, exchange, tgt, qty, avg in _DEFAULT:
                c.execute("INSERT OR IGNORE INTO assets VALUES (?,?,?,?,?)",
                          (_DEV_USER_ID, ticker, name, atype, exchange))
                c.execute("INSERT OR IGNORE INTO target_allocations VALUES (?,?,?)",
                          (_DEV_USER_ID, ticker, tgt))
                c.execute("INSERT OR IGNORE INTO user_positions VALUES (?,?,?,?)",
                          (_DEV_USER_ID, ticker, qty, avg or None))


def _sqlite_read_all(user_id: str) -> list[tuple]:
    with _conn() as c:
        return c.execute("""
            SELECT a.ticker, a.name, a.asset_type, a.exchange,
                   COALESCE(al.target_pct, 0), COALESCE(p.quantity, 0),
                   COALESCE(p.avg_buy_price, 0)
            FROM assets a
            LEFT JOIN target_allocations al ON al.user_id=a.user_id AND al.ticker=a.ticker
            LEFT JOIN user_positions p ON p.user_id=a.user_id AND p.ticker=a.ticker
            WHERE a.user_id=?
            ORDER BY a.ticker
        """, (user_id,)).fetchall()


def _sqlite_upsert_one(user_id: str, ticker: str, name: str, asset_type: str,
                       exchange: str, target_pct: float, quantity: float,
                       avg_buy_price: float) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO assets VALUES (?,?,?,?,?)",
                  (user_id, ticker, name, asset_type, exchange))
        c.execute("INSERT OR REPLACE INTO target_allocations VALUES (?,?,?)",
                  (user_id, ticker, target_pct))
        c.execute("INSERT OR REPLACE INTO user_positions VALUES (?,?,?,?)",
                  (user_id, ticker, quantity, avg_buy_price or None))


# ── Supabase helpers ──────────────────────────────────────────────────────

def _sb_load_all(sb, user_id: str) -> list[dict] | None:
    try:
        ar = (sb.table("otimizador_assets")
              .select("*").eq("user_id", user_id).order("ticker").execute())
        if not ar.data:
            return []
        alloc = {
            r["ticker"]: float(r["target_pct"])
            for r in (sb.table("otimizador_allocations")
                      .select("ticker,target_pct").eq("user_id", user_id).execute().data or [])
        }
        pos = {
            r["ticker"]: (float(r["quantity"]), r.get("avg_buy_price"))
            for r in (sb.table("otimizador_positions")
                      .select("ticker,quantity,avg_buy_price").eq("user_id", user_id).execute().data or [])
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


def _sb_upsert_one(sb, user_id: str, ticker: str, name: str, asset_type: str,
                   exchange: str, target_pct: float, quantity: float,
                   avg_buy_price: float) -> None:
    sb.table("otimizador_assets").upsert(
        {"user_id": user_id, "ticker": ticker, "name": name,
         "asset_type": asset_type, "exchange": exchange},
        on_conflict="user_id,ticker",
    ).execute()
    sb.table("otimizador_allocations").upsert(
        {"user_id": user_id, "ticker": ticker, "target_pct": target_pct},
        on_conflict="user_id,ticker",
    ).execute()
    sb.table("otimizador_positions").upsert(
        {"user_id": user_id, "ticker": ticker, "quantity": quantity,
         "avg_buy_price": avg_buy_price or None},
        on_conflict="user_id,ticker",
    ).execute()


def _sb_delete_one(sb, user_id: str, ticker: str) -> None:
    sb.table("otimizador_assets").delete().eq("user_id", user_id).eq("ticker", ticker).execute()


# ── Public API ────────────────────────────────────────────────────────────

def load_portfolio_full(user_id: str) -> pd.DataFrame:
    """Return all assets for user_id (7 columns).

    Reads SQLite first (local, fast); only calls Supabase when SQLite is empty
    for this user — i.e. fresh Render container or first login.
    """
    sb = _sb()

    # With Supabase: try SQLite cache first, fall back to Supabase
    if sb:
        data = _sqlite_read_all(user_id)
        if not data:
            rows = _sb_load_all(sb, user_id)
            if rows:
                for r in rows:
                    _sqlite_upsert_one(user_id, r["ticker"], r["name"], r["asset_type"],
                                       r["exchange"], r["target_pct"], r["quantity"],
                                       r["avg_buy_price"])
                data = _sqlite_read_all(user_id)
        if data:
            return pd.DataFrame(data, columns=FULL_COLS)
        return pd.DataFrame(columns=FULL_COLS)

    # Local dev (no Supabase): read from SQLite directly
    data = _sqlite_read_all(user_id)
    if not data:
        return pd.DataFrame(columns=FULL_COLS)
    return pd.DataFrame(data, columns=FULL_COLS)


def load_portfolio(user_id: str) -> pd.DataFrame:
    full = load_portfolio_full(user_id)
    if full.empty:
        return pd.DataFrame([
            {"Ticker": t, "Percentagem Alvo (%)": tgt, "Quantidade Detida": qty}
            for t, _, _, _, tgt, qty, _ in _DEFAULT
        ])
    return full[EDITOR_COLS].reset_index(drop=True)


def get_tickers(user_id: str) -> list[str]:
    sb = _sb()
    if sb:
        try:
            r = (sb.table("otimizador_assets")
                 .select("ticker").eq("user_id", user_id).order("ticker").execute())
            if r.data:
                return [row["ticker"] for row in r.data]
        except Exception:
            pass
    with _conn() as c:
        return [row[0] for row in
                c.execute("SELECT ticker FROM assets WHERE user_id=? ORDER BY ticker",
                          (user_id,)).fetchall()]


def get_asset(ticker: str, user_id: str) -> dict | None:
    sb = _sb()
    if sb:
        try:
            ar = (sb.table("otimizador_assets")
                  .select("*").eq("user_id", user_id).eq("ticker", ticker).execute())
            if not ar.data:
                return None
            a = ar.data[0]
            al = (sb.table("otimizador_allocations")
                  .select("target_pct").eq("user_id", user_id).eq("ticker", ticker).execute())
            po = (sb.table("otimizador_positions")
                  .select("quantity,avg_buy_price").eq("user_id", user_id).eq("ticker", ticker).execute())
            return {
                "ticker": a["ticker"], "name": a.get("name", ""),
                "asset_type": a.get("asset_type", "ETF"), "exchange": a.get("exchange", ""),
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
            LEFT JOIN target_allocations al ON al.user_id=a.user_id AND al.ticker=a.ticker
            LEFT JOIN user_positions p ON p.user_id=a.user_id AND p.ticker=a.ticker
            WHERE a.user_id=? AND a.ticker=?
        """, (user_id, ticker)).fetchone()
    if not row:
        return None
    return {"ticker": row[0], "name": row[1], "asset_type": row[2], "exchange": row[3],
            "target_pct": row[4], "quantity": row[5], "avg_buy_price": row[6]}


def upsert_asset(ticker: str, name: str = "", asset_type: str = "ETF",
                 exchange: str = "", target_pct: float = 0.0,
                 quantity: float = 0.0, avg_buy_price: float = 0.0,
                 user_id: str = _DEV_USER_ID) -> None:
    ticker = ticker.strip()
    _sqlite_upsert_one(user_id, ticker, name, asset_type, exchange,
                       target_pct, quantity, avg_buy_price)
    sb = _sb()
    if sb:
        try:
            _sb_upsert_one(sb, user_id, ticker, name, asset_type, exchange,
                           target_pct, quantity, avg_buy_price)
        except Exception:
            pass


def delete_asset(ticker: str, user_id: str = _DEV_USER_ID) -> None:
    with _conn() as c:
        c.execute("DELETE FROM assets WHERE user_id=? AND ticker=?", (user_id, ticker))
    sb = _sb()
    if sb:
        try:
            _sb_delete_one(sb, user_id, ticker)
        except Exception:
            pass


def save_portfolio_from_editor(df: pd.DataFrame, user_id: str = _DEV_USER_ID) -> None:
    full = load_portfolio_full(user_id)
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
            user_id=user_id,
        )
    for t in set(meta.keys()) - new_tickers:
        delete_asset(t, user_id)


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
