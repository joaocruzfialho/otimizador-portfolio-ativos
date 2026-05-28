from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf


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
                cur = t.info.get("currency")
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
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(list(tickers), period=period, auto_adjust=True,
                          progress=False, threads=True)
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
