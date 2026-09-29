import os
import math
import time
from datetime import datetime
from functools import lru_cache

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf

app = Flask(__name__)

# Simple server-side cache. This prevents repeated Analyse taps from
# immediately hitting Yahoo Finance again.
CACHE_SECONDS = 600  # 10 minutes
_data_cache = {}

def normalize_symbol(symbol):
    s = (symbol or "").strip().upper()
    if not s:
        return ""
    return s if "." in s else s + ".NS"

def _cached(symbol, period, interval):
    key = (symbol, period, interval)
    item = _data_cache.get(key)
    if item and (time.time() - item["time"] < CACHE_SECONDS):
        return item["data"].copy()
    return None

def _save_cache(symbol, period, interval, df):
    _data_cache[(symbol, period, interval)] = {
        "time": time.time(),
        "data": df.copy()
    }

def fetch_data(symbol, period="1y", interval="1d"):
    ticker = normalize_symbol(symbol)
    if not ticker:
        return "", pd.DataFrame(), "Please enter a stock symbol."

    cached = _cached(ticker, period, interval)
    if cached is not None:
        return ticker, cached, None

    last_error = None

    # Only a small number of retries. The important part is that a rate-limit
    # response becomes a normal JSON error instead of crashing the worker.
    for attempt in range(2):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df is None or df.empty:
                last_error = "Yahoo Finance returned no data."
                break

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            needed = ["Open", "High", "Low", "Close", "Volume"]
            for c in needed:
                if c not in df.columns:
                    df[c] = np.nan

            df = df[needed].dropna(subset=["Close"]).copy()

            if df.empty:
                last_error = "No usable price data was returned."
                break

            _save_cache(ticker, period, interval, df)
            return ticker, df, None

        except Exception as exc:
            last_error = str(exc)
            # Do not hammer a rate-limited endpoint.
            if "RateLimit" in last_error or "Too Many Requests" in last_error:
                break
            if attempt == 0:
                time.sleep(1)

    return ticker, pd.DataFrame(), (
        "Market-data provider is temporarily rate-limiting this app. "
        "Please wait a few minutes and try again. "
        f"Details: {last_error or 'unknown error'}"
    )

def indicators(df):
    x = df.copy()
    c = x["Close"]
    h, l = x["High"], x["Low"]

    x["SMA20"] = c.rolling(20).mean()
    x["SMA50"] = c.rolling(50).mean()
    x["EMA20"] = c.ewm(span=20, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = 100 - (100 / (1 + rs))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACDSignal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACDHist"] = x["MACD"] - x["MACDSignal"]

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    x["BBMid"] = mid
    x["BBUpper"] = mid + 2 * std
    x["BBLower"] = mid - 2 * std

    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1
    ).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()

    ret = c.pct_change()
    x["Volatility20"] = ret.rolling(20).std() * np.sqrt(252) * 100
    x["Return1D"] = ret * 100

    return x.dropna().copy()

def build_signal(x):
    if x.empty:
        return {"signal": "N/A", "score": 0, "confidence": 0, "reasons": []}

    r = x.iloc[-1]
    score = 0
    reasons = []

    tests = [
        (r["Close"] > r["SMA20"], "Price above SMA20"),
        (r["Close"] > r["SMA50"], "Price above SMA50"),
        (r["Close"] > r["EMA20"], "Price above EMA20"),
        (r["RSI"] >= 50, "RSI above 50"),
        (r["MACD"] > r["MACDSignal"], "MACD bullish"),
        (r["Close"] > r["BBMid"], "Price above Bollinger midline"),
    ]

    for ok, reason in tests:
        if bool(ok):
            score += 1
            reasons.append(reason)

    if score >= 5:
        signal = "BUY"
    elif score <= 1:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = round(min(95, 50 + abs(score - 3) * 12.5))
    return {
        "signal": signal,
        "score": score,
        "confidence": confidence,
        "reasons": reasons,
    }

def levels(x):
    r = x.iloc[-1]
    price = float(r["Close"])
    atr = float(r["ATR14"]) if pd.notna(r["ATR14"]) else price * 0.02
    support = float(x["Low"].tail(20).min())
    resistance = float(x["High"].tail(20).max())
    stop = max(0.01, price - 1.5 * atr)
    target = price + 2.0 * atr
    rr = (target - price) / max(price - stop, 1e-9)

    return {
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "atr": round(atr, 2),
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round(rr, 2),
    }

def serialize_chart(x):
    y = x.tail(180)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in y.index],
        "close": [round(float(v), 2) for v in y["Close"]],
        "sma20": [None if pd.isna(v) else round(float(v), 2) for v in y["SMA20"]],
        "sma50": [None if pd.isna(v) else round(float(v), 2) for v in y["SMA50"]],
        "bb_upper": [None if pd.isna(v) else round(float(v), 2) for v in y["BBUpper"]],
        "bb_lower": [None if pd.isna(v) else round(float(v), 2) for v in y["BBLower"]],
    }

def analyze(symbol, period="1y"):
    ticker, raw, error = fetch_data(symbol, period=period)
    if error:
        return {"error": error}

    x = indicators(raw)
    if x.empty:
        return {"error": "Not enough historical data to calculate indicators."}

    sig = build_signal(x)
    lev = levels(x)
    r = x.iloc[-1]

    return {
        "ticker": ticker,
        "symbol": symbol.upper(),
        "price": round(float(r["Close"]), 2),
        "rsi": round(float(r["RSI"]), 2),
        "macd": round(float(r["MACD"]), 4),
        "macd_signal": round(float(r["MACDSignal"]), 4),
        "volatility": round(float(r["Volatility20"]), 2),
        "trend": "Bullish" if r["Close"] > r["SMA50"] else "Bearish",
        **sig,
        **lev,
        "chart": serialize_chart(x),
        "updated": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }

@app.route("/")
def home():
    try:
        with open("index.html", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return jsonify({"error": f"Frontend file error: {exc}"}), 500

@app.post("/api/analyze")
def api_analyze():
    try:
        data = request.get_json(silent=True) or {}
        result = analyze(data.get("symbol", ""))
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Analyze endpoint failed")
        return jsonify({
            "error": "Analysis failed safely. Please try again later.",
            "details": str(exc)
        }), 500

@app.post("/api/chat")
def api_chat():
    try:
        data = request.get_json(silent=True) or {}
        symbol = data.get("symbol", "")
        question = (data.get("message", "") or "").strip().lower()
        result = analyze(symbol)

        if "error" in result:
            return jsonify({"reply": result["error"]})

        if "stop" in question or "sl" in question:
            reply = (
                f"For {result['symbol'].upper()}, ATR-based stop is around "
                f"₹{result['stop_loss']} and recent 20-day support is "
                f"₹{result['support']}. These are technical reference levels, "
                "not guarantees."
            )
        elif "target" in question:
            reply = (
                f"ATR-based technical target is around ₹{result['target']}; "
                f"recent 20-day resistance is ₹{result['resistance']}."
            )
        elif "why" in question or "signal" in question:
            reply = (
                f"The current technical signal is {result['signal']} with "
                f"score {result['score']}/6. "
                + ("; ".join(result["reasons"])
                   if result["reasons"] else
                   "Most tracked conditions are not bullish.")
            )
        elif "risk" in question:
            reply = (
                f"20-day annualized volatility is about {result['volatility']}%. "
                f"Risk/reward from the ATR reference levels is about "
                f"{result['risk_reward']}:1."
            )
        else:
            reply = (
                f"{result['symbol'].upper()} is ₹{result['price']}, "
                f"RSI {result['rsi']}, trend {result['trend']}, "
                f"technical signal {result['signal']} ({result['score']}/6). "
                "Ask about signal, stop-loss, target, support, resistance or risk."
            )

        return jsonify({"reply": reply})
    except Exception as exc:
        app.logger.exception("Chat endpoint failed")
        return jsonify({
            "reply": "Chat analysis failed safely. Please try again later.",
            "details": str(exc)
        }), 500

@app.post("/api/backtest")
def api_backtest():
    try:
        data = request.get_json(silent=True) or {}
        ticker, raw, error = fetch_data(data.get("symbol", ""), period="5y")

        if error:
            return jsonify({"error": error})

        x = indicators(raw)
        if len(x) < 60:
            return jsonify({"error": "Not enough data for backtest."})

        long_cond = (
            (x["Close"] > x["SMA20"])
            & (x["SMA20"] > x["SMA50"])
            & (x["RSI"] > 50)
            & (x["MACD"] > x["MACDSignal"])
        )

        pos = long_cond.shift(1).fillna(False).astype(int)
        daily = x["Close"].pct_change().fillna(0)
        strat = pos * daily
        equity = (1 + strat).cumprod()
        bh = (1 + daily).cumprod()

        total = (equity.iloc[-1] - 1) * 100
        bh_total = (bh.iloc[-1] - 1) * 100
        dd = equity / equity.cummax() - 1

        return jsonify({
            "ticker": ticker,
            "strategy_return": round(float(total), 2),
            "buy_hold_return": round(float(bh_total), 2),
            "max_drawdown": round(float(dd.min() * 100), 2),
            "trades": int(pos.diff().abs().sum() / 2),
            "message": (
                "Backtest uses next-day execution to reduce look-ahead bias; "
                "results are historical and not predictive."
            ),
        })
    except Exception as exc:
        app.logger.exception("Backtest endpoint failed")
        return jsonify({
            "error": "Backtest failed safely. Please try again later.",
            "details": str(exc)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    
