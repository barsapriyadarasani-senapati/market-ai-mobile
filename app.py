import os
import json
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

CACHE_SECONDS = 600
_data_cache = {}

def normalize_symbol(symbol):
    s = (symbol or "").strip().upper()
    if not s:
        return ""
    return s if "." in s else s + ".NS"

def period_seconds(period):
    return {
        "1mo": 31 * 86400,
        "3mo": 93 * 86400,
        "6mo": 186 * 86400,
        "1y": 366 * 86400,
        "2y": 2 * 366 * 86400,
        "5y": 5 * 366 * 86400,
    }.get(period, 366 * 86400)

def cache_get(key):
    item = _data_cache.get(key)
    if item and time.time() - item["time"] < CACHE_SECONDS:
        return item["data"].copy()
    return None

def cache_put(key, df):
    _data_cache[key] = {"time": time.time(), "data": df.copy()}

def fetch_data(symbol, period="1y", interval="1d"):
    ticker = normalize_symbol(symbol)
    if not ticker:
        return "", pd.DataFrame(), "Please enter a stock symbol."

    key = (ticker, period, interval)
    cached = cache_get(key)
    if cached is not None:
        return ticker, cached, None

    now = int(time.time())
    start = now - period_seconds(period)
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + ticker
        + f"?period1={start}&period2={now}&interval={interval}"
        + "&events=history&includeAdjustedClose=true"
    )

    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Android; Mobile) AppleWebKit/537.36 "
                          "Chrome/128.0 Mobile Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
        },
    )

    try:
        with urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))

        result = payload.get("chart", {}).get("result")
        if not result:
            err = payload.get("chart", {}).get("error")
            msg = err.get("description") if isinstance(err, dict) else None
            return ticker, pd.DataFrame(), msg or "Market-data provider returned no data."

        result = result[0]
        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]

        if not timestamps:
            return ticker, pd.DataFrame(), "No historical market data was returned."

        df = pd.DataFrame({
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        }, index=pd.to_datetime(timestamps, unit="s", utc=True))

        df.index = df.index.tz_convert(None)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])

        if df.empty:
            return ticker, pd.DataFrame(), "No usable closing prices were returned."

        cache_put(key, df)
        return ticker, df, None

    except HTTPError as exc:
        if exc.code == 429:
            return ticker, pd.DataFrame(), (
                "Market-data provider is temporarily rate-limiting this server. "
                "Please wait a few minutes before trying again."
            )
        return ticker, pd.DataFrame(), f"Market-data HTTP error {exc.code}."
    except URLError:
        return ticker, pd.DataFrame(), "Could not reach the market-data provider."
    except Exception as exc:
        return ticker, pd.DataFrame(), f"Market-data error: {exc}"

def indicators(df):
    x = df.copy()
    c, h, l = x["Close"], x["High"], x["Low"]

    x["SMA20"] = c.rolling(20).mean()
    x["SMA50"] = c.rolling(50).mean()
    x["EMA20"] = c.ewm(span=20, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = 100 - 100 / (1 + rs)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACDSignal"] = x["MACD"].ewm(span=9, adjust=False).mean()

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    x["BBMid"] = mid
    x["BBUpper"] = mid + 2 * std
    x["BBLower"] = mid - 2 * std

    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()

    x["Volatility20"] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100
    return x.dropna().copy()

def build_signal(x):
    r = x.iloc[-1]
    tests = [
        (r["Close"] > r["SMA20"], "Price above SMA20"),
        (r["Close"] > r["SMA50"], "Price above SMA50"),
        (r["Close"] > r["EMA20"], "Price above EMA20"),
        (r["RSI"] >= 50, "RSI above 50"),
        (r["MACD"] > r["MACDSignal"], "MACD bullish"),
        (r["Close"] > r["BBMid"], "Price above Bollinger midline"),
    ]
    reasons = [reason for ok, reason in tests if bool(ok)]
    score = len(reasons)
    signal = "BUY" if score >= 5 else "SELL" if score <= 1 else "HOLD"
    confidence = round(min(95, 50 + abs(score - 3) * 12.5))
    return {"signal": signal, "score": score, "confidence": confidence, "reasons": reasons}

def levels(x):
    r = x.iloc[-1]
    price = float(r["Close"])
    atr = float(r["ATR14"])
    support = float(x["Low"].tail(20).min())
    resistance = float(x["High"].tail(20).max())
    stop = max(0.01, price - 1.5 * atr)
    target = price + 2 * atr
    rr = (target - price) / max(price - stop, 1e-9)
    return {
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "atr": round(atr, 2),
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round(rr, 2),
    }

def chart_data(x):
    y = x.tail(180)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in y.index],
        "close": [round(float(v), 2) for v in y["Close"]],
        "sma20": [round(float(v), 2) for v in y["SMA20"]],
        "sma50": [round(float(v), 2) for v in y["SMA50"]],
        "bb_upper": [round(float(v), 2) for v in y["BBUpper"]],
        "bb_lower": [round(float(v), 2) for v in y["BBLower"]],
    }

def analyze(symbol, period="1y"):
    ticker, raw, error = fetch_data(symbol, period)
    if error:
        return {"error": error}
    x = indicators(raw)
    if x.empty:
        return {"error": "Not enough historical data to calculate indicators."}

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
        **build_signal(x),
        **levels(x),
        "chart": chart_data(x),
        "updated": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "data_source": "Yahoo Finance chart data",
    }

@app.route("/")
def home():
    try:
        return open("index.html", encoding="utf-8").read()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.post("/api/analyze")
def api_analyze():
    try:
        data = request.get_json(silent=True) or {}
        return jsonify(analyze(data.get("symbol", "")))
    except Exception as exc:
        app.logger.exception("Analyze failed")
        return jsonify({"error": f"Analysis failed safely: {exc}"}), 500

@app.post("/api/chat")
def api_chat():
    try:
        data = request.get_json(silent=True) or {}
        symbol = data.get("symbol", "")
        question = (data.get("message", "") or "").lower().strip()
        result = analyze(symbol)

        if "error" in result:
            return jsonify({"reply": result["error"]})

        if "stop" in question or "sl" in question:
            reply = f"ATR-based stop reference: ₹{result['stop_loss']}. Recent 20-day support: ₹{result['support']}."
        elif "target" in question:
            reply = f"ATR-based target reference: ₹{result['target']}. Recent 20-day resistance: ₹{result['resistance']}."
        elif "risk" in question:
            reply = f"20-day annualized volatility: {result['volatility']}%. Reference risk/reward: {result['risk_reward']}:1."
        elif "why" in question or "signal" in question:
            reply = f"Signal: {result['signal']} ({result['score']}/6). " + "; ".join(result["reasons"])
        else:
            reply = f"{result['symbol']} is ₹{result['price']}, RSI {result['rsi']}, trend {result['trend']}, signal {result['signal']} ({result['score']}/6)."
        return jsonify({"reply": reply})
    except Exception as exc:
        return jsonify({"reply": f"Chat failed safely: {exc}"}), 500

@app.post("/api/backtest")
def api_backtest():
    try:
        data = request.get_json(silent=True) or {}
        ticker, raw, error = fetch_data(data.get("symbol", ""), "5y")
        if error:
            return jsonify({"error": error})

        x = indicators(raw)
        if len(x) < 60:
            return jsonify({"error": "Not enough historical data for a 5-year backtest."})

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
        dd = equity / equity.cummax() - 1

        return jsonify({
            "ticker": ticker,
            "strategy_return": round(float((equity.iloc[-1]-1)*100), 2),
            "buy_hold_return": round(float((bh.iloc[-1]-1)*100), 2),
            "max_drawdown": round(float(dd.min()*100), 2),
            "trades": int(pos.diff().abs().sum()/2),
            "message": "Historical backtest only; results are not predictive."
        })
    except Exception as exc:
        return jsonify({"error": f"Backtest failed safely: {exc}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
    
