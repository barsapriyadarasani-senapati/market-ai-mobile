import os, math
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf

app = Flask(__name__)

def normalize_symbol(symbol):
    s = (symbol or "").strip().upper()
    if not s:
        return ""
    if "." in s:
        return s
    # Common Indian symbols default to NSE.
    return s + ".NS"

def fetch_data(symbol, period="1y", interval="1d"):
    ticker = normalize_symbol(symbol)
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return ticker, pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open","High","Low","Close","Volume"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    df = df[needed].dropna(subset=["Close"]).copy()
    return ticker, df

def indicators(df):
    x = df.copy()
    c = x["Close"]
    h, l = x["High"], x["Low"]

    x["SMA20"] = c.rolling(20).mean()
    x["SMA50"] = c.rolling(50).mean()
    x["EMA20"] = c.ewm(span=20, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
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

    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()

    ret = c.pct_change()
    x["Volatility20"] = ret.rolling(20).std() * np.sqrt(252) * 100
    x["Return1D"] = ret * 100
    return x.dropna().copy()

def build_signal(x):
    if x.empty:
        return {"signal":"N/A","score":0,"confidence":0,"reasons":[]}
    r = x.iloc[-1]
    score = 0
    reasons = []

    tests = [
        (r["Close"] > r["SMA20"], 1, "Price above SMA20"),
        (r["Close"] > r["SMA50"], 1, "Price above SMA50"),
        (r["Close"] > r["EMA20"], 1, "Price above EMA20"),
        (r["RSI"] >= 50, 1, "RSI above 50"),
        (r["MACD"] > r["MACDSignal"], 1, "MACD bullish"),
        (r["Close"] > r["BBMid"], 1, "Price above Bollinger midline"),
    ]
    for ok, pts, reason in tests:
        if bool(ok):
            score += pts
            reasons.append(reason)

    # A transparent 6-point technical score.
    if score >= 5:
        signal = "BUY"
    elif score <= 1:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = round(min(95, 50 + abs(score - 3) * 12.5))
    return {"signal": signal, "score": score, "confidence": confidence, "reasons": reasons}

def levels(x):
    r = x.iloc[-1]
    price = float(r["Close"])
    atr = float(r["ATR14"]) if pd.notna(r["ATR14"]) else price * .02
    support = float(x["Low"].tail(20).min())
    resistance = float(x["High"].tail(20).max())
    stop = max(0.01, price - 1.5 * atr)
    target = price + 2.0 * atr
    rr = (target-price) / max(price-stop, 1e-9)
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
        "close": [round(float(v),2) for v in y["Close"]],
        "sma20": [None if pd.isna(v) else round(float(v),2) for v in y["SMA20"]],
        "sma50": [None if pd.isna(v) else round(float(v),2) for v in y["SMA50"]],
        "bb_upper": [None if pd.isna(v) else round(float(v),2) for v in y["BBUpper"]],
        "bb_lower": [None if pd.isna(v) else round(float(v),2) for v in y["BBLower"]],
    }

def analyze(symbol):
    ticker, raw = fetch_data(symbol)
    if raw.empty:
        return {"error": f"No market data found for {symbol}. Try an NSE symbol such as TCS, RELIANCE, INFY or TCIEXP."}
    x = indicators(raw)
    if x.empty:
        return {"error": "Not enough historical data to calculate indicators."}
    sig = build_signal(x)
    lev = levels(x)
    r = x.iloc[-1]
    return {
        "ticker": ticker,
        "symbol": symbol.upper(),
        "price": round(float(r["Close"]),2),
        "rsi": round(float(r["RSI"]),2),
        "macd": round(float(r["MACD"]),4),
        "macd_signal": round(float(r["MACDSignal"]),4),
        "volatility": round(float(r["Volatility20"]),2),
        "trend": "Bullish" if r["Close"] > r["SMA50"] else "Bearish",
        **sig, **lev, "chart": serialize_chart(x),
        "updated": datetime.now().strftime("%d %b %Y, %I:%M %p")
    }

@app.route("/")
def home():
    return open("index.html", encoding="utf-8").read()

@app.post("/api/analyze")
def api_analyze():
    data = request.get_json(silent=True) or {}
    return jsonify(analyze(data.get("symbol","")))

@app.post("/api/chat")
def api_chat():
    data = request.get_json(silent=True) or {}
    symbol = data.get("symbol","")
    question = (data.get("message","") or "").strip().lower()
    result = analyze(symbol)
    if "error" in result:
        return jsonify({"reply": result["error"]})
    if "stop" in question or "sl" in question:
        reply = f"For {result['symbol'].upper()}, ATR-based stop is around ₹{result['stop_loss']} and the recent 20-day support is ₹{result['support']}. These are technical reference levels, not guarantees."
    elif "target" in question:
        reply = f"ATR-based technical target is around ₹{result['target']}; recent 20-day resistance is ₹{result['resistance']}."
    elif "why" in question or "signal" in question:
        reply = f"The current technical signal is {result['signal']} with score {result['score']}/6. " + ("; ".join(result["reasons"]) if result["reasons"] else "Most tracked conditions are not bullish.")
    elif "risk" in question:
        reply = f"20-day annualized volatility is about {result['volatility']}%. Risk/reward from the ATR reference levels is about {result['risk_reward']}:1."
    else:
        reply = f"{result['symbol'].upper()} is ₹{result['price']}, RSI {result['rsi']}, trend {result['trend']}, technical signal {result['signal']} ({result['score']}/6). Ask about signal, stop-loss, target, support, resistance or risk."
    return jsonify({"reply": reply})

@app.post("/api/backtest")
def api_backtest():
    data = request.get_json(silent=True) or {}
    ticker, raw = fetch_data(data.get("symbol",""), period="5y")
    if raw.empty:
        return jsonify({"error":"No historical data found."})
    x = indicators(raw)
    if len(x) < 60:
        return jsonify({"error":"Not enough data for backtest."})
    # Simple no-lookahead daily strategy: position is based only on prior day's indicators.
    long_cond = (
        (x["Close"] > x["SMA20"]) &
        (x["SMA20"] > x["SMA50"]) &
        (x["RSI"] > 50) &
        (x["MACD"] > x["MACDSignal"])
    )
    pos = long_cond.shift(1).fillna(False).astype(int)
    daily = x["Close"].pct_change().fillna(0)
    strat = pos * daily
    equity = (1 + strat).cumprod()
    bh = (1 + daily).cumprod()
    total = (equity.iloc[-1]-1)*100
    bh_total = (bh.iloc[-1]-1)*100
    dd = equity / equity.cummax() - 1
    return jsonify({
        "ticker": ticker, "strategy_return": round(float(total),2),
        "buy_hold_return": round(float(bh_total),2),
        "max_drawdown": round(float(dd.min()*100),2),
        "trades": int(pos.diff().abs().sum()/2),
        "message":"Backtest uses next-day execution to reduce look-ahead bias; results are historical and not predictive."
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
