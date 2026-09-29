import os
import math
from typing import Optional
import httpx
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Market AI Mobile", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(ROOT, "index.html")

def nse_symbol(symbol: str) -> str:
    s = symbol.upper().strip()
    if not s.endswith(".NS") and not s.endswith(".BO"):
        s += ".NS"
    return s

async def yahoo_chart(symbol: str, period="6mo", interval="1d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{nse_symbol(symbol)}"
    params = {"range": period, "interval": interval, "includePrePost": "false", "events": "div,splits"}
    async with httpx.AsyncClient(timeout=12, headers={"User-Agent":"Mozilla/5.0"}) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()["chart"]["result"][0]
    q = data["indicators"]["quote"][0]
    ts = data.get("timestamp", [])
    rows = []
    for i, t in enumerate(ts):
        rows.append({
            "date": pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%d"),
            "open": q["open"][i], "high": q["high"][i],
            "low": q["low"][i], "close": q["close"][i],
            "volume": q["volume"][i]
        })
    return pd.DataFrame(rows).dropna(subset=["close"])

def indicators(df):
    x = df.copy()
    c = x.close
    x["sma20"] = c.rolling(20).mean()
    x["sma50"] = c.rolling(50).mean()
    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["signal"] = x.macd.ewm(span=9, adjust=False).mean()
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    x["bb_upper"] = mid + 2*sd
    x["bb_lower"] = mid - 2*sd
    tr = pd.concat([(x.high-x.low), (x.high-c.shift()).abs(), (x.low-c.shift()).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    return x

def signal(row):
    score = 0
    reasons = []
    if pd.notna(row.sma20) and row.close > row.sma20: score += 1; reasons.append("Price above SMA20")
    else: reasons.append("Price below SMA20")
    if pd.notna(row.sma50) and row.close > row.sma50: score += 1; reasons.append("Price above SMA50")
    if pd.notna(row.ema20) and row.close > row.ema20: score += 1; reasons.append("Price above EMA20")
    if pd.notna(row.rsi):
        if row.rsi < 30: score += 2; reasons.append("RSI oversold")
        elif row.rsi > 70: score -= 2; reasons.append("RSI overbought")
        else: reasons.append(f"RSI neutral ({row.rsi:.1f})")
    if pd.notna(row.macd) and pd.notna(row.signal):
        if row.macd > row.signal: score += 1; reasons.append("MACD bullish")
        else: score -= 1; reasons.append("MACD bearish")
    label = "BUY" if score >= 3 else ("SELL" if score <= -2 else "HOLD")
    return label, score, reasons

@app.get("/")
async def home():
    return FileResponse(INDEX)

@app.get("/api/quote")
async def quote(symbol: str = Query(...)):
    try:
        df = indicators(await yahoo_chart(symbol, "5d", "1d"))
        r = df.iloc[-1]
        prev = df.iloc[-2].close if len(df) > 1 else r.close
        change = r.close - prev
        return {"symbol": symbol.upper(), "price": round(float(r.close),2),
                "change": round(float(change),2),
                "change_pct": round(float(change/prev*100),2) if prev else 0}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

@app.get("/api/chart")
async def chart(symbol: str = Query(...), period: str="6mo"):
    try:
        df = indicators(await yahoo_chart(symbol, period, "1d"))
        rows = df.tail(250).replace({np.nan: None}).to_dict("records")
        return {"symbol": symbol.upper(), "data": rows}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

@app.get("/api/analysis")
async def analysis(symbol: str = Query(...)):
    try:
        df = indicators(await yahoo_chart(symbol, "1y", "1d"))
        r = df.iloc[-1]
        label, score, reasons = signal(r)
        return {"symbol":symbol.upper(),"signal":label,"score":score,
                "price":round(float(r.close),2),
                "rsi":None if pd.isna(r.rsi) else round(float(r.rsi),2),
                "sma20":None if pd.isna(r.sma20) else round(float(r.sma20),2),
                "sma50":None if pd.isna(r.sma50) else round(float(r.sma50),2),
                "ema20":None if pd.isna(r.ema20) else round(float(r.ema20),2),
                "macd":None if pd.isna(r.macd) else round(float(r.macd),3),
                "atr":None if pd.isna(r.atr) else round(float(r.atr),2),
                "reasons":reasons}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

@app.get("/api/backtest")
async def backtest(symbol: str=Query(...), period: str="2y"):
    try:
        df = indicators(await yahoo_chart(symbol, period, "1d")).dropna(subset=["sma20","sma50"])
        position = 0
        entry = 0
        trades = []
        equity = 1.0
        for _, r in df.iterrows():
            buy = r.close > r.sma20 and r.sma20 > r.sma50 and r.rsi < 70
            sell = r.close < r.sma20 or (pd.notna(r.rsi) and r.rsi > 70)
            if position == 0 and buy:
                position, entry = 1, r.close
            elif position == 1 and sell:
                ret = r.close/entry - 1
                equity *= (1+ret)
                trades.append(ret)
                position = 0
        if position:
            ret = df.iloc[-1].close/entry - 1
            equity *= (1+ret)
            trades.append(ret)
        return {"symbol":symbol.upper(),"total_return_pct":round((equity-1)*100,2),
                "trades":len(trades),
                "win_rate_pct":round(sum(x>0 for x in trades)/len(trades)*100,1) if trades else 0}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

@app.get("/api/screener")
async def screener():
    symbols = ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","ITC","LT","BHARTIARTL","TATAMOTORS"]
    out=[]
    for s in symbols:
        try:
            a=await analysis(s)
            if isinstance(a, dict) and "signal" in a:
                out.append(a)
        except Exception:
            pass
    return {"results":out}

@app.get("/api/market-status")
async def market_status():
    return {"market":"NSE/BSE","status":"Live data endpoint available","note":"Market hours and holidays can affect quote freshness."}

@app.get("/api/chat")
async def chat(message: str=Query(...), symbol: Optional[str]=None):
    text=message.lower()
    if symbol:
        try:
            a=await analysis(symbol)
            return {"reply":f"{symbol.upper()}: {a['signal']} signal from the rule-based technical engine. Score {a['score']}. RSI {a['rsi']}. This is an analytical signal, not guaranteed future performance.","analysis":a}
        except Exception as e:
            return {"reply":f"I couldn't fetch {symbol.upper()} right now: {e}"}
    return {"reply":"Ask me about a stock, e.g. 'Analyse RELIANCE' or enter a symbol and use Analyse."}

@app.get("/api/risk")
async def risk(symbol: str=Query(...)):
    try:
        df=await yahoo_chart(symbol,"6mo","1d")
        ret=df.close.pct_change().dropna()
        vol=float(ret.std()*math.sqrt(252)*100)
        return {"symbol":symbol.upper(),"annualized_volatility_pct":round(vol,2),
                "risk":"High" if vol>35 else ("Medium" if vol>20 else "Lower")}
    except Exception as e:
        return JSONResponse({"error":str(e)},status_code=502)
