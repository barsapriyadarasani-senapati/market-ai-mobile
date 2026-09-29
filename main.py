from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os, math, json
import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
import requests
from datetime import datetime, timedelta

BASE = Path(__file__).resolve().parent
app = FastAPI(title="Market AI Mobile")

def sym(s):
    s = s.upper().strip()
    return s if s.endswith(".NS") else s + ".NS"

def history(symbol, period="1y", interval="1d"):
    d = yf.Ticker(sym(symbol)).history(period=period, interval=interval, auto_adjust=False)
    if d is None or d.empty:
        return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna(how="all")

def indicators(d):
    x=d.copy(); c=x.Close
    x["SMA20"]=c.rolling(20).mean()
    x["SMA50"]=c.rolling(50).mean()
    x["SMA200"]=c.rolling(200).mean()
    x["EMA20"]=c.ewm(span=20,adjust=False).mean()
    x["EMA50"]=c.ewm(span=50,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    x["RSI"]=100-(100/(1+(gain/loss.replace(0,np.nan))))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    x["MACD"]=e12-e26; x["MACDSignal"]=x.MACD.ewm(span=9,adjust=False).mean()
    tr=pd.concat([(x.High-x.Low),(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1)
    x["ATR"]=tr.rolling(14).mean()
    x["VOL20"]=x.Volume.rolling(20).mean()
    x["VOLR"]=x.Volume/x.VOL20
    mid=c.rolling(20).mean(); sd=c.rolling(20).std()
    x["BBU"]=mid+2*sd; x["BBL"]=mid-2*sd
    return x


def risk_calc(entry, stop, capital, risk_pct=1.0):
    entry=float(entry); stop=float(stop); capital=float(capital); risk_pct=float(risk_pct)
    per_share=abs(entry-stop)
    max_loss=capital*risk_pct/100
    qty=0 if per_share<=0 else math.floor(max_loss/per_share)
    return {"entry":entry,"stop":stop,"capital":capital,"riskPct":risk_pct,
            "riskPerShare":round(per_share,2),"maxLoss":round(max_loss,2),"maxQty":qty,
            "positionValue":round(qty*entry,2)}

def nse_json(url, referer="https://www.nseindia.com/"):
    headers={"User-Agent":"Mozilla/5.0 (Android 13; Mobile) AppleWebKit/537.36 Chrome/128 Mobile Safari/537.36",
             "Accept":"application/json,text/plain,*/*","Referer":referer}
    s=requests.Session()
    try:
        s.get("https://www.nseindia.com/",headers=headers,timeout=8)
        r=s.get(url,headers=headers,timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def safe(v):
    if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return None
    if isinstance(v,(np.integer,np.floating)): return float(v)
    return v

def analyze(d, info):
    if len(d)<60: return {"signal":"INSUFFICIENT DATA","score":0,"confidence":0,"reasons":[]}
    a=d.iloc[-1]; p=d.iloc[-2]; score=0; reasons=[]
    if a.Close>a.SMA50: score+=15; reasons.append("Price above 50-day average")
    else: score-=15; reasons.append("Price below 50-day average")
    if not pd.isna(a.SMA200):
        if a.Close>a.SMA200: score+=15; reasons.append("Price above 200-day average")
        else: score-=15; reasons.append("Price below 200-day average")
    r=float(a.RSI) if not pd.isna(a.RSI) else 50
    if 50<=r<=68: score+=15; reasons.append(f"RSI constructive ({r:.1f})")
    elif r>75: score-=10; reasons.append(f"RSI overheated ({r:.1f})")
    elif r<30: score+=5; reasons.append(f"RSI oversold ({r:.1f})")
    else: score-=5; reasons.append(f"RSI weak ({r:.1f})")
    if a.MACD> a.MACDSignal: score+=10; reasons.append("MACD above signal")
    else: score-=10; reasons.append("MACD below signal")
    vr=float(a.VOLR) if not pd.isna(a.VOLR) else 1
    if vr>1.5 and a.Close>p.Close: score+=10; reasons.append("Upside move has strong volume")
    pe=info.get("trailingPE"); roe=info.get("returnOnEquity"); debt=info.get("debtToEquity")
    if isinstance(pe,(int,float)):
        if 0<pe<25: score+=5; reasons.append(f"P/E {pe:.1f}")
        elif pe>60: score-=5; reasons.append(f"P/E elevated {pe:.1f}")
    if isinstance(roe,(int,float)) and roe>.15: score+=5; reasons.append(f"ROE {roe*100:.1f}%")
    if isinstance(debt,(int,float)) and debt>150: score-=5; reasons.append(f"Debt/equity {debt:.0f}%")
    score=max(-100,min(100,score))
    signal="BUY SETUP" if score>=45 else ("SELL / AVOID SETUP" if score<=-35 else "HOLD / WAIT")
    return {"signal":signal,"score":score,"confidence":min(95,max(5,int(50+abs(score)*.45))),
            "rsi":round(r,2),"atr":safe(a.ATR),"volRatio":safe(a.VOLR),"reasons":reasons}

@app.get("/api/quote")
def quote(symbol: str):
    s=sym(symbol); d=indicators(history(s,"1y")); info=yf.Ticker(s).info or {}
    if d.empty: return {"error":"No data"}
    a=d.iloc[-1]; p=d.iloc[-2]
    an=analyze(d,info)
    return {"symbol":s.replace(".NS",""),"price":safe(a.Close),"changePct":safe((a.Close/p.Close-1)*100),
            "high52":safe(d.Close.tail(252).max()),"low52":safe(d.Close.tail(252).min()),
            "analysis":an,
            "fundamentals":{k:safe(info.get(k)) for k in ["marketCap","trailingPE","forwardPE","priceToBook","returnOnEquity","debtToEquity","dividendYield"]}}

@app.get("/api/chart")
def chart(symbol: str, period: str="1y"):
    d=indicators(history(symbol,period))
    if d.empty:return {"error":"No data"}
    out=[]
    for idx,r in d.iterrows():
        out.append({"date":idx.strftime("%Y-%m-%d"),**{k:safe(r.get(k)) for k in ["Open","High","Low","Close","Volume","SMA20","SMA50","SMA200","EMA20","EMA50","RSI","MACD","MACDSignal","ATR","BBU","BBL"]}})
    return out

@app.get("/api/news")
def news(symbol: str):
    s=sym(symbol).replace(".NS","")
    feed=feedparser.parse(f"https://news.google.com/rss/search?q={s}%20stock%20India&hl=en-IN&gl=IN&ceid=IN:en")
    return [{"title":e.get("title",""),"link":e.get("link",""),"published":e.get("published","")} for e in feed.entries[:15]]

@app.get("/api/screener")
def screener(symbols: str="RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK,SBIN,ITC,LT,BHARTIARTL"):
    result=[]
    for s in [x.strip() for x in symbols.split(",") if x.strip()][:30]:
        try:
            d=indicators(history(s,"6mo")); info=yf.Ticker(sym(s)).info or {}
            if d.empty: continue
            a=d.iloc[-1]; an=analyze(d,info)
            result.append({"symbol":s.upper(),"price":safe(a.Close),"score":an["score"],"signal":an["signal"],"rsi":an.get("rsi")})
        except Exception: pass
    return sorted(result,key=lambda x:x["score"],reverse=True)

@app.get("/api/backtest")
def backtest(symbol: str):
    d=indicators(history(symbol,"5y")).dropna(subset=["SMA50","SMA200","RSI","MACDSignal"])
    if len(d)<100:return {"error":"Not enough data"}
    cash=100000.; shares=0; trades=0
    for i in range(1,len(d)):
        r=d.iloc[i]
        score=0
        score += 15 if r.Close>r.SMA50 else -15
        score += 15 if r.Close>r.SMA200 else -15
        score += 15 if 50<=r.RSI<=68 else (-10 if r.RSI>75 else (-5 if r.RSI>=30 else 5))
        score += 10 if r.MACD>r.MACDSignal else -10
        if score>=45 and shares==0:
            shares=cash/r.Close; cash=0; trades+=1
        elif score<=-35 and shares>0:
            cash=shares*r.Close; shares=0; trades+=1
    final=cash+(shares*d.iloc[-1].Close if shares else 0)
    return {"initial":100000,"final":round(float(final),2),"returnPct":round((final/100000-1)*100,2),"trades":trades}

@app.post("/api/chat")
def chat(payload: dict):
    q=payload.get("question",""); symbol=payload.get("symbol","RELIANCE")
    try:
        s=sym(symbol); d=indicators(history(s,"1y")); info=yf.Ticker(s).info or {}; an=analyze(d,info)
    except Exception:
        return {"answer":"Market data could not be loaded right now."}
    key=os.getenv("OPENAI_API_KEY")
    if key:
        try:
            from openai import OpenAI
            client=OpenAI(api_key=key)
            prompt=f"""You are a cautious Indian stock research assistant.
Do not claim certainty, guaranteed profit, or zero-error prediction.
Stock {s}; signal {an}; fundamentals { {k:info.get(k) for k in ['trailingPE','returnOnEquity','debtToEquity']} }.
Question: {q}
Explain the evidence and risks. Keep the answer concise."""
            r=client.responses.create(model="gpt-5-mini",input=prompt)
            return {"answer":r.output_text}
        except Exception: pass
    return {"answer":f"{s.replace('.NS','')}: {an['signal']} with score {an['score']}/100 and confidence {an['confidence']}%. Key factors: {', '.join(an['reasons'][:5])}. This is research support, not a guaranteed prediction."}


@app.get("/api/risk")
def risk(entry: float, stop: float, capital: float, riskPct: float=1.0):
    return risk_calc(entry,stop,capital,riskPct)

@app.get("/api/earnings")
def earnings(symbol: str):
    try:
        t=yf.Ticker(sym(symbol))
        cal=t.calendar
        if isinstance(cal,pd.DataFrame):
            return {"calendar":cal.to_dict()}
        if isinstance(cal,dict):
            return {k:(v.isoformat() if hasattr(v,"isoformat") else v) for k,v in cal.items()}
        return {"calendar":None}
    except Exception:
        return {"calendar":None}

@app.get("/api/options")
def options(symbol: str):
    try:
        t=yf.Ticker(sym(symbol))
        exps=list(t.options or [])
        if not exps:return {"expiries":[],"chains":[]}
        out=[]
        for exp in exps[:3]:
            try:
                oc=t.option_chain(exp)
                calls=oc.calls.nlargest(12,"openInterest")[["strike","lastPrice","bid","ask","volume","openInterest","impliedVolatility"]].to_dict("records")
                puts=oc.puts.nlargest(12,"openInterest")[["strike","lastPrice","bid","ask","volume","openInterest","impliedVolatility"]].to_dict("records")
                out.append({"expiry":exp,"calls":calls,"puts":puts})
            except Exception: pass
        return {"expiries":exps,"chains":out}
    except Exception:
        return {"expiries":[],"chains":[]}

@app.get("/api/fii-dii")
def fii_dii():
    # Public NSE endpoint; may be unavailable if NSE changes access controls.
    data=nse_json("https://www.nseindia.com/api/fiidiiTradeReact")
    if data is None:
        return {"available":False,"message":"FII/DII feed unavailable right now."}
    return {"available":True,"data":data}

@app.get("/api/market-status")
def market_status():
    data=nse_json("https://www.nseindia.com/api/marketStatus")
    return data if data is not None else {"marketState":[]}


# Serve frontend
app.mount("/static", StaticFiles(directory=str(BASE)), name="static")

@app.get("/")
def root():
    return FileResponse(BASE/"index.html")
