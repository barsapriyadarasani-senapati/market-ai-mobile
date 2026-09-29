# Market AI Mobile

A mobile-friendly FastAPI web app for Indian-market technical analysis.

## Features
- NSE/BSE-style symbol lookup through Yahoo Finance market data
- Price and chart view
- SMA20, SMA50, EMA20
- RSI, MACD, Bollinger Bands and ATR calculations
- Rule-based BUY/HOLD/SELL signal
- Volatility/risk snapshot
- Simple historical backtest
- Screener API
- AI-style chat endpoint
- Render-ready deployment
- No API key required for the initial market-data endpoint

## Render
Build:
`pip install -r requirements.txt`

Start:
`uvicorn main:app --host 0.0.0.0 --port $PORT`

The signal is an analytical rule, not a guarantee of future returns. Market data can be delayed or unavailable.
