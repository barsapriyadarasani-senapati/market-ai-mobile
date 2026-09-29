# Market AI Mobile V3

Mobile-friendly Flask stock-analysis dashboard for Indian markets.

## Features
- NSE/BSE-style ticker input (NSE is default)
- SMA20, SMA50, EMA20
- RSI, MACD
- Bollinger Bands
- ATR
- volatility
- support/resistance
- transparent 6-point technical signal
- ATR-based stop/target reference
- simple no-lookahead daily backtest
- built-in technical chat assistant
- Render-ready

## Run
pip install -r requirements.txt
python app.py

## Render
Build: `pip install -r requirements.txt`
Start: `gunicorn app:app`

This software is for research/education. It does not guarantee returns and is not financial advice.
