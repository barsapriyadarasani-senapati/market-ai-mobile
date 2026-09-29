# Market AI Mobile — NSE Research PWA

Mobile-first stock research application designed for Indian/NSE users.

## Features
- Mobile responsive dashboard
- NSE stock lookup
- Watchlist
- Candlestick chart
- SMA/EMA
- RSI
- MACD
- Bollinger Bands
- ATR and volatility
- Volume / relative volume
- Support/resistance approximation
- Fundamental snapshot
- Transparent multi-factor BUY / HOLD / SELL research signal
- Risk score and position-size calculator
- News feed
- Backtesting of the signal rules
- Portfolio tracker stored locally on the phone/browser
- Price alerts stored locally
- AI research chat through optional OpenAI API
- PWA manifest + service worker
- Dark/light mobile UI
- Offline shell caching

## Run locally

Backend:
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Android/Termux: python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open:
`http://127.0.0.1:8000`

## AI
Set `OPENAI_API_KEY` on the server if you want AI chat.
Without it, the app uses a deterministic research assistant.

## Deploy for phone use
Deploy the `backend` to a Python host and serve the `frontend` from the same host or a static host.
For an actual installable Android APK, wrap this PWA with a trusted Android WebView/TWA after deployment.

## Important
This is a research/decision-support application. It cannot guarantee zero errors, profits, or correct predictions.
Signals are generated from measurable data and should be verified before trading.


## V2 additions
- FII/DII feed integration with graceful fallback
- Market status
- Options chain snapshot
- Earnings calendar endpoint
- Risk-based position sizing
- Browser-local price alert storage
- Expanded mobile navigation
- Portfolio/watchlist remains local to device
- Backtesting endpoint
- AI chat with deterministic fallback

## Production data architecture
For a serious production deployment, replace the development yfinance/public feeds with properly licensed market-data and broker APIs. NSE states that real-time and other market-data products are supplied under data-use agreements and commercial licensing terms. See NSE's Market Data Policy and Real Time Data pages.

## Regulatory note
If this application is offered to other people as investment advice/research services, obtain Indian securities-law advice before launch. SEBI's framework assigns responsibility for AI outputs and investor-data security to regulated entities using AI, and requires relevant disclosures/records in advisory contexts.
