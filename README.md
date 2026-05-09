# Polymarket Paper Trading Bot

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-API-D97757?logo=anthropic&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

An autonomous paper trading bot for [Polymarket](https://polymarket.com) prediction markets. It ingests news from multiple sources in real time, uses an LLM (Claude or a local Ollama model) to infer the true probability of each market outcome, identifies mispriced markets, and simulates trades under strict risk management rules — all without touching real money.

> **Paper trading only.** No real funds are ever moved. All positions are virtual, backed by a configurable starting balance.

---

## How it works

The bot runs a continuous pipeline on a scheduler:

```
NewsAPI / GDELT / Telegram
         │
         ▼
    News Ingestor       ← deduplication, scoring, priority queue
         │
         ▼
  Sentiment Analyzer    ← LLM (Claude / Ollama): probability + edge + confidence
         │
         ▼
   Market Scanner       ← Polymarket Gamma API: filters by volume, spread, time
         │
         ▼
   Decision Engine      ← crosses news analysis with market price; sizes the trade
         │
         ▼
    Risk Manager        ← validates against configured limits (runs on every trade)
         │
         ▼
    Paper Trader        ← simulates execution with slippage and partial fills
         │
    ┌────┴─────┐
    ▼          ▼
 SQLite     Discord
    │
    ▼
Excel Report  ·  Streamlit Dashboard
```

Every 5 minutes the bot scans live markets, fetches fresh news, and makes decisions. Open positions are re-evaluated every 15 minutes. A daily Excel report and Discord summary are sent at 23:55.

---

## Features

- **Multi-source news ingestion** — NewsAPI, GDELT (no key required), and Telegram public channels via Telethon MTProto. Fuzzy deduplication prevents analyzing the same story twice.
- **LLM-powered analysis** — Structured JSON prompts ask the model for: consensus YES probability, price edge, sentiment score, impact magnitude, recommendation, and timeframe. Supports both Anthropic Claude (API) and Ollama (local, free).
- **Market scanning** — Polls the Polymarket Gamma + CLOB APIs; filters out low-volume, wide-spread, or near-expiry markets. Parallel processing for speed.
- **Decision engine** — Computes trade size as a fraction of the Kelly criterion. Detects duplicate/opposite open positions, applies low-info mode with stricter thresholds.
- **Risk management** — Position size cap, max simultaneous positions, stop-loss, take-profit, max drawdown with optional auto-pause. Three time-based exit tiers prevent slots from being locked indefinitely.
- **Sports in-play mode** — Optional separate sub-strategy for live sports markets with its own size and confidence limits.
- **Backtester** — Runs the full pipeline against already-resolved Polymarket markets. Two modes: `current` (today's news, fast calibration) and `replay` (historical GDELT news, realistic).
- **Streamlit dashboard** — Real-time view of balance curve, open positions, P&L per trade, LLM decision log, and manual override controls.
- **Excel reports** — Daily `.xlsx` with 5 sheets: Executive Summary, Detailed Trades, LLM Analyses, Decisions Log, Balance Curve.
- **Discord notifications** — Rich embeds for trade open/close, stop-loss trigger, drawdown alert, bot pause/resume, and daily summary.
- **Full test suite** — pytest with unit tests for all major modules and integration fixtures.

---

## Tech stack

| Layer | Libraries |
|---|---|
| Data validation | Pydantic v2 |
| LLM | `anthropic` SDK, Ollama HTTP API |
| Scheduling | APScheduler |
| Database | SQLite (`sqlite3` stdlib, WAL mode) |
| News | `newsapi-python`, `gdeltdoc`, Telethon |
| Dashboard | Streamlit, Plotly |
| Reports | openpyxl, xlsxwriter |
| Notifications | `discord-webhook` |
| HTTP | `httpx`, `requests`, `tenacity` (retries) |
| Deduplication | `rapidfuzz` |
| Logging | Loguru |
| Testing | pytest, pytest-mock, pytest-asyncio |

---

## Quick start

**Prerequisites:** Python 3.10+, and at least one of: Anthropic API key, or [Ollama](https://ollama.com) running locally.

```bash
# 1. Clone and install
git clone https://github.com/icabrera0/Polymarket-paper-trader.git
cd Polymarket-paper-trader
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Configure secrets
copy .env.example .env       # then fill in your keys

# 3. Review settings (balance, risk %, intervals, LLM provider, etc.)
#    config/settings.yaml — no code changes needed

# 4. Run the bot
python main.py

# 5. Open the dashboard (separate terminal)
streamlit run dashboard.py
```

### Environment variables (`.env`)

```
ANTHROPIC_API_KEY=sk-ant-...          # required if provider = anthropic
NEWSAPI_KEY=...                        # required if newsapi.enabled = true
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
TELEGRAM_API_ID=...                    # required if telegram.enabled = true
TELEGRAM_API_HASH=...
TELEGRAM_PHONE=+34...
```

GDELT is always available with no credentials required.

---

## Risk management

Defaults calibrated for a **€150** starting bankroll — fully adjustable in `settings.yaml`:

| Parameter | Default | Description |
|---|---|---|
| Max position size | 15% of balance | ~€22 per trade |
| Max open positions | 3 | |
| Min trade size | €5 | Below this the trade is skipped |
| Stop loss | −20% of entry | Automatic close |
| Take profit | +30% | Evaluated on each cycle |
| Max drawdown | 30% | Bot auto-pauses (configurable) |
| Min 24h market volume | $10,000 | Liquidity filter |
| Max bid/ask spread | 5 cents | Efficiency filter |
| Min price edge | 10% | Minimum gap between price and inferred probability |
| Time exit — tier 1 | 24 h | Tighten TP to +15% |
| Time exit — tier 2 | 48 h | Close immediately if profitable |
| Time exit — tier 3 | 72 h | Unconditional close |

---

## Project structure

```
├── main.py                  # Entry point
├── dashboard.py             # Streamlit dashboard
├── dev_runner.py            # Hot-reload wrapper for development
├── config/
│   └── settings.yaml        # All tuneable parameters
├── src/
│   ├── orchestrator.py      # Pipeline coordinator + APScheduler
│   ├── market_scanner.py    # Gamma/CLOB API client + filters
│   ├── news_ingestor.py     # Multi-source ingestion + deduplication
│   ├── sentiment_analyzer.py# LLM analysis (Claude / Ollama)
│   ├── decision_engine.py   # Trade sizing + entry logic
│   ├── risk_manager.py      # Limits, stop-loss, drawdown
│   ├── paper_trader.py      # Simulated execution
│   ├── report_generator.py  # Excel reports
│   ├── notification_system.py# Discord webhooks
│   ├── backtester.py        # Historical simulation
│   ├── database.py          # SQLite persistence
│   ├── models.py            # Pydantic data models
│   ├── config_loader.py     # YAML + .env loader
│   ├── llm_client.py        # Unified Anthropic/Ollama interface
│   └── ...                  # API clients (Gamma, CLOB, NewsAPI, GDELT, Telegram)
├── tests/                   # pytest suite
├── scripts/                 # Utility scripts (backtest runner, balance manager, etc.)
└── requirements.txt
```

---

## Disclaimer

This project is for **educational and research purposes only**. Paper trading results do not predict real trading performance. Polymarket may be restricted in your jurisdiction. The author accepts no liability for any use of this code.
