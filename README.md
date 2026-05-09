# Polymarket Paper Trading Bot

Automated **paper trading** bot for Polymarket that analyzes news in real time, detects discrepancies between market prices and inferred event probabilities, and simulates trades applying strict risk management rules. Generates a daily Excel report with the details of each trade.

> ⚠️ **Phase 1: paper trading only.** No real operations are executed on Polymarket. Everything is simulation with a virtual balance.

---

## Architecture

The bot is composed of 10 modules orchestrated in a pipeline:

1. **NEWS_INGESTOR** — News ingestion (NewsAPI + GDELT), deduplication and priority queue.
2. **SENTIMENT_ANALYZER** — Sentiment and impact analysis via Claude API.
3. **MARKET_SCANNER** — Polling the Polymarket Gamma API; filtering by volume, spread and time remaining.
4. **DECISION_ENGINE** — Cross-referencing news and markets; entry decision based on rules + LLM.
5. **PAPER_TRADER** — Order execution simulator with slippage and partial fills.
6. **RISK_MANAGER** — Validation of each trade against configured limits (cross-cutting).
7. **REPORT_GENERATOR** — Daily Excel report with 5 sheets (summary, trades, news, metrics, evolution).
8. **NOTIFICATION_SYSTEM** — Discord alerts on each trade, stop loss, drawdown and daily summary.
9. **ORCHESTRATOR** — `main.py` that coordinates all modules and schedules tasks.
10. **BACKTESTING** — Historical replay of news and markets to validate the strategy.

### Data flow

```
News  →  Ingestor  →  Analyzer  →  Scanner  →  Decision Engine
                                                       │
                                                       ▼
Excel  ◄──  Report  ◄──  SQLite  ◄──  Paper Trader  ◄  Risk Manager
                                                       │
                                                       ▼
                                                    Discord
```

---

## Project structure

```
Polymarket/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── settings.yaml          # All tuneable configuration
├── src/                       # Modules (to be created in subsequent parts)
├── data/                      # SQLite + news cache
├── reports/                   # Daily Excel files (YYYY-MM-DD_report.xlsx)
├── logs/                      # Logs of each LLM decision
└── tests/                     # Unit tests
```

---

## Initial setup

### 1. Requirements

- **Python 3.10+** (required for some modern type annotations)
- Account at [Anthropic Console](https://console.anthropic.com) with API key
- Free account at [NewsAPI.org](https://newsapi.org/register)
- Discord webhook in the server where you want notifications
- (GDELT does not require an API key)

### 2. Installation

```bash
cd E:\AI\Polymarket
python -m venv venv
venv\Scripts\activate           # On Windows
pip install -r requirements.txt
```

### 3. Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
copy .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
NEWSAPI_KEY=xxxxxxxx
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxxxx/xxxxx
```

### 4. Fine-tuning

Review `config/settings.yaml` to tune parameters without touching code (balance, risk %, intervals, etc.).

---

## Risk management rules

Calibrated for an initial bankroll of **€150**:

| Parameter | Value | Notes |
|---|---|---|
| Initial balance | €150 | Configurable in `settings.yaml` |
| Maximum position size | 15% of balance | ~€22 |
| Maximum simultaneous positions | 3 | |
| Minimum trade size | €5 | Avoids irrelevant operations |
| Stop loss | -20% of entry value | Automatic close |
| Take profit | +30% if confidence decreases | Evaluated close |
| Maximum drawdown | 30% of bankroll (€45) | Bot pauses automatically |
| Minimum 24h market volume | $10,000 | Reasonable liquidity |
| Maximum spread | 5 cents | Avoids inefficient markets |
| Minimum edge price vs probability | 10% | To enter |

---

## Execution (available after completing all modules)

```bash
python -m src.main
```

The bot:
- Scans Polymarket every 5 minutes
- Polls NewsAPI every 5 minutes and GDELT every 15
- Generates the daily Excel report at 23:55 (Madrid time)
- Sends Discord notifications on each relevant event

---

## Development plan

Incremental build, module by module:

- [x] **Part 1** — Base structure + configuration
- [ ] **Part 2** — RISK_MANAGER
- [ ] **Part 3** — MARKET_SCANNER
- [ ] **Part 4** — NEWS_INGESTOR
- [ ] **Part 5** — SENTIMENT_ANALYZER
- [ ] **Part 6** — DECISION_ENGINE
- [ ] **Part 7** — PAPER_TRADER + SQLite
- [ ] **Part 8** — REPORT_GENERATOR (Excel)
- [ ] **Part 9** — NOTIFICATION_SYSTEM (Discord) + ORCHESTRATOR
- [ ] **Part 10** — BACKTESTING

---

## Disclaimer

This software is exclusively for **educational and research purposes**. Paper trading does not guarantee returns in real operations. Polymarket may have legal restrictions in your jurisdiction — verify before any real use. The author takes no responsibility for losses resulting from the use of this code.
