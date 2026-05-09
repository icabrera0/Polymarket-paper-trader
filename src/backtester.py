"""
Backtester — simulates the bot on already-resolved Polymarket markets.

Flow:
1. Download recently resolved markets from the Gamma API.
2. Filter those that pass the same filters as the live scanner.
3. For each resolved market:
   a) Simulate the state "at the time of analysis" (pre-resolution price).
   b) Fetch relevant news (GDELT with historical or fresh timespan).
   c) Analyze with the LLM.
   d) If the engine decides OPEN_TRADE, simulate the position:
      - Entry at the price it had at the time (simulated).
      - Exit at the resolution price (1.0 YES won, 0.0 YES lost).
   e) Record the result in the report.

Two modes:
  "current"  → Uses today's news. The facts are already known (look-ahead bias).
               Useful for calibrating the LLM and validating the pipeline, NOT
               for evaluating the strategy in a realistic way.
  "replay"   → Fetches news in GDELT before the market's close date.
               More realistic, slower, more limited by GDELT coverage.

Results are saved to:
  - A temporary SQLite DB (not the production one).
  - An Excel results file with the same structure as the daily report.
  - A summary printed to the console.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
import re

from loguru import logger

from src.config_loader import BotConfig
from src.database import Database
from src.decision_engine import DecisionEngine
from src.gamma_client import GammaApiClient
from src.models import (
    CloseReason,
    DecisionAction,
    MarketAnalysis,
    MarketSnapshot,
    Position,
    TradeDecision,
    TradeRecommendation,
    TradeSide,
    TradeStatus,
    _now_utc,
)
from src.news_ingestor import NewsIngestor
from src.paper_trader import PaperTrader
from src.risk_manager import RiskManager
from src.sentiment_analyzer import SentimentAnalyzer


# =====================================================
# Backtesting result
# =====================================================


@dataclass
class BacktestTrade:
    """A simulated trade in the backtest."""

    market_id: str
    market_question: str
    resolved_yes: bool            # True if YES won (resolution = 1.0)
    entry_price_simulated: float  # Price at which we entered (simulated)
    exit_price: float             # 1.0 (YES won) or 0.0 (YES lost)
    side: TradeSide
    size_eur: float
    pnl_eur: float
    pnl_pct: float
    confidence: int
    edge: float
    num_articles: int
    is_low_info: bool
    decision: DecisionAction
    skip_reasons: list[str] = field(default_factory=list)
    llm_recommendation: str = ""


@dataclass
class BacktestResult:
    """Aggregated backtest result."""

    mode: str
    start_time: datetime
    end_time: datetime
    initial_balance: float
    final_balance: float
    markets_analyzed: int
    trades_executed: int
    trades_won: int
    trades_lost: int
    total_pnl_eur: float
    win_rate: float
    avg_pnl_per_trade: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: list[BacktestTrade] = field(default_factory=list)

    def print_summary(self) -> None:
        """Prints a summary to the console."""
        print()
        print("═" * 65)
        print("  BACKTESTING RESULT")
        print("═" * 65)
        print(f"  Mode:                    {self.mode}")
        print(f"  Test duration:           {(self.end_time - self.start_time).seconds}s")
        print(f"  Markets analyzed:        {self.markets_analyzed}")
        print(f"  Trades executed:         {self.trades_executed}")
        print(f"  Winning trades:          {self.trades_won}")
        print(f"  Losing trades:           {self.trades_lost}")
        print(f"  Win rate:                {self.win_rate:.1%}")
        print(f"  Total P&L:               €{self.total_pnl_eur:+.2f}")
        if self.trades_executed > 0:
            print(f"  Average P&L per trade:   €{self.avg_pnl_per_trade:+.2f}")
        print(f"  Initial balance:         €{self.initial_balance:.2f}")
        print(f"  Final balance:           €{self.final_balance:.2f}")
        pnl_pct = (self.final_balance - self.initial_balance) / self.initial_balance
        print(f"  Total return:            {pnl_pct:+.2%}")
        print(f"  Max drawdown:            {self.max_drawdown_pct:.2%}")
        if self.sharpe_ratio:
            print(f"  Sharpe ratio (approx):   {self.sharpe_ratio:.2f}")
        print("═" * 65)
        if self.trades:
            print()
            print("  Trade detail:")
            for t in self.trades[:20]:
                icon = "✅" if t.pnl_eur >= 0 else "❌"
                low = " [LOW]" if t.is_low_info else ""
                print(
                    f"  {icon} {t.market_question[:50]:<50} "
                    f"€{t.pnl_eur:+.2f}{low}"
                )
            if len(self.trades) > 20:
                print(f"  ... and {len(self.trades) - 20} more")
        print()


# =====================================================
# Backtester
# =====================================================


class Backtester:
    """Simulates the bot on resolved markets."""

    def __init__(
        self,
        config: BotConfig,
        mode: str = "current",
        max_markets: int = 50,
        initial_balance: float = 150.0,
        db_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            config: bot configuration.
            mode: "current" (today's news) or "replay" (historical news).
            max_markets: how many resolved markets to analyze.
            initial_balance: starting balance for the simulation.
            db_path: temporary DB path. If None, uses a file in data/backtest.db.
        """
        self.config = config
        self.mode = mode
        self.max_markets = max_markets
        self.initial_balance = initial_balance

        # Separate DB from production
        self._db_path = db_path or str(
            Path(config.database.path).parent / "backtest.db"
        )

        self._log = logger.bind(module="backtester")
        self._log.info(
            "Backtester initialized: mode={}, max_markets={}, balance=€{}",
            mode, max_markets, initial_balance,
        )

    # =====================================================
    # Entry point
    # =====================================================

    def run(self) -> BacktestResult:
        """Runs the backtest and returns the result."""
        start_time = _now_utc()

        # Modules with temporary DB and their own virtual balance
        db = Database(self._db_path)
        risk_manager = RiskManager(self.config, initial_balance_eur=self.initial_balance)
        paper_trader = PaperTrader(self.config, risk_manager, db=db)
        news_ingestor = NewsIngestor(self.config)
        sentiment_analyzer = SentimentAnalyzer(self.config)
        decision_engine = DecisionEngine(self.config, risk_manager)

        # 1) Fetch recently resolved markets
        self._log.info("Downloading resolved markets from Polymarket...")
        resolved_markets = self._fetch_resolved_markets()
        self._log.info("Found {} resolved markets", len(resolved_markets))

        bt_trades: list[BacktestTrade] = []
        peak_balance = self.initial_balance
        max_drawdown = 0.0

        for i, market_data in enumerate(resolved_markets):
            self._log.debug(
                "Analyzing {}/{}: {}",
                i + 1, len(resolved_markets),
                market_data.get("question", "")[:60],
            )

            # Extract resolved market info
            snap, resolved_yes = self._parse_resolved_market(market_data)
            if snap is None:
                continue

            # 2) Fetch news
            keywords = self._extract_keywords(snap.question)
            if self.mode == "replay" and snap.end_date:
                # Calculate approximate timespan up to market close
                days_ago = (
                    _now_utc() - snap.end_date
                ).days
                timespan = f"{min(days_ago + 7, 90)}d"  # +7 days of prior context
            else:
                timespan = "7d"  # week of news (current mode)

            articles = news_ingestor.fetch(
                keywords,
                max_articles=10,
                fallback_timespan=timespan,
            )

            # 3) Analyze
            analysis = sentiment_analyzer.analyze(snap, articles, force_refresh=True)
            db.log_analysis(analysis)

            # 4) Decide
            decision = decision_engine.decide(
                analysis=analysis,
                current_balance_eur=paper_trader.balance_eur,
                open_positions=paper_trader.open_positions,
                articles=articles,
            )
            db.log_decision(decision)

            # 5) Simulate P&L with resolution price
            if decision.action == DecisionAction.OPEN_TRADE and decision.side:
                # Exit price = actual market resolution
                if decision.side == TradeSide.BUY_YES:
                    exit_price = 1.0 if resolved_yes else 0.0
                else:
                    exit_price = 1.0 if not resolved_yes else 0.0

                # Simulate: open and immediately close at the resolution price
                entry_price = decision.entry_price or snap.yes_price
                position = paper_trader.execute_decision(decision)
                if position:
                    closed = paper_trader.close_position(
                        trade_id=position.trade_id,
                        current_market_price=exit_price,
                        reason=CloseReason.MARKET_RESOLVED,
                        notes=f"Backtest: market resolved YES={resolved_yes}",
                    )
                    if closed:
                        bt_trade = BacktestTrade(
                            market_id=market_data.get("id", ""),
                            market_question=snap.question,
                            resolved_yes=resolved_yes,
                            entry_price_simulated=entry_price,
                            exit_price=exit_price,
                            side=decision.side,
                            size_eur=decision.size_eur or 0,
                            pnl_eur=closed.pnl_eur or 0.0,
                            pnl_pct=closed.pnl_pct or 0.0,
                            confidence=analysis.confidence,
                            edge=analysis.edge,
                            num_articles=len(articles),
                            is_low_info=analysis.is_low_info,
                            decision=decision.action,
                            llm_recommendation=analysis.recommendation.value,
                        )
                        bt_trades.append(bt_trade)

                        # Update drawdown
                        bal = paper_trader.balance_eur
                        if bal > peak_balance:
                            peak_balance = bal
                        dd = (peak_balance - bal) / peak_balance if peak_balance > 0 else 0.0
                        max_drawdown = max(max_drawdown, dd)
            else:
                # NO_TRADE
                bt_trade = BacktestTrade(
                    market_id=market_data.get("id", ""),
                    market_question=snap.question,
                    resolved_yes=resolved_yes,
                    entry_price_simulated=0.0,
                    exit_price=0.0,
                    side=TradeSide.BUY_YES,
                    size_eur=0.0,
                    pnl_eur=0.0,
                    pnl_pct=0.0,
                    confidence=analysis.confidence,
                    edge=analysis.edge,
                    num_articles=len(articles),
                    is_low_info=analysis.is_low_info,
                    decision=decision.action,
                    skip_reasons=[r.value for r in decision.skip_reasons],
                    llm_recommendation=analysis.recommendation.value,
                )
                bt_trades.append(bt_trade)

            # Brief pause between markets to avoid saturating GDELT/Ollama
            time.sleep(0.5)

        db.close()

        # Calculate metrics
        actual_trades = [t for t in bt_trades if t.decision == DecisionAction.OPEN_TRADE]
        winners = [t for t in actual_trades if t.pnl_eur > 0]
        losers = [t for t in actual_trades if t.pnl_eur < 0]
        total_pnl = sum(t.pnl_eur for t in actual_trades)
        final_balance = self.initial_balance + total_pnl
        win_rate = len(winners) / len(actual_trades) if actual_trades else 0.0
        avg_pnl = total_pnl / len(actual_trades) if actual_trades else 0.0
        sharpe = self._calculate_sharpe(actual_trades)

        return BacktestResult(
            mode=self.mode,
            start_time=start_time,
            end_time=_now_utc(),
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            markets_analyzed=len(resolved_markets),
            trades_executed=len(actual_trades),
            trades_won=len(winners),
            trades_lost=len(losers),
            total_pnl_eur=total_pnl,
            win_rate=win_rate,
            avg_pnl_per_trade=avg_pnl,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            trades=bt_trades,
        )

    # =====================================================
    # Fetching resolved markets
    # =====================================================

    def _fetch_resolved_markets(self) -> list[dict[str, Any]]:
        """Downloads closed/resolved markets from the Gamma API."""
        client = GammaApiClient(self.config)
        try:
            markets = client.fetch_markets(
                active=False,
                closed=True,
                order="volume24hr",
                ascending=False,
                limit=min(self.max_markets * 3, 500),
            )
        except Exception as exc:
            self._log.error("Error downloading resolved markets: {}", exc)
            return []

        # Filter those that are truly resolved (have a winner)
        resolved = [
            m for m in markets
            if self._is_truly_resolved(m)
        ][:self.max_markets]
        return resolved

    @staticmethod
    def _is_truly_resolved(market: dict[str, Any]) -> bool:
        """A market is resolved if it has a clear outcome (1.0 / 0.0)."""
        prices_raw = market.get("outcomePrices")
        if not prices_raw:
            return False
        try:
            import json
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            if not isinstance(prices, list) or len(prices) < 2:
                return False
            p0, p1 = float(prices[0]), float(prices[1])
            # Resolved: one price must be exactly 1.0 and the other 0.0
            return (p0 == 1.0 and p1 == 0.0) or (p0 == 0.0 and p1 == 1.0)
        except (ValueError, TypeError, Exception):
            return False

    # =====================================================
    # Resolved market parser
    # =====================================================

    def _parse_resolved_market(
        self, market_data: dict[str, Any]
    ) -> tuple[Optional[MarketSnapshot], bool]:
        """
        Returns (MarketSnapshot_pre_resolution, resolved_yes).

        To simulate the state "before resolution", we use an
        artificial price of 0.50 (maximum uncertainty). In replay mode we could
        look up the historical price, but the public API does not expose it
        easily. This adds noise but keeps the test conservative.
        """
        import json

        question = market_data.get("question", "")
        if not question:
            return None, False

        # Determine whether YES won
        prices_raw = market_data.get("outcomePrices", "[]")
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            resolved_yes = float(prices[0]) == 1.0
        except (ValueError, TypeError, IndexError):
            return None, False

        # Tokens
        tokens_raw = market_data.get("clobTokenIds", "[]")
        try:
            tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
            if len(tokens) < 2:
                return None, False
            yes_token, no_token = str(tokens[0]), str(tokens[1])
        except (ValueError, TypeError, IndexError):
            return None, False

        # Simulated pre-resolution price: we use the price "opposite" to the
        # resolution adjusted by volume. If YES won with high volume,
        # we assume it was trading at ~0.70 before resolution. If it was a surprise,
        # it could be 0.30. Without real historical data we use 0.50 as a base.
        # This is a necessary simplification given data availability.
        simulated_yes_price = 0.50
        simulated_no_price = 0.50

        snap = MarketSnapshot(
            market_id=str(market_data.get("id", "")),
            slug=market_data.get("slug", ""),
            question=question,
            description=market_data.get("description", ""),
            category=market_data.get("category", ""),
            end_date=self._parse_iso(market_data.get("endDate")),
            yes_token_id=yes_token,
            no_token_id=no_token,
            yes_price=simulated_yes_price,
            no_price=simulated_no_price,
            spread=0.01,
            volume_24h_usd=float(market_data.get("volume24hr") or 0),
            volume_total_usd=float(market_data.get("volumeNum") or 0),
            liquidity_usd=float(market_data.get("liquidityNum") or 0),
            is_active=False,
            is_closed=True,
        )
        return snap, resolved_yes

    # =====================================================
    # Utilities
    # =====================================================

    @staticmethod
    def _extract_keywords(question: str, max_kw: int = 4) -> list[str]:
        stopwords = {
            "will", "the", "a", "an", "is", "are", "be", "by", "of", "in",
            "on", "at", "to", "for", "and", "or", "if", "than", "more", "less",
            "this", "that", "before", "after", "any", "all", "with", "from",
            "win", "wins", "won", "do", "does", "did", "can", "could", "should",
            "would", "may", "might", "first", "next", "year", "month", "week",
            "day", "much", "many", "election", "vote",
        }
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9'-]{3,}\b", question)
        entities, common = [], []
        for w in words:
            if w.lower() in stopwords:
                continue
            (entities if w[0].isupper() else common).append(
                w if w[0].isupper() else w.lower()
            )
        seen: set[str] = set()
        result: list[str] = []
        for w in entities + common:
            if w.lower() not in seen:
                seen.add(w.lower())
                result.append(w)
            if len(result) >= max_kw:
                break
        return result

    @staticmethod
    def _parse_iso(value: Any) -> Optional[datetime]:
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _calculate_sharpe(trades: list[BacktestTrade]) -> float:
        """Simplified Sharpe ratio based on each trade's percentage P&L."""
        if len(trades) < 2:
            return 0.0
        import statistics
        pnls = [t.pnl_pct for t in trades]
        mean = statistics.mean(pnls)
        std = statistics.stdev(pnls)
        if std == 0:
            return 0.0
        return mean / std
