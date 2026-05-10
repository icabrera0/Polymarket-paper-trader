"""
Orchestrator — the conductor of the bot.

Connects all modules in a pipeline and schedules them with APScheduler:

  Every 5 min  → main cycle: scan → keywords → news → analyze → decide → execute
  Every 15 min → re-evaluate open positions with updated prices
  Every 23:55  → generate daily Excel report + summary notification

The Orchestrator manages the full lifecycle:
  - Startup: initializes all modules, restores state from the DB,
    verifies Ollama if applicable, sends startup ping to Discord.
  - Loop: executes scheduled jobs indefinitely.
  - Shutdown: captures Ctrl+C / SIGTERM, closes the DB cleanly and
    sends a shutdown notification.

Error philosophy:
  - Errors inside a job are logged but do NOT stop the bot.
  - Only a CRITICAL error at startup can abort.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

_HOT_RELOAD_FLAG = Path(__file__).resolve().parent.parent / "data" / "hot_reload.flag"
_HOT_RELOAD_EXIT_CODE = 42
from loguru import logger

from src.clob_client import ClobApiClient
from src.config_loader import BotConfig, load_config, validate_secrets
from src.database import Database
from src.decision_engine import DecisionEngine
from src.market_scanner import MarketScanner
from src.models import CloseReason, DecisionAction, TradeDecision, TradeRecommendation, TradeSide
from src.news_ingestor import NewsIngestor
from src.social_ingestor import SocialIngestor
from src.notification_system import NotificationSystem
from src.paper_trader import PaperTrader
from src.report_generator import ReportGenerator
from src.risk_manager import RiskManager
from src.compound import CompoundEngine
from src.sentiment_analyzer import SentimentAnalyzer


# =====================================================
# Keyword extractor (same as the live script)
# =====================================================


import re

_STOPWORDS = {
    "will", "the", "a", "an", "is", "are", "be", "by", "of", "in", "on",
    "at", "to", "for", "and", "or", "if", "than", "more", "less", "this",
    "that", "before", "after", "any", "all", "with", "from", "into", "as",
    "have", "has", "had", "win", "wins", "won", "do", "does", "did",
    "can", "could", "should", "would", "may", "might", "first", "next",
    "year", "month", "week", "day", "much", "many", "make", "makes",
    "made", "election", "vote",
}


def extract_keywords(question: str, max_kw: int = 4) -> list[str]:
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9'-]{3,}\b", question)
    entities, common = [], []
    for w in words:
        if w.lower() in _STOPWORDS:
            continue
        (entities if w[0].isupper() else common).append(w if w[0].isupper() else w.lower())
    seen: set[str] = set()
    result: list[str] = []
    for w in entities + common:
        if w.lower() not in seen:
            seen.add(w.lower())
            result.append(w)
        if len(result) >= max_kw:
            break
    return result


# =====================================================
# Orchestrator
# =====================================================


class Orchestrator:
    """Coordinates all modules and schedules jobs."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._log = logger.bind(module="orchestrator")
        self._running = False
        self._scheduler: Optional[BackgroundScheduler] = None

        # Prevents concurrent main cycles (APScheduler doesn't track the direct
        # call in start(), so without this both can run simultaneously).
        self._cycle_lock = threading.Lock()
        # Lock that serialises position-closing across threads (price monitor
        # and position review must not evaluate the same position simultaneously)
        self._position_lock = threading.Lock()
        self._price_monitor_running = False
        # Drawdown notification cooldown — only alert once per 24h
        self._last_drawdown_alert_ts: float = 0.0
        # Dead-token zombie tracking: throttle Gamma resolution checks (once/60s)
        self._resolution_last_check: dict[str, float] = {}
        # When each zombie token was FIRST detected dead — never reset on
        # intermediate Gamma responses, only cleared on successful close.
        self._zombie_since: dict[str, float] = {}
        # Kill switch: set True when dashboard writes kill_switch_active=true
        self._kill_switch_active: bool = False

        # --- Modules ---
        self.clob_client = ClobApiClient()
        self.db = Database(config.database.path)
        self.risk_manager = RiskManager(config)
        self.paper_trader = PaperTrader(config, self.risk_manager, self.db)
        self.market_scanner = MarketScanner(config)
        self.news_ingestor = NewsIngestor(config)
        self.social_ingestor = SocialIngestor(config) if getattr(config, "social", None) and config.social.enabled else None
        self.compound = CompoundEngine(config, self.db)
        self.sentiment_analyzer = SentimentAnalyzer(config, compound=self.compound)
        self.decision_engine = DecisionEngine(config, self.risk_manager)
        self.report_generator = ReportGenerator(config, self.db)
        self.notifications = NotificationSystem(config)

        # Sports in-play: track trade_ids separately from main positions
        self._sports_trade_ids: set[str] = set()

        self._log.info(
            "Orchestrator initialized. Balance: €{:.2f}, "
            "Open positions: {}",
            self.paper_trader.balance_eur,
            self.paper_trader.num_open_positions,
        )

    # =====================================================
    # Lifecycle
    # =====================================================

    def start(self) -> None:
        """Starts the bot and blocks until it is stopped."""
        self._log.info("=" * 60)
        self._log.info("  POLYMARKET PAPER TRADING BOT — STARTING UP")
        self._log.info("=" * 60)

        # Verify Ollama if applicable
        if self.config.llm.provider == "ollama":
            from src.llm_client import OllamaClient, OllamaUnavailable
            try:
                OllamaClient(self.config).verify_setup()
            except OllamaUnavailable as exc:
                self._log.error("Ollama not available: {}", exc)
                self._log.error(
                    "Run 'ollama serve' and make sure the model "
                    "is downloaded. Aborting."
                )
                sys.exit(1)

        # Notify startup
        self.notifications.send_text(
            f"🤖 **Bot started** | Balance: €{self.paper_trader.balance_eur:.2f} "
            f"| Open positions: {self.paper_trader.num_open_positions}"
        )

        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Schedule jobs
        self._scheduler = BackgroundScheduler(
            timezone=self.config.app.timezone
        )

        # Job 1: Main cycle (every N seconds)
        self._scheduler.add_job(
            self._run_main_cycle,
            trigger=IntervalTrigger(
                seconds=self.config.polymarket.scan_interval_seconds
            ),
            id="main_cycle",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=60,
        )

        # Job 2: Re-evaluate positions (every 15 min)
        self._scheduler.add_job(
            self._run_position_review,
            trigger=IntervalTrigger(
                minutes=self.config.decision.reevaluate_open_positions_minutes
            ),
            id="position_review",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=120,
        )

        # Job 3: Daily report (time configured in settings.yaml)
        report_time = self.config.reports.generation_time  # "23:55"
        h, m = report_time.split(":")
        self._scheduler.add_job(
            self._run_daily_report,
            trigger=CronTrigger(hour=int(h), minute=int(m)),
            id="daily_report",
            max_instances=1,
        )

        self._scheduler.start()
        self._running = True

        # Start the price monitor as a daemon thread (10-second polling, no
        # scheduler overhead) — replaces the old 60-second APScheduler job
        self._price_monitor_running = True
        _pm_thread = threading.Thread(
            target=self._price_monitor_loop,
            name="price-monitor",
            daemon=True,
        )
        _pm_thread.start()

        self._log.info(
            "Scheduler started. Cycle every {}s, review every {}min, "
            "SL/TP monitor every 10s, report at {}",
            self.config.polymarket.scan_interval_seconds,
            self.config.decision.reevaluate_open_positions_minutes,
            report_time,
        )

        # Run the first cycle immediately without waiting for the interval
        self._log.info("Running initial cycle...")
        self._run_main_cycle()

        # Block on the main thread
        try:
            while self._running:
                if _HOT_RELOAD_FLAG.exists():
                    self._hot_reload()
                time.sleep(1)
        except KeyboardInterrupt:
            self._shutdown()

    def _handle_shutdown(self, signum, frame) -> None:
        self._log.info("Shutdown signal received ({}). Stopping bot...", signum)
        self._shutdown()

    def _shutdown(self) -> None:
        self._running = False
        self._price_monitor_running = False
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        self.notifications.send_text(
            f"🛑 **Bot stopped** | Final balance: €{self.paper_trader.balance_eur:.2f}"
        )
        self.db.close()
        self._log.info("Bot stopped cleanly.")
        sys.exit(0)

    def _hot_reload(self) -> None:
        """Graceful restart triggered by dev_runner.py writing hot_reload.flag."""
        _HOT_RELOAD_FLAG.unlink(missing_ok=True)
        self._log.info("Hot reload requested — restarting with new code...")
        self._running = False
        self._price_monitor_running = False
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=True)  # let current jobs finish
        self.db.close()
        sys.exit(_HOT_RELOAD_EXIT_CODE)

    # =====================================================
    # Jobs
    # =====================================================

    def _apply_overrides(self) -> None:
        """Apply dashboard overrides from data/overrides.json (written by the UI)."""
        overrides_path = Path(__file__).resolve().parent.parent / "data" / "overrides.json"
        if not overrides_path.exists():
            return
        try:
            data = json.loads(overrides_path.read_text())

            new_max = int(data.get("max_simultaneous_positions", self.config.risk.max_simultaneous_positions))
            if new_max != self.config.risk.max_simultaneous_positions:
                self._log.info(
                    "Override: max_simultaneous_positions {} → {}",
                    self.config.risk.max_simultaneous_positions,
                    new_max,
                )
                self.config.risk.max_simultaneous_positions = new_max

            new_par = int(data.get("llm_parallelism", self.config.llm.llm_parallelism))
            new_par = max(1, min(8, new_par))
            if new_par != self.config.llm.llm_parallelism:
                self._log.info(
                    "Override: llm_parallelism {} → {}",
                    self.config.llm.llm_parallelism,
                    new_par,
                )
                self.config.llm.llm_parallelism = new_par

            # Kill switch — close all positions immediately if active
            self._kill_switch_active = bool(data.get("kill_switch_active", False))

        except Exception as exc:
            self._log.warning("Error reading dashboard overrides: {}", exc)

    def _run_main_cycle(self) -> None:
        """Main cycle: scan → news → analyze → decide → execute."""
        if not self._cycle_lock.acquire(blocking=False):
            self._log.warning(
                "Previous cycle still running — skipping scheduler trigger"
            )
            return
        try:
            self._apply_overrides()

            # Kill switch: close all positions and skip the rest of this cycle
            if self._kill_switch_active:
                self._log.error(
                    "KILL SWITCH ACTIVE — closing all open positions, halting new trades"
                )
                self._close_all_positions_kill_switch()
                return

            self._log.info(
                "--- Main cycle | balance=€{:.2f} | positions={}",
                self.paper_trader.balance_eur,
                self.paper_trader.num_open_positions,
            )

            # 1) Check if the bot is paused
            if self.risk_manager.is_paused:
                self._log.warning(
                    "Bot paused due to drawdown. Skipping cycle. "
                    "Use 'python scripts/manage_balance.py status' to review."
                )
                return

            # 1b) Compound drawdown guard — warning only, does not block trades
            self.compound.drawdown_guard()

            # 2) Scan markets
            markets = self.market_scanner.scan()
            if not markets:
                self._log.info("No tradeable markets in this cycle.")
                return

            # 3) Re-ranking by category and candidate selection
            candidates = self.market_scanner.rank_for_analysis(
                markets,
                category_boost=self.config.decision.category_priority_boost,
                top_n=self.config.decision.markets_to_analyze_per_cycle,
            )

            # 3b) NO-hunt: append high-YES markets specifically to find
            #     overpriced YES / BUY_NO opportunities. Markets with
            #     YES >= threshold are often under-analyzed for the NO side.
            if self.config.decision.no_hunt_enabled:
                candidate_ids = {m.market_id for m in candidates}
                no_hunt = [
                    m for m in markets
                    if m.yes_price >= self.config.decision.no_hunt_min_yes_price
                    and m.market_id not in candidate_ids
                ]
                no_hunt.sort(key=lambda m: m.volume_24h_usd, reverse=True)
                no_hunt = no_hunt[:self.config.decision.no_hunt_max_candidates]
                if no_hunt:
                    self._log.info(
                        "NO hunt: +{} markets with YES≥{:.0%} added to analysis",
                        len(no_hunt),
                        self.config.decision.no_hunt_min_yes_price,
                    )
                    candidates = candidates + no_hunt

            # ── Pipeline: each worker fetches news + runs LLM for its own markets ──
            # Previously: Phase 1 fetched ALL news (3 workers), Phase 2 analyzed ALL
            # markets (N LLM workers). Problem: workers shared the same news pool and
            # caused GDELT rate-limit bursts. Now each worker owns its markets fully,
            # interleaving GDELT calls with Ollama calls — no shared state.
            parallelism = self.config.llm.llm_parallelism
            fallback_ts = (
                self.config.decision.fallback_news_lookback
                if self.config.decision.enable_fallback_search
                else None
            )

            pairs: list[tuple] = [None] * len(candidates)    # type: ignore[assignment]
            analyses: list = [None] * len(candidates)         # type: ignore[assignment]

            def _pipeline_task(market, idx: int):
                worker = threading.current_thread().name
                t0 = time.time()
                keywords = extract_keywords(market.question)
                articles = self.news_ingestor.fetch(
                    keywords, max_articles=10, fallback_timespan=fallback_ts
                )
                if self.social_ingestor is not None:
                    social_articles = self.social_ingestor.fetch_all(keywords)
                    articles = articles + social_articles
                self._log.info(
                    "[{}] start #{} '{}' ({} arts)",
                    worker, idx, market.question[:50], len(articles),
                )
                result = self.sentiment_analyzer.analyze(market, articles)
                elapsed = time.time() - t0
                self._log.info(
                    "[{}] done  #{} '{}' | rec={} conf={} | {:.1f}s",
                    worker, idx, market.question[:40],
                    result.recommendation.value, result.confidence, elapsed,
                )
                return articles, result

            t_pipeline = time.time()
            if parallelism > 1:
                with ThreadPoolExecutor(
                    max_workers=parallelism, thread_name_prefix="llm-worker"
                ) as pool:
                    future_to_idx = {
                        pool.submit(_pipeline_task, market, idx): idx
                        for idx, market in enumerate(candidates)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        market = candidates[idx]
                        try:
                            articles, analysis = future.result()
                            pairs[idx] = (market, articles)
                            analyses[idx] = analysis
                        except Exception as exc:
                            self._log.error(
                                "Pipeline error #{} '{}': {}",
                                idx, market.question[:40], exc,
                            )
                            pairs[idx] = (market, [])
                            analyses[idx] = self.sentiment_analyzer._make_insufficient_data(
                                market, [], reason=str(exc)
                            )
            else:
                for idx, market in enumerate(candidates):
                    try:
                        articles, analysis = _pipeline_task(market, idx)
                        pairs[idx] = (market, articles)
                        analyses[idx] = analysis
                    except Exception as exc:
                        self._log.error(
                            "Pipeline error #{} '{}': {}",
                            idx, market.question[:40], exc,
                        )
                        pairs[idx] = (market, [])
                        analyses[idx] = self.sentiment_analyzer._make_insufficient_data(
                            market, [], reason=str(exc)
                        )

            actionable = sum(
                1 for a in analyses
                if a and a.recommendation.value not in ("WAIT", "INSUFFICIENT_DATA")
            )
            self._log.info(
                "Pipeline complete: {} markets in {:.1f}s | {} actionable",
                len(candidates), time.time() - t_pipeline, actionable,
            )

            # ── Phase 3: Decide + execute (serial — position state must be consistent)
            new_trades = 0
            for (market, articles), analysis in zip(pairs, analyses):
                # Stop if we have reached the maximum number of positions
                if (
                    self.paper_trader.num_open_positions
                    >= self.config.risk.max_simultaneous_positions
                ):
                    self._log.info(
                        "Maximum positions reached ({}/{}). "
                        "No new trades this cycle — reviewing open positions.",
                        self.paper_trader.num_open_positions,
                        self.config.risk.max_simultaneous_positions,
                    )
                    self._run_position_review(scanned_markets=markets)
                    break

                self.db.log_analysis(analysis)

                decision = self.decision_engine.decide(
                    analysis=analysis,
                    current_balance_eur=self.paper_trader.balance_eur,
                    open_positions=self.paper_trader.open_positions,
                    articles=articles,
                )

                if decision.action == DecisionAction.OPEN_TRADE:
                    position = self.paper_trader.execute_decision(decision)
                    if position:
                        new_trades += 1
                        self.notifications.notify_trade_open(
                            position, self.paper_trader.balance_eur
                        )
                else:
                    self.db.log_decision(decision)

            if new_trades > 0:
                self._log.info(
                    "Cycle complete: {} trade(s) opened", new_trades
                )

            # 5) Check drawdown after the cycle
            self._check_drawdown()

            # 6) Secondary sports module (only if enabled)
            if self.config.sports_in_play.enabled:
                self._run_sports_cycle()

        except Exception as exc:
            self._log.error("Error in main cycle: {}", exc, exc_info=True)
            self.notifications.notify_error("main_cycle", str(exc))
        finally:
            self._cycle_lock.release()

    def _price_monitor_loop(self) -> None:
        """Daemon thread body: polls every 10 s while the bot is running."""
        self._log.info("Price monitor thread started (10s interval)")
        tick = 0
        while self._price_monitor_running and self._running:
            tick += 1
            try:
                self._run_price_monitor(tick=tick)
            except Exception as exc:
                self._log.error("Price monitor loop error: {}", exc, exc_info=True)
            time.sleep(10)
        self._log.info("Price monitor thread stopped")

    def _run_price_monitor(self, tick: int = 0) -> None:
        """Lightweight SL/TP monitor — called every 10 s from the daemon thread.

        Only fetches current prices for open positions and closes them if
        SL/TP is triggered. No news fetch, no LLM. Uses a non-blocking lock
        so it yields if _run_position_review is already evaluating positions.
        """
        if not self.paper_trader.open_positions:
            return

        if not self._position_lock.acquire(blocking=False):
            self._log.debug("Price monitor: position review in progress, skipping tick")
            return

        try:
            positions = self.paper_trader.open_positions
            if not positions:
                return

            token_ids = [p.token_id for p in positions]
            price_map = self.clob_client.fetch_midpoints(token_ids)

            # Compact tick log — one line per cycle, shows dead tokens explicitly
            from src.clob_client import _DEAD_TOKENS
            parts = []
            for pos in positions:
                p = price_map.get(pos.token_id)
                label = pos.market_question[:20].rstrip()
                if p is not None:
                    pnl = (p - pos.entry_price) / pos.entry_price
                    parts.append(f"{label}: {p:.4f} ({pnl:+.1%})")
                elif pos.token_id in _DEAD_TOKENS:
                    parts.append(f"{label}: DEAD (closing...)")
                else:
                    parts.append(f"{label}: no price")
            self._log.opt(colors=True).info(
                "<dim>[TICK #{tick}] {n} pos │ {prices}</dim>",
                tick=tick, n=len(positions), prices="  │  ".join(parts),
            )

            for position in positions:
                current_price = price_map.get(position.token_id)
                if current_price is None:
                    settlement = self._check_market_resolution(position)
                    if settlement is not None:
                        self._close_resolved_position(position, settlement)
                    continue

                close_decision = self.decision_engine.evaluate_open_position(
                    position=position,
                    current_price=current_price,
                    new_analysis=None,
                )

                if close_decision.should_close:
                    self._log.info(
                        "Price monitor: closing '{}' due to {} | price={:.4f}",
                        position.market_question[:40],
                        close_decision.reason,
                        current_price,
                    )
                    closed = self.paper_trader.close_position(
                        trade_id=position.trade_id,
                        current_market_price=current_price,
                        reason=close_decision.reason or CloseReason.MANUAL,
                        notes=close_decision.notes,
                    )
                    if closed:
                        self.compound.run_post_mortem(closed)
                        if close_decision.reason and "STOP" in close_decision.reason.value:
                            self.notifications.notify_stop_loss(
                                closed, self.paper_trader.balance_eur
                            )
                        else:
                            self.notifications.notify_trade_close(
                                closed, self.paper_trader.balance_eur
                            )
                else:
                    self._log.debug(
                        "Price monitor: {} | price={:.4f} | P&L={:+.2%}",
                        position.trade_id[:8],
                        current_price,
                        close_decision.pnl_pct,
                    )

        except Exception as exc:
            self._log.error("Error in price monitor: {}", exc, exc_info=True)
        finally:
            self._position_lock.release()

    def _run_sports_cycle(self) -> None:
        """Secondary module: looks for underdog opportunities in live matches.

        Only operates when sports_in_play.enabled=true. Maintains max 1
        simultaneous sports position with a fixed size and a specialized prompt.
        """
        cfg = self.config.sports_in_play
        open_ids = {p.trade_id for p in self.paper_trader.open_positions}

        # Remove from _sports_trade_ids any positions that are already closed
        self._sports_trade_ids &= open_ids

        if len(self._sports_trade_ids) >= cfg.max_positions:
            self._log.debug(
                "Sports: {}/{} positions open — skipping cycle",
                len(self._sports_trade_ids),
                cfg.max_positions,
            )
            return

        candidates = self.market_scanner.scan_sports_candidates()
        if not candidates:
            return

        now = datetime.now(timezone.utc)
        fresh_cutoff_s = cfg.min_fresh_news_minutes * 60
        slots = cfg.max_positions - len(self._sports_trade_ids)

        for market in candidates:
            if slots <= 0:
                break

            keywords = extract_keywords(market.question)
            articles = self.news_ingestor.fetch(keywords, max_articles=8)

            # Require at least 1 fresh article (<min_fresh_news_minutes)
            fresh = [
                a for a in articles
                if a.published_at
                and (now - a.published_at).total_seconds() < fresh_cutoff_s
            ]
            if not fresh:
                self._log.debug(
                    "Sports: '{}' — no fresh news (<{}min), discarded",
                    market.question[:40],
                    cfg.min_fresh_news_minutes,
                )
                continue

            analysis = self.sentiment_analyzer.analyze_sports(market, articles)
            self.db.log_analysis(analysis)

            if (
                analysis.recommendation != TradeRecommendation.BUY_NO
                or analysis.confidence < cfg.min_confidence
            ):
                continue

            # Calculate SL/TP specific to sports
            no_price = market.no_price
            sl_price = no_price * (1.0 - cfg.stop_loss_pct)
            tp_price = no_price * (1.0 + cfg.take_profit_pct)

            # Clamp within valid range
            sl_price = max(0.001, min(0.999, sl_price))
            tp_price = max(0.001, min(0.999, tp_price))

            decision = TradeDecision(
                action=DecisionAction.OPEN_TRADE,
                market_id=market.market_id,
                market_question=market.question,
                market_slug=market.slug,
                side=TradeSide.BUY_NO,
                token_id=market.no_token_id,
                entry_price=no_price,
                size_eur=cfg.position_size_eur,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                confidence=analysis.confidence,
                edge=analysis.edge,
                rationale=(
                    f"[SPORTS] conf={analysis.confidence} | "
                    f"YES={market.yes_price:.2f} | {analysis.summary[:120]}"
                ),
            )

            position = self.paper_trader.execute_decision(decision)
            if position:
                self._sports_trade_ids.add(position.trade_id)
                slots -= 1
                self._log.info(
                    "SPORTS trade opened: '{}' | NO@{:.4f} | conf={} | "
                    "SL={:.4f} TP={:.4f} | size=€{:.2f}",
                    market.question[:40],
                    no_price,
                    analysis.confidence,
                    sl_price,
                    tp_price,
                    cfg.position_size_eur,
                )
                self.notifications.notify_trade_open(
                    position, self.paper_trader.balance_eur
                )

    def _run_position_review(self, scanned_markets=None) -> None:
        """Reviews open positions and closes those that qualify.

        Args:
            scanned_markets: markets already scanned in the main cycle.
                If provided, they are reused (no additional scan). If None
                (autonomous 15-min job), a targeted query is made for only
                the tokens of open positions — no full 500-market scan.
        """
        self._position_lock.acquire()  # blocks if price monitor is mid-evaluation
        try:
            positions = self.paper_trader.open_positions
            if not positions:
                return

            self._log.info(
                "Reviewing {} open position(s)", len(positions)
            )

            if scanned_markets is not None:
                # Called from the main cycle: reuse already-fetched markets.
                markets: list = list(scanned_markets)
                price_map = self._build_price_map(markets)
                # Only fetch tokens that did not pass the scan filters.
                tokens_to_fetch = [
                    p.token_id for p in positions if p.token_id not in price_map
                ]
            else:
                # Autonomous 15-min job: targeted query, no full scan.
                markets = []
                price_map = {}
                tokens_to_fetch = [p.token_id for p in positions]

            if tokens_to_fetch:
                self._log.debug(
                    "CLOB midpoint lookup for {} token(s) from open positions",
                    len(tokens_to_fetch),
                )
                clob_prices = self.clob_client.fetch_midpoints(tokens_to_fetch)
                price_map.update(clob_prices)

            for position in positions:
                current_price = price_map.get(position.token_id)
                if current_price is None:
                    settlement = self._check_market_resolution(position)
                    if settlement is not None:
                        self._close_resolved_position(position, settlement)
                        continue
                    self._log.warning(
                        "No price for token {} ({}...) — using entry price "
                        "for SL/TP review",
                        position.token_id[:10],
                        position.market_question[:30],
                    )
                    current_price = position.entry_price

                # Fetch fresh analysis if there are recent news articles
                keywords = extract_keywords(position.market_question)
                articles = self.news_ingestor.fetch(keywords, max_articles=5)
                market_snap = next(
                    (m for m in markets if m.yes_token_id == position.token_id
                     or m.no_token_id == position.token_id),
                    None,
                )
                new_analysis = None
                if market_snap and articles:
                    new_analysis = self.sentiment_analyzer.analyze(
                        market_snap, articles
                    )

                close_decision = self.decision_engine.evaluate_open_position(
                    position=position,
                    current_price=current_price,
                    new_analysis=new_analysis,
                )

                if close_decision.should_close:
                    closed = self.paper_trader.close_position(
                        trade_id=position.trade_id,
                        current_market_price=current_price,
                        reason=close_decision.reason or CloseReason.MANUAL,
                        notes=close_decision.notes,
                    )
                    if closed:
                        self.compound.run_post_mortem(closed)
                        if close_decision.reason and "STOP" in close_decision.reason.value:
                            self.notifications.notify_stop_loss(
                                closed, self.paper_trader.balance_eur
                            )
                        else:
                            self.notifications.notify_trade_close(
                                closed, self.paper_trader.balance_eur
                            )
                else:
                    self._log.debug(
                        "Position {} held | price={:.4f} | P&L={:+.2%}",
                        position.trade_id[:8],
                        current_price,
                        close_decision.pnl_pct,
                    )

            self._check_drawdown()

        except Exception as exc:
            self._log.error("Error in position review: {}", exc, exc_info=True)
            self.notifications.notify_error("position_review", str(exc))
        finally:
            self._position_lock.release()

    def _run_daily_report(self) -> None:
        """Generates the Excel report, runs compound consolidation, and notifies Discord."""
        try:
            self._log.info("Generating daily report...")
            # Compound layer: compute metrics, prune KB, rebuild llm_report.md
            self.compound.nightly_consolidation(
                open_positions_count=self.paper_trader.num_open_positions
            )
            today = datetime.now(timezone.utc)
            report_path = self.report_generator.generate_daily_report(today)

            # Calculate basic KPIs for the Discord summary
            balance_history = self.db.get_balance_history()
            from src.report_generator import ReportGenerator
            from datetime import time, timedelta

            day_start = datetime.combine(
                today.date(), time.min, tzinfo=timezone.utc
            )
            day_end = day_start + timedelta(days=1)

            bal_start, bal_end = ReportGenerator._get_day_balance_bounds(
                balance_history, day_start, day_end
            )
            total_pnl = bal_end - bal_start

            all_trades = self.db.get_all_trades()
            closed_today = [
                t for t in all_trades
                if t.exit_timestamp and day_start <= t.exit_timestamp <= day_end
            ]
            winners = [t for t in closed_today if (t.pnl_eur or 0) > 0]
            win_rate = len(winners) / len(closed_today) if closed_today else 0.0

            self.notifications.notify_daily_summary(
                date_str=today.strftime("%Y-%m-%d"),
                balance_start=bal_start,
                balance_end=bal_end,
                total_pnl=total_pnl,
                num_trades=len(closed_today),
                win_rate=win_rate,
                report_path=str(report_path),
            )
            self._log.info("Daily report generated: {}", report_path)

        except Exception as exc:
            self._log.error("Error generating report: {}", exc, exc_info=True)
            self.notifications.notify_error("daily_report", str(exc))

    # =====================================================
    # Helpers
    # =====================================================

    def _close_all_positions_kill_switch(self) -> None:
        """Immediately closes all open positions when the kill switch is activated."""
        open_positions = list(self.paper_trader.open_positions)
        if not open_positions:
            self._log.info("Kill switch: no open positions to close")
            return

        self._log.warning(
            "Kill switch: closing {} open position(s)", len(open_positions)
        )
        for position in open_positions:
            try:
                # Fetch current price from the CLOB if available; fall back to entry price
                current_price = position.entry_price
                try:
                    price_map = self.clob_client.fetch_midpoints([position.token_id])
                    fetched = price_map.get(position.token_id)
                    if fetched is not None:
                        current_price = fetched
                except Exception:
                    pass

                closed = self.paper_trader.close_position(
                    trade_id=position.trade_id,
                    current_market_price=current_price,
                    reason=CloseReason.KILL_SWITCH,
                    notes="Kill switch activated from dashboard",
                )
                if closed:
                    self._log.info(
                        "Kill switch: closed trade={} pnl={:.2%}",
                        position.trade_id[:8],
                        closed.pnl_pct or 0,
                    )
            except Exception as exc:
                self._log.error(
                    "Kill switch: failed to close trade={}: {}",
                    position.trade_id[:8], exc
                )

    def _build_price_map(self, markets) -> dict[str, float]:
        """Builds {token_id: price} for active markets."""
        price_map: dict[str, float] = {}
        for m in markets:
            price_map[m.yes_token_id] = m.yes_price
            price_map[m.no_token_id] = m.no_price
        return price_map

    def _check_market_resolution(self, position) -> float | None:
        """If the position's token is dead (404), tries to resolve the market
        via the Gamma API. The zombie timer starts on the FIRST detection and is
        never reset by intermediate Gamma responses — avoids the bug where
        outcomePrices in a pending state would restart the countdown indefinitely.

        Returns:
            1.0  — bought side won
            0.0  — bought side lost
            None — not resolved / force-close already handled internally
        """
        from src.clob_client import _DEAD_TOKENS
        if position.token_id not in _DEAD_TOKENS:
            return None

        now = time.time()

        # Zombie timer: starts on first detection, never reset by intermediate
        # Gamma responses — only cleared when the position is closed.
        zombie_since = self._zombie_since.setdefault(position.token_id, now)
        zombie_elapsed = now - zombie_since

        # Force-close check: runs every tick once threshold is exceeded
        if zombie_elapsed >= 600:
            self._log.warning(
                "ZOMBIE FORCE-CLOSE: '{}' — {:.0f}min without resolution. "
                "Closing at entry price.",
                position.market_question[:50],
                zombie_elapsed / 60,
            )
            closed = self.paper_trader.close_position(
                trade_id=position.trade_id,
                current_market_price=position.entry_price,
                reason=CloseReason.MANUAL,
                notes="Zombie force-close: CLOB 404, market unresolvable after 10min",
            )
            if closed:
                self.compound.run_post_mortem(closed)
                self.notifications.notify_trade_close(
                    closed, self.paper_trader.balance_eur
                )
            self._zombie_since.pop(position.token_id, None)
            self._resolution_last_check.pop(position.token_id, None)
            return None

        # Throttle Gamma API calls to once per 60s per token
        last = self._resolution_last_check.get(position.token_id, 0.0)
        if now - last < 60.0:
            return None
        self._resolution_last_check[position.token_id] = now

        raw = self.market_scanner.client.fetch_market_by_token_id_raw(
            position.token_id
        )
        if raw is None:
            self._log.warning(
                "Token {}... ('{}') zombie | Gamma returned no data | {:.0f}s / 600s",
                position.token_id[:12],
                position.market_question[:30],
                zombie_elapsed,
            )
            return None

        # Gamma returned data — try to detect resolution.
        # We do NOT reset zombie_since here: if outcomePrices are intermediate,
        # the countdown continues until force-close.
        from src.market_scanner import MarketScanner
        yes_price, _ = MarketScanner._parse_price_pair(raw.get("outcomePrices"))
        if yes_price is None:
            self._log.warning(
                "Token {}... ('{}') zombie | Gamma OK but no outcomePrices | {:.0f}s / 600s",
                position.token_id[:12],
                position.market_question[:30],
                zombie_elapsed,
            )
            return None

        if yes_price >= 0.95:
            yes_won = True
        elif yes_price <= 0.05:
            yes_won = False
        else:
            self._log.warning(
                "Token {}... ('{}') zombie | outcomePrices={:.3f} (pending) | {:.0f}s / 600s",
                position.token_id[:12],
                position.market_question[:30],
                yes_price,
                zombie_elapsed,
            )
            return None  # Gamma still showing intermediate — countdown continues

        self._log.info(
            "Resolution detected — token {}... | '{}' | YES={:.3f} ({})",
            position.token_id[:12],
            position.market_question[:40],
            yes_price,
            "YES won" if yes_won else "NO won",
        )
        self._zombie_since.pop(position.token_id, None)
        self._resolution_last_check.pop(position.token_id, None)

        if position.side == TradeSide.BUY_YES:
            return 1.0 if yes_won else 0.0
        else:  # BUY_NO
            return 0.0 if yes_won else 1.0

    def _close_resolved_position(self, position, settlement: float) -> None:
        """Closes a position at the settlement price and notifies."""
        pnl_pct = (settlement - position.entry_price) / position.entry_price
        outcome = "WON" if settlement >= 0.99 else "LOST"
        self._log.info(
            "Market resolved — position {} | {} | settlement={:.3f} | "
            "entry={:.4f} | P&L={:+.1%}",
            outcome,
            position.market_question[:50],
            settlement,
            position.entry_price,
            pnl_pct,
        )
        closed = self.paper_trader.close_position(
            trade_id=position.trade_id,
            current_market_price=settlement,
            reason=CloseReason.MARKET_RESOLVED,
            notes=f"Market resolved — settlement {settlement:.3f}",
        )
        if closed:
            self.compound.run_post_mortem(closed)
            self.notifications.notify_trade_close(
                closed, self.paper_trader.balance_eur
            )

    def _check_drawdown(self) -> None:
        """Checks drawdown and notifies — at most once every 24 hours."""
        status = self.risk_manager.update_balance_and_check_drawdown(
            self.paper_trader.balance_eur
        )

        _24H = 86_400.0
        now = time.time()
        cooldown_active = (now - self._last_drawdown_alert_ts) < _24H

        if status.threshold_breached and not cooldown_active:
            self._last_drawdown_alert_ts = now
            self.notifications.notify_drawdown_warning(
                current_balance=status.current_balance_eur,
                peak_balance=status.peak_balance_eur,
                drawdown_pct=status.current_drawdown_pct,
            )

        if status.bot_should_pause and self.risk_manager.is_paused:
            self.notifications.notify_bot_paused(
                reason=self.risk_manager.pause_reason or "Maximum drawdown reached",
                balance=self.paper_trader.balance_eur,
            )


# =====================================================
# Entry point
# =====================================================


def setup_logging(config: BotConfig) -> None:
    """Configure loguru with color-coded console output and clean file logs."""
    import logging
    import sys
    from pathlib import Path as P

    log_dir = P(config.logging.log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    # ── Silence noisy third-party stdlib loggers ──────────────────────────────
    for lib in ("apscheduler", "urllib3", "httpx", "asyncio", "watchdog",
                "telethon", "requests", "hpack", "charset_normalizer"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    # ── Shared filter: adds a default module name for unbound loggers ─────────
    def _ensure_module(record: dict) -> bool:
        record["extra"].setdefault("module", record["name"].split(".")[-1][:18])
        return True

    # ── Console format (color-coded by level, module in cyan) ─────────────────
    # Colors are driven by loguru's <level> tag (INFO=green, WARNING=yellow,
    # ERROR=red, DEBUG=blue). Module name always cyan for easy scanning.
    FMT_CONSOLE = (
        "<green>{time:HH:mm:ss}</green>"
        "  <level>{level: <8}</level>"
        "  <cyan>{extra[module]:<18}</cyan>"
        "  {message}"
    )

    # ── File format (plain text, full timestamp, source location) ─────────────
    FMT_FILE = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS}"
        " | {level: <8}"
        " | {extra[module]:<18}"
        " | {name}:{line}"
        " | {message}"
    )

    logger.add(
        sys.stdout,
        level="INFO",
        format=FMT_CONSOLE,
        colorize=True,
        filter=_ensure_module,
    )

    logger.add(
        str(log_dir / "bot.log"),
        level=config.logging.level,
        format=FMT_FILE,
        rotation=f"{config.logging.rotation_size_mb} MB",
        retention=f"{config.logging.retention_days} days",
        encoding="utf-8",
        filter=_ensure_module,
    )

    if config.logging.log_llm_decisions:
        logger.add(
            str(log_dir / "llm_decisions.log"),
            level="DEBUG",
            format=FMT_FILE,
            filter=lambda r: r["extra"].get("module") in (
                "sentiment_analyzer", "decision_engine"
            ),
            rotation="50 MB",
            retention="30 days",
            encoding="utf-8",
        )


def main() -> None:
    """Main entry point of the bot."""
    config = load_config()

    # Set up logging first to capture everything
    setup_logging(config)
    log = logger.bind(module="main")

    # Validate secrets
    errors = validate_secrets(config)
    if errors:
        for err in errors:
            log.error("Config error: {}", err)
        log.error("Aborting due to configuration errors.")
        sys.exit(1)

    log.info("Configuration loaded. LLM provider: {} ({})",
             config.llm.provider, config.llm.model)

    # Create and start the orchestrator
    orchestrator = Orchestrator(config)
    orchestrator.start()


if __name__ == "__main__":
    main()
