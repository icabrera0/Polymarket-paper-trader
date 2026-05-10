"""
Compound layer — self-improving feedback loop.

After each trade closes, this module:
1. Runs a post-mortem using Ollama/Qwen (classify failure, extract lesson)
2. Saves the post-mortem to data/llm_outcomes.jsonl (append-only JSONL)
3. Persists lessons to the knowledge_base table in SQLite
4. Rebuilds data/llm_report.md (human-readable, fully rebuilt each time)

On every LLM analysis (via SentimentAnalyzer), relevant lessons are injected
into the user prompt so the LLM learns from past mistakes.

Nightly consolidation (23:55): recomputes performance metrics, saves a
PerformanceSnapshot, prunes low-confidence KB entries, and rebuilds the report.

Drawdown guard: blocks new trades if the current drawdown exceeds
DRAWDOWN_GUARD_PCT (8%). Independent and tighter than risk.max_drawdown_pct.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config_loader import BotConfig
from src.database import Database
from src.llm_client import LLMError, build_llm_client
from src.models import (
    FailureCategory,
    KnowledgeBaseEntry,
    PerformanceSnapshot,
    Position,
    PostMortem,
)

OUTCOMES_FILE = Path("data/llm_outcomes.jsonl")
REPORT_FILE = Path("data/llm_report.md")

DRAWDOWN_GUARD_PCT = 0.08  # Block new trades if drawdown > 8%

_POST_MORTEM_SYSTEM = (
    "You are a quantitative trading analyst reviewing closed paper trades on "
    "Polymarket prediction markets. Given a closed trade, classify the outcome "
    "and extract a concise, reusable lesson for future trades.\n\n"
    "Respond ONLY with a valid JSON object containing:\n"
    '- "failure_category": one of BAD_PREDICTION, BAD_TIMING, BAD_EXECUTION, '
    "EXTERNAL_SHOCK, NOT_A_LOSS\n"
    '- "root_cause": string (max 300 chars) — specific cause of the outcome\n'
    '- "lesson": string (max 200 chars) — actionable rule to apply in future trades\n'
    '- "market_pattern": string (max 150 chars) — abstract pattern this market '
    "belongs to (used for knowledge base lookup)\n"
    '- "category": one of politics, economics, sports, crypto, legal, science, '
    "entertainment, general — inferred from the market question and market slug"
)


class CompoundEngine:
    """Self-improving feedback loop for the paper trading bot."""

    def __init__(self, config: BotConfig, db: Database) -> None:
        self.config = config
        self.db = db
        self._log = logger.bind(module="compound")
        OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # Public: post-mortem after a trade closes
    # =====================================================

    def run_post_mortem(self, position: Position) -> Optional[PostMortem]:
        """Run an LLM post-mortem on a closed position.

        Appends an outcome record, updates the knowledge base, and rebuilds
        the report. Returns None on any LLM error (non-fatal).
        """
        if position.exit_price is None or position.pnl_pct is None:
            self._log.warning(
                "run_post_mortem: position {} has no exit data — skipping",
                position.trade_id[:8],
            )
            return None

        time_held_hours = 0.0
        if position.exit_timestamp and position.entry_timestamp:
            delta = position.exit_timestamp - position.entry_timestamp
            time_held_hours = delta.total_seconds() / 3600

        try:
            result = self._call_llm_for_post_mortem(position, time_held_hours)
        except (LLMError, Exception) as exc:
            self._log.warning(
                "Post-mortem LLM call failed for {}: {}", position.trade_id[:8], exc
            )
            return None

        try:
            failure_category = FailureCategory(
                result.get("failure_category", "BAD_PREDICTION")
            )
        except ValueError:
            failure_category = FailureCategory.BAD_PREDICTION

        pm = PostMortem(
            trade_id=position.trade_id,
            failure_category=failure_category,
            root_cause=(result.get("root_cause") or "")[:300],
            lesson=(result.get("lesson") or "")[:200],
            market_slug=position.market_slug,
            predicted_prob=0.0,
            actual_outcome=None,
            pnl_pct=position.pnl_pct,
            time_held_hours=time_held_hours,
        )

        self.db.save_post_mortem(pm)

        market_pattern = (
            result.get("market_pattern") or position.market_slug or "general"
        )[:150]
        _VALID_CATEGORIES = {
            "politics", "economics", "sports", "crypto",
            "legal", "science", "entertainment", "general",
        }
        category = (result.get("category") or "general").lower().strip()
        if category not in _VALID_CATEGORIES:
            category = "general"
        self._update_knowledge_base(pm, market_pattern, category)
        self._append_outcome_record(pm, position)
        self._rebuild_report()
        self._cull_knowledge_base()

        self._log.info(
            "Post-mortem | trade={} | category={} | pnl={:+.1%} | lesson={}",
            position.trade_id[:8],
            pm.failure_category.value,
            pm.pnl_pct,
            pm.lesson[:60],
        )
        return pm

    # =====================================================
    # Public: inject lessons into LLM prompt
    # =====================================================

    def get_relevant_lessons(
        self,
        market_question: str,
        market_slug: str = "",
        max_lessons: int = 3,
    ) -> str:
        """Returns a formatted string of KB lessons relevant to the market.

        Safe to embed in any prompt — returns empty string if nothing found.
        """
        entries = self.db.get_knowledge_base(limit=50)
        if not entries:
            return ""

        q_lower = market_question.lower()
        slug_words = set(market_slug.lower().replace("-", " ").split())

        def _relevance(entry: KnowledgeBaseEntry) -> float:
            pattern_lower = entry.market_pattern.lower()
            pattern_words = set(pattern_lower.replace("-", " ").split())
            overlap = len(slug_words & pattern_words)
            keyword_hit = any(w in q_lower for w in pattern_lower.split() if len(w) > 4)
            return entry.confidence * (
                1 + overlap * 0.3 + (0.2 if keyword_hit else 0)
            )

        market_category = self._infer_category(market_question)
        ranked = sorted(entries, key=_relevance, reverse=True)[:max_lessons * 3]
        # Fix 2: filter by category — only inject lessons that match this market's
        # category or are tagged "general" (universal lessons)
        ranked = [e for e in ranked if e.category in (market_category, "general")]
        # Fix 1: raise confidence threshold from 0.3 to 0.5
        ranked = [e for e in ranked if e.confidence >= 0.5]
        ranked = ranked[:max_lessons]
        if not ranked:
            return ""

        # Fix 3: contradiction detection — keyword-based, no LLM call
        _AVOID_KEYWORDS = {"avoid", "don't trade", "do not trade", "skip", "stay out"}
        _BUY_KEYWORDS = {"buy", "trade when", "enter when", "take position"}
        contradiction_warning = ""
        by_cat: dict[str, list[KnowledgeBaseEntry]] = {}
        for e in ranked:
            by_cat.setdefault(e.category, []).append(e)
        for cat, cat_entries in by_cat.items():
            has_avoid = any(
                any(kw in e.lesson.lower() for kw in _AVOID_KEYWORDS)
                for e in cat_entries
            )
            has_buy = any(
                any(kw in e.lesson.lower() for kw in _BUY_KEYWORDS)
                for e in cat_entries
            )
            if has_avoid and has_buy:
                contradiction_warning = (
                    f"\n\n⚠ Contradictory lessons found in category '{cat}'. "
                    f"Weight by confidence: higher confidence takes precedence. "
                    f"When in doubt, default to WAIT."
                )
                break

        lines = ["## Past lessons from similar markets"]
        for e in ranked:
            lines.append(
                f"- [{e.failure_category.value}] {e.lesson} "
                f"(confidence={e.confidence:.0%}, confirmed {e.times_confirmed}x)"
            )
        return "\n".join(lines) + contradiction_warning

    # =====================================================
    # Public: metrics
    # =====================================================

    def calculate_metrics(self, open_positions_count: int = 0) -> PerformanceSnapshot:
        """Compute performance metrics from closed trades in the last 90 days."""
        from datetime import date

        trades = self.db.get_closed_trades_in_window(days=90)
        snap = PerformanceSnapshot(
            snapshot_date=date.today(),
            total_trades=len(trades),
            open_positions=open_positions_count,
        )
        if not trades:
            return snap

        pnl_pcts = [t.get("pnl_pct") or 0.0 for t in trades]
        wins = [p for p in pnl_pcts if p > 0]
        losses = [p for p in pnl_pcts if p < 0]

        snap.win_rate = len(wins) / len(pnl_pcts)

        if len(pnl_pcts) >= 2:
            mean = sum(pnl_pcts) / len(pnl_pcts)
            variance = sum((x - mean) ** 2 for x in pnl_pcts) / (len(pnl_pcts) - 1)
            std = math.sqrt(variance) if variance > 0 else 0.0
            snap.sharpe_ratio = (mean / std * math.sqrt(252)) if std > 0 else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        if gross_loss > 0:
            snap.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            snap.profit_factor = 1.0

        history = self.db.get_balance_history()
        if history:
            balances = [float(r["balance_eur"]) for r in history]
            peak = balances[0]
            max_dd = 0.0
            for b in balances:
                if b > peak:
                    peak = b
                dd = (peak - b) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
            snap.max_drawdown = max_dd

        # Brier score from post_mortems with a known actual_outcome
        post_mortems = self.db.get_post_mortems_today()
        brier_samples = [
            (float(pm.get("predicted_prob") or 0.0), pm.get("actual_outcome"))
            for pm in post_mortems
            if pm.get("actual_outcome") is not None
        ]
        if brier_samples:
            snap.brier_score = (
                sum((pred - actual) ** 2 for pred, actual in brier_samples)
                / len(brier_samples)
            )

        return snap

    # =====================================================
    # Public: drawdown guard
    # =====================================================

    def drawdown_guard(self) -> bool:
        """Warns when drawdown exceeds threshold. Always returns False (monitoring only)."""
        history = self.db.get_balance_history()
        if not history or len(history) < 2:
            return False

        balances = [float(r["balance_eur"]) for r in history]
        peak = max(balances)
        current = balances[-1]
        if peak <= 0:
            return False

        drawdown = (peak - current) / peak
        if drawdown >= DRAWDOWN_GUARD_PCT:
            self._log.warning(
                "Drawdown alert: peak={:.2f}€, current={:.2f}€, "
                "drawdown={:.1%} >= {:.0%} threshold",
                peak,
                current,
                drawdown,
                DRAWDOWN_GUARD_PCT,
            )
        return False

    # =====================================================
    # Public: nightly consolidation
    # =====================================================

    def nightly_consolidation(self, open_positions_count: int = 0) -> None:
        """Compute metrics, save snapshot, prune KB, rebuild report."""
        self._log.info("Nightly consolidation started")

        snap = self.calculate_metrics(open_positions_count)
        self.db.save_performance_snapshot(snap)
        self._log.info(
            "Performance snapshot saved | trades={} | win_rate={:.1%} | "
            "sharpe={:.2f} | profit_factor={:.2f} | max_dd={:.1%}",
            snap.total_trades,
            snap.win_rate,
            snap.sharpe_ratio,
            snap.profit_factor,
            snap.max_drawdown,
        )

        self._prune_knowledge_base()
        self._rebuild_report()
        self._log.info("Nightly consolidation complete")

    # =====================================================
    # Private: LLM call
    # =====================================================

    def _call_llm_for_post_mortem(
        self, position: Position, time_held_hours: float
    ) -> dict:
        client = build_llm_client(self.config)

        outcome = "GAIN" if (position.pnl_pct or 0) >= 0 else "LOSS"
        close_reason = (
            position.close_reason.value if position.close_reason else "UNKNOWN"
        )
        user_prompt = (
            f"Trade outcome: {outcome}\n"
            f"Market: {position.market_question}\n"
            f"Side: {position.side.value}\n"
            f"Entry price: {position.entry_price:.4f}\n"
            f"Exit price: {position.exit_price:.4f}\n"
            f"P&L: {(position.pnl_pct or 0):+.1%}\n"
            f"Close reason: {close_reason}\n"
            f"Time held: {time_held_hours:.1f} hours\n"
            f"Entry reason: {(position.entry_reason or 'N/A')[:300]}\n"
            f"Exit reason: {(position.exit_reason_text or 'N/A')[:200]}\n"
            "\nClassify this trade and extract a reusable lesson. "
            "Return ONLY valid JSON."
        )

        result, _meta = client.complete_json(
            system_prompt=_POST_MORTEM_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=400,
            temperature=0.3,
        )
        return result

    # =====================================================
    # Private: knowledge base
    # =====================================================

    def _update_knowledge_base(
        self, pm: PostMortem, market_pattern: str, category: str = "general"
    ) -> None:
        """Upsert a lesson: reinforce existing pattern or insert new entry."""
        entries = self.db.get_knowledge_base(limit=200)
        existing = next(
            (e for e in entries if e.market_pattern.lower() == market_pattern.lower()),
            None,
        )

        if existing:
            new_confirmed = existing.times_confirmed + 1
            new_confidence = min(0.95, existing.confidence + 0.05 * (1 - existing.confidence))
            self.db.update_knowledge_entry_confidence(
                existing.id, new_confirmed, new_confidence
            )
        else:
            entry = KnowledgeBaseEntry(
                market_pattern=market_pattern,
                lesson=pm.lesson,
                failure_category=pm.failure_category,
                confidence=0.4,
                times_confirmed=1,
                category=category,
            )
            self.db.save_knowledge_entry(entry)

    def _infer_category(self, question: str) -> str:
        """Infer market category from question text using keyword matching."""
        q = question.lower()
        if any(w in q for w in (
            "election", "president", "senate", "congress", "vote", "minister",
            "parliament", "ballot", "party", "democrat", "republican",
        )):
            return "politics"
        if any(w in q for w in (
            "bitcoin", "eth", "crypto", "token", "coin", "blockchain",
            "defi", "nft", "solana", "binance",
        )):
            return "crypto"
        if any(w in q for w in (
            "match", "team", "player", "championship", "tournament", "vs",
            "league", "soccer", "football", "basketball", "tennis", "olympic",
        )):
            return "sports"
        if any(w in q for w in (
            "gdp", "inflation", "fed", "interest rate", "recession",
            "unemployment", "economy", "market cap", "earnings", "stock", "bond",
        )):
            return "economics"
        if any(w in q for w in (
            "court", "trial", "lawsuit", "verdict", "judge", "legal",
            "arrest", "indicted", "convicted",
        )):
            return "legal"
        if any(w in q for w in (
            "climate", "temperature", "earthquake", "hurricane", "scientific",
            "vaccine", "virus", "pandemic", "discovery",
        )):
            return "science"
        if any(w in q for w in (
            "oscar", "grammy", "movie", "film", "celebrity", "music",
            "album", "box office", "streaming",
        )):
            return "entertainment"
        return "general"

    def _cull_knowledge_base(self) -> None:
        """Demote or delete KB entries whose patterns have poor win rates across 5+ trades.

        - Loads all KB entries with times_confirmed >= 5.
        - Fetches associated post-mortems for each entry via fuzzy pattern match.
        - If win_rate (pnl_pct > 0 / total) < 0.40: halves confidence.
        - If halved confidence < 0.1: deletes the entry entirely.
        - All failures are non-fatal (logged at WARNING level).
        """
        try:
            entries = self.db.get_knowledge_base(limit=500)
            to_delete: list[str] = []
            to_update: list[tuple[str, int, float]] = []  # (id, times_confirmed, new_confidence)

            for entry in entries:
                if entry.times_confirmed < 5:
                    continue

                try:
                    post_mortems = self.db.get_post_mortems_by_pattern(entry.market_pattern)
                except Exception as exc:
                    self._log.warning(
                        "KB cull: could not fetch post-mortems for pattern '{}': {}",
                        entry.market_pattern, exc,
                    )
                    continue

                if not post_mortems:
                    continue

                wins = sum(1 for pm in post_mortems if (pm.get("pnl_pct") or 0.0) > 0)
                total = len(post_mortems)
                if total == 0:
                    continue

                win_rate = wins / total
                if win_rate < 0.40:
                    new_confidence = entry.confidence * 0.5
                    if new_confidence < 0.1:
                        to_delete.append(entry.id)
                        action = "DELETED"
                    else:
                        to_update.append((entry.id, entry.times_confirmed, new_confidence))
                        action = f"halved → {new_confidence:.2f}"
                    self._log.debug(
                        "KB cull: pattern='{}' win_rate={:.0%} → confidence {:.2f} → {}",
                        entry.market_pattern,
                        win_rate,
                        entry.confidence,
                        action,
                    )

            if to_delete:
                self.db.delete_knowledge_entries(to_delete)
                self._log.info("KB cull: deleted {} low-win-rate entries", len(to_delete))

            for entry_id, times_confirmed, new_confidence in to_update:
                try:
                    self.db.update_knowledge_entry_confidence(entry_id, times_confirmed, new_confidence)
                except Exception as exc:
                    self._log.warning("KB cull: failed to update entry {}: {}", entry_id, exc)

            if to_delete or to_update:
                self._log.info(
                    "KB cull complete: {} deleted, {} confidence-halved",
                    len(to_delete),
                    len(to_update),
                )
        except Exception as exc:
            self._log.warning("_cull_knowledge_base failed (non-fatal): {}", exc)

    def _prune_knowledge_base(self, min_confidence: float = 0.2) -> None:
        entries = self.db.get_knowledge_base(limit=500)
        to_delete = [e.id for e in entries if e.confidence < min_confidence]
        if to_delete:
            self.db.delete_knowledge_entries(to_delete)
            self._log.info("Pruned {} low-confidence KB entries", len(to_delete))

    # =====================================================
    # Private: JSONL + report
    # =====================================================

    def _append_outcome_record(self, pm: PostMortem, position: Position) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trade_id": pm.trade_id,
            "market_slug": pm.market_slug,
            "side": position.side.value,
            "pnl_pct": round(pm.pnl_pct, 4),
            "time_held_hours": round(pm.time_held_hours, 2),
            "failure_category": pm.failure_category.value,
            "root_cause": pm.root_cause,
            "lesson": pm.lesson,
            "close_reason": (
                position.close_reason.value if position.close_reason else ""
            ),
        }
        try:
            with OUTCOMES_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._log.warning("Could not write to {}: {}", OUTCOMES_FILE, exc)

    def _rebuild_report(self) -> None:
        """Rebuilds data/llm_report.md from scratch."""
        entries = self.db.get_knowledge_base(limit=100)
        snap = self.db.get_latest_performance_snapshot()

        lines: list[str] = [
            "# LLM Learning Report",
            f"\n_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n",
            "## Performance Summary",
        ]

        if snap:
            lines += [
                f"- **Date**: {snap.snapshot_date}",
                f"- **Total trades (90d)**: {snap.total_trades}",
                f"- **Win rate**: {snap.win_rate:.1%}",
                f"- **Sharpe ratio**: {snap.sharpe_ratio:.2f}",
                f"- **Profit factor**: {snap.profit_factor:.2f}",
                f"- **Max drawdown**: {snap.max_drawdown:.1%}",
                f"- **Brier score**: {snap.brier_score:.3f}",
            ]
        else:
            lines.append("_No performance snapshot yet._")

        lines.append("\n## Knowledge Base")
        if entries:
            lines.append(f"\n{len(entries)} lessons stored, sorted by confidence:\n")
            lines.append("| # | Pattern | Lesson | Category | Confidence | Confirmed |")
            lines.append("|---|---------|--------|----------|-----------|-----------|")
            for i, e in enumerate(entries, 1):
                lesson_esc = e.lesson.replace("|", "\\|")
                pattern_esc = e.market_pattern.replace("|", "\\|")
                lines.append(
                    f"| {i} | {pattern_esc} | {lesson_esc} | "
                    f"{e.failure_category.value} | {e.confidence:.0%} | "
                    f"{e.times_confirmed} |"
                )
        else:
            lines.append("\n_No lessons learned yet._")

        lines.append("\n## Recent Outcomes")
        lines.append("\nSee `data/llm_outcomes.jsonl` for the full machine-readable record.\n")
        if OUTCOMES_FILE.exists():
            try:
                recent = OUTCOMES_FILE.read_text(encoding="utf-8").splitlines()[-10:]
                if recent:
                    lines.append("Last 10 outcomes:\n\n```")
                    for raw in recent:
                        try:
                            r = json.loads(raw)
                            lines.append(
                                f"{r.get('ts','')[:10]} | "
                                f"{r.get('market_slug','')[:30]} | "
                                f"{r.get('side','')} | "
                                f"pnl={float(r.get('pnl_pct', 0)):+.1%} | "
                                f"{r.get('failure_category','')} | "
                                f"{r.get('lesson','')[:60]}"
                            )
                        except (json.JSONDecodeError, ValueError):
                            pass
                    lines.append("```")
            except OSError:
                pass

        lines.append("\n## Prompt Improvement Notes")
        lines.append(
            "\n_Use this file to identify systematic failures and improve LLM prompts._\n"
        )
        by_category: dict[str, int] = {}
        for e in entries:
            k = e.failure_category.value
            by_category[k] = by_category.get(k, 0) + 1
        if by_category:
            lines.append("### Failure frequency by category\n")
            for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
                lines.append(f"- **{cat}**: {count}")
        else:
            lines.append("- _No failures categorized yet_")

        try:
            REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            self._log.warning("Could not write report to {}: {}", REPORT_FILE, exc)
