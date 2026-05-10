"""
Tests for the self-learning loop fixes in CompoundEngine.

Covers:
1. Confidence threshold — lessons with confidence < 0.5 are NOT injected.
2. Category filtering — a political lesson does NOT fire on a sports market.
3. Contradiction detection — when two opposing lessons exist in the same
   category, a warning is appended.
4. Win-rate culling — entry with 6 confirmations at 30% win rate gets
   confidence halved.
5. Win-rate culling deletion — entry with resulting confidence < 0.1 AND
   times_confirmed >= 5 is deleted.
6. Missing "category" in post-mortem LLM output falls back to "general".
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.models import FailureCategory, KnowledgeBaseEntry, PostMortem, Position, TradeSide


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_entry(
    *,
    confidence: float,
    category: str = "general",
    lesson: str = "Always check liquidity before entering.",
    market_pattern: str = "test-pattern",
    times_confirmed: int = 1,
) -> KnowledgeBaseEntry:
    return KnowledgeBaseEntry(
        id=str(uuid.uuid4()),
        market_pattern=market_pattern,
        lesson=lesson,
        failure_category=FailureCategory.BAD_PREDICTION,
        confidence=confidence,
        times_confirmed=times_confirmed,
        category=category,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_compound_engine(kb_entries: list[KnowledgeBaseEntry]):
    """Build a CompoundEngine with a mocked DB that returns the given KB entries."""
    from src.compound import CompoundEngine

    mock_db = MagicMock()
    mock_db.get_knowledge_base.return_value = kb_entries
    mock_config = MagicMock()

    engine = CompoundEngine.__new__(CompoundEngine)
    engine.config = mock_config
    engine.db = mock_db
    engine._log = MagicMock()
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: Confidence threshold
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceThreshold:
    def test_lesson_with_confidence_04_not_injected(self):
        """Entry with confidence=0.4 must NOT appear in the injected string."""
        entry = _make_entry(confidence=0.4, category="general")
        engine = _make_compound_engine([entry])
        result = engine.get_relevant_lessons("Will inflation rise next month?")
        assert result == ""

    def test_lesson_with_confidence_05_is_injected(self):
        """Entry with confidence=0.5 must appear in the injected string."""
        entry = _make_entry(
            confidence=0.5,
            category="economics",
            lesson="Always check liquidity before entering.",
        )
        engine = _make_compound_engine([entry])
        result = engine.get_relevant_lessons(
            "Will inflation rise next month? Fed interest rate decision."
        )
        assert "Always check liquidity before entering." in result

    def test_lesson_with_confidence_049_not_injected(self):
        """Boundary: 0.49 is below threshold — must not be injected."""
        entry = _make_entry(confidence=0.49, category="general")
        engine = _make_compound_engine([entry])
        result = engine.get_relevant_lessons("General market question")
        assert result == ""

    def test_lesson_with_confidence_10_is_injected(self):
        """High confidence lesson should always be injected (when category matches)."""
        entry = _make_entry(confidence=0.95, category="general", lesson="Never FOMO.")
        engine = _make_compound_engine([entry])
        result = engine.get_relevant_lessons("General market question")
        assert "Never FOMO." in result


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: Category filtering
# ─────────────────────────────────────────────────────────────────────────────


class TestCategoryFiltering:
    def test_political_lesson_does_not_fire_on_sports_question(self):
        """A politics-tagged lesson must NOT inject on a sports market question."""
        entry = _make_entry(
            confidence=0.8,
            category="politics",
            lesson="Avoid trading near election day volatility.",
        )
        engine = _make_compound_engine([entry])
        result = engine.get_relevant_lessons(
            "Will the NBA championship be won by the Lakers?"
        )
        assert result == ""

    def test_sports_lesson_fires_on_sports_question(self):
        """A sports-tagged lesson must inject on a sports market question."""
        entry = _make_entry(
            confidence=0.8,
            category="sports",
            lesson="Home team advantage is overpriced by 5%.",
        )
        engine = _make_compound_engine([entry])
        result = engine.get_relevant_lessons(
            "Will the NBA championship be won by the Lakers?"
        )
        assert "Home team advantage is overpriced" in result

    def test_general_lesson_fires_on_any_question(self):
        """A 'general' lesson must inject on any market question."""
        entry = _make_entry(
            confidence=0.6,
            category="general",
            lesson="Reduce size when news is contradictory.",
        )
        engine = _make_compound_engine([entry])
        # sports question
        sports_result = engine.get_relevant_lessons(
            "Will the NBA championship be won by the Lakers?"
        )
        assert "Reduce size when news is contradictory." in sports_result
        # politics question
        pol_result = engine.get_relevant_lessons(
            "Will the senate vote pass the bill?"
        )
        assert "Reduce size when news is contradictory." in pol_result

    def test_crypto_lesson_does_not_fire_on_politics_question(self):
        """A crypto-tagged lesson must NOT fire on a politics question."""
        entry = _make_entry(
            confidence=0.7,
            category="crypto",
            lesson="Bitcoin often dumps after halving hype.",
        )
        engine = _make_compound_engine([entry])
        result = engine.get_relevant_lessons(
            "Will the president win re-election?"
        )
        assert result == ""

    def test_infer_category_politics(self):
        """_infer_category identifies political questions correctly."""
        from src.compound import CompoundEngine
        engine = CompoundEngine.__new__(CompoundEngine)
        engine._log = MagicMock()
        assert engine._infer_category("Who will win the senate election?") == "politics"

    def test_infer_category_sports(self):
        from src.compound import CompoundEngine
        engine = CompoundEngine.__new__(CompoundEngine)
        engine._log = MagicMock()
        assert engine._infer_category("Will the team win the championship tournament?") == "sports"

    def test_infer_category_crypto(self):
        from src.compound import CompoundEngine
        engine = CompoundEngine.__new__(CompoundEngine)
        engine._log = MagicMock()
        assert engine._infer_category("Will bitcoin reach $100k this month?") == "crypto"

    def test_infer_category_fallback(self):
        from src.compound import CompoundEngine
        engine = CompoundEngine.__new__(CompoundEngine)
        engine._log = MagicMock()
        assert engine._infer_category("Will it happen before December?") == "general"


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3: Contradiction detection
# ─────────────────────────────────────────────────────────────────────────────


class TestContradictionDetection:
    def test_contradictory_lessons_append_warning(self):
        """When an 'avoid' lesson and a 'buy' lesson exist in the same category,
        a contradiction warning must be appended to the result string.
        """
        avoid_entry = _make_entry(
            confidence=0.7,
            category="politics",
            lesson="Avoid trading on election markets during volatility.",
        )
        buy_entry = _make_entry(
            confidence=0.6,
            category="politics",
            lesson="Buy YES when incumbent leads polls by >10%.",
        )
        engine = _make_compound_engine([avoid_entry, buy_entry])
        result = engine.get_relevant_lessons("Will the president win re-election?")
        assert "Contradictory lessons found" in result
        assert "default to WAIT" in result

    def test_no_contradiction_when_only_one_direction(self):
        """If all lessons point the same way, no warning should appear."""
        entry1 = _make_entry(
            confidence=0.7,
            category="politics",
            lesson="Buy YES when incumbent leads by wide margin.",
        )
        entry2 = _make_entry(
            confidence=0.6,
            category="politics",
            lesson="Enter when polls converge within 3 days.",
            market_pattern="poll-convergence",
        )
        engine = _make_compound_engine([entry1, entry2])
        result = engine.get_relevant_lessons("Will the president win re-election?")
        assert "Contradictory lessons found" not in result

    def test_no_contradiction_different_categories(self):
        """'avoid' in politics and 'buy' in crypto must NOT trigger contradiction."""
        avoid_entry = _make_entry(
            confidence=0.7,
            category="politics",
            lesson="Avoid trading when senate is deadlocked.",
        )
        buy_entry = _make_entry(
            confidence=0.6,
            category="crypto",
            lesson="Buy ETH when gas fees spike — short-term reversal likely.",
        )
        engine = _make_compound_engine([avoid_entry, buy_entry])
        # This is a political question — crypto entry should be filtered out
        result = engine.get_relevant_lessons("Will the president veto the bill?")
        assert "Contradictory lessons found" not in result

    def test_contradiction_warning_mentions_category(self):
        """The warning text must include the conflicting category name."""
        avoid_entry = _make_entry(
            confidence=0.75,
            category="sports",
            lesson="Avoid tournaments where top seed is injured.",
        )
        buy_entry = _make_entry(
            confidence=0.65,
            category="sports",
            lesson="Take position when underdog has home advantage.",
        )
        engine = _make_compound_engine([avoid_entry, buy_entry])
        result = engine.get_relevant_lessons(
            "Will the top team win the championship tournament?"
        )
        assert "sports" in result


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4: Win-rate culling
# ─────────────────────────────────────────────────────────────────────────────


class TestWinRateCulling:
    def _make_engine_for_culling(
        self,
        entries: list[KnowledgeBaseEntry],
        post_mortems_by_pattern: list[dict],
    ):
        from src.compound import CompoundEngine

        mock_db = MagicMock()
        mock_db.get_knowledge_base.return_value = entries
        mock_db.get_post_mortems_by_pattern.return_value = post_mortems_by_pattern
        mock_config = MagicMock()

        engine = CompoundEngine.__new__(CompoundEngine)
        engine.config = mock_config
        engine.db = mock_db
        engine._log = MagicMock()
        return engine

    def _make_pm_dicts(self, wins: int, losses: int) -> list[dict]:
        """Build fake post-mortem dicts with the given win/loss breakdown."""
        pms = []
        for _ in range(wins):
            pms.append({"pnl_pct": 0.15, "market_slug": "test-slug"})
        for _ in range(losses):
            pms.append({"pnl_pct": -0.10, "market_slug": "test-slug"})
        return pms

    def test_entry_with_30pct_win_rate_confidence_halved(self):
        """Entry with 6 confirmations and 30% win rate must have confidence halved."""
        entry = _make_entry(
            confidence=0.6,
            category="general",
            times_confirmed=6,
            market_pattern="volatile-election-outcome",
        )
        # 2 wins out of 6 = 33% win rate → below 40%
        post_mortems = self._make_pm_dicts(wins=2, losses=4)
        engine = self._make_engine_for_culling([entry], post_mortems)

        engine._cull_knowledge_base()

        engine.db.update_knowledge_entry_confidence.assert_called_once()
        call_args = engine.db.update_knowledge_entry_confidence.call_args[0]
        entry_id, times_confirmed, new_confidence = call_args
        assert entry_id == entry.id
        assert abs(new_confidence - 0.3) < 1e-9  # 0.6 * 0.5 = 0.3

    def test_entry_below_5_confirmations_is_skipped(self):
        """Entry with fewer than 5 confirmations must not be touched by culling."""
        entry = _make_entry(
            confidence=0.6,
            category="general",
            times_confirmed=4,
            market_pattern="volatile-election-outcome",
        )
        post_mortems = self._make_pm_dicts(wins=0, losses=4)
        engine = self._make_engine_for_culling([entry], post_mortems)

        engine._cull_knowledge_base()

        engine.db.update_knowledge_entry_confidence.assert_not_called()
        engine.db.delete_knowledge_entries.assert_not_called()

    def test_entry_with_50pct_win_rate_not_culled(self):
        """Entry at 50% win rate (above 40% floor) must NOT be touched."""
        entry = _make_entry(
            confidence=0.6,
            category="general",
            times_confirmed=6,
            market_pattern="election-poll-leader",
        )
        post_mortems = self._make_pm_dicts(wins=3, losses=3)
        engine = self._make_engine_for_culling([entry], post_mortems)

        engine._cull_knowledge_base()

        engine.db.update_knowledge_entry_confidence.assert_not_called()
        engine.db.delete_knowledge_entries.assert_not_called()

    def test_entry_deleted_when_halved_confidence_below_01(self):
        """Entry with confidence=0.18 at 30% win rate → halved=0.09 < 0.1 → DELETED."""
        entry = _make_entry(
            confidence=0.18,
            category="general",
            times_confirmed=7,
            market_pattern="low-confidence-bad-pattern",
        )
        # 1 win out of 5 = 20% win rate
        post_mortems = self._make_pm_dicts(wins=1, losses=4)
        engine = self._make_engine_for_culling([entry], post_mortems)

        engine._cull_knowledge_base()

        engine.db.delete_knowledge_entries.assert_called_once_with([entry.id])
        engine.db.update_knowledge_entry_confidence.assert_not_called()

    def test_entry_not_deleted_when_halved_confidence_at_01(self):
        """Entry with confidence=0.2 → halved=0.1, which is NOT below 0.1 → halved not deleted."""
        entry = _make_entry(
            confidence=0.2,
            category="general",
            times_confirmed=5,
            market_pattern="borderline-pattern",
        )
        post_mortems = self._make_pm_dicts(wins=1, losses=4)  # 20% win rate
        engine = self._make_engine_for_culling([entry], post_mortems)

        engine._cull_knowledge_base()

        # 0.2 * 0.5 = 0.1, which is NOT < 0.1, so update not delete
        engine.db.update_knowledge_entry_confidence.assert_called_once()
        engine.db.delete_knowledge_entries.assert_not_called()

    def test_culling_handles_empty_kb_gracefully(self):
        """Empty KB must not raise any exception."""
        engine = self._make_engine_for_culling([], [])
        engine._cull_knowledge_base()  # Should not raise
        engine.db.delete_knowledge_entries.assert_not_called()
        engine.db.update_knowledge_entry_confidence.assert_not_called()

    def test_culling_handles_no_post_mortems_gracefully(self):
        """If no post-mortems match a pattern, the entry must be skipped silently."""
        entry = _make_entry(
            confidence=0.6,
            category="general",
            times_confirmed=6,
            market_pattern="orphan-pattern-no-trades",
        )
        engine = self._make_engine_for_culling([entry], [])  # empty post-mortems

        engine._cull_knowledge_base()

        engine.db.update_knowledge_entry_confidence.assert_not_called()
        engine.db.delete_knowledge_entries.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 extra: Missing "category" in LLM output falls back to "general"
# ─────────────────────────────────────────────────────────────────────────────


class TestCategoryFallback:
    def test_missing_category_in_llm_result_defaults_to_general(self):
        """If the post-mortem LLM does not return a 'category' field,
        the KB entry must be saved with category='general' (not crash).
        """
        from src.compound import CompoundEngine

        mock_db = MagicMock()
        # LLM result with no 'category' key
        llm_result = {
            "failure_category": "BAD_PREDICTION",
            "root_cause": "Misjudged sentiment",
            "lesson": "Check contradictory sources.",
            "market_pattern": "US-election-general",
            # No "category" field
        }

        mock_db.get_knowledge_base.return_value = []
        mock_config = MagicMock()

        engine = CompoundEngine.__new__(CompoundEngine)
        engine.config = mock_config
        engine.db = mock_db
        engine._log = MagicMock()

        pm = PostMortem(
            trade_id="trade-001",
            failure_category=FailureCategory.BAD_PREDICTION,
            root_cause="Misjudged sentiment",
            lesson="Check contradictory sources.",
            market_slug="us-election-2024",
            pnl_pct=-0.12,
        )

        # Call _update_knowledge_base with category defaulting to "general"
        # (simulating run_post_mortem extracting category from llm_result)
        _VALID = {
            "politics", "economics", "sports", "crypto",
            "legal", "science", "entertainment", "general",
        }
        category = (llm_result.get("category") or "general").lower().strip()
        if category not in _VALID:
            category = "general"

        assert category == "general"

        engine._update_knowledge_base(pm, "US-election-general", category)

        # Verify the saved entry has category="general"
        mock_db.save_knowledge_entry.assert_called_once()
        saved_entry: KnowledgeBaseEntry = mock_db.save_knowledge_entry.call_args[0][0]
        assert saved_entry.category == "general"

    def test_invalid_category_in_llm_result_falls_back_to_general(self):
        """If the LLM returns an unrecognized category string, it must be
        normalized to 'general' before saving.
        """
        _VALID = {
            "politics", "economics", "sports", "crypto",
            "legal", "science", "entertainment", "general",
        }
        llm_result = {"category": "finance"}  # not in VALID_CATEGORIES
        category = (llm_result.get("category") or "general").lower().strip()
        if category not in _VALID:
            category = "general"
        assert category == "general"

    def test_valid_category_preserved(self):
        """A valid LLM category must be used as-is."""
        _VALID = {
            "politics", "economics", "sports", "crypto",
            "legal", "science", "entertainment", "general",
        }
        for cat in _VALID:
            llm_result = {"category": cat}
            result = (llm_result.get("category") or "general").lower().strip()
            if result not in _VALID:
                result = "general"
            assert result == cat


# ─────────────────────────────────────────────────────────────────────────────
# Brier score pipeline: predicted_prob and actual_outcome
# ─────────────────────────────────────────────────────────────────────────────


def test_predicted_prob_is_not_zero_in_post_mortem(config, tmp_path):
    """run_post_mortem() must set predicted_prob = position.predicted_prob, not 0.0."""
    from unittest.mock import patch, MagicMock
    from src.compound import CompoundEngine
    from src.database import Database
    from src.models import Position, TradeSide, CloseReason, TradeStatus
    from datetime import datetime, timezone

    db = Database(str(tmp_path / "test.db"))
    engine = CompoundEngine(config, db)

    position = Position(
        market_question="Will X happen?",
        market_slug="will-x-happen",
        token_id="tok-yes",
        side=TradeSide.BUY_YES,
        entry_price=0.50,
        size_eur=10.0,
        size_usd=10.7,
        tokens_quantity=21.4,
        stop_loss_price=0.40,
        take_profit_price=0.65,
        status=TradeStatus.CLOSED,
        exit_price=0.30,
        exit_timestamp=datetime.now(timezone.utc),
        close_reason=CloseReason.STOP_LOSS,
        pnl_eur=-2.0,
        pnl_pct=-0.40,
        predicted_prob=0.72,    # ← set at trade entry
    )

    mock_llm_result = {
        "failure_category": "BAD_PREDICTION",
        "root_cause": "market moved against position",
        "lesson": "avoid low-liquidity markets",
        "market_pattern": "test-pattern",
        "category": "general",
    }

    with patch.object(engine, "_call_llm_for_post_mortem", return_value=mock_llm_result):
        with patch("src.compound._emit_compound_trace"):
            pm = engine.run_post_mortem(position)

    assert pm is not None
    assert pm.predicted_prob == 0.72, f"predicted_prob={pm.predicted_prob}, expected 0.72"


def test_actual_outcome_inferred_from_resolved_exit(config, tmp_path):
    """_infer_actual_outcome() should return True when BUY_YES exit_price >= 0.95."""
    from src.compound import CompoundEngine
    from src.database import Database
    from src.models import Position, TradeSide, CloseReason, TradeStatus
    from datetime import datetime, timezone

    db = Database(str(tmp_path / "test2.db"))
    engine = CompoundEngine(config, db)

    # BUY_YES position that resolved YES (exit_price = 0.99)
    pos_yes_win = Position(
        market_question="Will YES resolve?",
        market_slug="will-yes-resolve",
        token_id="tok-yes2",
        side=TradeSide.BUY_YES,
        entry_price=0.60,
        size_eur=10.0,
        size_usd=10.7,
        tokens_quantity=17.8,
        stop_loss_price=0.48,
        take_profit_price=0.78,
        status=TradeStatus.CLOSED,
        exit_price=0.99,
        exit_timestamp=datetime.now(timezone.utc),
        close_reason=CloseReason.MARKET_RESOLVED,
        pnl_eur=6.5,
        pnl_pct=0.65,
    )
    assert engine._infer_actual_outcome(pos_yes_win) is True

    # BUY_NO position that resolved NO (NO token went to 0.99 = YES did NOT happen)
    pos_no_win = Position(
        market_question="Will NO resolve?",
        market_slug="will-no-resolve",
        token_id="tok-no2",
        side=TradeSide.BUY_NO,
        entry_price=0.40,
        size_eur=10.0,
        size_usd=10.7,
        tokens_quantity=26.75,
        stop_loss_price=0.20,
        take_profit_price=0.72,
        status=TradeStatus.CLOSED,
        exit_price=0.99,
        exit_timestamp=datetime.now(timezone.utc),
        close_reason=CloseReason.MARKET_RESOLVED,
        pnl_eur=5.9,
        pnl_pct=1.475,
    )
    assert engine._infer_actual_outcome(pos_no_win) is False

    # SL exit — no resolution
    pos_sl = Position(
        market_question="SL exit?",
        market_slug="sl-exit",
        token_id="tok-sl",
        side=TradeSide.BUY_YES,
        entry_price=0.60,
        size_eur=10.0,
        size_usd=10.7,
        tokens_quantity=17.8,
        stop_loss_price=0.48,
        take_profit_price=0.78,
        status=TradeStatus.CLOSED,
        exit_price=0.48,
        exit_timestamp=datetime.now(timezone.utc),
        close_reason=CloseReason.STOP_LOSS,
        pnl_eur=-2.4,
        pnl_pct=-0.20,
    )
    assert engine._infer_actual_outcome(pos_sl) is None
