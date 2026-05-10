"""
Tests for the DecisionEngine.

Cover:
- Pre-decision filters: WAIT, INSUFFICIENT_DATA, low confidence, low edge.
- Dynamic sizing (more confidence/edge → larger size).
- Anti-duplication: same position and opposite position.
- Final validation by the RiskManager.
- Re-evaluation of open positions (stop loss, news reversal).
- require_news_for_entry.

No network. No LLM. Only deterministic logic of the engine.

Run:
    pytest tests/test_decision_engine.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest

from src.decision_engine import (
    MIN_CONFIDENCE_TO_OPEN,
    MIN_EDGE_TO_OPEN,
    DecisionEngine,
)
from src.models import (
    CloseReason,
    DecisionAction,
    MarketAnalysis,
    NewsArticle,
    NewsSource,
    Position,
    SkipReason,
    Timeframe,
    TradeRecommendation,
    TradeSide,
    _new_article_id,
)
from src.risk_manager import RiskManager


# =====================================================
# Helpers
# =====================================================


def make_analysis(
    market_id: str = "m1",
    question: str = "Will X happen?",
    yes_price: float = 0.40,
    no_price: float = 0.59,
    consensus: float = 0.60,
    confidence: int = 75,
    recommendation: TradeRecommendation = TradeRecommendation.BUY_YES,
    yes_token: str = "0xyes",
    no_token: str = "0xno",
) -> MarketAnalysis:
    return MarketAnalysis(
        market_id=market_id,
        market_question=question,
        yes_token_id=yes_token,
        no_token_id=no_token,
        current_yes_price=yes_price,
        current_no_price=no_price,
        consensus_probability_yes=consensus,
        edge=consensus - yes_price,
        confidence=confidence,
        sentiment_score=0.5,
        impact_score=70.0,
        recommendation=recommendation,
        timeframe=Timeframe.HOURS,
        contradictory_sources=False,
        summary="test",
        justification="test",
        article_ids_analyzed=["a1"],
        num_articles_analyzed=2,
    )


def make_article(title: str = "test news") -> NewsArticle:
    url = f"https://e.com/{title.replace(' ', '-')}"
    return NewsArticle(
        article_id=_new_article_id(url, title),
        source=NewsSource.NEWSAPI,
        source_name="Reuters",
        title=title,
        url=url,
        published_at=datetime.now(timezone.utc),
    )


def make_open_position(
    market_question: str = "Will X happen?",
    token_id: str = "0xyes",
    side: TradeSide = TradeSide.BUY_YES,
    entry_price: float = 0.40,
) -> Position:
    return Position(
        market_question=market_question,
        token_id=token_id,
        side=side,
        entry_price=entry_price,
        size_eur=20.0,
        size_usd=21.4,
        tokens_quantity=53.5,
        stop_loss_price=entry_price * 0.8,
        take_profit_price=min(0.999, entry_price * 1.3),
    )


@pytest.fixture
def risk_manager(config) -> RiskManager:
    return RiskManager(config, initial_balance_eur=150.0)


@pytest.fixture
def engine(config, risk_manager) -> DecisionEngine:
    return DecisionEngine(config, risk_manager)


# =====================================================
# Pre-decision filters
# =====================================================


class TestPreFilters:
    def test_no_trade_si_insufficient_data(self, engine, config_factory):
        # To avoid the require_news_for_entry filter, we disable it
        cfg = config_factory()
        cfg.decision.require_news_for_entry = False
        rm = RiskManager(cfg, 150.0)
        eng = DecisionEngine(cfg, rm)
        a = make_analysis(recommendation=TradeRecommendation.INSUFFICIENT_DATA)
        d = eng.decide(a, current_balance_eur=150.0, open_positions=[])
        assert d.action == DecisionAction.NO_TRADE
        assert SkipReason.LLM_INSUFFICIENT_DATA in d.skip_reasons

    def test_no_trade_si_esperar(self, engine):
        a = make_analysis(recommendation=TradeRecommendation.WAIT)
        d = engine.decide(a, 150.0, [])
        assert d.action == DecisionAction.NO_TRADE
        assert SkipReason.LLM_RECOMMENDS_WAIT in d.skip_reasons

    def test_no_trade_si_confianza_baja(self, engine):
        a = make_analysis(confidence=MIN_CONFIDENCE_TO_OPEN - 1)
        d = engine.decide(a, 150.0, [], articles=[make_article()])
        assert d.action == DecisionAction.NO_TRADE
        assert SkipReason.LLM_RECOMMENDS_WAIT in d.skip_reasons
        assert "Confidence" in d.rationale

    def test_no_trade_si_edge_pequeno(self, engine):
        # consensus 0.43, price 0.40 → edge 0.03 < 0.05
        a = make_analysis(yes_price=0.40, consensus=0.43, confidence=80)
        d = engine.decide(a, 150.0, [], articles=[make_article()])
        assert d.action == DecisionAction.NO_TRADE
        assert "Edge" in d.rationale

    def test_acepta_si_todo_ok(self, engine):
        a = make_analysis(yes_price=0.40, consensus=0.60, confidence=80)
        d = engine.decide(a, 150.0, [], articles=[make_article()])
        assert d.action == DecisionAction.OPEN_TRADE
        assert d.side == TradeSide.BUY_YES
        assert d.size_eur > 0


# =====================================================
# Require news
# =====================================================


class TestRequireNews:
    def test_rechaza_sin_noticias_cuando_se_requieren(self, config_factory):
        cfg = config_factory()
        cfg.decision.require_news_for_entry = True
        rm = RiskManager(cfg, 150.0)
        eng = DecisionEngine(cfg, rm)
        a = make_analysis(yes_price=0.40, consensus=0.60, confidence=80)
        d = eng.decide(a, 150.0, [], articles=None)
        assert d.action == DecisionAction.NO_TRADE
        assert SkipReason.REQUIRE_NEWS_BUT_NONE in d.skip_reasons

    def test_acepta_sin_noticias_cuando_no_se_requieren(self, config_factory):
        cfg = config_factory()
        cfg.decision.require_news_for_entry = False
        rm = RiskManager(cfg, 150.0)
        eng = DecisionEngine(cfg, rm)
        a = make_analysis(yes_price=0.40, consensus=0.60, confidence=80)
        d = eng.decide(a, 150.0, [], articles=None)
        assert d.action == DecisionAction.OPEN_TRADE


# =====================================================
# Anti-duplication
# =====================================================


class TestAntiDuplication:
    def test_rechaza_si_ya_hay_posicion_en_mismo_token(self, engine):
        existing = make_open_position(
            market_question="Will X happen?",
            token_id="0xyes",
        )
        a = make_analysis(
            question="Will X happen?",
            yes_token="0xyes",
            yes_price=0.40,
            consensus=0.60,
            confidence=80,
        )
        d = engine.decide(a, 150.0, [existing], articles=[make_article()])
        assert d.action == DecisionAction.NO_TRADE
        assert SkipReason.DUPLICATE_OPEN_POSITION in d.skip_reasons

    def test_rechaza_si_hay_posicion_en_lado_opuesto(self, engine):
        # I have BUY_NO, and the LLM recommends BUY_YES → contradictory
        existing = make_open_position(
            market_question="Will X happen?",
            token_id="0xno",
            side=TradeSide.BUY_NO,
        )
        a = make_analysis(
            question="Will X happen?",
            yes_token="0xyes",
            no_token="0xno",
            yes_price=0.40,
            consensus=0.60,
            confidence=80,
            recommendation=TradeRecommendation.BUY_YES,
        )
        d = engine.decide(a, 150.0, [existing], articles=[make_article()])
        assert d.action == DecisionAction.NO_TRADE
        assert SkipReason.OPPOSITE_OPEN_POSITION in d.skip_reasons

    def test_acepta_si_posicion_existe_pero_en_otro_mercado(self, engine):
        existing = make_open_position(
            market_question="Will Y happen?",
            token_id="0xother",
        )
        a = make_analysis(
            question="Will X happen?",
            yes_price=0.40,
            consensus=0.60,
            confidence=80,
        )
        d = engine.decide(a, 150.0, [existing], articles=[make_article()])
        assert d.action == DecisionAction.OPEN_TRADE


# =====================================================
# Dynamic sizing
# =====================================================


class TestSizing:
    def test_mayor_confianza_mayor_tamano(self, engine):
        # Same edge, different confidence → more confidence, larger size
        a_low = make_analysis(yes_price=0.40, consensus=0.60, confidence=60)
        a_hi = make_analysis(yes_price=0.40, consensus=0.60, confidence=100)
        d_low = engine.decide(a_low, 150.0, [], articles=[make_article()])
        d_hi = engine.decide(a_hi, 150.0, [], articles=[make_article()])
        assert d_low.action == DecisionAction.OPEN_TRADE
        assert d_hi.action == DecisionAction.OPEN_TRADE
        assert d_hi.size_eur > d_low.size_eur

    def test_mayor_edge_mayor_tamano(self, engine):
        a_small = make_analysis(yes_price=0.40, consensus=0.50, confidence=80)  # edge 0.10
        a_big = make_analysis(yes_price=0.40, consensus=0.70, confidence=80)    # edge 0.30
        d_small = engine.decide(a_small, 150.0, [], articles=[make_article()])
        d_big = engine.decide(a_big, 150.0, [], articles=[make_article()])
        assert d_big.size_eur > d_small.size_eur

    def test_size_no_supera_max_del_risk_manager(self, engine, risk_manager):
        # Confidence 100 + edge 0.5 → should cap at the maximum
        a = make_analysis(yes_price=0.30, consensus=0.80, confidence=100)
        d = engine.decide(a, 150.0, [], articles=[make_article()])
        max_allowed = risk_manager.calculate_max_position_size(150.0)
        assert d.size_eur <= max_allowed + 0.01

    def test_size_nunca_bajo_minimo(self, engine):
        # Extreme case: minimum edge + minimum confidence → should be near the min
        a = make_analysis(
            yes_price=0.40,
            consensus=0.45,  # edge 0.05 (exactly the limit)
            confidence=MIN_CONFIDENCE_TO_OPEN,
        )
        d = engine.decide(a, 150.0, [], articles=[make_article()])
        if d.action == DecisionAction.OPEN_TRADE:
            assert d.size_eur >= engine.risk_manager.risk.min_trade_size_eur


# =====================================================
# RiskManager veto
# =====================================================


class TestRiskManagerIntegration:
    def test_rechaza_si_risk_manager_rechaza(self, config_factory):
        cfg = config_factory()
        rm = RiskManager(cfg, 150.0)
        # Force pause by simulating drawdown
        rm.update_balance_and_check_drawdown(80.0)
        assert rm.is_paused
        eng = DecisionEngine(cfg, rm)
        a = make_analysis(yes_price=0.40, consensus=0.60, confidence=80)
        d = eng.decide(a, 80.0, [], articles=[make_article()])
        assert d.action == DecisionAction.NO_TRADE
        assert SkipReason.RISK_MANAGER_REJECTED in d.skip_reasons

    def test_rechaza_si_demasiadas_posiciones(self, engine):
        # Create 3 positions in different markets (at the limit)
        positions = [
            make_open_position(
                market_question=f"Will event {i} happen?",
                token_id=f"0xtok{i}",
            )
            for i in range(3)
        ]
        a = make_analysis(yes_price=0.40, consensus=0.60, confidence=80)
        d = engine.decide(a, 150.0, positions, articles=[make_article()])
        assert d.action == DecisionAction.NO_TRADE
        assert SkipReason.RISK_MANAGER_REJECTED in d.skip_reasons


# =====================================================
# Trade side (BUY_YES vs BUY_NO)
# =====================================================


class TestSideResolution:
    def test_compra_yes_usa_yes_token_y_yes_price(self, engine):
        a = make_analysis(
            yes_price=0.40,
            no_price=0.59,
            consensus=0.60,
            confidence=80,
            yes_token="0xyes_token",
            no_token="0xno_token",
            recommendation=TradeRecommendation.BUY_YES,
        )
        d = engine.decide(a, 150.0, [], articles=[make_article()])
        assert d.side == TradeSide.BUY_YES
        assert d.token_id == "0xyes_token"
        assert d.entry_price == pytest.approx(0.40)

    def test_compra_no_usa_no_token_y_no_price(self, engine):
        # consensus_yes 0.20 → implicit consensus_no 0.80; price_yes 0.40 →
        # negative edge (NO undervalued) → BUY_NO
        a = make_analysis(
            yes_price=0.40,
            no_price=0.59,
            consensus=0.20,           # implies edge -0.20
            confidence=80,
            yes_token="0xyes_token",
            no_token="0xno_token",
            recommendation=TradeRecommendation.BUY_NO,
        )
        d = engine.decide(a, 150.0, [], articles=[make_article()])
        assert d.side == TradeSide.BUY_NO
        assert d.token_id == "0xno_token"
        assert d.entry_price == pytest.approx(0.59)


# =====================================================
# Re-evaluation of open positions
# =====================================================


class TestEvaluateOpenPosition:
    def test_dispara_stop_loss(self, engine):
        position = make_open_position(entry_price=0.40)  # SL at 0.32
        decision = engine.evaluate_open_position(position, current_price=0.30)
        assert decision.should_close
        assert decision.reason == CloseReason.STOP_LOSS

    def test_dispara_take_profit(self, engine):
        position = make_open_position(entry_price=0.40)  # TP at 0.52
        decision = engine.evaluate_open_position(position, current_price=0.55)
        assert decision.should_close
        assert decision.reason == CloseReason.TAKE_PROFIT

    def test_news_reversal_cierra(self, engine):
        position = make_open_position(
            market_question="Will X?",
            side=TradeSide.BUY_YES,
            entry_price=0.40,
        )
        # New analysis recommends the opposite with high confidence
        new_a = make_analysis(
            question="Will X?",
            yes_price=0.42,
            consensus=0.20,
            confidence=85,
            recommendation=TradeRecommendation.BUY_NO,
        )
        decision = engine.evaluate_open_position(
            position, current_price=0.42, new_analysis=new_a
        )
        assert decision.should_close
        assert decision.reason == CloseReason.NEWS_REVERSAL

    def test_news_baja_confianza_no_cierra(self, engine):
        position = make_open_position(side=TradeSide.BUY_YES, entry_price=0.40)
        new_a = make_analysis(
            yes_price=0.42,
            consensus=0.20,
            confidence=50,  # < 70 → does not close
            recommendation=TradeRecommendation.BUY_NO,
        )
        decision = engine.evaluate_open_position(
            position, current_price=0.42, new_analysis=new_a
        )
        assert not decision.should_close

    def test_news_misma_direccion_no_cierra(self, engine):
        position = make_open_position(side=TradeSide.BUY_YES, entry_price=0.40)
        # New analysis confirms BUY_YES → does not close
        new_a = make_analysis(
            yes_price=0.42,
            consensus=0.70,
            confidence=85,
            recommendation=TradeRecommendation.BUY_YES,
        )
        decision = engine.evaluate_open_position(
            position, current_price=0.42, new_analysis=new_a
        )
        assert not decision.should_close

    def test_sin_nueva_info_solo_chequea_niveles(self, engine):
        position = make_open_position(entry_price=0.40)
        # Price within levels, no new info → does not close
        decision = engine.evaluate_open_position(position, current_price=0.42)
        assert not decision.should_close


# =====================================================
# Expected Value
# =====================================================


def test_expected_value_computed_for_buy_signal(config):
    """decide() should populate analysis.expected_value for BUY signals."""
    from src.decision_engine import DecisionEngine
    from src.risk_manager import RiskManager
    from src.models import (
        MarketAnalysis, TradeRecommendation, DecisionAction
    )

    rm = RiskManager(config, initial_balance_eur=150.0)
    engine = DecisionEngine(config, rm)

    # p=0.70, entry_price (YES) = 0.50 → b = (1/0.50)-1 = 1.0
    # EV = 0.70 * 1.0 - 0.30 = 0.40
    analysis = MarketAnalysis(
        market_id="m-ev",
        market_question="Will EV work?",
        yes_token_id="yes-ev",
        no_token_id="no-ev",
        current_yes_price=0.50,
        current_no_price=0.50,
        consensus_probability_yes=0.70,
        edge=0.20,
        confidence=75,
        sentiment_score=0.5,
        impact_score=60.0,
        recommendation=TradeRecommendation.BUY_YES,
    )

    decision = engine.decide(
        analysis,
        current_balance_eur=150.0,
        open_positions=[],
        articles=[make_article()],
    )
    assert decision.action == DecisionAction.OPEN_TRADE, f"Expected OPEN_TRADE, got {decision.action}: {decision.rationale}"
    assert abs(analysis.expected_value - 0.40) < 1e-4, (
        f"expected_value={analysis.expected_value}, expected ~0.40"
    )
