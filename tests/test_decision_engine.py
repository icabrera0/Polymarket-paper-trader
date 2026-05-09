"""
Tests del DecisionEngine.

Cubren:
- Filtros pre-decisión: ESPERAR, INSUFFICIENT_DATA, baja confianza, edge bajo.
- Sizing dinámico (más confianza/edge → más tamaño).
- Anti-duplicación: misma posición y posición opuesta.
- Validación final del RiskManager.
- Reevaluación de posiciones abiertas (stop loss, news reversal).
- require_news_for_entry.

Sin red. Sin LLM. Sólo lógica determinista del engine.

Ejecutar:
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
    recommendation: TradeRecommendation = TradeRecommendation.COMPRAR_YES,
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
        timeframe=Timeframe.HORAS,
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
# Filtros pre-decisión
# =====================================================


class TestPreFilters:
    def test_no_trade_si_insufficient_data(self, engine, config_factory):
        # Para evitar el filtro de require_news_for_entry, lo desactivamos
        cfg = config_factory()
        cfg.decision.require_news_for_entry = False
        rm = RiskManager(cfg, 150.0)
        eng = DecisionEngine(cfg, rm)
        a = make_analysis(recommendation=TradeRecommendation.INSUFFICIENT_DATA)
        d = eng.decide(a, current_balance_eur=150.0, open_positions=[])
        assert d.action == DecisionAction.NO_TRADE
        assert SkipReason.LLM_INSUFFICIENT_DATA in d.skip_reasons

    def test_no_trade_si_esperar(self, engine):
        a = make_analysis(recommendation=TradeRecommendation.ESPERAR)
        d = engine.decide(a, 150.0, [])
        assert d.action == DecisionAction.NO_TRADE
        assert SkipReason.LLM_RECOMMENDS_WAIT in d.skip_reasons

    def test_no_trade_si_confianza_baja(self, engine):
        a = make_analysis(confidence=MIN_CONFIDENCE_TO_OPEN - 1)
        d = engine.decide(a, 150.0, [], articles=[make_article()])
        assert d.action == DecisionAction.NO_TRADE
        assert SkipReason.LLM_RECOMMENDS_WAIT in d.skip_reasons
        assert "Confianza" in d.rationale

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
# Anti-duplicación
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
        # Tengo BUY_NO, y el LLM recomienda BUY_YES → contradictorio
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
            recommendation=TradeRecommendation.COMPRAR_YES,
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
# Sizing dinámico
# =====================================================


class TestSizing:
    def test_mayor_confianza_mayor_tamano(self, engine):
        # Mismo edge, diferente confianza → más confianza, más tamaño
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
        # Confianza 100 + edge 0.5 → debería topar en el máximo
        a = make_analysis(yes_price=0.30, consensus=0.80, confidence=100)
        d = engine.decide(a, 150.0, [], articles=[make_article()])
        max_allowed = risk_manager.calculate_max_position_size(150.0)
        assert d.size_eur <= max_allowed + 0.01

    def test_size_nunca_bajo_minimo(self, engine):
        # Caso extremo: edge mínimo + confianza mínima → debería ser cerca del mín
        a = make_analysis(
            yes_price=0.40,
            consensus=0.45,  # edge 0.05 (justo el límite)
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
        # Forzar pausa simulando drawdown
        rm.update_balance_and_check_drawdown(80.0)
        assert rm.is_paused
        eng = DecisionEngine(cfg, rm)
        a = make_analysis(yes_price=0.40, consensus=0.60, confidence=80)
        d = eng.decide(a, 80.0, [], articles=[make_article()])
        assert d.action == DecisionAction.NO_TRADE
        assert SkipReason.RISK_MANAGER_REJECTED in d.skip_reasons

    def test_rechaza_si_demasiadas_posiciones(self, engine):
        # Crear 3 posiciones en mercados distintos (al límite)
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
# Lado del trade (BUY_YES vs BUY_NO)
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
            recommendation=TradeRecommendation.COMPRAR_YES,
        )
        d = engine.decide(a, 150.0, [], articles=[make_article()])
        assert d.side == TradeSide.BUY_YES
        assert d.token_id == "0xyes_token"
        assert d.entry_price == pytest.approx(0.40)

    def test_compra_no_usa_no_token_y_no_price(self, engine):
        # consensus_yes 0.20 → consensus_no 0.80 implícito; precio_yes 0.40 →
        # edge negativo (NO infravalorado) → COMPRAR_NO
        a = make_analysis(
            yes_price=0.40,
            no_price=0.59,
            consensus=0.20,           # implica edge -0.20
            confidence=80,
            yes_token="0xyes_token",
            no_token="0xno_token",
            recommendation=TradeRecommendation.COMPRAR_NO,
        )
        d = engine.decide(a, 150.0, [], articles=[make_article()])
        assert d.side == TradeSide.BUY_NO
        assert d.token_id == "0xno_token"
        assert d.entry_price == pytest.approx(0.59)


# =====================================================
# Reevaluación de posiciones abiertas
# =====================================================


class TestEvaluateOpenPosition:
    def test_dispara_stop_loss(self, engine):
        position = make_open_position(entry_price=0.40)  # SL en 0.32
        decision = engine.evaluate_open_position(position, current_price=0.30)
        assert decision.should_close
        assert decision.reason == CloseReason.STOP_LOSS

    def test_dispara_take_profit(self, engine):
        position = make_open_position(entry_price=0.40)  # TP en 0.52
        decision = engine.evaluate_open_position(position, current_price=0.55)
        assert decision.should_close
        assert decision.reason == CloseReason.TAKE_PROFIT

    def test_news_reversal_cierra(self, engine):
        position = make_open_position(
            market_question="Will X?",
            side=TradeSide.BUY_YES,
            entry_price=0.40,
        )
        # Nuevo análisis recomienda lo contrario con alta confianza
        new_a = make_analysis(
            question="Will X?",
            yes_price=0.42,
            consensus=0.20,
            confidence=85,
            recommendation=TradeRecommendation.COMPRAR_NO,
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
            confidence=50,  # < 70 → no cierra
            recommendation=TradeRecommendation.COMPRAR_NO,
        )
        decision = engine.evaluate_open_position(
            position, current_price=0.42, new_analysis=new_a
        )
        assert not decision.should_close

    def test_news_misma_direccion_no_cierra(self, engine):
        position = make_open_position(side=TradeSide.BUY_YES, entry_price=0.40)
        # Nuevo análisis confirma BUY_YES → no cierra
        new_a = make_analysis(
            yes_price=0.42,
            consensus=0.70,
            confidence=85,
            recommendation=TradeRecommendation.COMPRAR_YES,
        )
        decision = engine.evaluate_open_position(
            position, current_price=0.42, new_analysis=new_a
        )
        assert not decision.should_close

    def test_sin_nueva_info_solo_chequea_niveles(self, engine):
        position = make_open_position(entry_price=0.40)
        # Precio dentro de niveles, sin nueva info → no cierra
        decision = engine.evaluate_open_position(position, current_price=0.42)
        assert not decision.should_close
