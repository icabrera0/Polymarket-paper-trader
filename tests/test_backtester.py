"""
Tests del Backtester.

Cubren:
- Detección de mercados resueltos vs no resueltos.
- Parsing de precios de resolución (YES ganó / NO ganó).
- Pipeline completo con clientes fake (sin red ni LLM real).
- Cálculo de métricas agregadas (win rate, P&L, Sharpe).

Sin red. Sin LLM real. Sin Polymarket.

Ejecutar:
    pytest tests/test_backtester.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.backtester import Backtester, BacktestTrade, BacktestResult
from src.models import DecisionAction, TradeSide


# =====================================================
# Helpers
# =====================================================


def make_resolved_market(
    market_id: str = "m1",
    question: str = "Will Spain win the Euro 2028?",
    yes_won: bool = True,
    volume_24h: float = 15000.0,
    volume_total: float = 100000.0,
) -> dict:
    yes_price = "1.0" if yes_won else "0.0"
    no_price = "0.0" if yes_won else "1.0"
    return {
        "id": market_id,
        "question": question,
        "slug": question.lower().replace(" ", "-"),
        "description": "",
        "category": "Sports",
        "endDate": "2026-07-01T22:00:00Z",
        "active": False,
        "closed": True,
        "clobTokenIds": json.dumps(["0xyes", "0xno"]),
        "outcomePrices": json.dumps([yes_price, no_price]),
        "volume24hr": str(volume_24h),
        "volumeNum": volume_total,
        "liquidityNum": 10000.0,
    }


# =====================================================
# Tests de detección de mercados resueltos
# =====================================================


class TestIsTrulyResolved:
    def test_yes_gano(self):
        m = make_resolved_market(yes_won=True)
        assert Backtester._is_truly_resolved(m) is True

    def test_no_gano(self):
        m = make_resolved_market(yes_won=False)
        assert Backtester._is_truly_resolved(m) is True

    def test_mercado_no_resuelto(self):
        m = make_resolved_market()
        m["outcomePrices"] = json.dumps(["0.6", "0.4"])
        assert Backtester._is_truly_resolved(m) is False

    def test_sin_precios(self):
        m = make_resolved_market()
        del m["outcomePrices"]
        assert Backtester._is_truly_resolved(m) is False

    def test_precios_malformados(self):
        m = make_resolved_market()
        m["outcomePrices"] = "not_json"
        assert Backtester._is_truly_resolved(m) is False


# =====================================================
# Tests de parsing
# =====================================================


class TestParseResolvedMarket:
    def test_parsea_mercado_yes_gano(self, config):
        bt = Backtester(config, mode="current", max_markets=1)
        m = make_resolved_market(yes_won=True)
        snap, resolved_yes = bt._parse_resolved_market(m)
        assert snap is not None
        assert resolved_yes is True
        assert snap.yes_price == 0.50  # precio simulado pre-resolución
        assert snap.yes_token_id == "0xyes"

    def test_parsea_mercado_no_gano(self, config):
        bt = Backtester(config, mode="current", max_markets=1)
        m = make_resolved_market(yes_won=False)
        snap, resolved_yes = bt._parse_resolved_market(m)
        assert snap is not None
        assert resolved_yes is False

    def test_descarta_mercado_sin_pregunta(self, config):
        bt = Backtester(config, mode="current", max_markets=1)
        m = make_resolved_market()
        m["question"] = ""
        snap, _ = bt._parse_resolved_market(m)
        assert snap is None

    def test_descarta_mercado_sin_tokens(self, config):
        bt = Backtester(config, mode="current", max_markets=1)
        m = make_resolved_market()
        m["clobTokenIds"] = json.dumps(["only_one"])  # solo 1 token
        snap, _ = bt._parse_resolved_market(m)
        assert snap is None


# =====================================================
# Tests de extracción de keywords
# =====================================================


class TestExtractKeywords:
    def test_extrae_entidades(self):
        kws = Backtester._extract_keywords("Will Trump win the 2028 election?")
        assert "Trump" in kws

    def test_excluye_stopwords(self):
        kws = Backtester._extract_keywords("Will the election happen?")
        assert "will" not in kws
        assert "the" not in kws

    def test_limite_de_keywords(self):
        kws = Backtester._extract_keywords(
            "Will Trump Biden Obama Clinton Santos win?", max_kw=3
        )
        assert len(kws) <= 3


# =====================================================
# Tests de métricas
# =====================================================


class TestMetrics:
    def test_sharpe_serie_uniforme(self):
        """Si todos los P&L son iguales, std=0 → Sharpe=0."""
        trades = [
            BacktestTrade(
                market_id=f"m{i}", market_question="Q",
                resolved_yes=True, entry_price_simulated=0.5,
                exit_price=1.0, side=TradeSide.BUY_YES,
                size_eur=10.0, pnl_eur=1.0, pnl_pct=0.1,
                confidence=80, edge=0.1, num_articles=2,
                is_low_info=False, decision=DecisionAction.OPEN_TRADE,
            )
            for i in range(5)
        ]
        assert Backtester._calculate_sharpe(trades) == 0.0

    def test_sharpe_con_varianza(self):
        """Con P&L variables debería calcular algo distinto de 0."""
        import random
        random.seed(42)
        trades = [
            BacktestTrade(
                market_id=f"m{i}", market_question="Q",
                resolved_yes=True, entry_price_simulated=0.5,
                exit_price=1.0, side=TradeSide.BUY_YES,
                size_eur=10.0,
                pnl_eur=random.uniform(-5, 10),
                pnl_pct=random.uniform(-0.5, 1.0),
                confidence=80, edge=0.1, num_articles=2,
                is_low_info=False, decision=DecisionAction.OPEN_TRADE,
            )
            for i in range(10)
        ]
        sharpe = Backtester._calculate_sharpe(trades)
        assert sharpe != 0.0

    def test_sharpe_menos_de_2_trades(self):
        trade = BacktestTrade(
            market_id="m1", market_question="Q",
            resolved_yes=True, entry_price_simulated=0.5,
            exit_price=1.0, side=TradeSide.BUY_YES,
            size_eur=10.0, pnl_eur=1.0, pnl_pct=0.1,
            confidence=80, edge=0.1, num_articles=2,
            is_low_info=False, decision=DecisionAction.OPEN_TRADE,
        )
        assert Backtester._calculate_sharpe([trade]) == 0.0
        assert Backtester._calculate_sharpe([]) == 0.0


# =====================================================
# Tests del BacktestResult
# =====================================================


class TestBacktestResult:
    def test_print_summary_no_crashea(self):
        """print_summary debe ejecutarse sin excepción aunque no haya trades."""
        result = BacktestResult(
            mode="current",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            initial_balance=150.0,
            final_balance=152.5,
            markets_analyzed=10,
            trades_executed=2,
            trades_won=1,
            trades_lost=1,
            total_pnl_eur=2.5,
            win_rate=0.5,
            avg_pnl_per_trade=1.25,
            max_drawdown_pct=0.05,
            sharpe_ratio=0.8,
            trades=[],
        )
        result.print_summary()  # No debe lanzar excepción
