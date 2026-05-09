"""
Tests del PaperTrader y Database.

Cubren:
- Apertura simulada con slippage correctamente aplicado
- Cálculo de P&L (gain y loss) con la fórmula de Polymarket
- Persistencia en SQLite (insert + update + queries)
- Anti-balance-negativo
- Restauración de posiciones abiertas tras reinicio
- Mass close

Usan SQLite en archivo temporal con tmp_path de pytest. Cada test es aislado.

Ejecutar:
    pytest tests/test_paper_trader.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.database import Database
from src.decision_engine import DecisionEngine
from src.models import (
    CloseReason,
    DecisionAction,
    Position,
    SkipReason,
    TradeDecision,
    TradeSide,
    TradeStatus,
)
from src.paper_trader import PaperTrader
from src.risk_manager import RiskManager


# =====================================================
# Fixtures
# =====================================================


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def db(db_path: Path) -> Database:
    return Database(db_path)


@pytest.fixture
def risk_manager(config) -> RiskManager:
    return RiskManager(config, initial_balance_eur=150.0)


@pytest.fixture
def trader(config, risk_manager, db_path) -> PaperTrader:
    db = Database(db_path)
    return PaperTrader(config, risk_manager, db=db)


def make_open_decision(
    market_question: str = "Will X happen?",
    side: TradeSide = TradeSide.BUY_YES,
    token_id: str = "0xyes",
    entry_price: float = 0.40,
    size_eur: float = 20.0,
    sl: float = 0.32,
    tp: float = 0.52,
) -> TradeDecision:
    return TradeDecision(
        action=DecisionAction.OPEN_TRADE,
        market_id="m1",
        market_question=market_question,
        side=side,
        token_id=token_id,
        entry_price=entry_price,
        size_eur=size_eur,
        stop_loss_price=sl,
        take_profit_price=tp,
        confidence=80,
        edge=0.20,
        rationale="test",
    )


# =====================================================
# Database
# =====================================================


class TestDatabase:
    def test_inserta_y_lee_trade(self, db):
        from datetime import datetime, timezone

        position = Position(
            market_question="Test market",
            token_id="0xabc",
            side=TradeSide.BUY_YES,
            entry_price=0.40,
            size_eur=20.0,
            size_usd=21.4,
            tokens_quantity=53.5,
            entry_timestamp=datetime.now(timezone.utc),
            stop_loss_price=0.32,
            take_profit_price=0.52,
            confidence=75,
        )
        assert db.insert_trade(position) is True

        all_trades = db.get_all_trades()
        assert len(all_trades) == 1
        assert all_trades[0].trade_id == position.trade_id
        assert all_trades[0].side == TradeSide.BUY_YES
        assert all_trades[0].confidence == 75

    def test_filtra_open_vs_closed(self, db):
        from datetime import datetime, timezone

        p_open = Position(
            market_question="open",
            token_id="0xa",
            side=TradeSide.BUY_YES,
            entry_price=0.50,
            size_eur=10.0, size_usd=10.7, tokens_quantity=21.4,
            entry_timestamp=datetime.now(timezone.utc),
            stop_loss_price=0.40, take_profit_price=0.65,
        )
        p_closed = Position(
            market_question="closed",
            token_id="0xb",
            side=TradeSide.BUY_NO,
            entry_price=0.50,
            size_eur=10.0, size_usd=10.7, tokens_quantity=21.4,
            entry_timestamp=datetime.now(timezone.utc),
            stop_loss_price=0.40, take_profit_price=0.65,
            status=TradeStatus.CLOSED,
            exit_price=0.45,
            exit_timestamp=datetime.now(timezone.utc),
            close_reason=CloseReason.STOP_LOSS,
            pnl_eur=-1.0, pnl_pct=-0.10,
        )
        db.insert_trade(p_open)
        db.insert_trade(p_closed)
        assert len(db.get_open_positions()) == 1
        assert db.get_open_positions()[0].market_question == "open"
        assert len(db.get_all_trades()) == 2

    def test_log_balance_se_persiste(self, db):
        assert db.log_balance(150.0, 150.0, 0.0, 0, "TEST") is True
        history = db.get_balance_history()
        assert len(history) == 1
        assert history[0]["balance_eur"] == 150.0

    def test_log_decision_se_persiste(self, db):
        decision = TradeDecision(
            action=DecisionAction.NO_TRADE,
            market_id="m1",
            market_question="test",
            skip_reasons=[SkipReason.LLM_RECOMMENDS_WAIT],
            rationale="LLM says wait",
        )
        assert db.log_decision(decision) is True


# =====================================================
# PaperTrader: apertura
# =====================================================


class TestExecuteDecision:
    def test_no_trade_no_abre_nada(self, trader):
        decision = TradeDecision(
            action=DecisionAction.NO_TRADE,
            market_id="m1",
            market_question="x",
        )
        result = trader.execute_decision(decision)
        assert result is None
        assert trader.num_open_positions == 0

    def test_open_trade_abre_posicion(self, trader):
        balance_before = trader.balance_eur
        decision = make_open_decision(size_eur=20.0)
        position = trader.execute_decision(decision)

        assert position is not None
        assert trader.num_open_positions == 1
        assert trader.balance_eur == pytest.approx(balance_before - 20.0)
        assert position.status == TradeStatus.OPEN

    def test_aplica_slippage_en_compra(self, trader):
        # config tiene simulated_slippage_pct=0.005 (0.5%)
        decision = make_open_decision(entry_price=0.40, size_eur=20.0)
        position = trader.execute_decision(decision)
        # Precio efectivo debe ser ligeramente superior al de la decisión
        assert position.entry_price > 0.40
        assert position.entry_price == pytest.approx(0.40 * 1.005)

    def test_calcula_tokens_correctamente(self, trader):
        decision = make_open_decision(entry_price=0.40, size_eur=20.0)
        position = trader.execute_decision(decision)
        # size_usd ~ 20 * 1.07 = 21.4; precio efectivo 0.402; tokens ~ 53.23
        expected_tokens = (20.0 * 1.07) / (0.40 * 1.005)
        assert position.tokens_quantity == pytest.approx(expected_tokens, rel=0.001)

    def test_balance_insuficiente_retorna_none(self, config_factory):
        # Balance muy pequeño
        cfg = config_factory(paper_trading_overrides={"initial_balance_eur": 10.0})
        rm = RiskManager(cfg, 10.0)
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db = Database(tf.name)
            trader = PaperTrader(cfg, rm, db=db)
            decision = make_open_decision(size_eur=50.0)  # más que el balance
            result = trader.execute_decision(decision)
            assert result is None


# =====================================================
# PaperTrader: cierre
# =====================================================


class TestClosePosition:
    def test_cierre_con_ganancia(self, trader):
        decision = make_open_decision(entry_price=0.40, size_eur=20.0)
        position = trader.execute_decision(decision)
        balance_after_open = trader.balance_eur
        # Precio sube a 0.55 → ganancia ~37%
        closed = trader.close_position(
            position.trade_id,
            current_market_price=0.55,
            reason=CloseReason.TAKE_PROFIT,
        )
        assert closed is not None
        assert closed.status == TradeStatus.CLOSED
        assert closed.pnl_eur > 0
        assert closed.close_reason == CloseReason.TAKE_PROFIT
        assert trader.balance_eur > balance_after_open + 20.0  # devuelve principal + ganancia
        assert trader.num_open_positions == 0

    def test_cierre_con_perdida(self, trader):
        decision = make_open_decision(entry_price=0.40, size_eur=20.0)
        position = trader.execute_decision(decision)
        balance_after_open = trader.balance_eur
        # Precio cae a 0.30 → pérdida ~25%
        closed = trader.close_position(
            position.trade_id,
            current_market_price=0.30,
            reason=CloseReason.STOP_LOSS,
        )
        assert closed.pnl_eur < 0
        assert closed.close_reason == CloseReason.STOP_LOSS
        # Recupera principal menos pérdida
        assert trader.balance_eur < balance_after_open + 20.0
        assert trader.balance_eur > balance_after_open  # pero algo recupera

    def test_close_inexistente_retorna_none(self, trader):
        result = trader.close_position(
            "trade-id-fantasma",
            current_market_price=0.50,
            reason=CloseReason.MANUAL,
        )
        assert result is None

    def test_close_aplica_slippage_adverso(self, trader):
        decision = make_open_decision(entry_price=0.40, size_eur=20.0)
        position = trader.execute_decision(decision)
        # Cerramos al mismo precio de mercado: el efectivo debe ser MENOR
        # (vendes más barato por slippage)
        closed = trader.close_position(
            position.trade_id,
            current_market_price=0.40,
            reason=CloseReason.MANUAL,
        )
        # El P&L debe ser negativo por slippage de ambos lados aunque
        # entry y exit "de mercado" sean iguales
        assert closed.pnl_eur < 0


# =====================================================
# PaperTrader: mass close
# =====================================================


class TestCloseAllPositions:
    def test_cierra_todas_con_precios_dados(self, trader):
        d1 = make_open_decision(token_id="0xa", market_question="A?")
        d2 = make_open_decision(token_id="0xb", market_question="B?")
        p1 = trader.execute_decision(d1)
        p2 = trader.execute_decision(d2)
        prices = {p1.token_id: 0.50, p2.token_id: 0.45}
        closed = trader.close_all_positions(prices, reason=CloseReason.MANUAL)
        assert len(closed) == 2
        assert trader.num_open_positions == 0

    def test_omite_si_no_hay_precio(self, trader):
        d1 = make_open_decision(token_id="0xa")
        p1 = trader.execute_decision(d1)
        # No proporcionamos precio para p1
        closed = trader.close_all_positions({"otro_token": 0.50})
        assert len(closed) == 0
        assert trader.num_open_positions == 1


# =====================================================
# Restauración tras reinicio
# =====================================================


class TestRestoration:
    def test_recupera_posiciones_abiertas_de_db(self, config, risk_manager, db_path):
        # Primera instancia: abre 2 posiciones
        db1 = Database(db_path)
        t1 = PaperTrader(config, risk_manager, db=db1)
        t1.execute_decision(make_open_decision(token_id="0x1"))
        t1.execute_decision(make_open_decision(token_id="0x2", market_question="B?"))
        balance_before_reset = t1.balance_eur
        db1.close()

        # Segunda instancia con el MISMO db_path: debe restaurar
        rm2 = RiskManager(config, 150.0)
        db2 = Database(db_path)
        t2 = PaperTrader(config, rm2, db=db2)

        assert t2.num_open_positions == 2
        # Balance restaurado del último snapshot
        assert t2.balance_eur == pytest.approx(balance_before_reset, abs=0.01)


# =====================================================
# Integración con DecisionEngine
# =====================================================


class TestIntegrationWithDecisionEngine:
    def test_pipeline_decision_a_position(self, config, risk_manager, db_path):
        """Encadena: TradeDecision → execute_decision → Position en DB."""
        from src.models import (
            MarketAnalysis,
            Timeframe,
            TradeRecommendation,
        )

        db = Database(db_path)
        engine = DecisionEngine(config, risk_manager)
        trader = PaperTrader(config, risk_manager, db=db)

        # MarketAnalysis con recomendación clara
        analysis = MarketAnalysis(
            market_id="m1",
            market_question="Will X happen?",
            yes_token_id="0xy",
            no_token_id="0xn",
            current_yes_price=0.40,
            current_no_price=0.59,
            consensus_probability_yes=0.65,
            edge=0.25,
            confidence=85,
            sentiment_score=0.6,
            impact_score=80.0,
            recommendation=TradeRecommendation.COMPRAR_YES,
            timeframe=Timeframe.HORAS,
        )
        from src.models import NewsArticle, NewsSource, _new_article_id
        from datetime import datetime, timezone

        article = NewsArticle(
            article_id=_new_article_id("u", "t"),
            source=NewsSource.NEWSAPI, source_name="Reuters",
            title="t", url="u",
            published_at=datetime.now(timezone.utc),
        )

        # Decisión
        decision = engine.decide(
            analysis,
            current_balance_eur=trader.balance_eur,
            open_positions=trader.open_positions,
            articles=[article],
        )
        assert decision.action == DecisionAction.OPEN_TRADE

        # Ejecución
        position = trader.execute_decision(decision)
        assert position is not None
        assert position.side == TradeSide.BUY_YES

        # Verificar que está en la DB
        all_trades = db.get_all_trades()
        assert len(all_trades) == 1
        assert all_trades[0].trade_id == position.trade_id
