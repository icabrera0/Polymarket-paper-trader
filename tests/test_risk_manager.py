"""
Tests del RiskManager.

Cubren las reglas calibradas para 150 € de bankroll:
- Validación de tamaño (mín 5 €, máx 15% del balance)
- Límite de 3 posiciones simultáneas
- Stop loss a -20% / Take profit a +30%
- Drawdown máximo 30% → pausa el bot
- Reanudación manual

Ejecutar desde la raíz del proyecto:
    pytest tests/test_risk_manager.py -v

La fixture `config` viene de tests/conftest.py y construye un BotConfig de
prueba programáticamente, sin depender de config/settings.yaml.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models import (
    CloseReason,
    Position,
    RejectReason,
    TradeSide,
)
from src.risk_manager import RiskManager


# =====================================================
# Fixtures locales
# =====================================================


@pytest.fixture
def risk_manager(config):
    """RiskManager con balance inicial de 150 €. La fixture `config` viene
    de tests/conftest.py."""
    return RiskManager(config, initial_balance_eur=150.0)


def make_position(
    entry_price: float = 0.40,
    size_eur: float = 20.0,
    side: TradeSide = TradeSide.BUY_YES,
    stop_loss_price: float = 0.32,
    take_profit_price: float = 0.52,
) -> Position:
    """Helper para construir posiciones de test rápidamente."""
    size_usd = size_eur * 1.07  # rate del config
    return Position(
        market_question="Test market?",
        token_id="0xtest",
        side=side,
        entry_price=entry_price,
        size_eur=size_eur,
        size_usd=size_usd,
        tokens_quantity=size_usd / entry_price,
        entry_timestamp=datetime.now(timezone.utc),
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        confidence=75,
    )


# =====================================================
# Validación de trades nuevos
# =====================================================


class TestValidateNewTrade:
    def test_aprueba_trade_valido(self, risk_manager):
        result = risk_manager.validate_new_trade(
            proposed_size_eur=20.0,
            current_balance_eur=150.0,
            open_positions_count=0,
            entry_price=0.45,
        )
        assert result.approved is True
        assert result.rejection_reasons == []

    def test_rechaza_si_demasiadas_posiciones(self, risk_manager):
        result = risk_manager.validate_new_trade(
            proposed_size_eur=15.0,
            current_balance_eur=100.0,
            open_positions_count=3,  # ya en el límite
            entry_price=0.50,
        )
        assert result.approved is False
        assert RejectReason.MAX_POSITIONS_REACHED in result.rejection_reasons

    def test_rechaza_si_tamano_bajo_minimo(self, risk_manager):
        # 5 € es el mínimo configurado, 4 € debe fallar
        result = risk_manager.validate_new_trade(
            proposed_size_eur=4.0,
            current_balance_eur=150.0,
            open_positions_count=0,
            entry_price=0.50,
        )
        assert result.approved is False
        assert RejectReason.SIZE_BELOW_MIN in result.rejection_reasons

    def test_acepta_minimo_exacto(self, risk_manager):
        result = risk_manager.validate_new_trade(
            proposed_size_eur=5.0,
            current_balance_eur=150.0,
            open_positions_count=0,
            entry_price=0.50,
        )
        assert result.approved is True

    def test_recorta_si_excede_maximo(self, risk_manager):
        # 15% de 150 € = 22.5 €. Si pedimos 50 € debe recortar.
        result = risk_manager.validate_new_trade(
            proposed_size_eur=50.0,
            current_balance_eur=150.0,
            open_positions_count=0,
            entry_price=0.50,
        )
        assert result.approved is True
        assert result.adjusted_size_eur == pytest.approx(22.5)
        assert len(result.warnings) == 1

    def test_rechaza_precio_invalido(self, risk_manager):
        # Precio fuera de (0, 1)
        for bad_price in [0.0, 1.0, -0.1, 1.5]:
            result = risk_manager.validate_new_trade(
                proposed_size_eur=15.0,
                current_balance_eur=150.0,
                open_positions_count=0,
                entry_price=bad_price,
            )
            assert result.approved is False
            assert RejectReason.INVALID_PRICE in result.rejection_reasons

    def test_rechaza_si_balance_insuficiente(self, risk_manager):
        # Balance bajo, pedimos algo que el max_position_size permite pero el
        # balance no cubre.
        result = risk_manager.validate_new_trade(
            proposed_size_eur=10.0,
            current_balance_eur=8.0,  # menor que el size pedido
            open_positions_count=0,
            entry_price=0.50,
        )
        assert result.approved is False
        assert RejectReason.INSUFFICIENT_BALANCE in result.rejection_reasons

    def test_rechaza_si_bot_pausado(self, risk_manager):
        # Forzar pausa simulando un drawdown grande
        risk_manager.update_balance_and_check_drawdown(100.0)  # peak 150, ahora 100
        # 100/150 = 33% drawdown, supera el 30%
        assert risk_manager.is_paused is True

        result = risk_manager.validate_new_trade(
            proposed_size_eur=10.0,
            current_balance_eur=100.0,
            open_positions_count=0,
            entry_price=0.50,
        )
        assert result.approved is False
        assert RejectReason.BOT_PAUSED in result.rejection_reasons


# =====================================================
# Cálculos de tamaño y niveles
# =====================================================


class TestCalculations:
    def test_max_position_size(self, risk_manager):
        # 15% de 150 € = 22.5 €
        assert risk_manager.calculate_max_position_size(150.0) == pytest.approx(22.5)
        # 15% de 100 € = 15 €
        assert risk_manager.calculate_max_position_size(100.0) == pytest.approx(15.0)

    def test_stop_loss_price(self, risk_manager):
        # entry 0.40 con SL 20% → 0.32
        assert risk_manager.calculate_stop_loss_price(0.40) == pytest.approx(0.32)
        # entry 0.50 → 0.40
        assert risk_manager.calculate_stop_loss_price(0.50) == pytest.approx(0.40)

    def test_take_profit_price(self, risk_manager):
        # entry 0.40 con TP 30% → 0.52
        assert risk_manager.calculate_take_profit_price(0.40) == pytest.approx(0.52)

    def test_take_profit_capped_at_one(self, risk_manager):
        # entry 0.85, TP 30% sería 1.105, debe capar a 0.999
        assert risk_manager.calculate_take_profit_price(0.85) == pytest.approx(0.999)


# =====================================================
# Cierre de posiciones
# =====================================================


class TestShouldClosePosition:
    def test_no_cierra_si_dentro_de_niveles(self, risk_manager):
        position = make_position(entry_price=0.40, stop_loss_price=0.32, take_profit_price=0.52)
        decision = risk_manager.should_close_position(position, current_price=0.42)
        assert decision.should_close is False
        assert decision.reason is None
        assert decision.pnl_pct == pytest.approx((0.42 - 0.40) / 0.40)

    def test_dispara_stop_loss_exacto(self, risk_manager):
        position = make_position(entry_price=0.40, stop_loss_price=0.32, take_profit_price=0.52)
        decision = risk_manager.should_close_position(position, current_price=0.32)
        assert decision.should_close is True
        assert decision.reason == CloseReason.STOP_LOSS
        assert decision.pnl_pct == pytest.approx(-0.20)

    def test_dispara_stop_loss_por_debajo(self, risk_manager):
        position = make_position(entry_price=0.40, stop_loss_price=0.32, take_profit_price=0.52)
        decision = risk_manager.should_close_position(position, current_price=0.25)
        assert decision.should_close is True
        assert decision.reason == CloseReason.STOP_LOSS
        assert decision.pnl_pct < -0.20

    def test_dispara_take_profit(self, risk_manager):
        position = make_position(entry_price=0.40, stop_loss_price=0.32, take_profit_price=0.52)
        decision = risk_manager.should_close_position(position, current_price=0.55)
        assert decision.should_close is True
        assert decision.reason == CloseReason.TAKE_PROFIT
        assert decision.pnl_pct >= 0.30

    def test_funciona_para_buy_no(self, risk_manager):
        # BUY_NO usa la misma fórmula porque cada lado es su propio token
        position = make_position(
            entry_price=0.60,
            side=TradeSide.BUY_NO,
            stop_loss_price=0.48,
            take_profit_price=0.78,
        )
        # Precio del token NO baja a 0.48 → stop loss
        decision = risk_manager.should_close_position(position, current_price=0.48)
        assert decision.should_close is True
        assert decision.reason == CloseReason.STOP_LOSS

    def test_pnl_eur_correcto(self, risk_manager):
        position = make_position(entry_price=0.40, size_eur=20.0)
        # Precio sube de 0.40 a 0.50 → +25%
        decision = risk_manager.should_close_position(position, current_price=0.50)
        assert decision.pnl_pct == pytest.approx(0.25)
        assert decision.pnl_eur == pytest.approx(20.0 * 0.25)  # 5 €


# =====================================================
# Drawdown
# =====================================================


class TestDrawdown:
    def test_peak_inicial_es_balance_inicial(self, risk_manager):
        assert risk_manager.peak_balance_eur == 150.0

    def test_peak_se_actualiza_al_alza(self, risk_manager):
        status = risk_manager.update_balance_and_check_drawdown(180.0)
        assert risk_manager.peak_balance_eur == 180.0
        assert status.current_drawdown_pct == 0.0
        assert status.threshold_breached is False

    def test_peak_no_baja(self, risk_manager):
        risk_manager.update_balance_and_check_drawdown(180.0)
        risk_manager.update_balance_and_check_drawdown(160.0)
        assert risk_manager.peak_balance_eur == 180.0

    def test_drawdown_calculo_correcto(self, risk_manager):
        # Peak 150, balance 120 → drawdown 20%
        status = risk_manager.update_balance_and_check_drawdown(120.0)
        assert status.current_drawdown_pct == pytest.approx(0.20)
        assert status.threshold_breached is False
        assert risk_manager.is_paused is False

    def test_drawdown_supera_umbral_pausa_bot(self, risk_manager):
        # Peak 150, balance 100 → drawdown ~33% > 30%
        status = risk_manager.update_balance_and_check_drawdown(100.0)
        assert status.current_drawdown_pct > 0.30
        assert status.threshold_breached is True
        assert status.bot_should_pause is True
        assert risk_manager.is_paused is True
        assert risk_manager.pause_reason is not None

    def test_drawdown_30_exacto_pausa(self, risk_manager):
        # 150 * 0.70 = 105 → drawdown exactamente 30%
        status = risk_manager.update_balance_and_check_drawdown(105.0)
        assert status.current_drawdown_pct == pytest.approx(0.30)
        assert status.threshold_breached is True

    def test_no_se_despausa_solo(self, risk_manager):
        # Pausar
        risk_manager.update_balance_and_check_drawdown(100.0)
        assert risk_manager.is_paused is True
        # Aunque el balance se recupere, el bot sigue pausado
        risk_manager.update_balance_and_check_drawdown(150.0)
        assert risk_manager.is_paused is True


# =====================================================
# Control manual
# =====================================================


class TestManualControl:
    def test_resume_despausa(self, risk_manager):
        risk_manager.update_balance_and_check_drawdown(100.0)
        assert risk_manager.is_paused is True

        risk_manager.manually_resume()
        assert risk_manager.is_paused is False
        assert risk_manager.pause_reason is None

    def test_resume_sin_pausa_es_no_op(self, risk_manager):
        # No debe lanzar excepción aunque no estuviera pausado
        risk_manager.manually_resume()
        assert risk_manager.is_paused is False

    def test_reset_peak(self, risk_manager):
        risk_manager.update_balance_and_check_drawdown(180.0)
        assert risk_manager.peak_balance_eur == 180.0

        risk_manager.reset_peak(120.0)
        assert risk_manager.peak_balance_eur == 120.0

    def test_reset_peak_invalido(self, risk_manager):
        with pytest.raises(ValueError):
            risk_manager.reset_peak(0.0)
        with pytest.raises(ValueError):
            risk_manager.reset_peak(-10.0)


# =====================================================
# Integración: flujo completo simulado
# =====================================================


class TestIntegrationFlow:
    def test_flujo_completo_trade_ganador(self, risk_manager):
        """Simula: validar → abrir → cerrar con TP → balance actualizado."""
        # 1) Validar trade nuevo de 20€ a precio 0.40
        result = risk_manager.validate_new_trade(
            proposed_size_eur=20.0,
            current_balance_eur=150.0,
            open_positions_count=0,
            entry_price=0.40,
        )
        assert result.approved is True

        # 2) Calcular niveles y abrir posición
        sl = risk_manager.calculate_stop_loss_price(0.40)
        tp = risk_manager.calculate_take_profit_price(0.40)
        assert sl == pytest.approx(0.32)
        assert tp == pytest.approx(0.52)

        position = make_position(entry_price=0.40, stop_loss_price=sl, take_profit_price=tp)

        # 3) Precio sube a 0.55 → TP
        decision = risk_manager.should_close_position(position, current_price=0.55)
        assert decision.should_close is True
        assert decision.reason == CloseReason.TAKE_PROFIT

        # 4) Actualizar balance (ganamos ~+37.5% sobre 20€ = 7.5€)
        new_balance = 150.0 + decision.pnl_eur
        status = risk_manager.update_balance_and_check_drawdown(new_balance)
        assert status.peak_balance_eur > 150.0
        assert risk_manager.is_paused is False
