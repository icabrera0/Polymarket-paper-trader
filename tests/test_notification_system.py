"""
Tests del NotificationSystem.

No envía nada a Discord real. Intercepta llamadas HTTP con pytest-mock
y verifica que los embeds se construyen correctamente.

Ejecutar:
    pytest tests/test_notification_system.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.models import CloseReason, Position, TradeSide, TradeStatus
from src.notification_system import NotificationSystem


@pytest.fixture
def notifier(config):
    # Asegurar que el webhook está configurado para los tests
    config.notifications.discord.enabled = True
    n = NotificationSystem(config)
    n.webhook_url = "https://discord.com/api/webhooks/test/test"
    return n


def make_position(
    pnl_eur: float = 5.0,
    pnl_pct: float = 0.25,
    close_reason: CloseReason = CloseReason.TAKE_PROFIT,
) -> Position:
    return Position(
        market_question="Will X happen?",
        token_id="0xyes",
        side=TradeSide.BUY_YES,
        entry_price=0.40,
        size_eur=20.0, size_usd=21.4, tokens_quantity=53.5,
        entry_timestamp=datetime.now(timezone.utc),
        stop_loss_price=0.32, take_profit_price=0.52,
        status=TradeStatus.CLOSED,
        exit_price=0.55,
        exit_timestamp=datetime.now(timezone.utc),
        close_reason=close_reason,
        pnl_eur=pnl_eur,
        pnl_pct=pnl_pct,
        confidence=80,
    )


class TestEmbedConstruction:
    def test_embed_tiene_campos_requeridos(self, notifier):
        embed = notifier._embed(
            title="Test",
            color=3066993,
            fields=[("Campo", "Valor", True)],
        )
        assert embed["title"] == "Test"
        assert embed["color"] == 3066993
        assert len(embed["fields"]) == 1
        assert embed["fields"][0]["name"] == "Campo"
        assert "timestamp" in embed
        assert "footer" in embed

    def test_trade_open_embed_contiene_precio_y_tamanio(self, notifier):
        pos = make_position()
        # Capturar el embed que se construiría
        embeds_sent = []
        original_send_async = notifier._send_async

        def capture(embed=None, content=None):
            if embed:
                embeds_sent.append(embed)
        notifier._send_async = capture

        notifier.notify_trade_open(pos, balance_after=130.0)
        assert len(embeds_sent) == 1
        embed = embeds_sent[0]
        field_values = {f["name"]: f["value"] for f in embed["fields"]}
        assert "Precio entrada" in field_values
        assert "Tamaño" in field_values
        assert "€20.00" in field_values["Tamaño"]

    def test_trade_close_ganador_usa_color_profit(self, notifier):
        pos = make_position(pnl_eur=5.0, pnl_pct=0.25)
        embeds_sent = []

        def capture(embed=None, content=None):
            if embed:
                embeds_sent.append(embed)
        notifier._send_async = capture

        notifier.notify_trade_close(pos, balance_after=155.0)
        assert embeds_sent[0]["color"] == notifier.cfg.color_profit

    def test_trade_close_perdedor_usa_color_loss(self, notifier):
        pos = make_position(
            pnl_eur=-3.0, pnl_pct=-0.15,
            close_reason=CloseReason.STOP_LOSS,
        )
        embeds_sent = []

        def capture(embed=None, content=None):
            if embed:
                embeds_sent.append(embed)
        notifier._send_async = capture

        notifier.notify_trade_close(pos, balance_after=147.0)
        assert embeds_sent[0]["color"] == notifier.cfg.color_loss

    def test_drawdown_usa_color_warning(self, notifier):
        embeds_sent = []

        def capture(embed=None, content=None):
            if embed:
                embeds_sent.append(embed)
        notifier._send_async = capture

        notifier.notify_drawdown_warning(120.0, 150.0, 0.20)
        assert embeds_sent[0]["color"] == notifier.cfg.color_warning

    def test_daily_summary_positivo_color_profit(self, notifier):
        embeds_sent = []

        def capture(embed=None, content=None):
            if embed:
                embeds_sent.append(embed)
        notifier._send_async = capture

        notifier.notify_daily_summary(
            "2026-04-30", 150.0, 157.5, 7.5, 3, 0.67
        )
        assert embeds_sent[0]["color"] == notifier.cfg.color_profit

    def test_daily_summary_negativo_color_loss(self, notifier):
        embeds_sent = []

        def capture(embed=None, content=None):
            if embed:
                embeds_sent.append(embed)
        notifier._send_async = capture

        notifier.notify_daily_summary(
            "2026-04-30", 150.0, 143.0, -7.0, 2, 0.0
        )
        assert embeds_sent[0]["color"] == notifier.cfg.color_loss


class TestDisabledBehavior:
    def test_no_envia_si_deshabilitado(self, config_factory):
        cfg = config_factory()
        cfg.notifications.discord.enabled = False
        n = NotificationSystem(cfg)
        # No debe lanzar excepción ni intentar enviar
        pos = make_position()
        n.notify_trade_open(pos, 150.0)  # Silencioso

    def test_no_envia_sin_webhook_url(self, config_factory):
        cfg = config_factory()
        cfg.notifications.discord.enabled = True
        n = NotificationSystem(cfg)
        n.webhook_url = ""
        pos = make_position()
        n.notify_trade_open(pos, 150.0)  # Silencioso, no lanza excepción
