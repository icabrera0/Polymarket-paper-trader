"""
Sistema de notificaciones Discord.

Envía mensajes al webhook configurado con embeds formateados:
- 🟢 Trade abierto (verde)
- 🔴 Trade cerrado con pérdida (rojo)
- ✅ Trade cerrado con ganancia (verde)
- ⚠️ Stop loss / drawdown warning (amarillo)
- ⛔ Bot pausado (rojo oscuro)
- 📊 Resumen diario (azul)

Diseño:
- Síncrono con requests. Llamadas en background thread para no bloquear el bot.
- Si el webhook falla, loguea warning y continúa. Las notificaciones son
  informativas, nunca críticas para la operativa.
- Rate limiting: Discord permite ~30 mensajes/minuto. Incluimos un sleep
  mínimo entre envíos para no saturar.
"""

from __future__ import annotations

import time
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from loguru import logger

from src.config_loader import BotConfig
from src.models import CloseReason, Position, TradeDecision


class NotificationSystem:
    """Envía notificaciones a Discord."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.cfg = config.notifications.discord
        self.webhook_url = config.discord_webhook_url or ""
        self._log = logger.bind(module="notifications")
        self._last_send_ts: float = 0.0
        self._min_interval: float = 2.0  # segundos entre envíos

        if self.cfg.enabled and not self.webhook_url:
            self._log.warning(
                "Discord habilitado pero DISCORD_WEBHOOK_URL no está en .env"
            )

    # =====================================================
    # Notificaciones de trading
    # =====================================================

    def notify_trade_open(self, position: Position, balance_after: float) -> None:
        if not self.cfg.notify_on_trade_open:
            return

        # Calcular SL y TP en euros para que sea visual
        eur_rate = self.config.paper_trading.eur_to_usd_rate
        sl_eur = (
            (position.stop_loss_price - position.entry_price)
            * position.tokens_quantity
            / eur_rate
        )
        tp_eur = (
            (position.take_profit_price - position.entry_price)
            * position.tokens_quantity
            / eur_rate
        )

        embed = self._embed(
            title="📈 Trade Abierto",
            color=self.cfg.color_profit,
            fields=[
                ("Mercado", position.market_question[:80], False),
                ("Lado", position.side.value, True),
                ("Precio entrada", f"{position.entry_price:.4f}", True),
                ("Tamaño", f"€{position.size_eur:.2f}", True),
                (
                    "🛑 Stop Loss",
                    f"{position.stop_loss_price:.4f}  →  **€{sl_eur:+.2f}**",
                    True,
                ),
                (
                    "✅ Take Profit",
                    f"{position.take_profit_price:.4f}  →  **€{tp_eur:+.2f}**",
                    True,
                ),
                ("Confianza", f"{position.confidence}/100", True),
                ("Balance tras trade", f"€{balance_after:.2f}", True),
            ],
        )
        self._send_async(embed)

    def notify_trade_close(self, position: Position, balance_after: float) -> None:
        if not self.cfg.notify_on_trade_close:
            return
        pnl = position.pnl_eur or 0.0
        pnl_pct = position.pnl_pct or 0.0
        is_gain = pnl >= 0
        icon = "✅" if is_gain else "❌"
        color = self.cfg.color_profit if is_gain else self.cfg.color_loss

        duration = ""
        if position.entry_timestamp and position.exit_timestamp:
            h = (position.exit_timestamp - position.entry_timestamp).total_seconds() / 3600
            duration = f"{h:.1f}h"

        embed = self._embed(
            title=f"{icon} Trade Cerrado — {'GANANCIA' if is_gain else 'PÉRDIDA'}",
            color=color,
            fields=[
                ("Mercado", position.market_question[:80], False),
                ("Lado", position.side.value, True),
                (
                    "Entrada → Salida",
                    f"{position.entry_price:.4f} → {position.exit_price:.4f}",
                    True,
                ),
                (
                    "P&L",
                    f"**€{pnl:+.2f}** ({pnl_pct:+.2%})",
                    True,
                ),
                ("Motivo", (position.close_reason.value if position.close_reason else "—"), True),
                ("Duración", duration or "—", True),
                ("Balance", f"€{balance_after:.2f}", True),
            ],
        )
        self._send_async(embed)

    def notify_stop_loss(self, position: Position, balance_after: float) -> None:
        if not self.cfg.notify_on_stop_loss:
            return
        pnl = position.pnl_eur or 0.0
        pnl_pct = position.pnl_pct or 0.0
        embed = self._embed(
            title="🛑 Stop Loss Disparado",
            color=self.cfg.color_warning,
            fields=[
                ("Mercado", position.market_question[:80], False),
                (
                    "Precio SL",
                    f"{position.stop_loss_price:.4f}",
                    True,
                ),
                (
                    "P&L",
                    f"**€{pnl:+.2f}** ({pnl_pct:+.2%})",
                    True,
                ),
                ("Balance", f"€{balance_after:.2f}", True),
            ],
        )
        self._send_async(embed)

    def notify_drawdown_warning(
        self, current_balance: float, peak_balance: float, drawdown_pct: float
    ) -> None:
        if not self.cfg.notify_on_drawdown_warning:
            return
        embed = self._embed(
            title="⚠️ Alerta de Drawdown",
            color=self.cfg.color_warning,
            fields=[
                ("Balance actual", f"€{current_balance:.2f}", True),
                ("Peak histórico", f"€{peak_balance:.2f}", True),
                ("Drawdown", f"{drawdown_pct:.2%}", True),
                ("Límite configurado",
                 f"{self.config.risk.max_drawdown_pct:.0%}", True),
            ],
        )
        self._send_async(embed)

    def notify_bot_paused(self, reason: str, balance: float) -> None:
        if not self.cfg.notify_on_bot_pause:
            return
        embed = self._embed(
            title="⛔ Bot Pausado",
            color=self.cfg.color_loss,
            fields=[
                ("Razón", reason, False),
                ("Balance al pausar", f"€{balance:.2f}", True),
                ("Acción requerida",
                 "Revisar la situación y reanudar manualmente con:\n"
                 "`python scripts/manage_balance.py status`", False),
            ],
        )
        self._send_async(embed)

    def notify_bot_resumed(self, balance: float) -> None:
        embed = self._embed(
            title="▶️ Bot Reanudado",
            color=self.cfg.color_info,
            fields=[
                ("Balance actual", f"€{balance:.2f}", True),
                ("Estado", "Operativo", True),
            ],
        )
        self._send_async(embed)

    def notify_daily_summary(
        self,
        date_str: str,
        balance_start: float,
        balance_end: float,
        total_pnl: float,
        num_trades: int,
        win_rate: float,
        report_path: str = "",
    ) -> None:
        if not self.cfg.notify_daily_summary:
            return
        is_positive = total_pnl >= 0
        color = self.cfg.color_profit if is_positive else self.cfg.color_loss
        fields = [
            ("Fecha", date_str, True),
            ("Balance inicial", f"€{balance_start:.2f}", True),
            ("Balance final", f"€{balance_end:.2f}", True),
            ("P&L día", f"€{total_pnl:+.2f}", True),
            ("Trades", str(num_trades), True),
            ("Win rate", f"{win_rate:.1%}", True),
        ]
        if report_path:
            fields.append(("Reporte", f"`{report_path}`", False))
        embed = self._embed(
            title=f"📊 Resumen Diario — {date_str}",
            color=color,
            fields=fields,
        )
        self._send_async(embed)

    def notify_error(self, module: str, error: str) -> None:
        """Notifica un error inesperado del bot."""
        embed = self._embed(
            title="💥 Error del Bot",
            color=self.cfg.color_loss,
            fields=[
                ("Módulo", module, True),
                ("Error", error[:500], False),
            ],
        )
        self._send_async(embed)

    def send_text(self, text: str) -> None:
        """Envía un mensaje de texto plano (sin embed)."""
        self._send_async(content=text)

    # =====================================================
    # Internals
    # =====================================================

    def _embed(
        self,
        title: str,
        color: int,
        fields: list[tuple[str, str, bool]],
        description: str = "",
    ) -> dict[str, Any]:
        """Construye un embed de Discord."""
        embed: dict[str, Any] = {
            "title": title,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "Polymarket Paper Trading Bot"},
        }
        if description:
            embed["description"] = description
        if fields:
            embed["fields"] = [
                {"name": name, "value": value, "inline": inline}
                for name, value, inline in fields
            ]
        return embed

    def _send_async(
        self,
        embed: Optional[dict[str, Any]] = None,
        content: Optional[str] = None,
    ) -> None:
        """Envía en un thread daemon para no bloquear el bot."""
        if not self.cfg.enabled or not self.webhook_url:
            return
        t = threading.Thread(
            target=self._send,
            args=(embed, content),
            daemon=True,
        )
        t.start()

    def _send(
        self,
        embed: Optional[dict[str, Any]] = None,
        content: Optional[str] = None,
    ) -> None:
        """Envía el mensaje al webhook. Incluye throttling."""
        # Rate limiting básico (Discord: 30 msg/min)
        elapsed = time.time() - self._last_send_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_send_ts = time.time()

        payload: dict[str, Any] = {
            "username": self.cfg.username,
        }
        if self.cfg.avatar_url:
            payload["avatar_url"] = self.cfg.avatar_url
        if content:
            payload["content"] = content
        if embed:
            payload["embeds"] = [embed]

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            if response.status_code == 204:
                return  # OK
            if response.status_code == 429:
                # Rate limited: esperar y reintentar una vez
                retry_after = float(
                    response.json().get("retry_after", 5)
                )
                self._log.warning(
                    "Discord rate limited. Esperando {:.1f}s", retry_after
                )
                time.sleep(retry_after)
                requests.post(self.webhook_url, json=payload, timeout=10)
            else:
                self._log.warning(
                    "Discord respondió {}: {}",
                    response.status_code,
                    response.text[:200],
                )
        except requests.RequestException as exc:
            self._log.warning("Envío a Discord falló: {}", exc)
