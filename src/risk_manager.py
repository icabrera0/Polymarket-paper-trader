"""
Risk Manager — el guardián del bot.

Responsabilidades:
1. Validar cada trade nuevo contra los límites configurados ANTES de ejecutarlo.
2. Calcular el tamaño máximo permitido por posición.
3. Calcular niveles de stop loss y take profit en el momento de abrir.
4. Decidir si una posición abierta debe cerrarse (stop loss / take profit).
5. Llevar el control del drawdown y pausar el bot si supera el umbral.

El RiskManager NO conoce noticias ni sentiment scores: solo aplica reglas
numéricas. Eso lo hace fácilmente testeable en aislamiento.

Reglas calibradas para 150 € de bankroll (config/settings.yaml → risk):
- Tamaño máximo por posición: 15% del balance (~22 €)
- Tamaño mínimo: 5 €
- Máx. 3 posiciones simultáneas
- Stop loss: -20% del precio de entrada
- Take profit: +30% del precio de entrada (señal para evaluar cierre)
- Drawdown máximo: 30% sobre el balance pico → bot pausado
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from src.config_loader import BotConfig
from src.models import (
    CloseDecision,
    CloseReason,
    DrawdownStatus,
    Position,
    RejectReason,
    RiskCheckResult,
)


class RiskManager:
    """Aplica las reglas de gestión de riesgo. Stateful: mantiene peak y pausa."""

    def __init__(
        self,
        config: BotConfig,
        initial_balance_eur: Optional[float] = None,
    ) -> None:
        """Inicializa el RiskManager.

        Args:
            config: configuración cargada con load_config().
            initial_balance_eur: balance inicial. Si es None, usa el del config.
        """
        self.config = config
        self.risk = config.risk

        self._initial_balance: float = (
            initial_balance_eur
            if initial_balance_eur is not None
            else config.paper_trading.initial_balance_eur
        )
        self._peak_balance: float = self._initial_balance
        self._is_paused: bool = False
        self._pause_reason: Optional[str] = None

        self._log = logger.bind(module="risk_manager")
        self._log.info(
            "RiskManager inicializado: balance={:.2f}€, max_pos={:.0%}, "
            "stop_loss={:.0%}, max_drawdown={:.0%}",
            self._initial_balance,
            self.risk.max_position_size_pct,
            self.risk.stop_loss_pct,
            self.risk.max_drawdown_pct,
        )

    # =====================================================
    # Estado
    # =====================================================

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def pause_reason(self) -> Optional[str]:
        return self._pause_reason

    @property
    def peak_balance_eur(self) -> float:
        return self._peak_balance

    @property
    def initial_balance_eur(self) -> float:
        return self._initial_balance

    # =====================================================
    # Validación de trades nuevos
    # =====================================================

    def validate_new_trade(
        self,
        proposed_size_eur: float,
        current_balance_eur: float,
        open_positions_count: int,
        entry_price: float,
    ) -> RiskCheckResult:
        """Valida si un trade nuevo puede abrirse.

        Args:
            proposed_size_eur: tamaño propuesto en €.
            current_balance_eur: balance disponible actual.
            open_positions_count: posiciones actualmente abiertas.
            entry_price: precio de entrada del token (entre 0 y 1).

        Returns:
            RiskCheckResult con `approved` y, si aplica, `adjusted_size_eur`.
        """
        rejections: list[RejectReason] = []
        warnings: list[str] = []
        adjusted_size: Optional[float] = None

        # 1) Bot pausado por drawdown
        if self._is_paused:
            rejections.append(RejectReason.BOT_PAUSED)
            self._log.warning(
                "Trade rechazado: bot pausado ({})", self._pause_reason
            )
            return RiskCheckResult(approved=False, rejection_reasons=rejections)

        # 2) Precio inválido (debe estar estrictamente entre 0 y 1)
        if not (0 < entry_price < 1):
            rejections.append(RejectReason.INVALID_PRICE)
            return RiskCheckResult(approved=False, rejection_reasons=rejections)

        # 3) Demasiadas posiciones abiertas
        if open_positions_count >= self.risk.max_simultaneous_positions:
            rejections.append(RejectReason.MAX_POSITIONS_REACHED)

        # 4) Tamaño por debajo del mínimo
        if proposed_size_eur < self.risk.min_trade_size_eur:
            rejections.append(RejectReason.SIZE_BELOW_MIN)

        # 5) Balance insuficiente: pedimos más € de los que tenemos.
        #    Se comprueba ANTES del recorte por max_position porque indica un
        #    error en la cadena que llamó al RiskManager.
        if proposed_size_eur > current_balance_eur:
            rejections.append(RejectReason.INSUFFICIENT_BALANCE)

        # 6) Tamaño por encima del máximo permitido → recortar (no rechazar)
        max_allowed = self.calculate_max_position_size(current_balance_eur)
        if proposed_size_eur > max_allowed:
            adjusted_size = max_allowed
            warnings.append(
                f"Tamaño solicitado ({proposed_size_eur:.2f}€) excede el máximo "
                f"({max_allowed:.2f}€). Ajustado a {adjusted_size:.2f}€."
            )
            # Si tras el recorte no llega al mínimo, rechazar
            if adjusted_size < self.risk.min_trade_size_eur:
                rejections.append(RejectReason.SIZE_ABOVE_MAX)

        approved = len(rejections) == 0
        result = RiskCheckResult(
            approved=approved,
            rejection_reasons=rejections,
            warnings=warnings,
            adjusted_size_eur=adjusted_size,
        )

        if not approved:
            self._log.warning(
                "Trade rechazado | size={:.2f}€ balance={:.2f}€ open={} "
                "price={:.3f} | razones: {}",
                proposed_size_eur,
                current_balance_eur,
                open_positions_count,
                entry_price,
                [r.value for r in rejections],
            )
        else:
            final_size = adjusted_size if adjusted_size is not None else proposed_size_eur
            self._log.info(
                "Trade aprobado | size={:.2f}€ (ajustado={}) price={:.3f}",
                final_size,
                adjusted_size is not None,
                entry_price,
            )

        return result

    # =====================================================
    # Cálculos de tamaño y niveles
    # =====================================================

    def calculate_max_position_size(self, current_balance_eur: float) -> float:
        """Tamaño máximo permitido por posición (€) sobre el balance actual."""
        return current_balance_eur * self.risk.max_position_size_pct

    def calculate_stop_loss_price(self, entry_price: float) -> float:
        """Precio del token al que disparar el stop loss.

        Si entry_price = $0.40 y stop_loss_pct = 0.20 → SL en $0.32 (-20%).
        Sirve igual para BUY_YES y BUY_NO porque cada uno es su propio token.
        """
        return round(entry_price * (1 - self.risk.stop_loss_pct), 4)

    def calculate_take_profit_price(self, entry_price: float) -> float:
        """Precio del token al que se considera tomar beneficios.

        En Polymarket el precio máximo es $1, así que cap a 0.999 para evitar
        precios imposibles cuando entry_price es alto.
        """
        target = entry_price * (1 + self.risk.take_profit_pct)
        return round(min(target, 0.999), 4)

    # =====================================================
    # Cierre de posiciones abiertas
    # =====================================================

    def should_close_position(
        self,
        position: Position,
        current_price: float,
    ) -> CloseDecision:
        """Decide si cerrar una posición según stop loss, take profit o tiempo.

        Orden de evaluación:
          1. Stop loss (protección — siempre primero)
          2. Tiempo: TP ajustado tras N horas (Tier 1)
          3. Take profit original
          4. Tiempo: cierre si en beneficio tras N horas (Tier 2)
          5. Tiempo: cierre forzado incondicional (Tier 3)

        Cierres por noticias contradictorias o resolución de mercado los gestiona
        el DECISION_ENGINE en módulos posteriores.
        """
        pnl_pct = position.current_pnl_pct(current_price)
        pnl_eur = position.current_pnl_eur(current_price)

        # Calcular horas transcurridas desde la entrada
        now = datetime.now(timezone.utc)
        entry = position.entry_timestamp
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=timezone.utc)
        hours_held = (now - entry).total_seconds() / 3600

        risk = self.risk

        # 1) Stop loss: precio cae hasta o por debajo del nivel calculado
        if current_price <= position.stop_loss_price:
            self._log.info(
                "STOP LOSS disparado | trade={} entry={:.3f} now={:.3f} pnl={:.2%}",
                position.trade_id[:8],
                position.entry_price,
                current_price,
                pnl_pct,
            )
            return CloseDecision(
                should_close=True,
                reason=CloseReason.STOP_LOSS,
                pnl_pct=pnl_pct,
                pnl_eur=pnl_eur,
                notes=f"Precio {current_price:.4f} <= SL {position.stop_loss_price:.4f}",
            )

        # 2) TIME EXIT — Tier 1: TP ajustado tras time_tighten_tp_hours
        if hours_held >= risk.time_tighten_tp_hours:
            tightened_tp = position.entry_price * (1 + risk.time_tighten_tp_pct)
            if current_price >= tightened_tp:
                self._log.info(
                    "TIME EXIT (TP ajustado) | trade={} held={:.1f}h "
                    "precio={:.4f} TP_tight={:.4f} pnl={:+.2%}",
                    position.trade_id[:8],
                    hours_held,
                    current_price,
                    tightened_tp,
                    pnl_pct,
                )
                return CloseDecision(
                    should_close=True,
                    reason=CloseReason.TIME_EXIT,
                    pnl_pct=pnl_pct,
                    pnl_eur=pnl_eur,
                    notes=(
                        f"TP ajustado por tiempo: {hours_held:.1f}h "
                        f">= {risk.time_tighten_tp_hours:.0f}h | "
                        f"precio={current_price:.4f} >= TP_ajustado={tightened_tp:.4f}"
                    ),
                )

        # 3) Take profit original: precio alcanza o supera el nivel calculado
        if current_price >= position.take_profit_price:
            self._log.info(
                "TAKE PROFIT alcanzado | trade={} entry={:.3f} now={:.3f} pnl={:+.2%}",
                position.trade_id[:8],
                position.entry_price,
                current_price,
                pnl_pct,
            )
            return CloseDecision(
                should_close=True,
                reason=CloseReason.TAKE_PROFIT,
                pnl_pct=pnl_pct,
                pnl_eur=pnl_eur,
                notes=f"Precio {current_price:.4f} >= TP {position.take_profit_price:.4f}",
            )

        # 4) TIME EXIT — Tier 2: cerrar si en beneficio tras time_exit_profit_hours
        if hours_held >= risk.time_exit_profit_hours and pnl_pct >= 0:
            self._log.info(
                "TIME EXIT (en beneficio) | trade={} held={:.1f}h pnl={:+.2%}",
                position.trade_id[:8],
                hours_held,
                pnl_pct,
            )
            return CloseDecision(
                should_close=True,
                reason=CloseReason.TIME_EXIT,
                pnl_pct=pnl_pct,
                pnl_eur=pnl_eur,
                notes=(
                    f"Cierre por tiempo en beneficio: {hours_held:.1f}h "
                    f">= {risk.time_exit_profit_hours:.0f}h | P&L={pnl_pct:+.2%}"
                ),
            )

        # 5) TIME EXIT — Tier 3: cierre forzado incondicional tras time_exit_hard_hours
        if hours_held >= risk.time_exit_hard_hours:
            self._log.info(
                "TIME EXIT (forzado) | trade={} held={:.1f}h pnl={:+.2%}",
                position.trade_id[:8],
                hours_held,
                pnl_pct,
            )
            return CloseDecision(
                should_close=True,
                reason=CloseReason.TIME_EXIT,
                pnl_pct=pnl_pct,
                pnl_eur=pnl_eur,
                notes=(
                    f"Cierre forzado por tiempo: {hours_held:.1f}h "
                    f">= {risk.time_exit_hard_hours:.0f}h | P&L={pnl_pct:+.2%}"
                ),
            )

        # Posición sigue dentro de los niveles
        return CloseDecision(
            should_close=False,
            reason=None,
            pnl_pct=pnl_pct,
            pnl_eur=pnl_eur,
            notes="Dentro de niveles",
        )

    # =====================================================
    # Drawdown protection
    # =====================================================

    def update_balance_and_check_drawdown(
        self, current_balance_eur: float
    ) -> DrawdownStatus:
        """Computes drawdown from the INITIAL balance (not peak).

        Peak is still tracked for reporting purposes (dashboard / daily report)
        but the alert threshold is measured from the fixed starting bankroll.
        This avoids false alarms caused by temporary equity highs — e.g. going
        150 → 160 → 115 reads as a 28% drawdown from peak even though you're
        only -23% from your actual starting capital.

        If pause_on_drawdown is False (recommended for disposable bankrolls)
        the bot never freezes — the threshold only triggers a notification.
        """
        if current_balance_eur > self._peak_balance:
            self._peak_balance = current_balance_eur

        # Measure from the fixed initial bankroll, not the floating peak
        if self._initial_balance > 0:
            drawdown = (self._initial_balance - current_balance_eur) / self._initial_balance
        else:
            drawdown = 0.0
        drawdown = max(drawdown, 0.0)

        threshold_breached = drawdown >= self.risk.max_drawdown_pct

        if threshold_breached and not self._is_paused:
            if self.risk.pause_on_drawdown:
                self._is_paused = True
                self._pause_reason = (
                    f"Drawdown {drawdown:.2%} supera el límite "
                    f"{self.risk.max_drawdown_pct:.0%} del capital inicial"
                )
                self._log.error(
                    "BOT PAUSADO | balance={:.2f}€ inicial={:.2f}€ drawdown={:.2%}",
                    current_balance_eur,
                    self._initial_balance,
                    drawdown,
                )
            else:
                self._log.warning(
                    "Drawdown alert | balance={:.2f}€ inicial={:.2f}€ drawdown={:.2%} "
                    "(monitoring-only — bot continúa operando)",
                    current_balance_eur,
                    self._initial_balance,
                    drawdown,
                )

        return DrawdownStatus(
            current_balance_eur=current_balance_eur,
            peak_balance_eur=self._peak_balance,
            current_drawdown_pct=drawdown,
            threshold_breached=threshold_breached,
            bot_should_pause=self._is_paused,
        )

    # =====================================================
    # Control manual del estado de pausa
    # =====================================================

    def manually_resume(self) -> None:
        """Reanuda el bot tras una pausa por drawdown.

        Solo debe llamarse desde una intervención manual del operador (CLI o
        Discord), tras revisar la situación.
        """
        if not self._is_paused:
            self._log.warning("manually_resume() llamado pero el bot no estaba pausado")
            return

        prev_reason = self._pause_reason
        self._is_paused = False
        self._pause_reason = None
        # Resetear el peak al balance actual evita auto-pausar inmediatamente
        # tras reanudar si seguimos en mínimos.
        # El operador puede llamar a reset_peak() también si lo prefiere.
        self._log.info("Bot reanudado manualmente. Razón previa: {}", prev_reason)

    def reset_peak(self, new_peak: float) -> None:
        """Resetea el peak balance. Útil al reanudar tras drawdown."""
        if new_peak <= 0:
            raise ValueError("El peak balance debe ser positivo")
        self._log.info(
            "Peak balance reseteado: {:.2f}€ → {:.2f}€",
            self._peak_balance,
            new_peak,
        )
        self._peak_balance = new_peak
