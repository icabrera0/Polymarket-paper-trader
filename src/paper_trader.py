"""
Paper Trader — el ejecutor de órdenes simuladas.

Responsabilidades:
1. Recibir un `TradeDecision` con `action=OPEN_TRADE` y simular la ejecución:
   - aplicar slippage configurado al precio de entrada
   - calcular `tokens_quantity = size_usd / precio_efectivo`
   - apartar el dinero del balance virtual
   - crear y persistir una `Position` abierta

2. Cerrar posiciones (por SL, TP, news reversal, manual):
   - aplicar slippage en sentido contrario
   - calcular P&L en EUR y %
   - devolver el dinero al balance + el P&L

3. Mantener balance virtual + lista de posiciones abiertas en memoria,
   reflejados en SQLite (recuperación tras reinicio).

4. Reanudación automática: al arrancar, lee posiciones abiertas del SQLite.

P&L en Polymarket (importante):
- Cada lado YES/NO es un token independiente con precio entre $0 y $1.
- Compras `Q = size_usd / price_in` tokens al abrir.
- Cierras vendiendo Q tokens al precio de salida.
- pnl_usd = Q * (price_out - price_in) * (1 - slippage_out) - Q * price_in * slippage_in
  Simplificado: usamos slippage adverso en ambos extremos (BUY más caro, SELL más barato).

Nota sobre comisiones: Polymarket no cobra comisiones por trade hoy en día,
solo gas en Polygon, que en paper trading no aplica. Si en el futuro se
introducen, basta con habilitar `polymarket.trading_fee_pct` > 0.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from src.config_loader import BotConfig
from src.database import Database
from src.models import (
    CloseReason,
    DecisionAction,
    Position,
    TradeDecision,
    TradeSide,
    TradeStatus,
)
from src.risk_manager import RiskManager


class PaperTrader:
    """Simulador de ejecución de órdenes y gestor de balance virtual."""

    def __init__(
        self,
        config: BotConfig,
        risk_manager: RiskManager,
        db: Optional[Database] = None,
    ) -> None:
        self.config = config
        self.risk_manager = risk_manager
        self.db = db if db is not None else Database(config.database.path)

        self._slippage = config.polymarket.simulated_slippage_pct
        self._fee = config.polymarket.trading_fee_pct
        self._eur_to_usd = config.paper_trading.eur_to_usd_rate

        # Balance virtual en EUR
        self._balance_eur: float = config.paper_trading.initial_balance_eur

        # Posiciones abiertas en memoria (también en DB)
        self._open_positions: dict[str, Position] = {}

        self._log = logger.bind(module="paper_trader")

        # Restaurar estado si hay datos previos
        self._restore_from_db()

        self._log.info(
            "PaperTrader inicializado: balance={:.2f}€, posiciones_abiertas={}",
            self._balance_eur,
            len(self._open_positions),
        )

    # =====================================================
    # Estado
    # =====================================================

    @property
    def balance_eur(self) -> float:
        return self._balance_eur

    @property
    def open_positions(self) -> list[Position]:
        return list(self._open_positions.values())

    @property
    def num_open_positions(self) -> int:
        return len(self._open_positions)

    # =====================================================
    # Ejecución: abrir
    # =====================================================

    def execute_decision(self, decision: TradeDecision) -> Optional[Position]:
        """Ejecuta un TradeDecision si action==OPEN_TRADE.

        Devuelve la Position creada, o None si la decision era NO_TRADE o si
        algo falló (no debería: el DECISION_ENGINE ya validó todo).
        """
        # Persistir la decisión SIEMPRE (sea OPEN_TRADE o NO_TRADE) para auditoría
        self.db.log_decision(decision)

        if decision.action != DecisionAction.OPEN_TRADE:
            return None

        # Comprobaciones defensivas (no deberían fallar si DecisionEngine hizo su trabajo)
        if (
            decision.side is None
            or decision.token_id is None
            or decision.entry_price is None
            or decision.size_eur is None
            or decision.stop_loss_price is None
            or decision.take_profit_price is None
        ):
            self._log.error(
                "TradeDecision OPEN_TRADE incompleta: {}", decision
            )
            return None

        if decision.size_eur > self._balance_eur:
            self._log.error(
                "Balance insuficiente: necesita {:.2f}€, hay {:.2f}€",
                decision.size_eur,
                self._balance_eur,
            )
            return None

        # Aplicar slippage al precio de entrada (paga más alto en BUY)
        effective_price = self._apply_buy_slippage(decision.entry_price)

        # Calcular cantidades
        size_usd = decision.size_eur * self._eur_to_usd
        # Comisión (0% por ahora en Polymarket, parametrizado por si cambia)
        size_usd_after_fee = size_usd * (1 - self._fee)
        tokens_quantity = size_usd_after_fee / effective_price

        # Construir posición
        position = Position(
            market_question=decision.market_question,
            market_slug=decision.market_slug,
            token_id=decision.token_id,
            side=decision.side,
            entry_price=effective_price,
            size_eur=decision.size_eur,
            size_usd=size_usd,
            tokens_quantity=tokens_quantity,
            entry_timestamp=datetime.now(timezone.utc),
            stop_loss_price=decision.stop_loss_price,
            take_profit_price=decision.take_profit_price,
            status=TradeStatus.OPEN,
            entry_reason=decision.rationale,
            confidence=decision.confidence,
        )

        # Mover dinero del balance a la posición
        self._balance_eur -= decision.size_eur
        self._open_positions[position.trade_id] = position

        # Persistir
        self.db.insert_trade(position)
        self._log_balance_event("TRADE_OPEN")

        self._log.info(
            "ABIERTA | trade={} | {} @ {:.4f} (slippage from {:.4f}) | "
            "size={:.2f}€ | tokens={:.2f} | balance restante={:.2f}€",
            position.trade_id[:8],
            position.side.value,
            effective_price,
            decision.entry_price,
            position.size_eur,
            position.tokens_quantity,
            self._balance_eur,
        )
        return position

    # =====================================================
    # Ejecución: cerrar
    # =====================================================

    def close_position(
        self,
        trade_id: str,
        current_market_price: float,
        reason: CloseReason,
        notes: str = "",
    ) -> Optional[Position]:
        """Cierra una posición y devuelve el dinero (con P&L) al balance.

        `current_market_price` es el precio del MERCADO (no el efectivo de
        salida). Aplicamos slippage adverso para obtener el precio real.
        """
        position = self._open_positions.get(trade_id)
        if position is None:
            self._log.warning("close_position: trade_id {} no abierto", trade_id)
            return None

        if not (0 < current_market_price < 1):
            self._log.warning(
                "close_position: precio {} fuera de rango (0,1). Ajustando.",
                current_market_price,
            )
            current_market_price = max(0.001, min(0.999, current_market_price))

        # Slippage adverso en SELL (cobras menos)
        effective_exit_price = self._apply_sell_slippage(current_market_price)

        # P&L sobre los tokens poseídos
        # NOTA: la fórmula usa los precios efectivos (slippage ya aplicado).
        # Esto garantiza que el slippage ya restó al P&L.
        pnl_per_token = effective_exit_price - position.entry_price
        pnl_usd = pnl_per_token * position.tokens_quantity
        # Restar comisión de salida si aplica
        pnl_usd_after_fee = pnl_usd - (
            position.tokens_quantity * effective_exit_price * self._fee
        )
        pnl_eur = pnl_usd_after_fee / self._eur_to_usd
        pnl_pct = (effective_exit_price - position.entry_price) / position.entry_price

        # Devolver el principal + P&L al balance
        self._balance_eur += position.size_eur + pnl_eur

        # Marcar la posición como cerrada
        position.status = TradeStatus.CLOSED
        position.exit_price = effective_exit_price
        position.exit_timestamp = datetime.now(timezone.utc)
        position.close_reason = reason
        position.pnl_eur = pnl_eur
        position.pnl_pct = pnl_pct
        position.exit_reason_text = notes or reason.value

        # Quitar del diccionario y persistir
        del self._open_positions[trade_id]
        self.db.update_trade_close(position)
        self._log_balance_event("TRADE_CLOSE")

        # Actualizar drawdown del RiskManager
        self.risk_manager.update_balance_and_check_drawdown(self._balance_eur)

        outcome = "GAIN" if pnl_eur >= 0 else "LOSS"
        self._log.info(
            "CERRADA | trade={} | {} | reason={} | entry={:.4f} → exit={:.4f} | "
            "P&L={:+.2f}€ ({:+.2%}) | balance={:.2f}€",
            position.trade_id[:8],
            outcome,
            reason.value,
            position.entry_price,
            effective_exit_price,
            pnl_eur,
            pnl_pct,
            self._balance_eur,
        )
        return position

    def close_all_positions(
        self,
        current_prices: dict[str, float],
        reason: CloseReason = CloseReason.MANUAL,
        notes: str = "Mass close",
    ) -> list[Position]:
        """Cierra todas las posiciones abiertas. `current_prices` es {token_id: price}."""
        closed: list[Position] = []
        for trade_id in list(self._open_positions.keys()):
            position = self._open_positions[trade_id]
            price = current_prices.get(position.token_id)
            if price is None:
                self._log.warning(
                    "close_all_positions: no hay precio para token {} (trade={})",
                    position.token_id,
                    trade_id[:8],
                )
                continue
            result = self.close_position(trade_id, price, reason, notes)
            if result:
                closed.append(result)
        return closed

    # =====================================================
    # Helpers
    # =====================================================

    def _apply_buy_slippage(self, market_price: float) -> float:
        """Comprando pagas más caro: precio efectivo > precio de mercado."""
        effective = market_price * (1 + self._slippage)
        # Cap a 0.999 para no salirse del rango
        return min(effective, 0.999)

    def _apply_sell_slippage(self, market_price: float) -> float:
        """Vendiendo cobras más barato: precio efectivo < precio de mercado."""
        effective = market_price * (1 - self._slippage)
        return max(effective, 0.001)

    def _log_balance_event(self, event: str) -> None:
        """Registra un snapshot del balance en la tabla balance_history."""
        try:
            status = self.risk_manager.update_balance_and_check_drawdown(
                self._balance_eur
            )
            self.db.log_balance(
                balance_eur=self._balance_eur,
                peak_balance=status.peak_balance_eur,
                drawdown_pct=status.current_drawdown_pct,
                open_positions=len(self._open_positions),
                event=event,
            )
        except Exception as exc:
            self._log.warning("_log_balance_event falló: {}", exc)

    # =====================================================
    # Restauración de estado
    # =====================================================

    def _restore_from_db(self) -> None:
        """Recupera posiciones abiertas y balance al arrancar."""
        # 1) Recuperar posiciones abiertas
        open_positions = self.db.get_open_positions()
        for pos in open_positions:
            self._open_positions[pos.trade_id] = pos

        if not open_positions:
            # Bot fresco: registrar balance inicial
            self._log_balance_event("INIT")
            return

        # 2) Recalcular balance: balance_inicial - sum(size_eur de posiciones abiertas)
        # Esto es una aproximación: si en el pasado cerramos posiciones con P&L,
        # el balance debería incluirlas. Lo correcto es buscar el último
        # snapshot de balance_history.
        history = self.db.get_balance_history()
        if history:
            self._balance_eur = float(history[-1]["balance_eur"])
            self._log.info(
                "Balance recuperado del último snapshot: {:.2f}€",
                self._balance_eur,
            )
        else:
            # Fallback defensivo
            consumed = sum(p.size_eur for p in open_positions)
            self._balance_eur = (
                self.config.paper_trading.initial_balance_eur - consumed
            )
            self._log.warning(
                "Sin balance_history en DB; balance recalculado: {:.2f}€",
                self._balance_eur,
            )

        # Sync drawdown
        self.risk_manager.update_balance_and_check_drawdown(self._balance_eur)
        self._log_balance_event("RESUME")
