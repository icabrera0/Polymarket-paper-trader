"""
Paper Trader — the simulated order executor.

Responsibilities:
1. Receive a `TradeDecision` with `action=OPEN_TRADE` and simulate execution:
   - apply configured slippage to the entry price
   - calculate `tokens_quantity = size_usd / effective_price`
   - set aside the money from the virtual balance
   - create and persist an open `Position`

2. Close positions (by SL, TP, news reversal, manual):
   - apply slippage in the opposite direction
   - calculate P&L in EUR and %
   - return the money to the balance + P&L

3. Maintain virtual balance + list of open positions in memory,
   reflected in SQLite (recovery after restart).

4. Automatic resumption: on startup, reads open positions from SQLite.

P&L on Polymarket (important):
- Each YES/NO side is an independent token with a price between $0 and $1.
- Buy `Q = size_usd / price_in` tokens when opening.
- Close by selling Q tokens at the exit price.
- pnl_usd = Q * (price_out - price_in) * (1 - slippage_out) - Q * price_in * slippage_in
  Simplified: we apply adverse slippage at both ends (BUY more expensive, SELL cheaper).

Note on fees: Polymarket currently charges no trading fees,
only gas on Polygon, which does not apply in paper trading. If fees are
introduced in the future, simply enable `polymarket.trading_fee_pct` > 0.
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
    """Simulated order executor and virtual balance manager."""

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

        # Virtual balance in EUR
        self._balance_eur: float = config.paper_trading.initial_balance_eur

        # Open positions in memory (also in DB)
        self._open_positions: dict[str, Position] = {}

        self._log = logger.bind(module="paper_trader")

        # Restore state if previous data exists
        self._restore_from_db()

        self._log.info(
            "PaperTrader initialized: balance={:.2f}€, open_positions={}",
            self._balance_eur,
            len(self._open_positions),
        )

    # =====================================================
    # State
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
    # Execution: open
    # =====================================================

    def execute_decision(self, decision: TradeDecision) -> Optional[Position]:
        """Executes a TradeDecision if action==OPEN_TRADE.

        Returns the created Position, or None if the decision was NO_TRADE or if
        something failed (should not happen: the DECISION_ENGINE already validated everything).
        """
        # Always persist the decision (whether OPEN_TRADE or NO_TRADE) for auditing
        self.db.log_decision(decision)

        if decision.action != DecisionAction.OPEN_TRADE:
            return None

        # Defensive checks (should not fail if DecisionEngine did its job)
        if (
            decision.side is None
            or decision.token_id is None
            or decision.entry_price is None
            or decision.size_eur is None
            or decision.stop_loss_price is None
            or decision.take_profit_price is None
        ):
            self._log.error(
                "Incomplete OPEN_TRADE TradeDecision: {}", decision
            )
            return None

        if decision.size_eur > self._balance_eur:
            self._log.error(
                "Insufficient balance: needs {:.2f}€, available {:.2f}€",
                decision.size_eur,
                self._balance_eur,
            )
            return None

        # Apply slippage to the entry price (pay more on BUY)
        effective_price = self._apply_buy_slippage(decision.entry_price)

        # Calculate quantities
        size_usd = decision.size_eur * self._eur_to_usd
        # Fee (currently 0% on Polymarket, parameterized in case it changes)
        size_usd_after_fee = size_usd * (1 - self._fee)
        tokens_quantity = size_usd_after_fee / effective_price

        # Build position
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
            predicted_prob=decision.predicted_prob,
        )

        # Move money from balance to position
        self._balance_eur -= decision.size_eur
        self._open_positions[position.trade_id] = position

        # Persist
        self.db.insert_trade(position)
        self._log_balance_event("TRADE_OPEN")

        self._log.info(
            "OPENED | trade={} | {} @ {:.4f} (slippage from {:.4f}) | "
            "size={:.2f}€ | tokens={:.2f} | remaining balance={:.2f}€",
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
    # Execution: close
    # =====================================================

    def close_position(
        self,
        trade_id: str,
        current_market_price: float,
        reason: CloseReason,
        notes: str = "",
    ) -> Optional[Position]:
        """Closes a position and returns the money (with P&L) to the balance.

        `current_market_price` is the MARKET price (not the effective exit price).
        Adverse slippage is applied to obtain the real price.
        """
        position = self._open_positions.get(trade_id)
        if position is None:
            self._log.warning("close_position: trade_id {} not open", trade_id)
            return None

        if not (0 < current_market_price < 1):
            self._log.warning(
                "close_position: price {} out of range (0,1). Clamping.",
                current_market_price,
            )
            current_market_price = max(0.001, min(0.999, current_market_price))

        # Adverse slippage on SELL (receive less)
        effective_exit_price = self._apply_sell_slippage(current_market_price)

        # P&L on held tokens
        # NOTE: the formula uses effective prices (slippage already applied).
        # This ensures slippage has already been deducted from the P&L.
        pnl_per_token = effective_exit_price - position.entry_price
        pnl_usd = pnl_per_token * position.tokens_quantity
        # Deduct exit fee if applicable
        pnl_usd_after_fee = pnl_usd - (
            position.tokens_quantity * effective_exit_price * self._fee
        )
        pnl_eur = pnl_usd_after_fee / self._eur_to_usd
        pnl_pct = (effective_exit_price - position.entry_price) / position.entry_price

        # Return principal + P&L to balance
        self._balance_eur += position.size_eur + pnl_eur

        # Profit sweep: cap trading balance at config value, move excess to consolidated profit
        cap = self.config.paper_trading.initial_balance_eur
        swept = 0.0
        if self._balance_eur > cap:
            swept = self._balance_eur - cap
            self._balance_eur = cap
            self._log.info(
                "Profit sweep: €{:.2f} moved to consolidated profit, balance reset to €{:.2f}",
                swept, cap,
            )

        # Mark position as closed
        position.status = TradeStatus.CLOSED
        position.exit_price = effective_exit_price
        position.exit_timestamp = datetime.now(timezone.utc)
        position.close_reason = reason
        position.pnl_eur = pnl_eur
        position.pnl_pct = pnl_pct
        position.exit_reason_text = notes or reason.value

        # Remove from dictionary and persist
        del self._open_positions[trade_id]
        self.db.update_trade_close(position)
        self._log_balance_event("TRADE_CLOSE", consolidated_profit_eur=swept)

        # Update RiskManager drawdown
        self.risk_manager.update_balance_and_check_drawdown(self._balance_eur)

        outcome = "GAIN" if pnl_eur >= 0 else "LOSS"
        self._log.info(
            "CLOSED | trade={} | {} | reason={} | entry={:.4f} → exit={:.4f} | "
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
        """Closes all open positions. `current_prices` is {token_id: price}."""
        closed: list[Position] = []
        for trade_id in list(self._open_positions.keys()):
            position = self._open_positions[trade_id]
            price = current_prices.get(position.token_id)
            if price is None:
                self._log.warning(
                    "close_all_positions: no price for token {} (trade={})",
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
        """Buying costs more: effective price > market price."""
        effective = market_price * (1 + self._slippage)
        # Cap at 0.999 to stay within valid range
        return min(effective, 0.999)

    def _apply_sell_slippage(self, market_price: float) -> float:
        """Selling receives less: effective price < market price."""
        effective = market_price * (1 - self._slippage)
        return max(effective, 0.001)

    def _log_balance_event(self, event: str, consolidated_profit_eur: float = 0.0) -> None:
        """Records a balance snapshot in the balance_history table."""
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
                consolidated_profit_eur=consolidated_profit_eur,
            )
        except Exception as exc:
            self._log.warning("_log_balance_event failed: {}", exc)

    # =====================================================
    # State restoration
    # =====================================================

    def _restore_from_db(self) -> None:
        """Recovers open positions and balance on startup."""
        # 1) Recover open positions
        open_positions = self.db.get_open_positions()
        for pos in open_positions:
            self._open_positions[pos.trade_id] = pos

        if not open_positions:
            # Fresh bot: record initial balance
            self._log_balance_event("INIT")
            return

        # 2) Recalculate balance: initial_balance - sum(size_eur of open positions)
        # This is an approximation: if we previously closed positions with P&L,
        # the balance should include them. The correct approach is to look up
        # the last balance_history snapshot.
        history = self.db.get_balance_history()
        if history:
            self._balance_eur = float(history[-1]["balance_eur"])
            self._log.info(
                "Balance recovered from last snapshot: {:.2f}€",
                self._balance_eur,
            )
        else:
            # Defensive fallback
            consumed = sum(p.size_eur for p in open_positions)
            self._balance_eur = (
                self.config.paper_trading.initial_balance_eur - consumed
            )
            self._log.warning(
                "No balance_history in DB; balance recalculated: {:.2f}€",
                self._balance_eur,
            )

        # Sync drawdown
        self.risk_manager.update_balance_and_check_drawdown(self._balance_eur)
        self._log_balance_event("RESUME")
