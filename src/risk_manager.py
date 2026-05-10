"""
Risk Manager — the guardian of the bot.

Responsibilities:
1. Validate each new trade against configured limits BEFORE executing it.
2. Calculate the maximum allowed size per position.
3. Calculate stop loss and take profit levels at the time of opening.
4. Decide whether an open position should be closed (stop loss / take profit).
5. Track drawdown and pause the bot if it exceeds the threshold.

The RiskManager has no knowledge of news or sentiment scores: it only applies
numerical rules. This makes it easily testable in isolation.

Rules calibrated for a €150 bankroll (config/settings.yaml → risk):
- Maximum position size: 15% of balance (~€22)
- Minimum size: €5
- Max 3 simultaneous positions
- Stop loss: -20% of entry price
- Take profit: +30% of entry price (signal to evaluate closing)
- Maximum drawdown: 30% over peak balance → bot paused
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
    """Applies risk management rules. Stateful: tracks peak and pause state."""

    def __init__(
        self,
        config: BotConfig,
        initial_balance_eur: Optional[float] = None,
    ) -> None:
        """Initializes the RiskManager.

        Args:
            config: configuration loaded with load_config().
            initial_balance_eur: initial balance. If None, uses the value from config.
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
            "RiskManager initialized: balance={:.2f}€, max_pos={:.0%}, "
            "stop_loss={:.0%}, max_drawdown={:.0%}",
            self._initial_balance,
            self.risk.max_position_size_pct,
            self.risk.stop_loss_pct,
            self.risk.max_drawdown_pct,
        )

    # =====================================================
    # State
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
    # New trade validation
    # =====================================================

    def validate_new_trade(
        self,
        proposed_size_eur: float,
        current_balance_eur: float,
        open_positions_count: int,
        entry_price: float,
    ) -> RiskCheckResult:
        """Validates whether a new trade can be opened.

        Args:
            proposed_size_eur: proposed size in €.
            current_balance_eur: current available balance.
            open_positions_count: currently open positions.
            entry_price: token entry price (between 0 and 1).

        Returns:
            RiskCheckResult with `approved` and, if applicable, `adjusted_size_eur`.
        """
        rejections: list[RejectReason] = []
        warnings: list[str] = []
        adjusted_size: Optional[float] = None

        # 1) Bot paused due to drawdown
        if self._is_paused:
            rejections.append(RejectReason.BOT_PAUSED)
            self._log.warning(
                "Trade rejected: bot paused ({})", self._pause_reason
            )
            return RiskCheckResult(approved=False, rejection_reasons=rejections)

        # 2) Invalid price (must be strictly between 0 and 1)
        if not (0 < entry_price < 1):
            rejections.append(RejectReason.INVALID_PRICE)
            return RiskCheckResult(approved=False, rejection_reasons=rejections)

        # 3) Too many open positions
        if open_positions_count >= self.risk.max_simultaneous_positions:
            rejections.append(RejectReason.MAX_POSITIONS_REACHED)

        # 4) Size below minimum
        if proposed_size_eur < self.risk.min_trade_size_eur:
            rejections.append(RejectReason.SIZE_BELOW_MIN)

        # 5) Insufficient balance: requesting more € than available.
        #    Checked BEFORE the max_position trim because it indicates an
        #    error in the chain that called the RiskManager.
        if proposed_size_eur > current_balance_eur:
            rejections.append(RejectReason.INSUFFICIENT_BALANCE)

        # 6) Size above maximum allowed → trim (do not reject)
        max_allowed = self.calculate_max_position_size(current_balance_eur)
        if proposed_size_eur > max_allowed:
            adjusted_size = max_allowed
            warnings.append(
                f"Requested size ({proposed_size_eur:.2f}€) exceeds the maximum "
                f"({max_allowed:.2f}€). Adjusted to {adjusted_size:.2f}€."
            )
            # If after trimming it falls below the minimum, reject
            if adjusted_size < self.risk.min_trade_size_eur:
                rejections.append(RejectReason.SIZE_ABOVE_MAX)

        # 7) VaR check — 95% confidence, 1-day parametric VaR
        #    daily_var = 1.645 * sigma * effective_size
        #    Reject if daily_var > balance * var_daily_limit_pct
        sigma = self.risk.var_sigma_assumption
        check_size = adjusted_size if adjusted_size is not None else proposed_size_eur
        daily_var = 1.645 * sigma * check_size
        var_limit = current_balance_eur * self.risk.var_daily_limit_pct
        if daily_var > var_limit:
            rejections.append(RejectReason.VAR_LIMIT_EXCEEDED)
            warnings.append(
                f"VaR check: daily VaR ({daily_var:.2f}€) exceeds limit "
                f"({var_limit:.2f}€ = {self.risk.var_daily_limit_pct:.0%} of balance)"
            )

        approved = len(rejections) == 0
        result = RiskCheckResult(
            approved=approved,
            rejection_reasons=rejections,
            warnings=warnings,
            adjusted_size_eur=adjusted_size,
        )

        if not approved:
            self._log.warning(
                "Trade rejected | size={:.2f}€ balance={:.2f}€ open={} "
                "price={:.3f} | reasons: {}",
                proposed_size_eur,
                current_balance_eur,
                open_positions_count,
                entry_price,
                [r.value for r in rejections],
            )
        else:
            final_size = adjusted_size if adjusted_size is not None else proposed_size_eur
            self._log.info(
                "Trade approved | size={:.2f}€ (adjusted={}) price={:.3f}",
                final_size,
                adjusted_size is not None,
                entry_price,
            )

        return result

    # =====================================================
    # Size and level calculations
    # =====================================================

    def calculate_max_position_size(self, current_balance_eur: float) -> float:
        """Maximum allowed position size (€) based on current balance."""
        return current_balance_eur * self.risk.max_position_size_pct

    def calculate_stop_loss_price(self, entry_price: float) -> float:
        """Token price at which to trigger the stop loss.

        If entry_price = $0.40 and stop_loss_pct = 0.20 → SL at $0.32 (-20%).
        Works the same for BUY_YES and BUY_NO since each is its own token.
        """
        return round(entry_price * (1 - self.risk.stop_loss_pct), 4)

    def calculate_take_profit_price(self, entry_price: float) -> float:
        """Token price at which taking profits is considered.

        On Polymarket the maximum price is $1, so cap at 0.999 to avoid
        impossible prices when entry_price is high.
        """
        target = entry_price * (1 + self.risk.take_profit_pct)
        return round(min(target, 0.999), 4)

    # =====================================================
    # Open position closing
    # =====================================================

    def should_close_position(
        self,
        position: Position,
        current_price: float,
    ) -> CloseDecision:
        """Decides whether to close a position based on stop loss, take profit, or time.

        Evaluation order:
          1. Stop loss (protection — always first)
          2. Time: tightened TP after N hours (Tier 1)
          3. Original take profit
          4. Time: close if in profit after N hours (Tier 2)
          5. Time: unconditional forced close (Tier 3)

        Closes due to contradictory news or market resolution are handled
        by the DECISION_ENGINE in downstream modules.
        """
        pnl_pct = position.current_pnl_pct(current_price)
        pnl_eur = position.current_pnl_eur(current_price)

        # Calculate hours held since entry
        now = datetime.now(timezone.utc)
        entry = position.entry_timestamp
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=timezone.utc)
        hours_held = (now - entry).total_seconds() / 3600

        risk = self.risk

        # 1) Stop loss: price falls to or below the calculated level
        if current_price <= position.stop_loss_price:
            self._log.info(
                "STOP LOSS triggered | trade={} entry={:.3f} now={:.3f} pnl={:.2%}",
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
                notes=f"Price {current_price:.4f} <= SL {position.stop_loss_price:.4f}",
            )

        # 2) TIME EXIT — Tier 1: tightened TP after time_tighten_tp_hours
        if hours_held >= risk.time_tighten_tp_hours:
            tightened_tp = position.entry_price * (1 + risk.time_tighten_tp_pct)
            if current_price >= tightened_tp:
                self._log.info(
                    "TIME EXIT (tightened TP) | trade={} held={:.1f}h "
                    "price={:.4f} TP_tight={:.4f} pnl={:+.2%}",
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
                        f"TP tightened by time: {hours_held:.1f}h "
                        f">= {risk.time_tighten_tp_hours:.0f}h | "
                        f"price={current_price:.4f} >= TP_tightened={tightened_tp:.4f}"
                    ),
                )

        # 3) Original take profit: price reaches or exceeds the calculated level
        if current_price >= position.take_profit_price:
            self._log.info(
                "TAKE PROFIT reached | trade={} entry={:.3f} now={:.3f} pnl={:+.2%}",
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
                notes=f"Price {current_price:.4f} >= TP {position.take_profit_price:.4f}",
            )

        # 4) TIME EXIT — Tier 2: close if in profit after time_exit_profit_hours
        if hours_held >= risk.time_exit_profit_hours and pnl_pct >= 0:
            self._log.info(
                "TIME EXIT (in profit) | trade={} held={:.1f}h pnl={:+.2%}",
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
                    f"Time-based close in profit: {hours_held:.1f}h "
                    f">= {risk.time_exit_profit_hours:.0f}h | P&L={pnl_pct:+.2%}"
                ),
            )

        # 5) TIME EXIT — Tier 3: unconditional forced close after time_exit_hard_hours
        if hours_held >= risk.time_exit_hard_hours:
            self._log.info(
                "TIME EXIT (forced) | trade={} held={:.1f}h pnl={:+.2%}",
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
                    f"Forced time-based close: {hours_held:.1f}h "
                    f">= {risk.time_exit_hard_hours:.0f}h | P&L={pnl_pct:+.2%}"
                ),
            )

        # Position is still within levels
        return CloseDecision(
            should_close=False,
            reason=None,
            pnl_pct=pnl_pct,
            pnl_eur=pnl_eur,
            notes="Within levels",
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
                    f"Drawdown {drawdown:.2%} exceeds the limit "
                    f"{self.risk.max_drawdown_pct:.0%} of initial capital"
                )
                self._log.error(
                    "BOT PAUSED | balance={:.2f}€ initial={:.2f}€ drawdown={:.2%}",
                    current_balance_eur,
                    self._initial_balance,
                    drawdown,
                )
            else:
                self._log.warning(
                    "Drawdown alert | balance={:.2f}€ initial={:.2f}€ drawdown={:.2%} "
                    "(monitoring-only — bot continues operating)",
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
    # Manual pause state control
    # =====================================================

    def manually_resume(self) -> None:
        """Resumes the bot after a drawdown pause.

        Should only be called from a manual operator intervention (CLI or
        Discord), after reviewing the situation.
        """
        if not self._is_paused:
            self._log.warning("manually_resume() called but the bot was not paused")
            return

        prev_reason = self._pause_reason
        self._is_paused = False
        self._pause_reason = None
        # Resetting the peak to the current balance avoids immediately re-pausing
        # after resuming if we are still at the low.
        # The operator can also call reset_peak() if preferred.
        self._log.info("Bot resumed manually. Previous reason: {}", prev_reason)

    def reset_peak(self, new_peak: float) -> None:
        """Resets the peak balance. Useful when resuming after a drawdown."""
        if new_peak <= 0:
            raise ValueError("Peak balance must be positive")
        self._log.info(
            "Peak balance reset: {:.2f}€ → {:.2f}€",
            self._peak_balance,
            new_peak,
        )
        self._peak_balance = new_peak
