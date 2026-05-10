"""
Decision Engine — the last filter before the paper trader.

Receives a `MarketAnalysis` (output of the SENTIMENT_ANALYZER) + balance + open
positions, and produces a concrete `TradeDecision`:
- open a trade (with side, size, stop loss and take profit calculated), OR
- NO_TRADE with a clear reason for the log.

Also periodically evaluates open positions and decides whether to close them:
- by stop loss / take profit (delegated to RiskManager)
- by a new contradictory news article with high confidence
- by market resolution (when implemented)

Design:
- Stateless where possible. State is held by the RiskManager (drawdown, pause).
- Chains 4 validation layers before saying OPEN_TRADE:
   1. LLM recommendation must be BUY_YES or BUY_NO.
   2. Do not open if there is already a position on the same market and same side.
   3. Cancel if there is already a position on the OPPOSITE side (would be contradictory).
   4. Final RiskManager.validate_new_trade().
- Dynamic sizing: the base size is the RiskManager maximum, scaled down
  by (confidence × |edge|). Higher confidence and higher edge → closer to the maximum.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from src.config_loader import BotConfig
from src.models import (
    CloseDecision,
    DecisionAction,
    MarketAnalysis,
    NewsArticle,
    Position,
    SkipReason,
    TradeDecision,
    TradeRecommendation,
    TradeSide,
)
from src.risk_manager import RiskManager


# Minimum confidence to open a trade. Independent of the LLM threshold
# (which is already applied by the SENTIMENT_ANALYZER). This is a second line of defense.
MIN_CONFIDENCE_TO_OPEN = 60

# Minimum absolute edge to open a trade.
MIN_EDGE_TO_OPEN = 0.05

# To size the position we use a "confidence factor" that ranges from 0 to 1.
# If the LLM returns confidence=100 and |edge|>=0.30, the factor is 1.0 → maximum size.
# If confidence=60 and |edge|=0.05, the factor is low → size near the minimum.
EDGE_REFERENCE_FOR_FULL_SIZE = 0.30


class DecisionEngine:
    """Converts LLM analysis into concrete trading decisions."""

    def __init__(
        self,
        config: BotConfig,
        risk_manager: RiskManager,
    ) -> None:
        self.config = config
        self.risk_manager = risk_manager
        self._log = logger.bind(module="decision_engine")
        self._log.info(
            "DecisionEngine initialized: min_confidence={}, min_edge={}",
            MIN_CONFIDENCE_TO_OPEN,
            MIN_EDGE_TO_OPEN,
        )

    # =====================================================
    # Opening decision
    # =====================================================

    def decide(
        self,
        analysis: MarketAnalysis,
        current_balance_eur: float,
        open_positions: list[Position],
        articles: Optional[list[NewsArticle]] = None,
    ) -> TradeDecision:
        """Decides whether to open a trade given the analysis and current state.

        Args:
            analysis: the output of the SENTIMENT_ANALYZER.
            current_balance_eur: current available balance.
            open_positions: list of open positions (from any market).
            articles: associated news articles (optional, for require_news_for_entry).
        """
        skip_reasons: list[SkipReason] = []
        rationale_parts: list[str] = []

        # 1) Filter: the LLM must recommend buying
        if analysis.recommendation == TradeRecommendation.INSUFFICIENT_DATA:
            skip_reasons.append(SkipReason.LLM_INSUFFICIENT_DATA)
            rationale_parts.append("The LLM does not have enough data")
            return self._no_trade(analysis, skip_reasons, rationale_parts)

        if analysis.recommendation == TradeRecommendation.WAIT:
            skip_reasons.append(SkipReason.LLM_RECOMMENDS_WAIT)
            rationale_parts.append("The LLM recommends waiting")
            return self._no_trade(analysis, skip_reasons, rationale_parts)

        # 2) Filter: minimum confidence and edge (second line of defense).
        #    If the analysis is low_info, a higher threshold is required.
        cfg_dec = self.config.decision
        effective_min_confidence = (
            cfg_dec.low_info_min_confidence
            if analysis.is_low_info
            else MIN_CONFIDENCE_TO_OPEN
        )
        if analysis.confidence < effective_min_confidence:
            skip_reasons.append(SkipReason.LLM_RECOMMENDS_WAIT)
            mode = "low_info" if analysis.is_low_info else "normal"
            rationale_parts.append(
                f"Confidence {analysis.confidence} < threshold {effective_min_confidence} "
                f"(mode {mode})"
            )
            return self._no_trade(analysis, skip_reasons, rationale_parts)

        if abs(analysis.edge) < MIN_EDGE_TO_OPEN:
            skip_reasons.append(SkipReason.LLM_RECOMMENDS_WAIT)
            rationale_parts.append(
                f"|Edge| {abs(analysis.edge):.3f} < threshold {MIN_EDGE_TO_OPEN}"
            )
            return self._no_trade(analysis, skip_reasons, rationale_parts)

        # 3) Filter: require news if config demands it.
        #    In low_info mode this filter is relaxed (1 article is enough).
        if self.config.decision.require_news_for_entry and not analysis.is_low_info:
            if not articles or len(articles) == 0:
                skip_reasons.append(SkipReason.REQUIRE_NEWS_BUT_NONE)
                rationale_parts.append(
                    "Config requires news to open, but none were provided"
                )
                return self._no_trade(analysis, skip_reasons, rationale_parts)

        # 4) Determine trade side
        side, token_id, entry_price = self._resolve_side_and_token(analysis)

        # Compute expected value and store on analysis object
        p = analysis.consensus_probability_yes
        b = (1.0 / entry_price) - 1.0   # net decimal odds
        ev = p * b - (1.0 - p)
        analysis.expected_value = round(ev, 4)

        # 5) Anti-duplication: is there already an open position in this market?
        for pos in open_positions:
            if pos.token_id != token_id and self._same_market(pos, analysis):
                # Same question but opposite side → contradictory position
                skip_reasons.append(SkipReason.OPPOSITE_OPEN_POSITION)
                rationale_parts.append(
                    f"There is already an open position on the opposite side of the market "
                    f"({pos.trade_id[:8]})"
                )
                return self._no_trade(analysis, skip_reasons, rationale_parts)
            if pos.token_id == token_id:
                skip_reasons.append(SkipReason.DUPLICATE_OPEN_POSITION)
                rationale_parts.append(
                    f"There is already an open position on this token "
                    f"({pos.trade_id[:8]})"
                )
                return self._no_trade(analysis, skip_reasons, rationale_parts)

        # 6) Dynamic sizing (with reduction if low_info)
        proposed_size_eur = self._calculate_position_size(
            current_balance_eur=current_balance_eur,
            confidence=analysis.confidence,
            edge=analysis.edge,
            is_low_info=analysis.is_low_info,
        )

        # 7) Final RiskManager validation
        risk_check = self.risk_manager.validate_new_trade(
            proposed_size_eur=proposed_size_eur,
            current_balance_eur=current_balance_eur,
            open_positions_count=len(open_positions),
            entry_price=entry_price,
        )
        if not risk_check.approved:
            skip_reasons.append(SkipReason.RISK_MANAGER_REJECTED)
            rationale_parts.append(
                f"RiskManager rejected: {[r.value for r in risk_check.rejection_reasons]}"
            )
            return self._no_trade(analysis, skip_reasons, rationale_parts)

        # Apply the adjustment if one was made
        final_size = (
            risk_check.adjusted_size_eur
            if risk_check.adjusted_size_eur is not None
            else proposed_size_eur
        )

        # Check that after adjustment it is still above the minimum
        if final_size < self.risk_manager.risk.min_trade_size_eur:
            skip_reasons.append(SkipReason.SIZE_BELOW_MIN_AFTER_SIZING)
            rationale_parts.append(
                f"Size after adjustments ({final_size:.2f}€) < minimum "
                f"({self.risk_manager.risk.min_trade_size_eur}€)"
            )
            return self._no_trade(analysis, skip_reasons, rationale_parts)

        # 8) Approved! Calculate levels
        sl = self.risk_manager.calculate_stop_loss_price(entry_price)
        tp = self.risk_manager.calculate_take_profit_price(entry_price)

        rationale = (
            f"LLM={analysis.recommendation.value}, "
            f"edge={analysis.edge:+.3f}, "
            f"EV={analysis.expected_value:+.3f}, "
            f"confidence={analysis.confidence}/100, "
            f"size={final_size:.2f}€ "
            f"(SL={sl:.3f}, TP={tp:.3f})"
        )
        self._log.info("OPEN_TRADE | {} | {}", analysis.market_question[:60], rationale)

        return TradeDecision(
            action=DecisionAction.OPEN_TRADE,
            market_id=analysis.market_id,
            market_question=analysis.market_question,
            market_slug=analysis.market_slug,
            side=side,
            token_id=token_id,
            entry_price=entry_price,
            size_eur=final_size,
            stop_loss_price=sl,
            take_profit_price=tp,
            confidence=analysis.confidence,
            edge=analysis.edge,
            rationale=rationale,
            analysis_id=analysis.market_id,  # placeholder
        )

    # =====================================================
    # Open position re-evaluation
    # =====================================================

    def evaluate_open_position(
        self,
        position: Position,
        current_price: float,
        new_analysis: Optional[MarketAnalysis] = None,
    ) -> CloseDecision:
        """Decides whether to close an open position.

        Combines:
        1. Stop loss / take profit by price (delegated to RiskManager).
        2. Reversal by new news: if the LLM now recommends the opposite
           with confidence > 70%, close the position.
        """
        # 1) RiskManager check
        rm_decision = self.risk_manager.should_close_position(position, current_price)
        if rm_decision.should_close:
            return rm_decision

        # 2) News reversal
        if new_analysis is None:
            return rm_decision  # No new info, hold

        if new_analysis.confidence < 70:
            return rm_decision

        is_long_yes = position.side == TradeSide.BUY_YES
        contradicts = (
            (is_long_yes and new_analysis.recommendation == TradeRecommendation.BUY_NO)
            or (not is_long_yes and new_analysis.recommendation == TradeRecommendation.BUY_YES)
        )
        if contradicts:
            from src.models import CloseReason

            self._log.info(
                "CLOSE due to news reversal | trade={} new_rec={} confidence={}",
                position.trade_id[:8],
                new_analysis.recommendation.value,
                new_analysis.confidence,
            )
            pnl_pct = position.current_pnl_pct(current_price)
            return CloseDecision(
                should_close=True,
                reason=CloseReason.NEWS_REVERSAL,
                pnl_pct=pnl_pct,
                pnl_eur=position.current_pnl_eur(current_price),
                notes=(
                    f"Reversal: new analysis recommends "
                    f"{new_analysis.recommendation.value} with confidence "
                    f"{new_analysis.confidence}/100"
                ),
            )

        return rm_decision

    # =====================================================
    # Internals
    # =====================================================

    def _resolve_side_and_token(
        self, analysis: MarketAnalysis
    ) -> tuple[TradeSide, str, float]:
        """Returns (side, token_id, entry_price) based on the recommendation."""
        if analysis.recommendation == TradeRecommendation.BUY_YES:
            return (
                TradeSide.BUY_YES,
                analysis.yes_token_id,
                analysis.current_yes_price,
            )
        if analysis.recommendation == TradeRecommendation.BUY_NO:
            return (
                TradeSide.BUY_NO,
                analysis.no_token_id,
                analysis.current_no_price,
            )
        # Defensive: should not reach here because the prior check discards it
        raise ValueError(
            f"_resolve_side_and_token called with invalid recommendation: "
            f"{analysis.recommendation}"
        )

    def _calculate_position_size(
        self,
        current_balance_eur: float,
        confidence: int,
        edge: float,
        is_low_info: bool = False,
    ) -> float:
        """Dynamic sizing: (confidence × edge) factor applied to the maximum allowed.

        If is_low_info=True, applies the multiplier from config.decision to
        further reduce the size (typically 50%).

        Always guarantees at least the RiskManager's minimum trade size.
        At most, the maximum allowed position size.
        """
        max_size = self.risk_manager.calculate_max_position_size(current_balance_eur)
        min_size = self.risk_manager.risk.min_trade_size_eur

        # Confidence factor: 0 when confidence=0, 1 when confidence=100
        confidence_factor = confidence / 100.0
        # Edge factor: 0 when edge=0, 1 when |edge| >= EDGE_REFERENCE
        edge_factor = min(1.0, abs(edge) / EDGE_REFERENCE_FOR_FULL_SIZE)
        # Combination: product. If either is low, the size drops significantly.
        sizing_factor = confidence_factor * edge_factor

        # Additional reduction for low_info mode
        if is_low_info:
            low_info_mult = self.config.decision.low_info_size_multiplier
            sizing_factor *= low_info_mult
            self._log.debug(
                "Sizing low_info: applying multiplier {:.2f}", low_info_mult
            )

        proposed = max_size * sizing_factor
        return max(min_size, proposed)

    @staticmethod
    def _same_market(position: Position, analysis: MarketAnalysis) -> bool:
        """Heuristic: compare by market_question. The stable Polymarket market_id
        could be stored in Position in the future for greater robustness."""
        return position.market_question == analysis.market_question

    def _no_trade(
        self,
        analysis: MarketAnalysis,
        skip_reasons: list[SkipReason],
        rationale_parts: list[str],
    ) -> TradeDecision:
        rationale = " | ".join(rationale_parts) if rationale_parts else "(no reason)"
        self._log.debug(
            "NO_TRADE | {} | {}",
            analysis.market_question[:60],
            rationale,
        )
        return TradeDecision(
            action=DecisionAction.NO_TRADE,
            market_id=analysis.market_id,
            market_question=analysis.market_question,
            edge=analysis.edge,
            confidence=analysis.confidence,
            skip_reasons=skip_reasons,
            rationale=rationale,
            analysis_id=analysis.market_id,
        )
