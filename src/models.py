"""
Core data models for the bot.

Shared across modules (risk_manager, paper_trader, decision_engine,
report_generator). Defined with Pydantic for automatic validation.

Conventions:
- Token prices on Polymarket are between $0 and $1 (implied probability).
- P&L is calculated on the price of the token we hold:
      pnl_pct = (current_price - entry_price) / entry_price
  This formula is valid for both BUY_YES and BUY_NO because each side
  is an independent token with its own price.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# =====================================================
# Enums
# =====================================================


class TradeSide(str, Enum):
    """Trade side. On Polymarket each side is an independent token."""

    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class CloseReason(str, Enum):
    """Reason why a position is closed."""

    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    MANUAL = "MANUAL"
    NEWS_REVERSAL = "NEWS_REVERSAL"
    MARKET_RESOLVED = "MARKET_RESOLVED"
    TIME_EXIT = "TIME_EXIT"


class RejectReason(str, Enum):
    """Reasons why the RiskManager rejects a new trade."""

    BOT_PAUSED = "BOT_PAUSED"
    MAX_POSITIONS_REACHED = "MAX_POSITIONS_REACHED"
    SIZE_BELOW_MIN = "SIZE_BELOW_MIN"
    SIZE_ABOVE_MAX = "SIZE_ABOVE_MAX"
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
    INVALID_PRICE = "INVALID_PRICE"


class NewsSource(str, Enum):
    """Source from which a news article originates."""

    NEWSAPI = "NEWSAPI"
    GDELT = "GDELT"
    TELEGRAM = "TELEGRAM"
    RSS = "RSS"


# =====================================================
# Helpers
# =====================================================


def _new_trade_id() -> str:
    return str(uuid.uuid4())


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# =====================================================
# Position / Trade
# =====================================================


class Position(BaseModel):
    """A paper trading position, open or closed."""

    model_config = ConfigDict(use_enum_values=False)

    trade_id: str = Field(default_factory=_new_trade_id)
    market_question: str
    market_slug: str = ""
    token_id: str
    side: TradeSide

    # Entry
    entry_price: float = Field(gt=0, lt=1, description="Token price between $0 and $1")
    size_eur: float = Field(gt=0, description="Amount invested in EUR")
    size_usd: float = Field(gt=0, description="Amount in USDC (Polymarket operates in USDC)")
    tokens_quantity: float = Field(gt=0, description="size_usd / entry_price")
    entry_timestamp: datetime = Field(default_factory=_now_utc)

    # Levels calculated by the RiskManager
    stop_loss_price: float = Field(gt=0, lt=1)
    take_profit_price: float = Field(gt=0)

    # Exit (None while open)
    status: TradeStatus = TradeStatus.OPEN
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    close_reason: Optional[CloseReason] = None
    pnl_eur: Optional[float] = None
    pnl_pct: Optional[float] = None

    # Traceability
    entry_reason: str = ""
    exit_reason_text: str = ""
    confidence: int = Field(default=0, ge=0, le=100)

    def current_pnl_pct(self, current_price: float) -> float:
        """Percentage P&L on the price of the token we hold."""
        return (current_price - self.entry_price) / self.entry_price

    def current_pnl_eur(self, current_price: float) -> float:
        """Absolute P&L in euros at the current price."""
        return self.current_pnl_pct(current_price) * self.size_eur


# =====================================================
# RiskManager results
# =====================================================


class RiskCheckResult(BaseModel):
    """Result of validating a new trade."""

    approved: bool
    rejection_reasons: list[RejectReason] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    adjusted_size_eur: Optional[float] = Field(
        default=None,
        description="Adjusted size if the requested amount exceeded the allowed maximum",
    )

    @property
    def is_rejected(self) -> bool:
        return not self.approved


class CloseDecision(BaseModel):
    """Decision on whether an open position should be closed."""

    should_close: bool
    reason: Optional[CloseReason] = None
    pnl_pct: float
    pnl_eur: float
    notes: str = ""


class DrawdownStatus(BaseModel):
    """Current drawdown status."""

    current_balance_eur: float
    peak_balance_eur: float
    current_drawdown_pct: float = Field(ge=0)
    threshold_breached: bool
    bot_should_pause: bool


# =====================================================
# Market (snapshot)
# =====================================================


class MarketSnapshot(BaseModel):
    """Instant snapshot of a Polymarket market.

    Generated by the MARKET_SCANNER from the Gamma API. Not the source of
    truth for executing orders (that is done by the CLOB), only for identifying
    which markets are tradeable and at what price.
    """

    # Identification
    market_id: str
    slug: str = ""
    question: str
    description: str = ""
    category: str = ""
    end_date: Optional[datetime] = None

    # CLOB tokens (one per YES/NO side)
    yes_token_id: str
    no_token_id: str

    # Current prices (between 0 and 1)
    yes_price: float = Field(ge=0, le=1)
    no_price: float = Field(ge=0, le=1)

    # Order book top (optional, for real spread)
    best_bid: Optional[float] = Field(default=None, ge=0, le=1)
    best_ask: Optional[float] = Field(default=None, ge=0, le=1)
    spread: float = Field(ge=0)

    # Liquidity
    volume_24h_usd: float = Field(ge=0)
    volume_total_usd: float = Field(ge=0)
    liquidity_usd: float = Field(ge=0)

    # Status
    is_active: bool = True
    is_closed: bool = False

    # Snapshot timestamp (UTC)
    snapshot_timestamp: datetime = Field(default_factory=_now_utc)

    @property
    def time_to_close_hours(self) -> Optional[float]:
        """Hours until market close, or None if there is no end_date."""
        if self.end_date is None:
            return None
        delta = self.end_date - datetime.now(timezone.utc)
        return delta.total_seconds() / 3600

    @property
    def implied_yes_probability(self) -> float:
        """Implied YES probability = YES token price."""
        return self.yes_price


# =====================================================
# News
# =====================================================


def _new_article_id(url: str, title: str) -> str:
    """Stable article ID (short hash of url + title)."""
    import hashlib

    return hashlib.sha1(f"{url}|{title}".encode("utf-8")).hexdigest()[:16]


class NewsArticle(BaseModel):
    """Normalized news article, independent of source."""

    article_id: str
    source: NewsSource
    source_name: str = ""        # Publisher name: "Reuters", "Bloomberg", etc.
    title: str
    description: str = ""
    content: str = ""
    url: str
    author: Optional[str] = None
    language: str = ""

    published_at: Optional[datetime] = None
    ingested_at: datetime = Field(default_factory=_now_utc)

    # Heuristic score calculated by the NEWS_INGESTOR before passing to the LLM.
    # The LLM (SENTIMENT_ANALYZER module) will recalculate a more refined real score.
    preliminary_impact_score: float = Field(default=0.0, ge=0, le=100)
    matched_keywords: list[str] = Field(default_factory=list)


# =====================================================
# Market analysis (output of SENTIMENT_ANALYZER)
# =====================================================


class TradeRecommendation(str, Enum):
    """Recommendation issued by the SENTIMENT_ANALYZER to the DECISION_ENGINE."""

    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    WAIT = "WAIT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # No news available or news is irrelevant


class Timeframe(str, Enum):
    """Estimated time horizon of the news impact."""

    IMMEDIATE = "IMMEDIATE"   # < 1h
    HOURS = "HOURS"           # 1h - 24h
    DAYS = "DAYS"             # 24h - 7d
    UNKNOWN = "UNKNOWN"


class MarketAnalysis(BaseModel):
    """Quantitative analysis of a market based on associated news.

    Structured output from the SENTIMENT_ANALYZER that the DECISION_ENGINE will use
    to decide whether to open a trade.
    """

    # Analysis identification
    market_id: str
    market_question: str
    market_slug: str = ""
    yes_token_id: str
    no_token_id: str

    # Current market prices at the time of analysis
    current_yes_price: float = Field(ge=0, le=1)
    current_no_price: float = Field(ge=0, le=1)

    # LLM output
    consensus_probability_yes: float = Field(
        ge=0.0, le=1.0,
        description="Consensus probability that YES is the outcome, "
                    "based on the news analysis.",
    )
    edge: float = Field(
        description="consensus_probability_yes - current_yes_price. "
                    "Positive → YES undervalued. Negative → NO undervalued.",
    )
    confidence: int = Field(
        ge=0, le=100,
        description="LLM confidence in its own analysis (0-100).",
    )
    sentiment_score: float = Field(
        ge=-1.0, le=1.0,
        description="Aggregated news sentiment toward YES (-1 to +1).",
    )
    impact_score: float = Field(
        ge=0.0, le=100.0,
        description="Combined magnitude of news impact (0-100).",
    )

    recommendation: TradeRecommendation
    timeframe: Timeframe = Timeframe.UNKNOWN
    contradictory_sources: bool = False

    # Traceability
    summary: str = ""              # LLM executive summary (2-3 sentences)
    justification: str = ""        # Recommendation reasoning
    article_ids_analyzed: list[str] = Field(default_factory=list)
    num_articles_analyzed: int = 0

    # Low-info mode: analysis performed with fewer news articles than usual.
    # The DECISION_ENGINE will apply stricter thresholds and reduced sizing.
    is_low_info: bool = False

    # Metadata
    analyzed_at: datetime = Field(default_factory=_now_utc)
    llm_model: str = ""
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


# =====================================================
# Trading decision (output of DECISION_ENGINE)
# =====================================================


class DecisionAction(str, Enum):
    """What to do according to the DECISION_ENGINE decision."""

    OPEN_TRADE = "OPEN_TRADE"
    NO_TRADE = "NO_TRADE"


class SkipReason(str, Enum):
    """Reasons why the DECISION_ENGINE discards opening a trade."""

    LLM_RECOMMENDS_WAIT = "LLM_RECOMMENDS_WAIT"
    LLM_INSUFFICIENT_DATA = "LLM_INSUFFICIENT_DATA"
    DUPLICATE_OPEN_POSITION = "DUPLICATE_OPEN_POSITION"
    OPPOSITE_OPEN_POSITION = "OPPOSITE_OPEN_POSITION"
    SIZE_BELOW_MIN_AFTER_SIZING = "SIZE_BELOW_MIN_AFTER_SIZING"
    RISK_MANAGER_REJECTED = "RISK_MANAGER_REJECTED"
    REQUIRE_NEWS_BUT_NONE = "REQUIRE_NEWS_BUT_NONE"


class TradeDecision(BaseModel):
    """Final decision ready for the PAPER_TRADER to execute (or not)."""

    action: DecisionAction
    market_id: str
    market_question: str
    market_slug: str = ""

    # If action == OPEN_TRADE, all of the following fields are populated:
    side: Optional[TradeSide] = None
    token_id: Optional[str] = None
    entry_price: Optional[float] = None
    size_eur: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    confidence: int = 0

    # Traceability
    edge: float = 0.0
    skip_reasons: list[SkipReason] = Field(default_factory=list)
    rationale: str = ""
    analysis_id: str = ""              # To correlate with MarketAnalysis
    decided_at: datetime = Field(default_factory=_now_utc)

    @property
    def should_execute(self) -> bool:
        return self.action == DecisionAction.OPEN_TRADE
