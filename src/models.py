"""
Modelos de datos centrales del bot.

Compartidos entre módulos (risk_manager, paper_trader, decision_engine,
report_generator). Definidos con Pydantic para validación automática.

Convenciones:
- Los precios de los tokens en Polymarket están entre $0 y $1 (probabilidad implícita).
- El P&L se calcula sobre el precio del token que poseemos:
      pnl_pct = (precio_actual - precio_entrada) / precio_entrada
  Esta fórmula es válida tanto para BUY_YES como para BUY_NO porque cada lado
  es un token independiente con su propio precio.
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
    """Lado del trade. En Polymarket cada lado es un token independiente."""

    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class CloseReason(str, Enum):
    """Motivo por el que se cierra una posición."""

    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    MANUAL = "MANUAL"
    NEWS_REVERSAL = "NEWS_REVERSAL"
    MARKET_RESOLVED = "MARKET_RESOLVED"
    TIME_EXIT = "TIME_EXIT"


class RejectReason(str, Enum):
    """Motivos por los que el RiskManager rechaza un trade nuevo."""

    BOT_PAUSED = "BOT_PAUSED"
    MAX_POSITIONS_REACHED = "MAX_POSITIONS_REACHED"
    SIZE_BELOW_MIN = "SIZE_BELOW_MIN"
    SIZE_ABOVE_MAX = "SIZE_ABOVE_MAX"
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
    INVALID_PRICE = "INVALID_PRICE"


class NewsSource(str, Enum):
    """Fuente de la que procede un artículo de noticias."""

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
# Posición / Trade
# =====================================================


class Position(BaseModel):
    """Una posición de paper trading, abierta o cerrada."""

    model_config = ConfigDict(use_enum_values=False)

    trade_id: str = Field(default_factory=_new_trade_id)
    market_question: str
    market_slug: str = ""
    token_id: str
    side: TradeSide

    # Entrada
    entry_price: float = Field(gt=0, lt=1, description="Precio del token entre $0 y $1")
    size_eur: float = Field(gt=0, description="Importe invertido en €")
    size_usd: float = Field(gt=0, description="Importe en USDC (Polymarket opera en USDC)")
    tokens_quantity: float = Field(gt=0, description="size_usd / entry_price")
    entry_timestamp: datetime = Field(default_factory=_now_utc)

    # Niveles calculados por el RiskManager
    stop_loss_price: float = Field(gt=0, lt=1)
    take_profit_price: float = Field(gt=0)

    # Salida (None mientras esté abierta)
    status: TradeStatus = TradeStatus.OPEN
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    close_reason: Optional[CloseReason] = None
    pnl_eur: Optional[float] = None
    pnl_pct: Optional[float] = None

    # Trazabilidad
    entry_reason: str = ""
    exit_reason_text: str = ""
    confidence: int = Field(default=0, ge=0, le=100)

    def current_pnl_pct(self, current_price: float) -> float:
        """P&L porcentual sobre el precio del token que poseemos."""
        return (current_price - self.entry_price) / self.entry_price

    def current_pnl_eur(self, current_price: float) -> float:
        """P&L absoluto en euros para el precio actual."""
        return self.current_pnl_pct(current_price) * self.size_eur


# =====================================================
# Resultados del RiskManager
# =====================================================


class RiskCheckResult(BaseModel):
    """Resultado de validar un trade nuevo."""

    approved: bool
    rejection_reasons: list[RejectReason] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    adjusted_size_eur: Optional[float] = Field(
        default=None,
        description="Tamaño ajustado si el solicitado superaba el máximo permitido",
    )

    @property
    def is_rejected(self) -> bool:
        return not self.approved


class CloseDecision(BaseModel):
    """Decisión sobre si una posición abierta debe cerrarse."""

    should_close: bool
    reason: Optional[CloseReason] = None
    pnl_pct: float
    pnl_eur: float
    notes: str = ""


class DrawdownStatus(BaseModel):
    """Estado actual del drawdown."""

    current_balance_eur: float
    peak_balance_eur: float
    current_drawdown_pct: float = Field(ge=0)
    threshold_breached: bool
    bot_should_pause: bool


# =====================================================
# Mercado (snapshot)
# =====================================================


class MarketSnapshot(BaseModel):
    """Foto instantánea de un mercado de Polymarket.

    Generado por el MARKET_SCANNER a partir de la Gamma API. No es la fuente de
    verdad para ejecutar órdenes (eso lo hace el CLOB), solo para identificar
    qué mercados son operables y a qué precio están.
    """

    # Identificación
    market_id: str
    slug: str = ""
    question: str
    description: str = ""
    category: str = ""
    end_date: Optional[datetime] = None

    # Tokens CLOB (uno por lado YES/NO)
    yes_token_id: str
    no_token_id: str

    # Precios actuales (entre 0 y 1)
    yes_price: float = Field(ge=0, le=1)
    no_price: float = Field(ge=0, le=1)

    # Order book top (opcional, para spread real)
    best_bid: Optional[float] = Field(default=None, ge=0, le=1)
    best_ask: Optional[float] = Field(default=None, ge=0, le=1)
    spread: float = Field(ge=0)

    # Liquidez
    volume_24h_usd: float = Field(ge=0)
    volume_total_usd: float = Field(ge=0)
    liquidity_usd: float = Field(ge=0)

    # Estado
    is_active: bool = True
    is_closed: bool = False

    # Timestamp del snapshot (UTC)
    snapshot_timestamp: datetime = Field(default_factory=_now_utc)

    @property
    def time_to_close_hours(self) -> Optional[float]:
        """Horas hasta el cierre del mercado, o None si no hay end_date."""
        if self.end_date is None:
            return None
        delta = self.end_date - datetime.now(timezone.utc)
        return delta.total_seconds() / 3600

    @property
    def implied_yes_probability(self) -> float:
        """Probabilidad implícita YES = precio del token YES."""
        return self.yes_price


# =====================================================
# Noticias
# =====================================================


def _new_article_id(url: str, title: str) -> str:
    """ID estable de un artículo (hash corto de url + título)."""
    import hashlib

    return hashlib.sha1(f"{url}|{title}".encode("utf-8")).hexdigest()[:16]


class NewsArticle(BaseModel):
    """Artículo de noticia normalizado, independiente de la fuente."""

    article_id: str
    source: NewsSource
    source_name: str = ""        # Nombre publicador: "Reuters", "Bloomberg", etc.
    title: str
    description: str = ""
    content: str = ""
    url: str
    author: Optional[str] = None
    language: str = ""

    published_at: Optional[datetime] = None
    ingested_at: datetime = Field(default_factory=_now_utc)

    # Score heurístico calculado por el NEWS_INGESTOR antes de pasar al LLM.
    # El LLM (módulo SENTIMENT_ANALYZER) recalculará un score real más fino.
    preliminary_impact_score: float = Field(default=0.0, ge=0, le=100)
    matched_keywords: list[str] = Field(default_factory=list)


# =====================================================
# Análisis de mercado (output del SENTIMENT_ANALYZER)
# =====================================================


class TradeRecommendation(str, Enum):
    """Recomendación que emite el SENTIMENT_ANALYZER al DECISION_ENGINE."""

    COMPRAR_YES = "COMPRAR_YES"
    COMPRAR_NO = "COMPRAR_NO"
    ESPERAR = "ESPERAR"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # No hay noticias o son irrelevantes


class Timeframe(str, Enum):
    """Horizonte temporal estimado del impacto de la noticia."""

    INMEDIATO = "INMEDIATO"   # < 1h
    HORAS = "HORAS"           # 1h - 24h
    DIAS = "DIAS"             # 24h - 7d
    DESCONOCIDO = "DESCONOCIDO"


class MarketAnalysis(BaseModel):
    """Análisis cuantitativo de un mercado a partir de las noticias asociadas.

    Output estructurado del SENTIMENT_ANALYZER que el DECISION_ENGINE usará
    para decidir si abrir un trade.
    """

    # Identificación del análisis
    market_id: str
    market_question: str
    market_slug: str = ""
    yes_token_id: str
    no_token_id: str

    # Precios actuales del mercado en el momento del análisis
    current_yes_price: float = Field(ge=0, le=1)
    current_no_price: float = Field(ge=0, le=1)

    # Salida del LLM
    consensus_probability_yes: float = Field(
        ge=0.0, le=1.0,
        description="Probabilidad consensuada de que YES sea el resultado, "
                    "según el análisis de las noticias.",
    )
    edge: float = Field(
        description="consensus_probability_yes - current_yes_price. "
                    "Positivo → YES infravalorado. Negativo → NO infravalorado.",
    )
    confidence: int = Field(
        ge=0, le=100,
        description="Confianza del LLM en su propio análisis (0-100).",
    )
    sentiment_score: float = Field(
        ge=-1.0, le=1.0,
        description="Sentimiento agregado de las noticias hacia el YES (-1 a +1).",
    )
    impact_score: float = Field(
        ge=0.0, le=100.0,
        description="Magnitud combinada del impacto de las noticias (0-100).",
    )

    recommendation: TradeRecommendation
    timeframe: Timeframe = Timeframe.DESCONOCIDO
    contradictory_sources: bool = False

    # Trazabilidad
    summary: str = ""              # Resumen ejecutivo del LLM (2-3 oraciones)
    justification: str = ""        # Razonamiento de la recomendación
    article_ids_analyzed: list[str] = Field(default_factory=list)
    num_articles_analyzed: int = 0

    # Modo low-info: análisis hecho con menos noticias de las habituales.
    # El DECISION_ENGINE aplicará umbrales más estrictos y sizing reducido.
    is_low_info: bool = False

    # Metadatos
    analyzed_at: datetime = Field(default_factory=_now_utc)
    llm_model: str = ""
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


# =====================================================
# Decisión de trading (output del DECISION_ENGINE)
# =====================================================


class DecisionAction(str, Enum):
    """Qué hacer según la decisión del DECISION_ENGINE."""

    OPEN_TRADE = "OPEN_TRADE"
    NO_TRADE = "NO_TRADE"


class SkipReason(str, Enum):
    """Razones por las que el DECISION_ENGINE descarta abrir un trade."""

    LLM_RECOMMENDS_WAIT = "LLM_RECOMMENDS_WAIT"
    LLM_INSUFFICIENT_DATA = "LLM_INSUFFICIENT_DATA"
    DUPLICATE_OPEN_POSITION = "DUPLICATE_OPEN_POSITION"
    OPPOSITE_OPEN_POSITION = "OPPOSITE_OPEN_POSITION"
    SIZE_BELOW_MIN_AFTER_SIZING = "SIZE_BELOW_MIN_AFTER_SIZING"
    RISK_MANAGER_REJECTED = "RISK_MANAGER_REJECTED"
    REQUIRE_NEWS_BUT_NONE = "REQUIRE_NEWS_BUT_NONE"


class TradeDecision(BaseModel):
    """Decisión final lista para que el PAPER_TRADER la ejecute (o no)."""

    action: DecisionAction
    market_id: str
    market_question: str
    market_slug: str = ""

    # Si action == OPEN_TRADE, los siguientes campos están todos rellenos:
    side: Optional[TradeSide] = None
    token_id: Optional[str] = None
    entry_price: Optional[float] = None
    size_eur: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    confidence: int = 0

    # Trazabilidad
    edge: float = 0.0
    skip_reasons: list[SkipReason] = Field(default_factory=list)
    rationale: str = ""
    analysis_id: str = ""              # Para correlacionar con MarketAnalysis
    decided_at: datetime = Field(default_factory=_now_utc)

    @property
    def should_execute(self) -> bool:
        return self.action == DecisionAction.OPEN_TRADE

