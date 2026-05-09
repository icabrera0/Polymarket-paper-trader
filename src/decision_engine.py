"""
Decision Engine — el último filtro antes del paper trader.

Recibe un `MarketAnalysis` (output del SENTIMENT_ANALYZER) + balance + posiciones
abiertas, y produce un `TradeDecision` concreto:
- abrir un trade (con lado, tamaño, stop loss y take profit calculados), O
- NO_TRADE con razón clara para el log.

También evalúa periódicamente las posiciones abiertas y decide si hay que
cerrarlas:
- por stop loss / take profit (delegado al RiskManager)
- por noticia contradictoria nueva con alta confianza
- por resolución del mercado (cuando se implemente)

Diseño:
- Stateless en lo posible. El estado lo guarda el RiskManager (drawdown, pausa).
- Encadena 4 capas de validación antes de decir OPEN_TRADE:
   1. Recomendación del LLM debe ser COMPRAR_YES o COMPRAR_NO.
   2. No abrir si ya hay posición en el mismo mercado y mismo lado.
   3. Cancelar si ya hay posición en el LADO OPUESTO (sería contradictoria).
   4. RiskManager.validate_new_trade() final.
- Sizing dinámico: el tamaño base es el máximo del RiskManager, recortado
  por (confianza × |edge|). Más confianza y más edge → más cerca del máximo.
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


# Confidence mínima para abrir trade. Es independiente del threshold del LLM
# (que ya lo aplica el SENTIMENT_ANALYZER). Esta es una segunda línea de defensa.
MIN_CONFIDENCE_TO_OPEN = 60

# Edge mínimo absoluto para abrir trade.
MIN_EDGE_TO_OPEN = 0.05

# Para dimensionar la posición usamos un "factor de confianza" que va de 0 a 1.
# Si el LLM devuelve confidence=100 y |edge|>=0.30, el factor es 1.0 → tamaño máximo.
# Si confidence=60 y |edge|=0.05, el factor es bajo → tamaño cerca del mínimo.
EDGE_REFERENCE_FOR_FULL_SIZE = 0.30


class DecisionEngine:
    """Convierte análisis del LLM en decisiones de trading concretas."""

    def __init__(
        self,
        config: BotConfig,
        risk_manager: RiskManager,
    ) -> None:
        self.config = config
        self.risk_manager = risk_manager
        self._log = logger.bind(module="decision_engine")
        self._log.info(
            "DecisionEngine inicializado: min_confidence={}, min_edge={}",
            MIN_CONFIDENCE_TO_OPEN,
            MIN_EDGE_TO_OPEN,
        )

    # =====================================================
    # Decisión de apertura
    # =====================================================

    def decide(
        self,
        analysis: MarketAnalysis,
        current_balance_eur: float,
        open_positions: list[Position],
        articles: Optional[list[NewsArticle]] = None,
    ) -> TradeDecision:
        """Decide si abrir un trade dado el análisis y el estado actual.

        Args:
            analysis: el output del SENTIMENT_ANALYZER.
            current_balance_eur: balance disponible actual.
            open_positions: lista de posiciones abiertas (de cualquier mercado).
            articles: noticias asociadas (opcional, para require_news_for_entry).
        """
        skip_reasons: list[SkipReason] = []
        rationale_parts: list[str] = []

        # 1) Filtro: el LLM tiene que recomendar comprar
        if analysis.recommendation == TradeRecommendation.INSUFFICIENT_DATA:
            skip_reasons.append(SkipReason.LLM_INSUFFICIENT_DATA)
            rationale_parts.append("El LLM no tiene datos suficientes")
            return self._no_trade(analysis, skip_reasons, rationale_parts)

        if analysis.recommendation == TradeRecommendation.ESPERAR:
            skip_reasons.append(SkipReason.LLM_RECOMMENDS_WAIT)
            rationale_parts.append("El LLM recomienda esperar")
            return self._no_trade(analysis, skip_reasons, rationale_parts)

        # 2) Filtro: confianza y edge mínimos (segunda línea de defensa).
        #    Si el análisis es low_info, exigimos un threshold más alto.
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
                f"Confianza {analysis.confidence} < umbral {effective_min_confidence} "
                f"(modo {mode})"
            )
            return self._no_trade(analysis, skip_reasons, rationale_parts)

        if abs(analysis.edge) < MIN_EDGE_TO_OPEN:
            skip_reasons.append(SkipReason.LLM_RECOMMENDS_WAIT)
            rationale_parts.append(
                f"|Edge| {abs(analysis.edge):.3f} < umbral {MIN_EDGE_TO_OPEN}"
            )
            return self._no_trade(analysis, skip_reasons, rationale_parts)

        # 3) Filtro: requerir noticias si el config lo exige.
        #    En modo low_info este filtro se relaja (basta con 1 noticia).
        if self.config.decision.require_news_for_entry and not analysis.is_low_info:
            if not articles or len(articles) == 0:
                skip_reasons.append(SkipReason.REQUIRE_NEWS_BUT_NONE)
                rationale_parts.append(
                    "Config exige noticias para abrir, pero no se aportaron"
                )
                return self._no_trade(analysis, skip_reasons, rationale_parts)

        # 4) Determinar lado del trade
        side, token_id, entry_price = self._resolve_side_and_token(analysis)

        # 5) Anti-duplicación: ya hay posición abierta en este mercado?
        for pos in open_positions:
            if pos.token_id != token_id and self._same_market(pos, analysis):
                # Misma pregunta pero lado opuesto → posición contradictoria
                skip_reasons.append(SkipReason.OPPOSITE_OPEN_POSITION)
                rationale_parts.append(
                    f"Ya hay posición abierta en el lado opuesto del mercado "
                    f"({pos.trade_id[:8]})"
                )
                return self._no_trade(analysis, skip_reasons, rationale_parts)
            if pos.token_id == token_id:
                skip_reasons.append(SkipReason.DUPLICATE_OPEN_POSITION)
                rationale_parts.append(
                    f"Ya hay posición abierta en este token "
                    f"({pos.trade_id[:8]})"
                )
                return self._no_trade(analysis, skip_reasons, rationale_parts)

        # 6) Sizing dinámico (con reducción si es low_info)
        proposed_size_eur = self._calculate_position_size(
            current_balance_eur=current_balance_eur,
            confidence=analysis.confidence,
            edge=analysis.edge,
            is_low_info=analysis.is_low_info,
        )

        # 7) Validación final del RiskManager
        risk_check = self.risk_manager.validate_new_trade(
            proposed_size_eur=proposed_size_eur,
            current_balance_eur=current_balance_eur,
            open_positions_count=len(open_positions),
            entry_price=entry_price,
        )
        if not risk_check.approved:
            skip_reasons.append(SkipReason.RISK_MANAGER_REJECTED)
            rationale_parts.append(
                f"RiskManager rechazó: {[r.value for r in risk_check.rejection_reasons]}"
            )
            return self._no_trade(analysis, skip_reasons, rationale_parts)

        # Aplicar el ajuste si lo hubo
        final_size = (
            risk_check.adjusted_size_eur
            if risk_check.adjusted_size_eur is not None
            else proposed_size_eur
        )

        # Comprobar que tras el ajuste sigue por encima del mínimo
        if final_size < self.risk_manager.risk.min_trade_size_eur:
            skip_reasons.append(SkipReason.SIZE_BELOW_MIN_AFTER_SIZING)
            rationale_parts.append(
                f"Tamaño tras ajustes ({final_size:.2f}€) < mínimo "
                f"({self.risk_manager.risk.min_trade_size_eur}€)"
            )
            return self._no_trade(analysis, skip_reasons, rationale_parts)

        # 8) ¡Aprobado! Calcular niveles
        sl = self.risk_manager.calculate_stop_loss_price(entry_price)
        tp = self.risk_manager.calculate_take_profit_price(entry_price)

        rationale = (
            f"LLM={analysis.recommendation.value}, "
            f"edge={analysis.edge:+.3f}, "
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
    # Reevaluación de posiciones abiertas
    # =====================================================

    def evaluate_open_position(
        self,
        position: Position,
        current_price: float,
        new_analysis: Optional[MarketAnalysis] = None,
    ) -> CloseDecision:
        """Decide si cerrar una posición abierta.

        Combina:
        1. Stop loss / take profit por precio (delegado al RiskManager).
        2. Reversal por nueva noticia: si el LLM ahora recomienda lo contrario
           con confianza > 70%, cierra la posición.
        """
        # 1) Chequeo del RiskManager
        rm_decision = self.risk_manager.should_close_position(position, current_price)
        if rm_decision.should_close:
            return rm_decision

        # 2) Reversal por noticia
        if new_analysis is None:
            return rm_decision  # No hay nueva info, mantener

        if new_analysis.confidence < 70:
            return rm_decision

        is_long_yes = position.side == TradeSide.BUY_YES
        contradicts = (
            (is_long_yes and new_analysis.recommendation == TradeRecommendation.COMPRAR_NO)
            or (not is_long_yes and new_analysis.recommendation == TradeRecommendation.COMPRAR_YES)
        )
        if contradicts:
            from src.models import CloseReason

            self._log.info(
                "CIERRE por reversal de noticias | trade={} new_rec={} confidence={}",
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
                    f"Reversal: nuevo análisis recomienda "
                    f"{new_analysis.recommendation.value} con confianza "
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
        """Devuelve (side, token_id, entry_price) según la recomendación."""
        if analysis.recommendation == TradeRecommendation.COMPRAR_YES:
            return (
                TradeSide.BUY_YES,
                analysis.yes_token_id,
                analysis.current_yes_price,
            )
        if analysis.recommendation == TradeRecommendation.COMPRAR_NO:
            return (
                TradeSide.BUY_NO,
                analysis.no_token_id,
                analysis.current_no_price,
            )
        # Defensivo: no debería llegar aquí porque el chequeo previo lo descarta
        raise ValueError(
            f"_resolve_side_and_token llamado con recomendación inválida: "
            f"{analysis.recommendation}"
        )

    def _calculate_position_size(
        self,
        current_balance_eur: float,
        confidence: int,
        edge: float,
        is_low_info: bool = False,
    ) -> float:
        """Tamaño dinámico: factor (confianza × edge) sobre el máximo permitido.

        Si is_low_info=True, aplica el multiplicador de config.decision para
        recortar el tamaño aún más (típicamente 50%).

        Garantiza siempre como mínimo el tamaño mínimo de trade del RiskManager.
        Como máximo, el tamaño máximo permitido por posición.
        """
        max_size = self.risk_manager.calculate_max_position_size(current_balance_eur)
        min_size = self.risk_manager.risk.min_trade_size_eur

        # Factor de confianza: 0 cuando confidence=0, 1 cuando confidence=100
        confidence_factor = confidence / 100.0
        # Factor de edge: 0 cuando edge=0, 1 cuando |edge| >= EDGE_REFERENCE
        edge_factor = min(1.0, abs(edge) / EDGE_REFERENCE_FOR_FULL_SIZE)
        # Combinación: producto. Si cualquiera es bajo, el tamaño cae mucho.
        sizing_factor = confidence_factor * edge_factor

        # Reducción adicional por modo low_info
        if is_low_info:
            low_info_mult = self.config.decision.low_info_size_multiplier
            sizing_factor *= low_info_mult
            self._log.debug(
                "Sizing low_info: aplicando multiplier {:.2f}", low_info_mult
            )

        proposed = max_size * sizing_factor
        return max(min_size, proposed)

    @staticmethod
    def _same_market(position: Position, analysis: MarketAnalysis) -> bool:
        """Heurística: comparamos por market_question. El market_id estable de
        Polymarket podríamos almacenarlo en Position en el futuro para mayor
        robustez."""
        return position.market_question == analysis.market_question

    def _no_trade(
        self,
        analysis: MarketAnalysis,
        skip_reasons: list[SkipReason],
        rationale_parts: list[str],
    ) -> TradeDecision:
        rationale = " | ".join(rationale_parts) if rationale_parts else "(sin razón)"
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
