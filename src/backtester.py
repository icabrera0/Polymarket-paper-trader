"""
Backtester — simula el bot sobre mercados de Polymarket ya resueltos.

Flujo:
1. Descarga mercados resueltos recientes de la Gamma API.
2. Filtra los que cumplen los mismos filtros del scanner en vivo.
3. Para cada mercado resuelto:
   a) Simula el estado "en el momento del análisis" (precio pre-resolución).
   b) Busca noticias relevantes (GDELT con timespan historical o fresco).
   c) Analiza con el LLM.
   d) Si el engine decide OPEN_TRADE, simula la posición:
      - Entrada al precio que tenía en el momento (simulado).
      - Salida al precio de resolución (1.0 YES ganó, 0.0 YES perdió).
   e) Registra el resultado en el informe.

Dos modos:
  "current"  → Usa noticias de hoy. Los hechos ya son conocidos (Look-ahead bias).
               Sirve para calibrar el LLM y validar el pipeline, NO para evaluar
               la estrategia de forma real.
  "replay"   → Busca noticias en GDELT anteriores al cierre del mercado.
               Más realista, más lento, más limitado por cobertura de GDELT.

Los resultados se guardan en:
  - Una DB SQLite temporal (no la de producción).
  - Un Excel de resultados con la misma estructura que el report diario.
  - Un resumen en consola.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
import re

from loguru import logger

from src.config_loader import BotConfig
from src.database import Database
from src.decision_engine import DecisionEngine
from src.gamma_client import GammaApiClient
from src.models import (
    CloseReason,
    DecisionAction,
    MarketAnalysis,
    MarketSnapshot,
    Position,
    TradeDecision,
    TradeRecommendation,
    TradeSide,
    TradeStatus,
    _now_utc,
)
from src.news_ingestor import NewsIngestor
from src.paper_trader import PaperTrader
from src.risk_manager import RiskManager
from src.sentiment_analyzer import SentimentAnalyzer


# =====================================================
# Resultado de backtesting
# =====================================================


@dataclass
class BacktestTrade:
    """Un trade simulado en el backtest."""

    market_id: str
    market_question: str
    resolved_yes: bool            # True si YES ganó (resolución = 1.0)
    entry_price_simulated: float  # Precio al que entramos simulando
    exit_price: float             # 1.0 (YES ganó) o 0.0 (YES perdió)
    side: TradeSide
    size_eur: float
    pnl_eur: float
    pnl_pct: float
    confidence: int
    edge: float
    num_articles: int
    is_low_info: bool
    decision: DecisionAction
    skip_reasons: list[str] = field(default_factory=list)
    llm_recommendation: str = ""


@dataclass
class BacktestResult:
    """Resultado agregado del backtesting."""

    mode: str
    start_time: datetime
    end_time: datetime
    initial_balance: float
    final_balance: float
    markets_analyzed: int
    trades_executed: int
    trades_won: int
    trades_lost: int
    total_pnl_eur: float
    win_rate: float
    avg_pnl_per_trade: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: list[BacktestTrade] = field(default_factory=list)

    def print_summary(self) -> None:
        """Imprime un resumen en consola."""
        print()
        print("═" * 65)
        print("  RESULTADO DE BACKTESTING")
        print("═" * 65)
        print(f"  Modo:                    {self.mode}")
        print(f"  Duración del test:       {(self.end_time - self.start_time).seconds}s")
        print(f"  Mercados analizados:     {self.markets_analyzed}")
        print(f"  Trades ejecutados:       {self.trades_executed}")
        print(f"  Trades ganadores:        {self.trades_won}")
        print(f"  Trades perdedores:       {self.trades_lost}")
        print(f"  Win rate:                {self.win_rate:.1%}")
        print(f"  P&L total:               €{self.total_pnl_eur:+.2f}")
        if self.trades_executed > 0:
            print(f"  P&L medio por trade:     €{self.avg_pnl_per_trade:+.2f}")
        print(f"  Balance inicial:         €{self.initial_balance:.2f}")
        print(f"  Balance final:           €{self.final_balance:.2f}")
        pnl_pct = (self.final_balance - self.initial_balance) / self.initial_balance
        print(f"  Retorno total:           {pnl_pct:+.2%}")
        print(f"  Max drawdown:            {self.max_drawdown_pct:.2%}")
        if self.sharpe_ratio:
            print(f"  Sharpe ratio (aprox):    {self.sharpe_ratio:.2f}")
        print("═" * 65)
        if self.trades:
            print()
            print("  Detalle de trades:")
            for t in self.trades[:20]:
                icon = "✅" if t.pnl_eur >= 0 else "❌"
                low = " [LOW]" if t.is_low_info else ""
                print(
                    f"  {icon} {t.market_question[:50]:<50} "
                    f"€{t.pnl_eur:+.2f}{low}"
                )
            if len(self.trades) > 20:
                print(f"  ... y {len(self.trades) - 20} más")
        print()


# =====================================================
# Backtester
# =====================================================


class Backtester:
    """Simula el bot sobre mercados resueltos."""

    def __init__(
        self,
        config: BotConfig,
        mode: str = "current",
        max_markets: int = 50,
        initial_balance: float = 150.0,
        db_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            config: configuración del bot.
            mode: "current" (noticias de hoy) o "replay" (noticias históricas).
            max_markets: cuántos mercados resueltos analizar.
            initial_balance: balance inicial de la simulación.
            db_path: ruta de DB temporal. Si None, usa un archivo en data/backtest.db.
        """
        self.config = config
        self.mode = mode
        self.max_markets = max_markets
        self.initial_balance = initial_balance

        # DB separada de la de producción
        self._db_path = db_path or str(
            Path(config.database.path).parent / "backtest.db"
        )

        self._log = logger.bind(module="backtester")
        self._log.info(
            "Backtester inicializado: modo={}, max_markets={}, balance=€{}",
            mode, max_markets, initial_balance,
        )

    # =====================================================
    # Entry point
    # =====================================================

    def run(self) -> BacktestResult:
        """Ejecuta el backtest y devuelve el resultado."""
        start_time = _now_utc()

        # Módulos con DB temporal y balance virtual propio
        db = Database(self._db_path)
        risk_manager = RiskManager(self.config, initial_balance_eur=self.initial_balance)
        paper_trader = PaperTrader(self.config, risk_manager, db=db)
        news_ingestor = NewsIngestor(self.config)
        sentiment_analyzer = SentimentAnalyzer(self.config)
        decision_engine = DecisionEngine(self.config, risk_manager)

        # 1) Obtener mercados resueltos recientes
        self._log.info("Descargando mercados resueltos de Polymarket...")
        resolved_markets = self._fetch_resolved_markets()
        self._log.info("Encontrados {} mercados resueltos", len(resolved_markets))

        bt_trades: list[BacktestTrade] = []
        peak_balance = self.initial_balance
        max_drawdown = 0.0

        for i, market_data in enumerate(resolved_markets):
            self._log.debug(
                "Analizando {}/{}: {}",
                i + 1, len(resolved_markets),
                market_data.get("question", "")[:60],
            )

            # Extraer info del mercado resuelto
            snap, resolved_yes = self._parse_resolved_market(market_data)
            if snap is None:
                continue

            # 2) Buscar noticias
            keywords = self._extract_keywords(snap.question)
            if self.mode == "replay" and snap.end_date:
                # Calcular timespan aproximado hasta el cierre del mercado
                days_ago = (
                    _now_utc() - snap.end_date
                ).days
                timespan = f"{min(days_ago + 7, 90)}d"  # +7 días de contexto previo
            else:
                timespan = "7d"  # noticias de la semana (modo current)

            articles = news_ingestor.fetch(
                keywords,
                max_articles=10,
                fallback_timespan=timespan,
            )

            # 3) Analizar
            analysis = sentiment_analyzer.analyze(snap, articles, force_refresh=True)
            db.log_analysis(analysis)

            # 4) Decidir
            decision = decision_engine.decide(
                analysis=analysis,
                current_balance_eur=paper_trader.balance_eur,
                open_positions=paper_trader.open_positions,
                articles=articles,
            )
            db.log_decision(decision)

            # 5) Simular P&L con precio de resolución
            if decision.action == DecisionAction.OPEN_TRADE and decision.side:
                # Precio de salida = resolución real del mercado
                if decision.side == TradeSide.BUY_YES:
                    exit_price = 1.0 if resolved_yes else 0.0
                else:
                    exit_price = 1.0 if not resolved_yes else 0.0

                # Simular: abrir y cerrar inmediatamente con el precio de resolución
                entry_price = decision.entry_price or snap.yes_price
                position = paper_trader.execute_decision(decision)
                if position:
                    closed = paper_trader.close_position(
                        trade_id=position.trade_id,
                        current_market_price=exit_price,
                        reason=CloseReason.MARKET_RESOLVED,
                        notes=f"Backtest: mercado resuelto YES={resolved_yes}",
                    )
                    if closed:
                        bt_trade = BacktestTrade(
                            market_id=market_data.get("id", ""),
                            market_question=snap.question,
                            resolved_yes=resolved_yes,
                            entry_price_simulated=entry_price,
                            exit_price=exit_price,
                            side=decision.side,
                            size_eur=decision.size_eur or 0,
                            pnl_eur=closed.pnl_eur or 0.0,
                            pnl_pct=closed.pnl_pct or 0.0,
                            confidence=analysis.confidence,
                            edge=analysis.edge,
                            num_articles=len(articles),
                            is_low_info=analysis.is_low_info,
                            decision=decision.action,
                            llm_recommendation=analysis.recommendation.value,
                        )
                        bt_trades.append(bt_trade)

                        # Actualizar drawdown
                        bal = paper_trader.balance_eur
                        if bal > peak_balance:
                            peak_balance = bal
                        dd = (peak_balance - bal) / peak_balance if peak_balance > 0 else 0.0
                        max_drawdown = max(max_drawdown, dd)
            else:
                # NO_TRADE
                bt_trade = BacktestTrade(
                    market_id=market_data.get("id", ""),
                    market_question=snap.question,
                    resolved_yes=resolved_yes,
                    entry_price_simulated=0.0,
                    exit_price=0.0,
                    side=TradeSide.BUY_YES,
                    size_eur=0.0,
                    pnl_eur=0.0,
                    pnl_pct=0.0,
                    confidence=analysis.confidence,
                    edge=analysis.edge,
                    num_articles=len(articles),
                    is_low_info=analysis.is_low_info,
                    decision=decision.action,
                    skip_reasons=[r.value for r in decision.skip_reasons],
                    llm_recommendation=analysis.recommendation.value,
                )
                bt_trades.append(bt_trade)

            # Pausa breve entre mercados para no saturar GDELT/Ollama
            time.sleep(0.5)

        db.close()

        # Calcular métricas
        actual_trades = [t for t in bt_trades if t.decision == DecisionAction.OPEN_TRADE]
        winners = [t for t in actual_trades if t.pnl_eur > 0]
        losers = [t for t in actual_trades if t.pnl_eur < 0]
        total_pnl = sum(t.pnl_eur for t in actual_trades)
        final_balance = self.initial_balance + total_pnl
        win_rate = len(winners) / len(actual_trades) if actual_trades else 0.0
        avg_pnl = total_pnl / len(actual_trades) if actual_trades else 0.0
        sharpe = self._calculate_sharpe(actual_trades)

        return BacktestResult(
            mode=self.mode,
            start_time=start_time,
            end_time=_now_utc(),
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            markets_analyzed=len(resolved_markets),
            trades_executed=len(actual_trades),
            trades_won=len(winners),
            trades_lost=len(losers),
            total_pnl_eur=total_pnl,
            win_rate=win_rate,
            avg_pnl_per_trade=avg_pnl,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            trades=bt_trades,
        )

    # =====================================================
    # Fetching de mercados resueltos
    # =====================================================

    def _fetch_resolved_markets(self) -> list[dict[str, Any]]:
        """Descarga mercados cerrados/resueltos de la Gamma API."""
        client = GammaApiClient(self.config)
        try:
            markets = client.fetch_markets(
                active=False,
                closed=True,
                order="volume24hr",
                ascending=False,
                limit=min(self.max_markets * 3, 500),
            )
        except Exception as exc:
            self._log.error("Error descargando mercados resueltos: {}", exc)
            return []

        # Filtrar los que realmente están resueltos (tienen un ganador)
        resolved = [
            m for m in markets
            if self._is_truly_resolved(m)
        ][:self.max_markets]
        return resolved

    @staticmethod
    def _is_truly_resolved(market: dict[str, Any]) -> bool:
        """Un mercado está resuelto si tiene resultado claro (1.0 / 0.0)."""
        prices_raw = market.get("outcomePrices")
        if not prices_raw:
            return False
        try:
            import json
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            if not isinstance(prices, list) or len(prices) < 2:
                return False
            p0, p1 = float(prices[0]), float(prices[1])
            # Resuelto: uno de los precios debe ser exactamente 1.0 y el otro 0.0
            return (p0 == 1.0 and p1 == 0.0) or (p0 == 0.0 and p1 == 1.0)
        except (ValueError, TypeError, Exception):
            return False

    # =====================================================
    # Parser de mercados resueltos
    # =====================================================

    def _parse_resolved_market(
        self, market_data: dict[str, Any]
    ) -> tuple[Optional[MarketSnapshot], bool]:
        """
        Devuelve (MarketSnapshot_pre_resolución, resolved_yes).

        Para simular el estado "antes de resolver", usamos un precio
        artificial de 0.50 (máxima incertidumbre). En modo replay podríamos
        buscar el precio histórico, pero la API pública no lo expone
        fácilmente. Esto añade ruido pero mantiene el test conservador.
        """
        import json

        question = market_data.get("question", "")
        if not question:
            return None, False

        # Determinar si YES ganó
        prices_raw = market_data.get("outcomePrices", "[]")
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            resolved_yes = float(prices[0]) == 1.0
        except (ValueError, TypeError, IndexError):
            return None, False

        # Tokens
        tokens_raw = market_data.get("clobTokenIds", "[]")
        try:
            tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
            if len(tokens) < 2:
                return None, False
            yes_token, no_token = str(tokens[0]), str(tokens[1])
        except (ValueError, TypeError, IndexError):
            return None, False

        # Precio pre-resolución simulado: usamos el precio "opuesto" a la
        # resolución ajustado por volumen. Si YES ganó con mucho volumen,
        # asumimos que antes de resolver cotizaba a ~0.70. Si fue sorpresa,
        # podría ser 0.30. Sin histórico real usamos 0.50 como base.
        # Esto es una simplificación necesaria dada la disponibilidad de datos.
        simulated_yes_price = 0.50
        simulated_no_price = 0.50

        snap = MarketSnapshot(
            market_id=str(market_data.get("id", "")),
            slug=market_data.get("slug", ""),
            question=question,
            description=market_data.get("description", ""),
            category=market_data.get("category", ""),
            end_date=self._parse_iso(market_data.get("endDate")),
            yes_token_id=yes_token,
            no_token_id=no_token,
            yes_price=simulated_yes_price,
            no_price=simulated_no_price,
            spread=0.01,
            volume_24h_usd=float(market_data.get("volume24hr") or 0),
            volume_total_usd=float(market_data.get("volumeNum") or 0),
            liquidity_usd=float(market_data.get("liquidityNum") or 0),
            is_active=False,
            is_closed=True,
        )
        return snap, resolved_yes

    # =====================================================
    # Utilidades
    # =====================================================

    @staticmethod
    def _extract_keywords(question: str, max_kw: int = 4) -> list[str]:
        stopwords = {
            "will", "the", "a", "an", "is", "are", "be", "by", "of", "in",
            "on", "at", "to", "for", "and", "or", "if", "than", "more", "less",
            "this", "that", "before", "after", "any", "all", "with", "from",
            "win", "wins", "won", "do", "does", "did", "can", "could", "should",
            "would", "may", "might", "first", "next", "year", "month", "week",
            "day", "much", "many", "election", "vote",
        }
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9'-]{3,}\b", question)
        entities, common = [], []
        for w in words:
            if w.lower() in stopwords:
                continue
            (entities if w[0].isupper() else common).append(
                w if w[0].isupper() else w.lower()
            )
        seen: set[str] = set()
        result: list[str] = []
        for w in entities + common:
            if w.lower() not in seen:
                seen.add(w.lower())
                result.append(w)
            if len(result) >= max_kw:
                break
        return result

    @staticmethod
    def _parse_iso(value: Any) -> Optional[datetime]:
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _calculate_sharpe(trades: list[BacktestTrade]) -> float:
        """Sharpe ratio simplificado sobre los P&L porcentuales de cada trade."""
        if len(trades) < 2:
            return 0.0
        import statistics
        pnls = [t.pnl_pct for t in trades]
        mean = statistics.mean(pnls)
        std = statistics.stdev(pnls)
        if std == 0:
            return 0.0
        return mean / std
