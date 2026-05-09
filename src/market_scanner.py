"""
Market Scanner — descubre y filtra mercados operables en Polymarket.

Responsabilidades:
1. Pedir mercados crudos a la Gamma API (vía GammaApiClient).
2. Parsearlos a `MarketSnapshot` validados, descartando los mal formados.
3. Aplicar los filtros del config (volumen, spread, tiempo a cierre, categoría).
4. Cachear el resultado durante un TTL corto para no machacar la API.
5. Exponer una búsqueda por palabras clave para que el DECISION_ENGINE pueda
   cruzar mercados con noticias.

El scanner NO toma decisiones de trading. Solo identifica el universo de
mercados sobre los que vale la pena pensar.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from src.config_loader import BotConfig
from src.gamma_client import GammaApiClient, GammaApiError
from src.models import MarketSnapshot

# TTL del caché de escaneo (segundos). Independiente del intervalo de polling
# del config (que controla cada cuánto el orquestador decide escanear).
DEFAULT_SCAN_CACHE_TTL = 60.0


class MarketScanner:
    """Descubre mercados de Polymarket que cumplen los filtros configurados."""

    def __init__(
        self,
        config: BotConfig,
        client: Optional[GammaApiClient] = None,
        cache_ttl_seconds: float = DEFAULT_SCAN_CACHE_TTL,
    ) -> None:
        self.config = config
        self.filters = config.market_filters
        self.client = client if client is not None else GammaApiClient(config)
        self._cache_ttl = cache_ttl_seconds

        self._cache: list[MarketSnapshot] = []
        self._cache_ts: float = 0.0
        self._sports_cache: list[MarketSnapshot] = []
        self._sports_cache_ts: float = 0.0

        self._log = logger.bind(module="market_scanner")

    # =====================================================
    # Parser
    # =====================================================

    def parse_market(self, raw: dict[str, Any]) -> Optional[MarketSnapshot]:
        """Convierte un dict crudo de la Gamma API en un MarketSnapshot.

        Devuelve None si el mercado está mal formado (campos faltantes,
        formatos inesperados). Nunca lanza excepción.
        """
        try:
            market_id = str(raw["id"])
            question = raw.get("question") or ""
            if not question:
                return None

            # Tokens CLOB. Pueden venir como lista o como string JSON.
            yes_token, no_token = self._parse_token_pair(raw.get("clobTokenIds"))
            if yes_token is None or no_token is None:
                return None

            # Precios YES/NO. Mismo problema: lista o string JSON.
            yes_price, no_price = self._parse_price_pair(raw.get("outcomePrices"))
            if yes_price is None or no_price is None:
                return None

            # Validación: precios deben estar en (0, 1) — Polymarket no permite 0/1
            # exactos en mercados activos. Si los vemos, el mercado probablemente
            # ya está resuelto.
            if not (0 < yes_price < 1) or not (0 < no_price < 1):
                return None

            # Mejor bid/ask si están disponibles
            best_bid = self._safe_float(raw.get("bestBid"))
            best_ask = self._safe_float(raw.get("bestAsk"))
            if best_bid is not None and not (0 < best_bid < 1):
                best_bid = None
            if best_ask is not None and not (0 < best_ask < 1):
                best_ask = None

            # Spread: si hay bid/ask reales, ese es el spread. Si no, lo
            # aproximamos como (1 - yes - no), que mide la inconsistencia
            # entre los lados.
            if best_bid is not None and best_ask is not None and best_ask >= best_bid:
                spread = best_ask - best_bid
            else:
                spread = max(0.0, 1.0 - yes_price - no_price)

            # Volumen y liquidez
            volume_24h = self._safe_float(raw.get("volume24hr")) or 0.0
            volume_total = self._safe_float(raw.get("volumeNum")) or 0.0
            liquidity = self._safe_float(raw.get("liquidityNum")) or 0.0

            # Fecha de cierre
            end_date = self._parse_iso_datetime(raw.get("endDate"))

            return MarketSnapshot(
                market_id=market_id,
                slug=raw.get("slug") or "",
                question=question,
                description=raw.get("description") or "",
                category=raw.get("category") or "",
                end_date=end_date,
                yes_token_id=str(yes_token),
                no_token_id=str(no_token),
                yes_price=yes_price,
                no_price=no_price,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                volume_24h_usd=volume_24h,
                volume_total_usd=volume_total,
                liquidity_usd=liquidity,
                is_active=bool(raw.get("active", False)),
                is_closed=bool(raw.get("closed", False)),
                snapshot_timestamp=datetime.now(timezone.utc),
            )
        except (KeyError, ValueError, TypeError) as exc:
            self._log.debug(
                "Mercado mal formado, descartado: {} | raw_id={}",
                exc,
                raw.get("id"),
            )
            return None

    # =====================================================
    # Filtros
    # =====================================================

    def is_tradeable(self, market: MarketSnapshot) -> tuple[bool, list[str]]:
        """Decide si un mercado pasa los filtros. Devuelve (ok, motivos_rechazo)."""
        reasons: list[str] = []

        if not market.is_active or market.is_closed:
            reasons.append("inactive_or_closed")

        if market.volume_24h_usd < self.filters.min_volume_24h_usd:
            reasons.append("low_volume")

        if market.spread > self.filters.max_spread_cents:
            reasons.append("wide_spread")

        # Tiempo a cierre
        ttc = market.time_to_close_hours
        if ttc is not None:
            if ttc < self.filters.min_time_to_close_hours:
                reasons.append("closing_too_soon")
            elif ttc > self.filters.max_time_to_close_days * 24:
                reasons.append("closing_too_far")

        # Categorías excluidas
        if market.category and self.filters.exclude_categories:
            excluded = [c.lower() for c in self.filters.exclude_categories]
            if market.category.lower() in excluded:
                reasons.append("excluded_category")

        return len(reasons) == 0, reasons

    # =====================================================
    # Escaneo principal (con caché)
    # =====================================================

    def scan(self, force_refresh: bool = False) -> list[MarketSnapshot]:
        """Escanea, parsea y filtra. Devuelve mercados operables.

        Usa caché interno con TTL. Pasa `force_refresh=True` para ignorarlo.
        Si la API falla, devuelve la última lista cacheada (o vacía si no hay).
        """
        now = time.time()
        if (
            not force_refresh
            and self._cache
            and (now - self._cache_ts) < self._cache_ttl
        ):
            return self._cache

        try:
            raw_markets = self.client.fetch_all_active_markets()
        except GammaApiError as exc:
            self._log.error("Escaneo abortado por error de API: {}", exc)
            return self._cache  # Devolvemos lo último cacheado, aunque esté viejo

        snapshots: list[MarketSnapshot] = []
        for raw in raw_markets:
            snap = self.parse_market(raw)
            if snap is not None:
                snapshots.append(snap)

        tradeable: list[MarketSnapshot] = []
        rejection_counter: dict[str, int] = {}
        for snap in snapshots:
            ok, reasons = self.is_tradeable(snap)
            if ok:
                tradeable.append(snap)
            else:
                for r in reasons:
                    rejection_counter[r] = rejection_counter.get(r, 0) + 1

        self._log.info(
            "Escaneo: {} crudos → {} parseados → {} operables. Rechazos: {}",
            len(raw_markets),
            len(snapshots),
            len(tradeable),
            rejection_counter or "ninguno",
        )

        self._cache = tradeable
        self._cache_ts = now
        return tradeable

    # =====================================================
    # Selección inteligente de candidatos
    # =====================================================

    def select_candidates(
        self,
        markets: list[MarketSnapshot],
        max_candidates: int = 10,
        min_uncertainty: float = 0.10,
    ) -> list[MarketSnapshot]:
        """Selecciona mercados que MERECEN ser analizados.

        Filtra:
        - Mercados con precios pegados a 0 o 1 (resueltos de facto, sin edge).
        - Mercados con incertidumbre menor a `min_uncertainty` (ej: <10% gap).

        Ordena por:
        - Incertidumbre (cercanía a 0.5) DESC: cuanto más incierto, más
          probable que las noticias muevan el precio.
        - Dentro del mismo nivel, volumen 24h DESC (más volumen = mejor
          ejecución posible).

        Esto evita gastar análisis en mercados como "Cubs vs Padres a YES=1.0".
        """
        candidates = []
        for m in markets:
            # Distancia al 0.5 — un mercado a 0.5 tiene incertidumbre máxima
            uncertainty = 0.5 - abs(m.yes_price - 0.5)
            if uncertainty < min_uncertainty:
                continue
            candidates.append((uncertainty, m.volume_24h_usd, m))

        # Ordenar por uncertainty desc, volumen desc
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        result = [m for (_, _, m) in candidates[:max_candidates]]

        self._log.info(
            "select_candidates: {} mercados → {} candidatos analizables "
            "(min_uncertainty={:.2f})",
            len(markets),
            len(result),
            min_uncertainty,
        )
        return result

    # =====================================================
    # Escaneo deportivo (módulo secundario sports_in_play)
    # =====================================================

    def scan_sports_candidates(
        self,
        force_refresh: bool = False,
    ) -> list[MarketSnapshot]:
        """Escanea mercados de partidos deportivos en directo para el módulo sports_in_play.

        Completamente independiente del scan() principal — bypasea exclude_categories
        y aplica sus propios filtros específicos de deportes.
        Tiene su propio caché con el mismo TTL que el scan principal.
        """
        cfg = self.config.sports_in_play
        now = time.time()
        if (
            not force_refresh
            and self._sports_cache
            and (now - self._sports_cache_ts) < self._cache_ttl
        ):
            return self._sports_cache

        try:
            raw_markets = self.client.fetch_all_active_markets(max_markets=500)
        except GammaApiError as exc:
            self._log.warning("Sports scan abortado: {}", exc)
            return self._sports_cache

        cats_lower = {c.lower() for c in cfg.scan_categories}
        candidates: list[MarketSnapshot] = []

        for raw in raw_markets:
            snap = self.parse_market(raw)
            if snap is None:
                continue

            # Solo categorías deportivas
            if snap.category.lower() not in cats_lower:
                continue

            # YES dentro del rango para apostar NO al underdog
            if not (cfg.min_yes_price <= snap.yes_price <= cfg.max_yes_price):
                continue

            # Partido cercano al cierre (en curso o inminente)
            ttc = snap.time_to_close_hours
            if ttc is None or ttc > cfg.max_time_to_close_hours:
                continue

            # Liquidez mínima
            if snap.volume_24h_usd < cfg.min_volume_24h_usd:
                continue

            candidates.append(snap)

        # Ordenar por YES price desc (favorito más dominante → mejor asimetría en NO)
        candidates.sort(key=lambda m: m.yes_price, reverse=True)

        self._log.info(
            "Sports scan: {} candidatos in-play (YES={:.0%}–{:.0%}, cierre<{:.0f}h)",
            len(candidates),
            cfg.min_yes_price,
            cfg.max_yes_price,
            cfg.max_time_to_close_hours,
        )
        self._sports_cache = candidates
        self._sports_cache_ts = now
        return candidates

    # =====================================================
    # Búsqueda por keywords (cruce con noticias)
    # =====================================================

    def search_by_keywords(
        self,
        markets: list[MarketSnapshot],
        keywords: list[str],
        match_in_description: bool = True,
    ) -> list[MarketSnapshot]:
        """Filtra mercados que mencionen alguno de los keywords.

        Búsqueda case-insensitive sobre la pregunta y opcionalmente la
        descripción. Para cruces más sofisticados (entidades nombradas,
        embeddings), el DECISION_ENGINE decidirá qué hacer en módulos
        posteriores; aquí basta con filtrado por substring.
        """
        if not keywords:
            return []
        kws_lower = [k.lower().strip() for k in keywords if k.strip()]
        if not kws_lower:
            return []

        results: list[MarketSnapshot] = []
        for m in markets:
            haystack = m.question.lower()
            if match_in_description and m.description:
                haystack += " " + m.description.lower()
            if any(kw in haystack for kw in kws_lower):
                results.append(m)
        return results

    # =====================================================
    # Re-ranking con sesgo de categoría (cobertura mediática)
    # =====================================================

    def rank_for_analysis(
        self,
        markets: list[MarketSnapshot],
        category_boost: dict[str, float],
        top_n: int = 15,
    ) -> list[MarketSnapshot]:
        """Reordena los mercados aplicando un boost de categoría.

        Los mercados con categorías que tienen alta cobertura mediática
        (Politics, Crypto, Geopolitics) se priorizan sobre Esports/Sports
        regionales que casi no salen en GDELT.

        Score = log(volume_24h) × boost_categoria

        Devuelve los top_n para mandar al sentiment analyzer. Si el universo
        de mercados es menor que top_n, devuelve todos.
        """
        import math

        if not markets:
            return []

        def score(m: MarketSnapshot) -> float:
            base = math.log10(max(1.0, m.volume_24h_usd))
            boost = 1.0
            if m.category and category_boost:
                # Match case-insensitive
                cat_lower = m.category.lower()
                for cat_key, mult in category_boost.items():
                    if cat_key.lower() == cat_lower:
                        boost = mult
                        break
            return base * boost

        ranked = sorted(markets, key=score, reverse=True)
        return ranked[:top_n]

    # =====================================================
    # Helpers privados
    # =====================================================

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Convierte a float devolviendo None si no es posible."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_token_pair(value: Any) -> tuple[Optional[str], Optional[str]]:
        """Devuelve (yes_token_id, no_token_id) o (None, None)."""
        items = MarketScanner._parse_list_or_json(value)
        if items is None or len(items) < 2:
            return None, None
        yes, no = items[0], items[1]
        if not yes or not no:
            return None, None
        return str(yes), str(no)

    @staticmethod
    def _parse_price_pair(value: Any) -> tuple[Optional[float], Optional[float]]:
        """Devuelve (yes_price, no_price) o (None, None)."""
        items = MarketScanner._parse_list_or_json(value)
        if items is None or len(items) < 2:
            return None, None
        try:
            return float(items[0]), float(items[1])
        except (TypeError, ValueError):
            return None, None

    @staticmethod
    def _parse_list_or_json(value: Any) -> Optional[list[Any]]:
        """La Gamma API a veces devuelve listas como string JSON. Normalizamos."""
        if value is None:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    @staticmethod
    def _parse_iso_datetime(value: Any) -> Optional[datetime]:
        """Parsea una cadena ISO 8601 a datetime con tz UTC. Tolerante."""
        if not value or not isinstance(value, str):
            return None
        try:
            # Soporta 'Z' al final
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None
