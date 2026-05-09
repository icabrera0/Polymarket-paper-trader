"""
News Ingestor — orquesta las fuentes de noticias, deduplica y prioriza.

Flujo:
1. Recibe una lista de keywords (típicamente extraídos de mercados activos).
2. Pide artículos a NewsAPI y/o GDELT en paralelo (si están habilitados).
3. Calcula un score heurístico de impacto a cada artículo (0-100).
4. Deduplica:
   a) por URL exacta (mismo artículo recogido por dos fuentes)
   b) por similitud difusa de títulos (rapidfuzz, umbral configurable)
5. Devuelve la lista ordenada por score descendente.

El score que devuelve es una HEURÍSTICA. El módulo SENTIMENT_ANALYZER
recalculará un score real con el LLM. Esta heurística sirve para:
- Decidir el orden en que se procesan las noticias por el LLM (limitado por
  rate limits y coste).
- Filtrar ruido obvio antes de gastar tokens del LLM.

Componentes del score:
  Recencia            (max 30) — decae linealmente sobre 24h
  Reputación fuente   (max 30) — Reuters/AP/Bloomberg etc. > otros
  Match de keywords   (max 30) — 10 pts por keyword distinto, hasta 3
  Urgencia del título (max 10) — BREAKING, URGENT, JUST IN, etc.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from loguru import logger
from rapidfuzz import fuzz

from src.config_loader import BotConfig
from src.models import NewsArticle


# Fuentes que consideramos de alta reputación por defecto. Se complementa con
# las definidas en config.news.newsapi.sources.
DEFAULT_HIGH_REPUTATION_SOURCES: tuple[str, ...] = (
    "reuters", "associated press", "ap news", "bloomberg",
    "bbc", "cnn", "wall street journal", "wsj",
    "financial times", "ft.com", "the new york times", "nyt",
    "washington post", "the economist", "nbc news", "cbs news",
    "abc news", "the guardian", "politico", "axios",
)

URGENCY_MARKERS: tuple[str, ...] = (
    "BREAKING", "URGENT", "JUST IN", "ALERT", "EXCLUSIVE", "LIVE",
)


# =====================================================
# Protocolo de cliente (para inyección de dependencias)
# =====================================================


class NewsClientProtocol(Protocol):
    """Protocolo común que cumplen NewsApiClient y GdeltClient."""

    def fetch_articles(
        self, keywords: list[str], **kwargs: Any
    ) -> list[NewsArticle]: ...


# =====================================================
# NewsIngestor
# =====================================================


class NewsIngestor:
    """Orquesta las fuentes de noticias y devuelve un stream priorizado."""

    def __init__(
        self,
        config: BotConfig,
        newsapi_client: Optional[NewsClientProtocol] = None,
        gdelt_client: Optional[NewsClientProtocol] = None,
        telegram_client: Optional[NewsClientProtocol] = None,
        cache_ttl_seconds: Optional[float] = None,
    ) -> None:
        self.config = config
        self.cache_ttl = cache_ttl_seconds or config.news.cache_ttl_seconds
        self.dedup_threshold = config.news.dedup_similarity_threshold

        # Inyección de clientes. Si no se pasan, se crean a partir del config
        # solo si la fuente está habilitada Y tenemos credenciales.
        self.newsapi_client = newsapi_client
        self.gdelt_client = gdelt_client
        self.telegram_client = telegram_client

        if (
            self.newsapi_client is None
            and config.news.newsapi.enabled
            and config.newsapi_key
        ):
            from src.newsapi_client import NewsApiClient
            self.newsapi_client = NewsApiClient(config)
        if self.gdelt_client is None and config.news.gdelt.enabled:
            from src.gdelt_client import GdeltClient
            self.gdelt_client = GdeltClient(config)
        if (
            self.telegram_client is None
            and config.news.telegram.enabled
            and config.telegram_api_id
            and config.telegram_api_hash
        ):
            from src.telegram_client import TelegramClient
            self.telegram_client = TelegramClient(config)

        # Set de fuentes de alta reputación, normalizado en lower.
        configured = [
            s.replace("-", " ").lower()
            for s in (config.news.newsapi.sources or [])
        ]
        self._high_reputation = set(DEFAULT_HIGH_REPUTATION_SOURCES) | set(configured)

        # Caché: keywords (tuple) → (timestamp, articles)
        self._cache: dict[tuple[str, ...], tuple[float, list[NewsArticle]]] = {}

        self._log = logger.bind(module="news_ingestor")
        active = [
            name for name, c in [
                ("NewsAPI", self.newsapi_client),
                ("GDELT", self.gdelt_client),
                ("Telegram", self.telegram_client),
            ] if c is not None
        ]
        self._log.info(
            "NewsIngestor inicializado: fuentes activas={}, dedup={:.2f}",
            active or "ninguna",
            self.dedup_threshold,
        )

    # =====================================================
    # API pública
    # =====================================================

    def fetch(
        self,
        keywords: list[str],
        max_articles: int = 50,
        force_refresh: bool = False,
        fallback_timespan: Optional[str] = None,
    ) -> list[NewsArticle]:
        """Devuelve artículos relevantes a `keywords`, deduplicados y priorizados.

        Si `fallback_timespan` está dado y la búsqueda normal no devuelve nada,
        se reintenta con ese timespan ampliado en GDELT (ej. "7d").
        """
        if not keywords:
            return []

        cache_key = tuple(sorted(k.lower().strip() for k in keywords if k.strip()))
        if not cache_key:
            return []

        # Caché
        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached and (time.time() - cached[0]) < self.cache_ttl:
                self._log.debug("Cache hit para {} keywords", len(cache_key))
                return cached[1][:max_articles]

        # Fetch desde todas las fuentes habilitadas. Los fallos individuales
        # no impiden que las otras fuentes contribuyan.
        all_raw: list[NewsArticle] = []
        clients = [
            ("NewsAPI", self.newsapi_client),
            ("GDELT", self.gdelt_client),
            ("Telegram", self.telegram_client),
        ]
        for name, client in clients:
            if client is None:
                continue
            try:
                all_raw.extend(client.fetch_articles(list(keywords)))
            except Exception as exc:
                self._log.error("{} fetch falló: {}", name, exc)

        # FALLBACK: si no hubo nada y se nos dio timespan extendido, reintentar
        # con GDELT (es el que más se beneficia: lookback más largo = más coverage)
        if not all_raw and fallback_timespan and self.gdelt_client is not None:
            self._log.info(
                "Sin noticias frescas. Reintentando GDELT con timespan='{}'",
                fallback_timespan,
            )
            try:
                # gdelt_client acepta timespan kwarg
                fallback_articles = self.gdelt_client.fetch_articles(
                    list(keywords),
                    timespan=fallback_timespan,
                )
                all_raw.extend(fallback_articles)
                if fallback_articles:
                    self._log.info(
                        "Fallback con timespan {} encontró {} artículos",
                        fallback_timespan,
                        len(fallback_articles),
                    )
            except Exception as exc:
                self._log.warning("GDELT fallback falló: {}", exc)

        # Score
        for art in all_raw:
            art.preliminary_impact_score = self._score(art, keywords)

        # Dedupe + sort
        unique = self._deduplicate(all_raw)
        unique.sort(key=lambda a: a.preliminary_impact_score, reverse=True)

        # Cache
        self._cache[cache_key] = (time.time(), unique)

        self._log.info(
            "Procesados {} artículos crudos → {} únicos tras dedup",
            len(all_raw),
            len(unique),
        )
        return unique[:max_articles]

    # =====================================================
    # Scoring
    # =====================================================

    def _score(self, article: NewsArticle, keywords: list[str]) -> float:
        score = 0.0

        # 1) Recencia (max 30)
        if article.published_at:
            delta = datetime.now(timezone.utc) - article.published_at
            hours_ago = max(0.0, delta.total_seconds() / 3600)
            score += max(0.0, 30 - hours_ago * (30 / 24))

        # 2) Reputación de la fuente (max 30, mín 10)
        if self._is_high_reputation(article.source_name):
            score += 30
        else:
            score += 10

        # 3) Match de keywords (max 30; 10 pts por keyword único, tope 3)
        text = f"{article.title} {article.description}".lower()
        matched: list[str] = []
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if kw_lower and kw_lower in text and kw_lower not in matched:
                matched.append(kw_lower)
                if len(matched) >= 3:
                    break
        score += len(matched) * 10
        article.matched_keywords = matched

        # 4) Marcadores de urgencia en el título (max 10)
        title_upper = article.title.upper()
        if any(marker in title_upper for marker in URGENCY_MARKERS):
            score += 10

        return min(100.0, score)

    def _is_high_reputation(self, source_name: str) -> bool:
        if not source_name:
            return False
        name_lower = source_name.lower()
        for rep in self._high_reputation:
            if rep in name_lower or name_lower in rep:
                return True
        return False

    # =====================================================
    # Deduplicación
    # =====================================================

    def _deduplicate(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        if not articles:
            return []

        # Paso 1: dedupe exacto por URL. Si hay duplicados, conserva el de mayor score.
        by_url: dict[str, NewsArticle] = {}
        for art in articles:
            existing = by_url.get(art.url)
            if existing is None or art.preliminary_impact_score > existing.preliminary_impact_score:
                by_url[art.url] = art
        unique_url = list(by_url.values())

        # Paso 2: dedupe difuso por título. Procesamos en orden descendente de
        # score para que el "ganador" del cluster sea siempre el de mayor score.
        unique_url.sort(key=lambda a: a.preliminary_impact_score, reverse=True)
        threshold_pct = self.dedup_threshold * 100  # rapidfuzz devuelve 0-100
        result: list[NewsArticle] = []
        seen_titles: list[str] = []
        for art in unique_url:
            is_dup = False
            for prev in seen_titles:
                if fuzz.token_set_ratio(art.title, prev) >= threshold_pct:
                    is_dup = True
                    break
            if not is_dup:
                result.append(art)
                seen_titles.append(art.title)
        return result
