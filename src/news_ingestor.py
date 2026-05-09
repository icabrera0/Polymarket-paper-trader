"""
News Ingestor — orchestrates news sources, deduplicates, and prioritizes.

Flow:
1. Receives a list of keywords (typically extracted from active markets).
2. Requests articles from NewsAPI and/or GDELT in parallel (if enabled).
3. Computes a heuristic impact score for each article (0-100).
4. Deduplicates:
   a) by exact URL (same article collected by two sources)
   b) by fuzzy title similarity (rapidfuzz, configurable threshold)
5. Returns the list sorted by score descending.

The score returned is a HEURISTIC. The SENTIMENT_ANALYZER module
will recalculate a real score using the LLM. This heuristic serves to:
- Decide the order in which news items are processed by the LLM (limited by
  rate limits and cost).
- Filter obvious noise before spending LLM tokens.

Score components:
  Recency            (max 30) — decays linearly over 24h
  Source reputation  (max 30) — Reuters/AP/Bloomberg etc. > others
  Keyword match      (max 30) — 10 pts per distinct keyword, up to 3
  Title urgency      (max 10) — BREAKING, URGENT, JUST IN, etc.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from loguru import logger
from rapidfuzz import fuzz

from src.config_loader import BotConfig
from src.models import NewsArticle


# Sources considered high-reputation by default. Complemented by those
# defined in config.news.newsapi.sources.
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
# Client protocol (for dependency injection)
# =====================================================


class NewsClientProtocol(Protocol):
    """Common protocol implemented by NewsApiClient and GdeltClient."""

    def fetch_articles(
        self, keywords: list[str], **kwargs: Any
    ) -> list[NewsArticle]: ...


# =====================================================
# NewsIngestor
# =====================================================


class NewsIngestor:
    """Orchestrates news sources and returns a prioritized stream."""

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

        # Client injection. If not provided, they are created from config
        # only if the source is enabled AND we have credentials.
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

        # Set of high-reputation sources, normalized to lowercase.
        configured = [
            s.replace("-", " ").lower()
            for s in (config.news.newsapi.sources or [])
        ]
        self._high_reputation = set(DEFAULT_HIGH_REPUTATION_SOURCES) | set(configured)

        # Cache: keywords (tuple) → (timestamp, articles)
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
            "NewsIngestor initialized: active sources={}, dedup={:.2f}",
            active or "none",
            self.dedup_threshold,
        )

    # =====================================================
    # Public API
    # =====================================================

    def fetch(
        self,
        keywords: list[str],
        max_articles: int = 50,
        force_refresh: bool = False,
        fallback_timespan: Optional[str] = None,
    ) -> list[NewsArticle]:
        """Returns articles relevant to `keywords`, deduplicated and prioritized.

        If `fallback_timespan` is given and the normal search returns nothing,
        it retries with that expanded timespan in GDELT (e.g. "7d").
        """
        if not keywords:
            return []

        cache_key = tuple(sorted(k.lower().strip() for k in keywords if k.strip()))
        if not cache_key:
            return []

        # Cache
        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached and (time.time() - cached[0]) < self.cache_ttl:
                self._log.debug("Cache hit for {} keywords", len(cache_key))
                return cached[1][:max_articles]

        # Fetch from all enabled sources. Individual failures do not prevent
        # other sources from contributing.
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
                self._log.error("{} fetch failed: {}", name, exc)

        # FALLBACK: if nothing was returned and an extended timespan was given,
        # retry with GDELT (it benefits the most: longer lookback = more coverage)
        if not all_raw and fallback_timespan and self.gdelt_client is not None:
            self._log.info(
                "No fresh news found. Retrying GDELT with timespan='{}'",
                fallback_timespan,
            )
            try:
                # gdelt_client accepts a timespan kwarg
                fallback_articles = self.gdelt_client.fetch_articles(
                    list(keywords),
                    timespan=fallback_timespan,
                )
                all_raw.extend(fallback_articles)
                if fallback_articles:
                    self._log.info(
                        "Fallback with timespan {} found {} articles",
                        fallback_timespan,
                        len(fallback_articles),
                    )
            except Exception as exc:
                self._log.warning("GDELT fallback failed: {}", exc)

        # Score
        for art in all_raw:
            art.preliminary_impact_score = self._score(art, keywords)

        # Dedupe + sort
        unique = self._deduplicate(all_raw)
        unique.sort(key=lambda a: a.preliminary_impact_score, reverse=True)

        # Cache
        self._cache[cache_key] = (time.time(), unique)

        self._log.info(
            "Processed {} raw articles → {} unique after dedup",
            len(all_raw),
            len(unique),
        )
        return unique[:max_articles]

    # =====================================================
    # Scoring
    # =====================================================

    def _score(self, article: NewsArticle, keywords: list[str]) -> float:
        score = 0.0

        # 1) Recency (max 30)
        if article.published_at:
            delta = datetime.now(timezone.utc) - article.published_at
            hours_ago = max(0.0, delta.total_seconds() / 3600)
            score += max(0.0, 30 - hours_ago * (30 / 24))

        # 2) Source reputation (max 30, min 10)
        if self._is_high_reputation(article.source_name):
            score += 30
        else:
            score += 10

        # 3) Keyword match (max 30; 10 pts per unique keyword, cap at 3)
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

        # 4) Urgency markers in the title (max 10)
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
    # Deduplication
    # =====================================================

    def _deduplicate(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        if not articles:
            return []

        # Step 1: exact dedup by URL. If duplicates exist, keep the one with the higher score.
        by_url: dict[str, NewsArticle] = {}
        for art in articles:
            existing = by_url.get(art.url)
            if existing is None or art.preliminary_impact_score > existing.preliminary_impact_score:
                by_url[art.url] = art
        unique_url = list(by_url.values())

        # Step 2: fuzzy dedup by title. Process in descending score order so
        # the cluster "winner" is always the one with the highest score.
        unique_url.sort(key=lambda a: a.preliminary_impact_score, reverse=True)
        threshold_pct = self.dedup_threshold * 100  # rapidfuzz returns 0-100
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