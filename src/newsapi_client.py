"""
NewsAPI client (https://newsapi.org).

Wrapper around the official `newsapi-python` library. Converts raw responses
into normalized `NewsArticle` objects.

Free plan (Developer) limitations:
- 100 requests/day
- Articles published in the last month
- Possible 24h delay on some articles

The bot consumes ~288 requests/day polling every 5 min, so the free plan
is NOT sufficient for production. It is intended for development and testing.
For 24/7 operation you need to upgrade to a paid plan or reduce polling.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from loguru import logger

from src.config_loader import BotConfig
from src.models import NewsArticle, NewsSource, _new_article_id


class NewsApiClient:
    """Wrapper around newsapi-python that returns NewsArticle objects."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.cfg = config.news.newsapi
        self._log = logger.bind(module="newsapi_client")
        self._client = None  # Created lazily

        if not config.newsapi_key:
            self._log.warning(
                "NEWSAPI_KEY is not set in .env; the client is disabled"
            )

    # =====================================================
    # Lazy initialization of the real client
    # =====================================================

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.config.newsapi_key:
            return None
        try:
            from newsapi import NewsApiClient as RawClient

            self._client = RawClient(api_key=self.config.newsapi_key)
            return self._client
        except ImportError:
            self._log.error(
                "newsapi-python is not installed. pip install newsapi-python"
            )
            return None

    # =====================================================
    # Public API
    # =====================================================

    def fetch_articles(
        self,
        keywords: list[str],
        max_results: int = 50,
        hours_lookback: int = 1,
    ) -> list[NewsArticle]:
        """Searches for articles mentioning any of the keywords.

        Builds a query of the form `"keyword1" OR "keyword2"` (with quotes
        for compound phrases). If there are more than 10 keywords it truncates
        to avoid exceeding the NewsAPI URL length limit.
        """
        client = self._ensure_client()
        if client is None or not keywords:
            return []

        query_kws = [k.strip() for k in keywords[:10] if k.strip()]
        if not query_kws:
            return []
        query = " OR ".join(f'"{k}"' for k in query_kws)

        from_param = (
            datetime.now(timezone.utc) - timedelta(hours=hours_lookback)
        ).isoformat(timespec="seconds")

        # NewsAPI expects a single language per call. If multiple are configured,
        # we iterate.
        all_articles: list[NewsArticle] = []
        for lang in self.cfg.languages or ["en"]:
            articles = self._fetch_language(
                client=client,
                query=query,
                language=lang,
                from_param=from_param,
                page_size=min(max_results, 100),
            )
            all_articles.extend(articles)

        self._log.info(
            "NewsAPI: {} articles in {} languages for {} keywords",
            len(all_articles),
            len(self.cfg.languages or ["en"]),
            len(query_kws),
        )
        return all_articles

    # =====================================================
    # Internals
    # =====================================================

    def _fetch_language(
        self,
        client: Any,
        query: str,
        language: str,
        from_param: str,
        page_size: int,
    ) -> list[NewsArticle]:
        try:
            response = client.get_everything(
                q=query,
                language=language,
                sort_by="publishedAt",
                page_size=page_size,
                from_param=from_param,
            )
        except Exception as exc:
            # newsapi-python can raise several exception classes
            self._log.warning("NewsAPI error ({}): {}", language, exc)
            return []

        if not isinstance(response, dict):
            return []
        if response.get("status") != "ok":
            self._log.warning(
                "NewsAPI status={} message={}",
                response.get("status"),
                response.get("message"),
            )
            return []

        result: list[NewsArticle] = []
        for raw in response.get("articles", []):
            article = self._parse_article(raw, language)
            if article is not None:
                result.append(article)
        return result

    def _parse_article(
        self, raw: dict[str, Any], language: str
    ) -> Optional[NewsArticle]:
        try:
            url = raw.get("url") or ""
            title = raw.get("title") or ""
            if not url or not title:
                return None

            source_name = ""
            source_dict = raw.get("source") or {}
            if isinstance(source_dict, dict):
                source_name = source_dict.get("name") or ""

            published_at = self._parse_iso(raw.get("publishedAt"))

            return NewsArticle(
                article_id=_new_article_id(url, title),
                source=NewsSource.NEWSAPI,
                source_name=source_name,
                title=title,
                description=raw.get("description") or "",
                content=raw.get("content") or "",
                url=url,
                author=raw.get("author"),
                language=language,
                published_at=published_at,
            )
        except (TypeError, ValueError) as exc:
            self._log.debug("Malformed NewsAPI article: {}", exc)
            return None

    @staticmethod
    def _parse_iso(value: Any) -> Optional[datetime]:
        if not value or not isinstance(value, str):
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None
