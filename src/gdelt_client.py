"""
GDELT 2.0 DOC API client (https://api.gdeltproject.org/api/v2/doc/doc).

We talk directly to the GDELT HTTP API instead of using the `gdeltdoc` library:
it is simpler, more predictable, and avoids adding a dependency that could
break with upstream changes. The GDELT DOC API:

- Is public, no API key required.
- Returns JSON with articles seen in the last N minutes/hours/days.
- Supports boolean queries and filters by language/country/domain.
- Is very fast (~ms) but does not return description/content, only URL+title.

Operational notes (lessons learned from testing):
- GDELT rejects timespans shorter than ~1h with "Timespan is too short". If
  you configure 15min, we automatically raise it to 1h.
- GDELT requires PARENTHESES around queries with OR of 3+ terms:
  CORRECT: ("trump" OR "biden" OR "spain")
  INCORRECT: "trump" OR "biden" OR "spain"  (returns a text error)
- GDELT has aggressive rate limits: we get 429 if we send batches without
  sleep. We wait GDELT_BATCH_DELAY_SECONDS between each batch.
- GDELT accepts up to ~5-7 OR terms per query before starting to fail.
  We do batching when there are more keywords.
- A query with no results is a valid empty response (not an error).

Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config_loader import BotConfig
from src.models import NewsArticle, NewsSource, _new_article_id

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Minimum accepted by GDELT in practice
MIN_TIMESPAN = "1h"
KEYWORDS_PER_QUERY = 5
GDELT_BATCH_DELAY_SECONDS = 1.5         # To avoid HTTP 429
GDELT_RATE_LIMIT_BACKOFF = 5.0          # If we receive a 429, we wait this extra time


class GdeltApiError(Exception):
    """Error communicating with GDELT."""


class GdeltClient:
    """GDELT client that returns NewsArticle objects."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.cfg = config.news.gdelt
        self.timeout = config.polymarket.request_timeout_seconds  # reused

        self._log = logger.bind(module="gdelt_client")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": f"PolymarketBot/{config.app.version}",
                "Accept": "application/json",
            }
        )

    # =====================================================
    # Public API
    # =====================================================

    def fetch_articles(
        self,
        keywords: list[str],
        max_results: Optional[int] = None,
        timespan: Optional[str] = None,
    ) -> list[NewsArticle]:
        """Searches for articles seen by GDELT that mention the keywords.

        If there are more than KEYWORDS_PER_QUERY keywords, it makes multiple
        batched queries and merges the results (deduplicated by URL). It waits
        between batches to respect GDELT's rate limit.
        """
        if not keywords:
            return []

        max_results = max_results or self.cfg.max_records
        timespan = self._normalize_timespan(timespan or self.cfg.timespan)

        clean_kws = [k.strip() for k in keywords if k.strip()]
        if not clean_kws:
            return []

        # Batches of KEYWORDS_PER_QUERY
        batches = [
            clean_kws[i : i + KEYWORDS_PER_QUERY]
            for i in range(0, len(clean_kws), KEYWORDS_PER_QUERY)
        ]

        all_articles: list[NewsArticle] = []
        seen_urls: set[str] = set()
        for i, batch in enumerate(batches):
            # Sleep between batches (not before the first one)
            if i > 0:
                time.sleep(GDELT_BATCH_DELAY_SECONDS)

            # Query with parentheses: GDELT requires them when there are 3+ OR terms.
            # With a single term it is not necessary but does not hurt.
            quoted = [f'"{k}"' for k in batch]
            if len(quoted) == 1:
                query = quoted[0]
            else:
                query = "(" + " OR ".join(quoted) + ")"

            try:
                raw_articles = self._call(
                    query=query,
                    timespan=timespan,
                    max_records=max_results,
                )
            except GdeltApiError as exc:
                # If it was a 429, wait extra and retry ONE more time
                if "429" in str(exc):
                    self._log.warning(
                        "GDELT rate limited on batch {}, waiting {}s and "
                        "retrying once",
                        i + 1,
                        GDELT_RATE_LIMIT_BACKOFF,
                    )
                    time.sleep(GDELT_RATE_LIMIT_BACKOFF)
                    try:
                        raw_articles = self._call(
                            query=query,
                            timespan=timespan,
                            max_records=max_results,
                        )
                    except GdeltApiError as exc2:
                        self._log.warning(
                            "GDELT batch {} failed after retry: {}",
                            i + 1,
                            exc2,
                        )
                        continue
                else:
                    self._log.warning(
                        "GDELT batch {} failed ({}): {}",
                        i + 1,
                        batch,
                        exc,
                    )
                    continue

            for raw in raw_articles:
                art = self._parse_article(raw)
                if art is None or art.url in seen_urls:
                    continue
                seen_urls.add(art.url)
                all_articles.append(art)

        self._log.info(
            "GDELT: {} articles for {} keywords in {} batches (timespan={})",
            len(all_articles),
            len(clean_kws),
            len(batches),
            timespan,
        )
        return all_articles

    # =====================================================
    # HTTP
    # =====================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _call(
        self,
        query: str,
        timespan: str,
        max_records: int,
    ) -> list[dict[str, Any]]:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "timespan": timespan,
            "maxrecords": max_records,
            "sort": "DateDesc",
        }
        try:
            response = self._session.get(
                GDELT_BASE_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GdeltApiError(str(exc)) from exc

        # GDELT sometimes returns 200 OK with an empty body or a text error
        # such as "Timespan is too short.". We detect those cases.
        text = response.text.strip()
        if not text:
            return []
        # Detect text error response
        if not text.startswith("{") and not text.startswith("["):
            self._log.debug("GDELT returned non-JSON text: {}", text[:200])
            # If it is a timespan error, propagate so it is known
            if "timespan" in text.lower():
                raise GdeltApiError(f"GDELT rejected the query: {text[:200]}")
            return []
        try:
            data = response.json()
        except ValueError:
            return []

        if not isinstance(data, dict):
            return []
        articles = data.get("articles")
        if not isinstance(articles, list):
            return []
        return articles

    # =====================================================
    # Helpers
    # =====================================================

    @staticmethod
    def _normalize_timespan(timespan: str) -> str:
        """Raises very short timespans to the minimum accepted by GDELT.

        GDELT rejects values like '15min' with "Timespan is too short".
        The empirically safe minimum is '1h'.
        """
        if not timespan:
            return MIN_TIMESPAN
        # Pattern: number + unit (min, h, d, w, m)
        match = re.match(r"^(\d+)\s*(min|h|d|w|m)$", timespan.strip().lower())
        if not match:
            return MIN_TIMESPAN
        n, unit = int(match.group(1)), match.group(2)
        if unit == "min" and n < 60:
            return MIN_TIMESPAN
        return timespan

    # =====================================================
    # Parser
    # =====================================================

    def _parse_article(self, raw: dict[str, Any]) -> Optional[NewsArticle]:
        try:
            url = raw.get("url") or ""
            title = raw.get("title") or ""
            if not url or not title:
                return None

            published_at = self._parse_seendate(raw.get("seendate"))
            source_name = raw.get("domain") or ""
            language = (raw.get("language") or "").lower()

            return NewsArticle(
                article_id=_new_article_id(url, title),
                source=NewsSource.GDELT,
                source_name=source_name,
                title=title,
                description="",  # GDELT does not include description
                content="",
                url=url,
                author=None,
                language=language,
                published_at=published_at,
            )
        except (TypeError, ValueError) as exc:
            self._log.debug("Malformed GDELT article: {}", exc)
            return None

    @staticmethod
    def _parse_seendate(value: Any) -> Optional[datetime]:
        """GDELT uses format YYYYMMDDTHHMMSSZ (e.g.: 20240315T120000Z)."""
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            return None
