"""
HTTP client for the Polymarket Gamma API.

The Gamma API (https://gamma-api.polymarket.com) exposes market metadata:
questions, dates, CLOB tokens, current prices, and volume. It is the first
entry point of the bot — the MARKET_SCANNER consumes this raw data and applies
the configured filters.

This client does NOT execute orders or access the full order book (that is the
CLOB API, which will be integrated in the PAPER_TRADER module).

Features:
- Exponential retries with tenacity (3 attempts, 2-10s backoff).
- Automatic pagination to collect all active markets.
- Configurable identifiable User-Agent and timeout.
- Network errors are logged and re-raised as GammaApiError.
"""

from __future__ import annotations

from typing import Any, Optional

import requests
from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config_loader import BotConfig


class GammaApiError(Exception):
    """Error communicating with the Gamma API."""


class GammaApiClient:
    """Synchronous wrapper around the Polymarket Gamma API."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.base_url = config.polymarket.gamma_api_url.rstrip("/")
        self.timeout = config.polymarket.request_timeout_seconds

        self._log = logger.bind(module="gamma_client")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": f"PolymarketBot/{config.app.version}",
                "Accept": "application/json",
            }
        )

    # =====================================================
    # Generic GET with retries
    # =====================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            self._log.warning("GET {} failed: {}", url, exc)
            raise

    # =====================================================
    # High-level endpoints
    # =====================================================

    def fetch_markets(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        order: str = "volume24hr",
        ascending: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Returns raw markets (list of dicts) sorted by 24h volume.

        Args:
            active: active markets only.
            closed: include closed markets.
            order: sort field (default 24h volume).
            ascending: ascending or descending order.
            limit: page size (max ~500 per API).
            offset: for pagination.
        """
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower(),
            "limit": limit,
            "offset": offset,
        }
        try:
            data = self._get("/markets", params=params)
        except (requests.RequestException, RetryError) as exc:
            raise GammaApiError(f"Could not retrieve markets: {exc}") from exc

        if not isinstance(data, list):
            self._log.warning("Unexpected response (not a list): {}", type(data))
            return []
        return data

    def fetch_all_active_markets(
        self,
        max_markets: int = 500,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Paginates up to `max_markets` active markets sorted by volume.

        Stops as soon as the API returns fewer results than the page_size
        (signal that we have reached the end).
        """
        all_markets: list[dict[str, Any]] = []
        offset = 0
        while len(all_markets) < max_markets:
            remaining = max_markets - len(all_markets)
            batch_limit = min(page_size, remaining)
            batch = self.fetch_markets(limit=batch_limit, offset=offset)
            if not batch:
                break
            all_markets.extend(batch)
            if len(batch) < batch_limit:
                # No more results
                break
            offset += batch_limit

        self._log.debug(
            "fetch_all_active_markets: {} markets collected in {} pages",
            len(all_markets),
            (offset // page_size) + 1,
        )
        return all_markets

    def fetch_markets_by_token_ids(
        self, token_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Targeted lookup: returns only the markets that contain the given CLOB
        token IDs. Use this instead of a full 500-market scan when you only
        need prices for a small number of known positions."""
        if not token_ids:
            return []
        params = {
            "clobTokenIds": ",".join(token_ids),
            "active": "true",
            "closed": "false",
        }
        try:
            data = self._get("/markets", params=params)
            return data if isinstance(data, list) else []
        except (requests.RequestException, RetryError) as exc:
            self._log.warning(
                "Could not retrieve markets by token IDs: {}", exc
            )
            return []

    def fetch_market_by_id(self, market_id: str) -> Optional[dict[str, Any]]:
        """Returns a single market by ID, or None if it does not exist."""
        try:
            data = self._get(f"/markets/{market_id}")
        except (requests.RequestException, RetryError) as exc:
            self._log.warning("Could not retrieve market {}: {}", market_id, exc)
            return None
        if isinstance(data, dict):
            return data
        return None

    def fetch_market_by_token_id_raw(self, token_id: str) -> Optional[dict[str, Any]]:
        """Returns the market containing this CLOB token, including resolved ones.

        Tries first without a filter, then with closed=true — the Gamma API may
        filter resolved markets by default if not explicitly specified.
        """
        for extra in ({}, {"closed": "true"}):
            params = {"clobTokenIds": token_id, **extra}
            try:
                data = self._get("/markets", params=params)
                if isinstance(data, list) and data:
                    return data[0]
            except (requests.RequestException, RetryError) as exc:
                self._log.warning(
                    "fetch_market_by_token_id_raw {}: {}", token_id[:12], exc
                )
                return None  # network error, don't retry with different params
        return None
