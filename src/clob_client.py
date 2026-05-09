"""
CLOB API client for real-time midpoint price lookups.

The Gamma API's clobTokenIds filter is non-functional (silently ignored).
This client calls the CLOB API directly, which is the authoritative
real-time price source for Polymarket tokens.

Endpoint: GET /midpoint?token_id={token_id} → {"mid": "0.365"}
No auth required for public price endpoints.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Module-level set: tokens that returned 404 (market resolved / token gone).
# Persists across ClobApiClient instances for the lifetime of the process so
# we don't hammer the same dead endpoint on every dashboard refresh.
_DEAD_TOKENS: set[str] = set()


class ClobApiClient:
    """Fetches real-time midpoint prices from the Polymarket CLOB API."""

    BASE_URL = "https://clob.polymarket.com"

    def __init__(self, timeout: int = 8) -> None:
        self._timeout = timeout
        self._log = logger.bind(module="clob_client")
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "PolymarketBot/0.1"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=False,
    )
    def fetch_midpoint(self, token_id: str) -> float | None:
        """Return the current midpoint price for a CLOB token ID, or None on error."""
        if token_id in _DEAD_TOKENS:
            return None
        try:
            r = self._session.get(
                f"{self.BASE_URL}/midpoint",
                params={"token_id": token_id},
                timeout=self._timeout,
            )
            r.raise_for_status()
            mid = r.json().get("mid")
            if mid is None:
                return None
            price = float(mid)
            return price if 0.0 < price < 1.0 else None
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                _DEAD_TOKENS.add(token_id)
                self._log.warning(
                    "Token {}… no encontrado en CLOB (404) — mercado posiblemente "
                    "resuelto. Se omitirá en futuros ciclos.",
                    token_id[:12],
                )
            else:
                self._log.debug("midpoint failed for {}…: {}", token_id[:12], exc)
            return None
        except Exception as exc:
            self._log.debug("midpoint failed for {}…: {}", token_id[:12], exc)
            return None

    def fetch_midpoints(
        self, token_ids: list[str], max_workers: int = 5
    ) -> dict[str, float]:
        """Fetch midpoint prices for multiple tokens concurrently.

        Returns {token_id: price} only for tokens where the fetch succeeded.
        """
        if not token_ids:
            return {}
        results: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_token = {
                pool.submit(self.fetch_midpoint, tok): tok for tok in token_ids
            }
            for future in as_completed(future_to_token):
                token = future_to_token[future]
                try:
                    price = future.result()
                    if price is not None:
                        results[token] = price
                except Exception as exc:
                    self._log.debug("midpoint worker error {}: {}", token[:12], exc)
        return results
