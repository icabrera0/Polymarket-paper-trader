"""
Market Scanner — discovers and filters tradeable markets on Polymarket.

Responsibilities:
1. Request raw markets from the Gamma API (via GammaApiClient).
2. Parse them into validated `MarketSnapshot` objects, discarding malformed ones.
3. Apply config filters (volume, spread, time to close, category).
4. Cache the result for a short TTL to avoid hammering the API.
5. Expose a keyword search so the DECISION_ENGINE can cross-reference
   markets with news.

The scanner does NOT make trading decisions. It only identifies the universe of
markets worth thinking about.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.config_loader import BotConfig
from src.gamma_client import GammaApiClient, GammaApiError
from src.models import MarketSnapshot

# Path for persisting filter stats for the dashboard
_FILTER_STATS_PATH = Path(__file__).resolve().parent.parent / "data" / "filter_stats.json"

# Scan cache TTL (seconds). Independent of the polling interval in config
# (which controls how often the orchestrator decides to scan).
DEFAULT_SCAN_CACHE_TTL = 60.0


class MarketScanner:
    """Discovers Polymarket markets that pass the configured filters."""

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
        """Converts a raw dict from the Gamma API into a MarketSnapshot.

        Returns None if the market is malformed (missing fields,
        unexpected formats). Never raises an exception.
        """
        try:
            market_id = str(raw["id"])
            question = raw.get("question") or ""
            if not question:
                return None

            # CLOB tokens. May come as a list or as a JSON string.
            yes_token, no_token = self._parse_token_pair(raw.get("clobTokenIds"))
            if yes_token is None or no_token is None:
                return None

            # YES/NO prices. Same issue: list or JSON string.
            yes_price, no_price = self._parse_price_pair(raw.get("outcomePrices"))
            if yes_price is None or no_price is None:
                return None

            # Validation: prices must be in (0, 1) — Polymarket does not allow exact 0/1
            # in active markets. If we see them, the market is likely already resolved.
            if not (0 < yes_price < 1) or not (0 < no_price < 1):
                return None

            # Best bid/ask if available
            best_bid = self._safe_float(raw.get("bestBid"))
            best_ask = self._safe_float(raw.get("bestAsk"))
            if best_bid is not None and not (0 < best_bid < 1):
                best_bid = None
            if best_ask is not None and not (0 < best_ask < 1):
                best_ask = None

            # Spread: if real bid/ask are available, that is the spread. Otherwise,
            # we approximate it as (1 - yes - no), which measures inconsistency
            # between the two sides.
            if best_bid is not None and best_ask is not None and best_ask >= best_bid:
                spread = best_ask - best_bid
            else:
                spread = max(0.0, 1.0 - yes_price - no_price)

            # Volume and liquidity
            volume_24h = self._safe_float(raw.get("volume24hr")) or 0.0
            volume_total = self._safe_float(raw.get("volumeNum")) or 0.0
            liquidity = self._safe_float(raw.get("liquidityNum")) or 0.0

            # Close date
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
                "Malformed market, discarded: {} | raw_id={}",
                exc,
                raw.get("id"),
            )
            return None

    # =====================================================
    # Filters
    # =====================================================

    def _passes_filters(self, market: MarketSnapshot) -> tuple[bool, str]:
        """Pre-analysis gate: fast structural filters run BEFORE the LLM is called.

        Returns (passes, rejection_reason). rejection_reason is an empty string
        when the market passes. All fields are treated as Optional — missing data
        means that specific filter is skipped, never a crash.

        Filter thresholds are read exclusively from config so they can be tuned
        in settings.yaml without touching code.
        """
        filters = self.filters

        # --- 1. Volume filter ---
        min_vol = filters.min_volume_usd
        if market.volume_24h_usd is not None and market.volume_24h_usd < min_vol:
            return False, f"low_volume:{market.volume_24h_usd:.0f}<{min_vol:.0f}"

        # --- 2. Time-to-resolution filter ---
        min_hours = filters.min_hours_to_close
        ttc = market.time_to_close_hours
        if ttc is not None and ttc < min_hours:
            return False, f"closing_soon:{ttc:.1f}h<{min_hours:.0f}h"

        # --- 3. Spread/liquidity filter (proxy: 1 - yes_price - no_price) ---
        # Only applied when both prices are available (they always are in a parsed
        # MarketSnapshot, but we guard defensively in case of future model changes).
        max_spread_cost = filters.max_spread_cost
        yes_p = getattr(market, "yes_price", None)
        no_p = getattr(market, "no_price", None)
        if yes_p is not None and no_p is not None:
            spread_cost = round(1.0 - yes_p - no_p, 6)
            if spread_cost > max_spread_cost:
                return False, f"wide_spread:{spread_cost:.3f}>{max_spread_cost:.3f}"

        return True, ""

    def is_tradeable(self, market: MarketSnapshot) -> tuple[bool, list[str]]:
        """Decides whether a market passes the filters. Returns (ok, rejection_reasons)."""
        reasons: list[str] = []

        if not market.is_active or market.is_closed:
            reasons.append("inactive_or_closed")

        if market.volume_24h_usd < self.filters.min_volume_24h_usd:
            reasons.append("low_volume")

        if market.spread > self.filters.max_spread_cents:
            reasons.append("wide_spread")

        # Time to close
        ttc = market.time_to_close_hours
        if ttc is not None:
            if ttc < self.filters.min_time_to_close_hours:
                reasons.append("closing_too_soon")
            elif ttc > self.filters.max_time_to_close_days * 24:
                reasons.append("closing_too_far")

        # Excluded categories
        if market.category and self.filters.exclude_categories:
            excluded = [c.lower() for c in self.filters.exclude_categories]
            if market.category.lower() in excluded:
                reasons.append("excluded_category")

        # Pre-analysis structural filters (_passes_filters)
        passes, pre_reason = self._passes_filters(market)
        if not passes:
            # Extract the prefix (e.g. "low_volume" from "low_volume:3000<5000")
            prefix = pre_reason.split(":")[0]
            reasons.append(prefix)

        return len(reasons) == 0, reasons

    # =====================================================
    # Main scan (with cache)
    # =====================================================

    def scan(self, force_refresh: bool = False) -> list[MarketSnapshot]:
        """Scans, parses, and filters. Returns tradeable markets.

        Uses internal cache with TTL. Pass `force_refresh=True` to bypass it.
        If the API fails, returns the last cached list (or empty if none).
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
            self._log.error("Scan aborted due to API error: {}", exc)
            return self._cache  # Return the last cached result, even if stale

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
            "Scan: {} raw → {} parsed → {} tradeable. Rejections: {}",
            len(raw_markets),
            len(snapshots),
            len(tradeable),
            rejection_counter or "none",
        )
        self._write_filter_stats(
            scanned=len(snapshots),
            passed=len(tradeable),
            rejections=rejection_counter,
        )

        self._cache = tradeable
        self._cache_ts = now
        return tradeable

    # =====================================================
    # Smart candidate selection
    # =====================================================

    def select_candidates(
        self,
        markets: list[MarketSnapshot],
        max_candidates: int = 10,
        min_uncertainty: float = 0.10,
    ) -> list[MarketSnapshot]:
        """Selects markets that DESERVE to be analyzed.

        Filters out:
        - Markets with prices stuck at 0 or 1 (de-facto resolved, no edge).
        - Markets with uncertainty below `min_uncertainty` (e.g. <10% gap).

        Sorts by:
        - Uncertainty (closeness to 0.5) DESC: the more uncertain, the more
          likely news will move the price.
        - Within the same level, volume 24h DESC (more volume = better
          possible execution).

        This avoids wasting analysis on markets like "Cubs vs Padres at YES=1.0".
        """
        candidates = []
        for m in markets:
            # Distance to 0.5 — a market at 0.5 has maximum uncertainty
            uncertainty = 0.5 - abs(m.yes_price - 0.5)
            if uncertainty < min_uncertainty:
                continue
            candidates.append((uncertainty, m.volume_24h_usd, m))

        # Sort by uncertainty desc, volume desc
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        result = [m for (_, _, m) in candidates[:max_candidates]]

        self._log.info(
            "select_candidates: {} markets → {} analyzable candidates "
            "(min_uncertainty={:.2f})",
            len(markets),
            len(result),
            min_uncertainty,
        )
        return result

    # =====================================================
    # Sports scan (secondary sports_in_play module)
    # =====================================================

    def scan_sports_candidates(
        self,
        force_refresh: bool = False,
    ) -> list[MarketSnapshot]:
        """Scans live sports match markets for the sports_in_play module.

        Completely independent of the main scan() — bypasses exclude_categories
        and applies its own sports-specific filters.
        Has its own cache with the same TTL as the main scan.
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
            self._log.warning("Sports scan aborted: {}", exc)
            return self._sports_cache

        cats_lower = {c.lower() for c in cfg.scan_categories}
        candidates: list[MarketSnapshot] = []

        for raw in raw_markets:
            snap = self.parse_market(raw)
            if snap is None:
                continue

            # Only sports categories
            if snap.category.lower() not in cats_lower:
                continue

            # YES within range to bet NO on the underdog
            if not (cfg.min_yes_price <= snap.yes_price <= cfg.max_yes_price):
                continue

            # Match close to closing (in progress or imminent)
            ttc = snap.time_to_close_hours
            if ttc is None or ttc > cfg.max_time_to_close_hours:
                continue

            # Minimum liquidity
            if snap.volume_24h_usd < cfg.min_volume_24h_usd:
                continue

            candidates.append(snap)

        # Sort by YES price desc (most dominant favorite → best asymmetry on NO)
        candidates.sort(key=lambda m: m.yes_price, reverse=True)

        self._log.info(
            "Sports scan: {} in-play candidates (YES={:.0%}–{:.0%}, close<{:.0f}h)",
            len(candidates),
            cfg.min_yes_price,
            cfg.max_yes_price,
            cfg.max_time_to_close_hours,
        )
        self._sports_cache = candidates
        self._sports_cache_ts = now
        return candidates

    # =====================================================
    # Keyword search (cross-reference with news)
    # =====================================================

    def search_by_keywords(
        self,
        markets: list[MarketSnapshot],
        keywords: list[str],
        match_in_description: bool = True,
    ) -> list[MarketSnapshot]:
        """Filters markets that mention any of the keywords.

        Case-insensitive search over the question and optionally the
        description. For more sophisticated matching (named entities,
        embeddings), the DECISION_ENGINE will decide what to do in later
        modules; here substring filtering is sufficient.
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
    # Re-ranking with category bias (media coverage)
    # =====================================================

    def rank_for_analysis(
        self,
        markets: list[MarketSnapshot],
        category_boost: dict[str, float],
        top_n: int = 15,
    ) -> list[MarketSnapshot]:
        """Reorders markets by applying a category boost.

        Markets in categories with high media coverage
        (Politics, Crypto, Geopolitics) are prioritized over regional Esports/Sports
        that barely appear in GDELT.

        Score = log(volume_24h) × category_boost

        Returns the top_n to send to the sentiment analyzer. If the market
        universe is smaller than top_n, returns all of them.
        """
        import math

        if not markets:
            return []

        def score(m: MarketSnapshot) -> float:
            base = math.log10(max(1.0, m.volume_24h_usd))
            boost = 1.0
            if m.category and category_boost:
                # Case-insensitive match
                cat_lower = m.category.lower()
                for cat_key, mult in category_boost.items():
                    if cat_key.lower() == cat_lower:
                        boost = mult
                        break
            return base * boost

        ranked = sorted(markets, key=score, reverse=True)
        return ranked[:top_n]

    # =====================================================
    # Private helpers
    # =====================================================

    def _write_filter_stats(
        self,
        scanned: int,
        passed: int,
        rejections: dict[str, int],
    ) -> None:
        """Persists filter stats to data/filter_stats.json for the dashboard.

        Writes atomically via a temp-and-rename pattern where possible.
        Silently swallows any I/O errors so a stats write never blocks the scan.
        """
        try:
            _FILTER_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
            stats = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "markets_scanned": scanned,
                "markets_passed": passed,
                "markets_rejected": scanned - passed,
                "rejections": rejections,
                "config": {
                    "min_volume_usd": self.filters.min_volume_usd,
                    "min_hours_to_close": self.filters.min_hours_to_close,
                    "max_spread_cost": self.filters.max_spread_cost,
                },
            }
            tmp = _FILTER_STATS_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            tmp.replace(_FILTER_STATS_PATH)
        except Exception as exc:
            self._log.debug("filter_stats.json write failed (non-fatal): {}", exc)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Converts to float, returning None if not possible."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_token_pair(value: Any) -> tuple[Optional[str], Optional[str]]:
        """Returns (yes_token_id, no_token_id) or (None, None)."""
        items = MarketScanner._parse_list_or_json(value)
        if items is None or len(items) < 2:
            return None, None
        yes, no = items[0], items[1]
        if not yes or not no:
            return None, None
        return str(yes), str(no)

    @staticmethod
    def _parse_price_pair(value: Any) -> tuple[Optional[float], Optional[float]]:
        """Returns (yes_price, no_price) or (None, None)."""
        items = MarketScanner._parse_list_or_json(value)
        if items is None or len(items) < 2:
            return None, None
        try:
            return float(items[0]), float(items[1])
        except (TypeError, ValueError):
            return None, None

    @staticmethod
    def _parse_list_or_json(value: Any) -> Optional[list[Any]]:
        """The Gamma API sometimes returns lists as JSON strings. We normalize."""
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
        """Parses an ISO 8601 string to a UTC-aware datetime. Tolerant."""
        if not value or not isinstance(value, str):
            return None
        try:
            # Supports trailing 'Z'
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None
