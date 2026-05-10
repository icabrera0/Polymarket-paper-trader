"""
Tests for MarketScanner pre-analysis filters (_passes_filters).

Covers:
1. Volume filter: market with volume=3000 and min_volume=5000 → rejected
2. Volume filter: market with volume=10000 → passes
3. Time-to-resolution filter: closing in 6h with min_hours=12 → rejected
4. Spread filter: yes=0.45, no=0.45 (spread=0.10 > 0.06) → rejected
5. Spread filter: yes=0.50, no=0.46 (spread=0.04 < 0.06) → passes
6. Volume filter: volume_24h_usd is None → passes (missing data = no skip)
7. All 3 filters pass → returns (True, "")
8. Filter stats accumulate correctly across multiple markets via is_tradeable()

No real API calls. Uses MarketSnapshot fixture objects directly.

Run:
    pytest tests/test_market_filters.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.config_loader import MarketFiltersConfig
from src.market_scanner import MarketScanner
from src.models import MarketSnapshot


# =====================================================
# Helpers / fixtures
# =====================================================


def _make_filters(
    min_volume_usd: float = 5000.0,
    min_hours_to_close: float = 12.0,
    max_spread_cost: float = 0.06,
) -> MarketFiltersConfig:
    """Build a MarketFiltersConfig with controlled pre-analysis filter thresholds."""
    return MarketFiltersConfig(
        min_volume_24h_usd=1.0,         # set low so legacy filter doesn't interfere
        max_spread_cents=1.0,           # set high so legacy spread filter doesn't interfere
        min_probability_edge=0.10,
        min_time_to_close_hours=0.0,    # set to 0 so legacy time filter doesn't interfere
        max_time_to_close_days=365,
        exclude_categories=[],
        min_volume_usd=min_volume_usd,
        min_hours_to_close=min_hours_to_close,
        max_spread_cost=max_spread_cost,
    )


def _make_snapshot(
    yes_price: float = 0.50,
    no_price: float = 0.46,
    volume_24h_usd: Optional[float] = 10000.0,
    hours_to_close: Optional[float] = 48.0,
    is_active: bool = True,
    is_closed: bool = False,
) -> MarketSnapshot:
    """Build a MarketSnapshot with controlled values for filter testing.

    `hours_to_close` drives `end_date` so that `time_to_close_hours` (property) returns
    the expected value. Passing None sets `end_date=None` so the property returns None.
    """
    end_date: Optional[datetime] = None
    if hours_to_close is not None:
        end_date = datetime.now(timezone.utc) + timedelta(hours=hours_to_close)

    # volume_24h_usd is a required field with ge=0 — we can't pass None directly.
    # For the "None" test case we use the scanner._passes_filters which reads the
    # field via getattr — but MarketSnapshot enforces ge=0. To test the "missing
    # data" guard we patch the attribute after construction.
    vol = volume_24h_usd if volume_24h_usd is not None else 0.0

    return MarketSnapshot(
        market_id="test-market",
        slug="test-market",
        question="Will this test pass?",
        description="",
        category="Test",
        end_date=end_date,
        yes_token_id="0xyes",
        no_token_id="0xno",
        yes_price=yes_price,
        no_price=no_price,
        spread=max(0.0, 1.0 - yes_price - no_price),
        volume_24h_usd=vol,
        volume_total_usd=vol * 10,
        liquidity_usd=1000.0,
        is_active=is_active,
        is_closed=is_closed,
    )


def _make_scanner(filters: MarketFiltersConfig) -> MarketScanner:
    """Build a MarketScanner with the given filters and a dummy client."""
    config = MagicMock()
    config.market_filters = filters
    # The scanner only accesses config.sports_in_play in scan_sports_candidates,
    # which we don't test here.
    scanner = MarketScanner.__new__(MarketScanner)
    scanner.config = config
    scanner.filters = filters
    scanner.client = MagicMock()
    scanner._cache = []
    scanner._cache_ts = 0.0
    scanner._sports_cache = []
    scanner._sports_cache_ts = 0.0
    from loguru import logger
    scanner._log = logger.bind(module="test_market_filters")
    return scanner


# =====================================================
# Test cases: _passes_filters
# =====================================================


class TestPassesFilters:
    """Direct unit tests for the _passes_filters private method."""

    # --- Volume filter ---

    def test_volume_below_min_is_rejected(self):
        """Case 1: volume=3000 < min_volume=5000 → rejected with low_volume prefix."""
        filters = _make_filters(min_volume_usd=5000.0)
        scanner = _make_scanner(filters)
        snap = _make_snapshot(volume_24h_usd=3000.0, hours_to_close=48.0,
                              yes_price=0.50, no_price=0.46)

        passes, reason = scanner._passes_filters(snap)

        assert passes is False
        assert reason.startswith("low_volume"), f"Expected low_volume prefix, got: {reason}"
        assert "3000" in reason
        assert "5000" in reason

    def test_volume_above_min_passes(self):
        """Case 2: volume=10000 >= min_volume=5000 → passes volume filter."""
        filters = _make_filters(min_volume_usd=5000.0)
        scanner = _make_scanner(filters)
        snap = _make_snapshot(volume_24h_usd=10000.0, hours_to_close=48.0,
                              yes_price=0.50, no_price=0.46)

        passes, reason = scanner._passes_filters(snap)

        assert passes is True
        assert reason == ""

    def test_volume_equal_to_min_passes(self):
        """Edge case: volume exactly at min → passes (boundary inclusive)."""
        filters = _make_filters(min_volume_usd=5000.0)
        scanner = _make_scanner(filters)
        snap = _make_snapshot(volume_24h_usd=5000.0, hours_to_close=48.0,
                              yes_price=0.50, no_price=0.46)

        passes, reason = scanner._passes_filters(snap)

        assert passes is True
        assert reason == ""

    # --- Time-to-resolution filter ---

    def test_closing_in_6h_with_min_12h_is_rejected(self):
        """Case 3: closing in 6h < min_hours=12 → rejected with closing_soon prefix."""
        filters = _make_filters(min_hours_to_close=12.0)
        scanner = _make_scanner(filters)
        snap = _make_snapshot(hours_to_close=6.0, volume_24h_usd=10000.0,
                              yes_price=0.50, no_price=0.46)

        passes, reason = scanner._passes_filters(snap)

        assert passes is False
        assert reason.startswith("closing_soon"), f"Expected closing_soon prefix, got: {reason}"
        assert "6.0h" in reason
        assert "12h" in reason

    def test_closing_in_24h_with_min_12h_passes(self):
        """Markets closing in 24h with min=12h should pass."""
        filters = _make_filters(min_hours_to_close=12.0)
        scanner = _make_scanner(filters)
        snap = _make_snapshot(hours_to_close=24.0, volume_24h_usd=10000.0,
                              yes_price=0.50, no_price=0.46)

        passes, reason = scanner._passes_filters(snap)

        assert passes is True
        assert reason == ""

    def test_no_end_date_skips_time_filter(self):
        """Case 6 variant: time_to_close_hours=None (no end_date) → filter is skipped."""
        filters = _make_filters(min_hours_to_close=12.0)
        scanner = _make_scanner(filters)
        snap = _make_snapshot(hours_to_close=None, volume_24h_usd=10000.0,
                              yes_price=0.50, no_price=0.46)

        passes, reason = scanner._passes_filters(snap)

        assert passes is True
        assert reason == ""

    # --- Spread filter ---

    def test_wide_spread_yes_045_no_045_rejected(self):
        """Case 4: yes=0.45, no=0.45 → spread_cost=0.10 > 0.06 → rejected."""
        filters = _make_filters(max_spread_cost=0.06)
        scanner = _make_scanner(filters)
        snap = _make_snapshot(yes_price=0.45, no_price=0.45, volume_24h_usd=10000.0,
                              hours_to_close=48.0)

        passes, reason = scanner._passes_filters(snap)

        assert passes is False
        assert reason.startswith("wide_spread"), f"Expected wide_spread prefix, got: {reason}"
        # spread_cost = 1 - 0.45 - 0.45 = 0.10
        assert "0.100" in reason or "0.10" in reason

    def test_tight_spread_yes_050_no_046_passes(self):
        """Case 5: yes=0.50, no=0.46 → spread_cost=0.04 < 0.06 → passes."""
        filters = _make_filters(max_spread_cost=0.06)
        scanner = _make_scanner(filters)
        snap = _make_snapshot(yes_price=0.50, no_price=0.46, volume_24h_usd=10000.0,
                              hours_to_close=48.0)

        passes, reason = scanner._passes_filters(snap)

        assert passes is True
        assert reason == ""

    def test_spread_exactly_at_max_passes(self):
        """Edge case: spread_cost exactly == max_spread_cost → passes (boundary inclusive)."""
        filters = _make_filters(max_spread_cost=0.06)
        scanner = _make_scanner(filters)
        # yes=0.47, no=0.47 → spread=0.06 exactly
        snap = _make_snapshot(yes_price=0.47, no_price=0.47, volume_24h_usd=10000.0,
                              hours_to_close=48.0)

        passes, reason = scanner._passes_filters(snap)

        assert passes is True
        assert reason == ""

    # --- None / missing data cases ---

    def test_volume_none_attribute_skips_volume_filter(self):
        """Case 6: volume_24h_usd=None → volume filter skipped, market not rejected.

        MarketSnapshot enforces ge=0 so volume_24h_usd can't actually be None.
        We simulate by patching the attribute to None to test the guard path.
        """
        filters = _make_filters(min_volume_usd=5000.0)
        scanner = _make_scanner(filters)
        snap = _make_snapshot(volume_24h_usd=0.0, hours_to_close=48.0,
                              yes_price=0.50, no_price=0.46)
        # Patch the attribute to None to exercise the None guard
        snap.__dict__["volume_24h_usd"] = None

        passes, reason = scanner._passes_filters(snap)

        # With volume=None the filter is skipped; other filters pass too
        assert passes is True
        assert reason == ""

    # --- All 3 filters pass ---

    def test_all_filters_pass_returns_true_empty_reason(self):
        """Case 7: good volume, good time, tight spread → (True, '')."""
        filters = _make_filters(min_volume_usd=5000.0, min_hours_to_close=12.0,
                                max_spread_cost=0.06)
        scanner = _make_scanner(filters)
        snap = _make_snapshot(
            yes_price=0.50,
            no_price=0.46,
            volume_24h_usd=20000.0,
            hours_to_close=48.0,
        )

        passes, reason = scanner._passes_filters(snap)

        assert passes is True
        assert reason == ""


# =====================================================
# Parametrized pass/fail
# =====================================================


@pytest.mark.parametrize("volume,hours,yes_p,no_p,should_pass,expected_prefix", [
    # Reject: low volume
    (3000.0,  48.0,  0.50, 0.46,  False, "low_volume"),
    # Pass: sufficient volume
    (10000.0, 48.0,  0.50, 0.46,  True,  ""),
    # Reject: closing too soon
    (10000.0,  6.0,  0.50, 0.46,  False, "closing_soon"),
    # Reject: wide spread (0.10 > 0.06)
    (10000.0, 48.0,  0.45, 0.45,  False, "wide_spread"),
    # Pass: tight spread (0.04 < 0.06)
    (10000.0, 48.0,  0.50, 0.46,  True,  ""),
    # Pass: all borderline values that still pass (use 12.1h to avoid timing race on 12.0)
    (5000.0,  12.1,  0.47, 0.47,  True,  ""),
])
def test_passes_filters_parametrized(volume, hours, yes_p, no_p,
                                     should_pass, expected_prefix):
    """Parametrized matrix covering pass/fail combinations."""
    filters = _make_filters(min_volume_usd=5000.0, min_hours_to_close=12.0,
                            max_spread_cost=0.06)
    scanner = _make_scanner(filters)
    snap = _make_snapshot(yes_price=yes_p, no_price=no_p,
                          volume_24h_usd=volume, hours_to_close=hours)

    passes, reason = scanner._passes_filters(snap)

    assert passes is should_pass, (
        f"Expected passes={should_pass} for vol={volume} hrs={hours} "
        f"yes={yes_p} no={no_p}, got passes={passes} reason={reason!r}"
    )
    if expected_prefix:
        assert reason.startswith(expected_prefix), (
            f"Expected reason prefix {expected_prefix!r}, got {reason!r}"
        )
    else:
        assert reason == ""


# =====================================================
# Filter stats accumulate correctly (Case 8)
# =====================================================


class TestFilterStatsAccumulation:
    """Verifies that rejection counts accumulate correctly across multiple markets.

    We test via is_tradeable() (which calls _passes_filters internally) to
    confirm end-to-end wiring, and verify the rejection reason prefixes.
    """

    def test_rejection_counts_accumulate(self):
        """Case 8: process multiple markets; counts per rejection prefix are accurate."""
        # Use strict thresholds so we control exactly which filter triggers
        filters = MarketFiltersConfig(
            min_volume_24h_usd=1.0,       # legacy filter: won't fire
            max_spread_cents=1.0,         # legacy filter: won't fire
            min_probability_edge=0.10,
            min_time_to_close_hours=0.0,  # legacy filter: won't fire
            max_time_to_close_days=365,
            exclude_categories=[],
            min_volume_usd=5000.0,
            min_hours_to_close=12.0,
            max_spread_cost=0.06,
        )
        scanner = _make_scanner(filters)

        markets = [
            # passes all
            _make_snapshot(volume_24h_usd=10000.0, hours_to_close=48.0,
                           yes_price=0.50, no_price=0.46),
            # fails volume
            _make_snapshot(volume_24h_usd=3000.0, hours_to_close=48.0,
                           yes_price=0.50, no_price=0.46),
            # fails closing_soon
            _make_snapshot(volume_24h_usd=10000.0, hours_to_close=6.0,
                           yes_price=0.50, no_price=0.46),
            # fails wide_spread
            _make_snapshot(volume_24h_usd=10000.0, hours_to_close=48.0,
                           yes_price=0.45, no_price=0.45),
            # fails volume again
            _make_snapshot(volume_24h_usd=1000.0, hours_to_close=48.0,
                           yes_price=0.50, no_price=0.46),
        ]

        rejection_counter: dict[str, int] = {}
        passed_count = 0

        for snap in markets:
            ok, reasons = scanner.is_tradeable(snap)
            if ok:
                passed_count += 1
            else:
                for r in reasons:
                    rejection_counter[r] = rejection_counter.get(r, 0) + 1

        assert passed_count == 1, f"Expected 1 market to pass, got {passed_count}"
        assert rejection_counter.get("low_volume", 0) == 2, (
            f"Expected 2 low_volume rejections, got {rejection_counter}"
        )
        assert rejection_counter.get("closing_soon", 0) == 1, (
            f"Expected 1 closing_soon rejection, got {rejection_counter}"
        )
        assert rejection_counter.get("wide_spread", 0) == 1, (
            f"Expected 1 wide_spread rejection, got {rejection_counter}"
        )

    def test_all_pass_yields_zero_rejections(self):
        """If all markets pass filters, rejection_counter stays empty."""
        filters = _make_filters(min_volume_usd=5000.0, min_hours_to_close=12.0,
                                max_spread_cost=0.06)
        scanner = _make_scanner(filters)

        markets = [
            _make_snapshot(volume_24h_usd=10000.0, hours_to_close=48.0,
                           yes_price=0.50, no_price=0.46),
            _make_snapshot(volume_24h_usd=20000.0, hours_to_close=72.0,
                           yes_price=0.40, no_price=0.58),
        ]

        rejection_counter: dict[str, int] = {}
        for snap in markets:
            ok, reasons = scanner.is_tradeable(snap)
            if not ok:
                for r in reasons:
                    rejection_counter[r] = rejection_counter.get(r, 0) + 1

        assert rejection_counter == {}, (
            f"Expected no rejections but got: {rejection_counter}"
        )
