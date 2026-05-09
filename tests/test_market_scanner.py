"""
Tests for the MarketScanner.

Cover:
- Defensive parsing of raw responses from the Gamma API (lists or JSON strings,
  missing fields, out-of-range prices).
- Tradeability filters (volume, spread, time to close, category).
- Keyword search (case-insensitive, in question and description).
- Scan cache.
- Behavior on API failures.

No network. Injects a fake client into MarketScanner.

Run:
    pytest tests/test_market_scanner.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest

from src.gamma_client import GammaApiError
from src.market_scanner import MarketScanner
from src.models import MarketSnapshot


# =====================================================
# Fake client to inject into MarketScanner
# =====================================================


class FakeGammaClient:
    """Implements the same interface as GammaApiClient for tests."""

    def __init__(
        self,
        markets: Optional[list[dict[str, Any]]] = None,
        raise_on_fetch: Optional[Exception] = None,
    ) -> None:
        self.markets = markets or []
        self.raise_on_fetch = raise_on_fetch
        self.fetch_calls = 0

    def fetch_all_active_markets(
        self,
        max_markets: int = 500,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        self.fetch_calls += 1
        if self.raise_on_fetch is not None:
            raise self.raise_on_fetch
        return self.markets[:max_markets]


# =====================================================
# Helpers
# =====================================================


def _future_iso(hours: float) -> str:
    """ISO string for `now + hours`."""
    return (
        datetime.now(timezone.utc) + timedelta(hours=hours)
    ).isoformat().replace("+00:00", "Z")


def make_raw_market(
    market_id: str = "1",
    question: str = "Will Spain win the World Cup 2030?",
    yes_price: float = 0.40,
    no_price: float = 0.58,
    volume_24h: float = 25000.0,
    end_hours_from_now: float = 240.0,  # 10 days
    category: str = "Sports",
    active: bool = True,
    closed: bool = False,
    best_bid: Optional[float] = None,
    best_ask: Optional[float] = None,
    use_string_arrays: bool = False,
    description: str = "",
    yes_token: str = "0xyes",
    no_token: str = "0xno",
) -> dict[str, Any]:
    """Builds a raw market simulating the Gamma API format."""
    tokens = [yes_token, no_token]
    prices = [str(yes_price), str(no_price)]
    if use_string_arrays:
        tokens = json.dumps(tokens)
        prices = json.dumps(prices)

    raw: dict[str, Any] = {
        "id": market_id,
        "question": question,
        "slug": question.lower().replace(" ", "-").replace("?", ""),
        "description": description,
        "category": category,
        "endDate": _future_iso(end_hours_from_now),
        "active": active,
        "closed": closed,
        "clobTokenIds": tokens,
        "outcomePrices": prices,
        "volume24hr": str(volume_24h),
        "volumeNum": volume_24h * 10,
        "liquidityNum": 10000.0,
    }
    if best_bid is not None:
        raw["bestBid"] = str(best_bid)
    if best_ask is not None:
        raw["bestAsk"] = str(best_ask)
    return raw


@pytest.fixture
def scanner(config) -> MarketScanner:
    """MarketScanner with an empty FakeGammaClient by default."""
    return MarketScanner(config, client=FakeGammaClient())


# =====================================================
# Parser
# =====================================================


class TestParseMarket:
    def test_parsea_mercado_valido(self, scanner):
        raw = make_raw_market()
        snap = scanner.parse_market(raw)
        assert snap is not None
        assert snap.market_id == "1"
        assert snap.yes_price == pytest.approx(0.40)
        assert snap.no_price == pytest.approx(0.58)
        assert snap.volume_24h_usd == pytest.approx(25000.0)
        assert snap.yes_token_id == "0xyes"
        assert snap.no_token_id == "0xno"

    def test_parsea_arrays_como_string_json(self, scanner):
        raw = make_raw_market(use_string_arrays=True)
        snap = scanner.parse_market(raw)
        assert snap is not None
        assert snap.yes_token_id == "0xyes"
        assert snap.yes_price == pytest.approx(0.40)

    def test_descarta_si_faltan_tokens(self, scanner):
        raw = make_raw_market()
        del raw["clobTokenIds"]
        assert scanner.parse_market(raw) is None

    def test_descarta_si_faltan_precios(self, scanner):
        raw = make_raw_market()
        del raw["outcomePrices"]
        assert scanner.parse_market(raw) is None

    def test_descarta_si_precio_fuera_de_rango(self, scanner):
        # yes=0 and no=1 indicates a resolved market
        raw = make_raw_market(yes_price=0.0, no_price=1.0)
        assert scanner.parse_market(raw) is None

    def test_descarta_si_pregunta_vacia(self, scanner):
        raw = make_raw_market(question="")
        assert scanner.parse_market(raw) is None

    def test_calcula_spread_desde_bid_ask(self, scanner):
        raw = make_raw_market(best_bid=0.39, best_ask=0.41)
        snap = scanner.parse_market(raw)
        assert snap is not None
        assert snap.spread == pytest.approx(0.02)

    def test_spread_fallback_cuando_no_hay_bid_ask(self, scanner):
        # yes 0.40 + no 0.58 = 0.98 → implicit spread 0.02
        raw = make_raw_market(yes_price=0.40, no_price=0.58)
        snap = scanner.parse_market(raw)
        assert snap is not None
        assert snap.spread == pytest.approx(0.02)

    def test_tolera_end_date_invalido(self, scanner):
        raw = make_raw_market()
        raw["endDate"] = "bad-date"
        snap = scanner.parse_market(raw)
        assert snap is not None
        assert snap.end_date is None

    def test_tolera_volumen_no_numerico(self, scanner):
        raw = make_raw_market()
        raw["volume24hr"] = "not-a-number"
        snap = scanner.parse_market(raw)
        assert snap is not None
        assert snap.volume_24h_usd == 0.0


# =====================================================
# Tradeability filters
# =====================================================


class TestIsTradeable:
    def test_pasa_todos_los_filtros(self, scanner):
        raw = make_raw_market(
            volume_24h=25000.0,
            yes_price=0.40,
            no_price=0.58,
            end_hours_from_now=240.0,
        )
        snap = scanner.parse_market(raw)
        assert snap is not None
        ok, reasons = scanner.is_tradeable(snap)
        assert ok, f"Should be tradeable, but rejected because: {reasons}"

    def test_rechaza_volumen_bajo(self, scanner):
        raw = make_raw_market(volume_24h=5000.0)  # < 10000
        snap = scanner.parse_market(raw)
        ok, reasons = scanner.is_tradeable(snap)
        assert not ok
        assert "low_volume" in reasons

    def test_rechaza_spread_grande(self, scanner):
        # yes 0.30 + no 0.50 = 0.80 → spread 0.20 > 0.05
        raw = make_raw_market(yes_price=0.30, no_price=0.50)
        snap = scanner.parse_market(raw)
        ok, reasons = scanner.is_tradeable(snap)
        assert not ok
        assert "wide_spread" in reasons

    def test_rechaza_cerrando_muy_pronto(self, scanner):
        # Closes in 1h, minimum is 2h
        raw = make_raw_market(end_hours_from_now=1.0)
        snap = scanner.parse_market(raw)
        ok, reasons = scanner.is_tradeable(snap)
        assert not ok
        assert "closing_too_soon" in reasons

    def test_rechaza_cerrando_muy_lejos(self, scanner):
        # Closes in 60 days, maximum is 30
        raw = make_raw_market(end_hours_from_now=60 * 24)
        snap = scanner.parse_market(raw)
        ok, reasons = scanner.is_tradeable(snap)
        assert not ok
        assert "closing_too_far" in reasons

    def test_rechaza_inactivo(self, scanner):
        raw = make_raw_market(active=False)
        snap = scanner.parse_market(raw)
        ok, reasons = scanner.is_tradeable(snap)
        assert not ok
        assert "inactive_or_closed" in reasons

    def test_rechaza_categoria_excluida(self, config_factory):
        cfg = config_factory()
        cfg.market_filters.exclude_categories = ["crypto"]
        scanner_excl = MarketScanner(cfg, client=FakeGammaClient())
        raw = make_raw_market(category="Crypto")
        snap = scanner_excl.parse_market(raw)
        ok, reasons = scanner_excl.is_tradeable(snap)
        assert not ok
        assert "excluded_category" in reasons


# =====================================================
# Keyword search
# =====================================================


class TestSearchByKeywords:
    @pytest.fixture
    def sample_markets(self, scanner) -> list[MarketSnapshot]:
        return [
            scanner.parse_market(
                make_raw_market(market_id="1", question="Will Trump win the 2028 election?")
            ),
            scanner.parse_market(
                make_raw_market(
                    market_id="2",
                    question="Will Bitcoin hit $200k in 2026?",
                    description="Crypto market on BTC reaching new highs",
                )
            ),
            scanner.parse_market(
                make_raw_market(market_id="3", question="Will Spain win the Euro 2028?")
            ),
        ]

    def test_encuentra_por_palabra(self, scanner, sample_markets):
        results = scanner.search_by_keywords(sample_markets, ["Trump"])
        assert len(results) == 1
        assert results[0].market_id == "1"

    def test_case_insensitive(self, scanner, sample_markets):
        results = scanner.search_by_keywords(sample_markets, ["trump"])
        assert len(results) == 1

    def test_busca_en_descripcion(self, scanner, sample_markets):
        results = scanner.search_by_keywords(sample_markets, ["BTC"])
        assert len(results) == 1
        assert results[0].market_id == "2"

    def test_no_busca_en_descripcion_si_se_desactiva(self, scanner, sample_markets):
        results = scanner.search_by_keywords(
            sample_markets, ["BTC"], match_in_description=False
        )
        assert len(results) == 0

    def test_keywords_vacios_devuelve_vacio(self, scanner, sample_markets):
        assert scanner.search_by_keywords(sample_markets, []) == []
        assert scanner.search_by_keywords(sample_markets, ["", "  "]) == []

    def test_multiples_keywords(self, scanner, sample_markets):
        results = scanner.search_by_keywords(sample_markets, ["Trump", "Spain"])
        ids = {m.market_id for m in results}
        assert ids == {"1", "3"}


# =====================================================
# Scan (cache and errors)
# =====================================================


class TestScan:
    def test_devuelve_solo_operables(self, config):
        client = FakeGammaClient(
            markets=[
                make_raw_market(market_id="ok", volume_24h=25000.0),
                make_raw_market(market_id="low_vol", volume_24h=1000.0),
            ]
        )
        scanner = MarketScanner(config, client=client)
        result = scanner.scan(force_refresh=True)
        ids = [m.market_id for m in result]
        assert ids == ["ok"]

    def test_cache_evita_segunda_llamada(self, config):
        client = FakeGammaClient(
            markets=[make_raw_market(market_id="ok", volume_24h=25000.0)]
        )
        scanner = MarketScanner(config, client=client, cache_ttl_seconds=60.0)
        scanner.scan()
        scanner.scan()
        scanner.scan()
        assert client.fetch_calls == 1

    def test_force_refresh_rompe_cache(self, config):
        client = FakeGammaClient(
            markets=[make_raw_market(market_id="ok", volume_24h=25000.0)]
        )
        scanner = MarketScanner(config, client=client, cache_ttl_seconds=60.0)
        scanner.scan()
        scanner.scan(force_refresh=True)
        assert client.fetch_calls == 2

    def test_devuelve_cache_si_api_falla(self, config):
        client = FakeGammaClient(
            markets=[make_raw_market(market_id="ok", volume_24h=25000.0)]
        )
        scanner = MarketScanner(config, client=client)
        first = scanner.scan()
        assert len(first) == 1

        # Now the API fails; should return the previous cache
        client.raise_on_fetch = GammaApiError("simulated outage")
        result = scanner.scan(force_refresh=True)
        assert len(result) == 1
        assert result[0].market_id == "ok"

    def test_devuelve_vacio_si_api_falla_sin_cache(self, config):
        client = FakeGammaClient(raise_on_fetch=GammaApiError("boom"))
        scanner = MarketScanner(config, client=client)
        result = scanner.scan()
        assert result == []

    def test_descarta_mercados_mal_formados(self, config):
        client = FakeGammaClient(
            markets=[
                make_raw_market(market_id="ok", volume_24h=25000.0),
                {"id": "broken"},  # missing almost everything
                make_raw_market(market_id="ok2", volume_24h=30000.0),
            ]
        )
        scanner = MarketScanner(config, client=client)
        result = scanner.scan()
        ids = sorted([m.market_id for m in result])
        assert ids == ["ok", "ok2"]
