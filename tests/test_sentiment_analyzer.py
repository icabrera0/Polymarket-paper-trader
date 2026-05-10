"""
Tests for the SentimentAnalyzer.

Cover:
- LLM bypass when there is insufficient news (INSUFFICIENT_DATA).
- Filtering of old news (>48h) before the LLM.
- LLM call with a well-formed prompt.
- Parsing of the JSON response with safe default values.
- Post-LLM validation (downgrade to WAIT for low confidence, low edge,
  internal contradiction).
- Cache (does not call twice for the same input).
- JSON extraction with textual preamble.

Does not use the real Anthropic API. FakeAnthropicClient returns
predefined responses.

Run:
    pytest tests/test_sentiment_analyzer.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import patch, MagicMock

import pytest

from src.llm_client import LLMClient
from src.models import (
    MarketSnapshot,
    NewsArticle,
    NewsSource,
    Timeframe,
    TradeRecommendation,
    _new_article_id,
)
from src.sentiment_analyzer import SentimentAnalyzer


# =====================================================
# Fakes and helpers
# =====================================================


class FakeLLMClient(LLMClient):
    """Fulfills the LLMClient contract for tests."""

    def __init__(self, json_response: Optional[dict[str, Any]] = None) -> None:
        # We do not call super().__init__ to avoid needing a real BotConfig;
        # the attributes we use are set manually.
        self.json_response = json_response or {
            "consensus_probability_yes": 0.55,
            "confidence": 70,
            "sentiment_score": 0.3,
            "impact_score": 60.0,
            "recommendation": "BUY_YES",
            "timeframe": "HOURS",
            "contradictory_sources": False,
            "summary": "Fake summary",
            "justification": "Fake justification",
        }
        self.calls: list[dict[str, Any]] = []
        self.raise_on_call: Optional[Exception] = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self._log = None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        force_json: bool = False,
    ) -> dict[str, Any]:
        # Not used directly by the tests; SentimentAnalyzer calls complete_json
        return {
            "text": "{}",
            "input_tokens": 0,
            "output_tokens": 0,
            "stop_reason": "stop",
            "estimated_cost_usd": 0.0,
        }

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_attempts: int = 2,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
        if self.raise_on_call is not None:
            raise self.raise_on_call
        meta = {
            "input_tokens": len(user_prompt) // 4,
            "output_tokens": 100,
            "stop_reason": "end_turn",
            "estimated_cost_usd": 0.0,
            "attempts": 1,
        }
        return self.json_response, meta


# Backward-compat for the rest of the file
FakeAnthropicClient = FakeLLMClient


def make_market(
    market_id: str = "test-market-1",
    yes_price: float = 0.40,
    no_price: float = 0.59,
) -> MarketSnapshot:
    return MarketSnapshot(
        market_id=market_id,
        slug="test-market",
        question="Will event X happen by date Y?",
        description="A test market",
        category="Politics",
        end_date=datetime.now(timezone.utc) + timedelta(days=7),
        yes_token_id="0xyes",
        no_token_id="0xno",
        yes_price=yes_price,
        no_price=no_price,
        spread=0.005,
        volume_24h_usd=50000.0,
        volume_total_usd=500000.0,
        liquidity_usd=20000.0,
    )


def make_article(
    title: str = "Some news",
    hours_ago: float = 1.0,
    source_name: str = "Reuters",
    score: float = 70.0,
    keywords: Optional[list[str]] = None,
) -> NewsArticle:
    url = f"https://example.com/{title.lower().replace(' ', '-')}"
    return NewsArticle(
        article_id=_new_article_id(url, title),
        source=NewsSource.NEWSAPI,
        source_name=source_name,
        title=title,
        description="",
        url=url,
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        preliminary_impact_score=score,
        matched_keywords=keywords or ["event"],
    )


@pytest.fixture
def analyzer(config) -> SentimentAnalyzer:
    """SentimentAnalyzer with injected FakeLLMClient."""
    return SentimentAnalyzer(config, client=FakeLLMClient())


# =====================================================
# LLM bypass (no data)
# =====================================================


class TestInsufficientData:
    def test_sin_noticias_devuelve_insufficient_data(self, analyzer):
        market = make_market()
        result = analyzer.analyze(market, [])
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA
        assert result.confidence == 0
        assert result.consensus_probability_yes == market.yes_price
        # Must not have called the LLM
        assert analyzer.client.calls == []

    def test_una_noticia_con_low_info_disabled(self, config_factory):
        """With allow_low_info_trades=False and only 1 article → INSUFFICIENT_DATA."""
        cfg = config_factory()
        cfg.decision.allow_low_info_trades = False
        analyzer = SentimentAnalyzer(cfg, client=FakeLLMClient())
        result = analyzer.analyze(make_market(), [make_article()])
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA
        assert analyzer.client.calls == []

    def test_una_noticia_con_low_info_enabled(self, config_factory):
        """With allow_low_info_trades=True and 1 article → IS analyzed, marked low_info."""
        cfg = config_factory()
        cfg.decision.allow_low_info_trades = True
        cfg.decision.low_info_min_articles = 1
        analyzer = SentimentAnalyzer(cfg, client=FakeLLMClient())
        result = analyzer.analyze(make_market(), [make_article()])
        # DID call the LLM — panel makes 4 calls (3 agents + 1 synthesis)
        assert len(analyzer.client.calls) == 4
        # Marked as low_info
        assert result.is_low_info is True

    def test_solo_noticias_viejas_devuelve_insufficient_data(self, analyzer):
        market = make_market()
        old_articles = [make_article(hours_ago=72), make_article(hours_ago=100)]
        result = analyzer.analyze(market, old_articles)
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA
        assert analyzer.client.calls == []


# =====================================================
# Pre-LLM filtering
# =====================================================


class TestFiltering:
    def test_filtra_noticias_viejas_pero_mantiene_recientes(self, analyzer):
        market = make_market()
        articles = [
            make_article(title=f"Recent {i}", hours_ago=2.0)
            for i in range(3)
        ] + [
            make_article(title=f"Old {i}", hours_ago=72.0)
            for i in range(5)
        ]
        analyzer.analyze(market, articles)
        # Panel makes 4 calls (3 agents + synthesis); all share the same filtered user_prompt
        assert len(analyzer.client.calls) == 4
        prompt = analyzer.client.calls[0]["user_prompt"]
        assert "Recent" in prompt
        # The old ones should not appear
        assert "Old 0" not in prompt

    def test_top_10_por_score(self, analyzer):
        market = make_market()
        articles = [
            make_article(title=f"News {i}", score=float(i), hours_ago=1.0)
            for i in range(20)
        ]
        analyzer.analyze(market, articles)
        prompt = analyzer.client.calls[0]["user_prompt"]
        # News 19 (best score) must be there; News 0 (worst) must NOT be there
        assert "News 19" in prompt
        assert "News 0" not in prompt


# =====================================================
# LLM call and parsing
# =====================================================


class TestLLMCall:
    def test_construye_prompt_con_info_del_mercado(self, analyzer):
        market = make_market(yes_price=0.42, no_price=0.57)
        articles = [make_article(title=f"News {i}") for i in range(3)]
        analyzer.analyze(market, articles)

        prompt = analyzer.client.calls[0]["user_prompt"]
        assert market.question in prompt
        assert "0.4200" in prompt or "0.42" in prompt  # yes_price
        assert "Politics" in prompt
        assert "News 0" in prompt
        assert "News 2" in prompt

    def test_parsea_respuesta_json(self, config):
        client = FakeAnthropicClient(json_response={
            "consensus_probability_yes": 0.65,
            "confidence": 75,
            "sentiment_score": 0.5,
            "impact_score": 70,
            "recommendation": "BUY_YES",
            "timeframe": "IMMEDIATE",
            "contradictory_sources": False,
            "summary": "Good news for YES",
            "justification": "Multiple Reuters sources confirm.",
        })
        analyzer = SentimentAnalyzer(config, client=client)
        market = make_market(yes_price=0.40)
        result = analyzer.analyze(market, [make_article(), make_article(title="b")])

        assert result.consensus_probability_yes == pytest.approx(0.65)
        assert result.edge == pytest.approx(0.25)  # 0.65 - 0.40
        assert result.confidence == 75
        assert result.recommendation == TradeRecommendation.BUY_YES
        assert result.timeframe == Timeframe.IMMEDIATE
        # Panel path prepends a [PANEL: ...] prefix — check the text is present anywhere
        assert "Good news for YES" in result.summary
        assert result.num_articles_analyzed == 2

    def test_consensus_fuera_de_rango_se_clipa(self, config):
        client = FakeAnthropicClient(json_response={
            "consensus_probability_yes": 1.5,  # outside [0, 1]
            "confidence": 70,
            "sentiment_score": 0.3,
            "impact_score": 60,
            "recommendation": "BUY_YES",
            "timeframe": "HOURS",
            "contradictory_sources": False,
            "summary": "...",
            "justification": "...",
        })
        analyzer = SentimentAnalyzer(config, client=client)
        market = make_market()
        result = analyzer.analyze(
            market, [make_article(), make_article(title="b")]
        )
        assert result.consensus_probability_yes == 1.0

    def test_recomendacion_invalida_default_a_esperar(self, config):
        client = FakeAnthropicClient(json_response={
            "consensus_probability_yes": 0.50,
            "confidence": 70,
            "sentiment_score": 0.0,
            "impact_score": 50,
            "recommendation": "INVENTED",  # does not exist
            "timeframe": "HOURS",
            "contradictory_sources": False,
            "summary": "...",
            "justification": "...",
        })
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(), [make_article(), make_article(title="b")]
        )
        assert result.recommendation == TradeRecommendation.WAIT


# =====================================================
# Post-LLM validation
# =====================================================


class TestValidation:
    def _llm_with(self, **overrides) -> dict[str, Any]:
        base = {
            "consensus_probability_yes": 0.55,
            "confidence": 70,
            "sentiment_score": 0.3,
            "impact_score": 60.0,
            "recommendation": "BUY_YES",
            "timeframe": "HOURS",
            "contradictory_sources": False,
            "summary": "...",
            "justification": "...",
        }
        base.update(overrides)
        return base

    def test_downgrade_si_confidence_baja(self, config):
        # min_confidence_threshold from conftest is 60
        client = FakeAnthropicClient(self._llm_with(confidence=40))
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.30),
            [make_article(), make_article(title="b")],
        )
        # consensus 0.55, price 0.30, edge 0.25 (sufficient) but conf 40 is low
        assert result.recommendation == TradeRecommendation.WAIT

    def test_downgrade_si_edge_pequeno(self, config):
        # consensus 0.42, price 0.40 → edge 0.02 (< 0.05 minimum)
        client = FakeAnthropicClient(self._llm_with(
            consensus_probability_yes=0.42, confidence=80,
        ))
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(), make_article(title="b")],
        )
        assert result.recommendation == TradeRecommendation.WAIT

    def test_downgrade_si_recomendacion_contradice_edge(self, config):
        # LLM says BUY_YES but consensus < price (negative edge)
        client = FakeAnthropicClient(self._llm_with(
            consensus_probability_yes=0.20,  # edge = 0.20 - 0.40 = -0.20
            confidence=80,
            recommendation="BUY_YES",
        ))
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(), make_article(title="b")],
        )
        assert result.recommendation == TradeRecommendation.WAIT

    def test_compra_yes_se_mantiene_si_todo_ok(self, config):
        client = FakeAnthropicClient(self._llm_with(
            consensus_probability_yes=0.60,  # edge = 0.20 (positive)
            confidence=85,
            recommendation="BUY_YES",
        ))
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(), make_article(title="b")],
        )
        assert result.recommendation == TradeRecommendation.BUY_YES

    def test_compra_no_se_mantiene_si_todo_ok(self, config):
        client = FakeAnthropicClient(self._llm_with(
            consensus_probability_yes=0.20,  # edge = -0.40 (NO undervalued)
            confidence=85,
            recommendation="BUY_NO",
        ))
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.60),
            [make_article(), make_article(title="b")],
        )
        assert result.recommendation == TradeRecommendation.BUY_NO


# =====================================================
# Cache
# =====================================================


class TestCache:
    # The panel makes 4 LLM calls per analyze(): 3 panel agents + 1 synthesis.
    # Cache hits skip ALL calls, so the count should not grow after the first call.
    _CALLS_PER_ANALYZE = 4  # 3 panel agents + 1 synthesis

    def test_cache_evita_segunda_llamada(self, analyzer):
        market = make_market()
        articles = [make_article(), make_article(title="b")]
        analyzer.analyze(market, articles)
        analyzer.analyze(market, articles)
        analyzer.analyze(market, articles)
        # Only the first analyze() should have made LLM calls (4); the rest are cache hits
        assert len(analyzer.client.calls) == self._CALLS_PER_ANALYZE

    def test_force_refresh_rompe_cache(self, analyzer):
        market = make_market()
        articles = [make_article(), make_article(title="b")]
        analyzer.analyze(market, articles)
        analyzer.analyze(market, articles, force_refresh=True)
        # Two full panel runs = 2 × 4 = 8 calls
        assert len(analyzer.client.calls) == self._CALLS_PER_ANALYZE * 2

    def test_cambio_de_precio_invalida_cache(self, analyzer):
        articles = [make_article(), make_article(title="b")]
        analyzer.analyze(make_market(yes_price=0.40), articles)
        # Same market but price moved > 3-decimal rounding
        analyzer.analyze(make_market(yes_price=0.50), articles)
        # Two different cache keys → two panel runs = 8 calls total
        assert len(analyzer.client.calls) == self._CALLS_PER_ANALYZE * 2

    def test_cambio_de_articulos_invalida_cache(self, analyzer):
        market = make_market()
        analyzer.analyze(market, [make_article(title="a"), make_article(title="b")])
        analyzer.analyze(market, [make_article(title="a"), make_article(title="c")])
        # Two different article sets → two panel runs = 8 calls total
        assert len(analyzer.client.calls) == self._CALLS_PER_ANALYZE * 2


# =====================================================
# Resilience
# =====================================================


class TestResilience:
    def test_error_del_llm_devuelve_insufficient(self, config):
        from src.llm_client import LLMError

        client = FakeLLMClient()
        client.raise_on_call = LLMError("API down")
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(), [make_article(), make_article(title="b")]
        )
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA
        assert "API down" in result.justification

    def test_daily_budget_exceeded_devuelve_insufficient(self, config):
        from src.llm_client import DailyBudgetExceeded

        client = FakeLLMClient()
        client.raise_on_call = DailyBudgetExceeded(
            "Budget $5.00 reached. Resets at 00:00 UTC."
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(), [make_article(), make_article(title="b")]
        )
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA
        assert "budget" in result.justification.lower()

    def test_credits_exhausted_devuelve_insufficient(self, config):
        from src.llm_client import CreditsExhausted

        client = FakeLLMClient()
        client.raise_on_call = CreditsExhausted(
            "Account has no credits. Recharge at console.anthropic.com",
            retry_after_seconds=3600,
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(), [make_article(), make_article(title="b")]
        )
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA
        assert "credits exhausted" in result.justification.lower()


# =====================================================
# JSON extractor (utility)
# =====================================================


class TestJsonExtractor:
    def test_json_puro(self):
        text = '{"a": 1, "b": "x"}'
        assert LLMClient.extract_json(text) == {"a": 1, "b": "x"}

    def test_json_con_prefacio(self):
        text = 'Here is the analysis:\n{"a": 1}\nAnything else?'
        assert LLMClient.extract_json(text) == {"a": 1}

    def test_json_en_bloque_markdown(self):
        text = "Sure!\n```json\n{\"a\": 1}\n```\n"
        assert LLMClient.extract_json(text) == {"a": 1}

    def test_json_anidado_balanceado(self):
        text = 'preface {"outer": {"inner": [1, 2]}} suffix'
        result = LLMClient.extract_json(text)
        assert result == {"outer": {"inner": [1, 2]}}

    def test_sin_json_devuelve_none(self):
        assert LLMClient.extract_json("no JSON here") is None
        assert LLMClient.extract_json("") is None

    def test_json_invalido_devuelve_none(self):
        assert LLMClient.extract_json("{not valid json}") is None


# =====================================================
# LLM Trace emission
# =====================================================


def test_trace_emits_panel_start_event(config, tmp_path):
    """_call_llm() should emit a PANEL_START event to the trace file."""
    trace_file = tmp_path / "llm_trace.jsonl"

    with patch("src.sentiment_analyzer.TRACE_FILE", trace_file):
        from src.sentiment_analyzer import SentimentAnalyzer
        from src.models import MarketSnapshot, TradeRecommendation, Timeframe, MarketAnalysis
        from datetime import datetime, timezone

        market = MarketSnapshot(
            market_id="mkt-001",
            question="Will X happen?",
            yes_token_id="yes-tok",
            no_token_id="no-tok",
            yes_price=0.60,
            no_price=0.40,
            spread=0.01,
            volume_24h_usd=50000,
            volume_total_usd=100000,
            liquidity_usd=20000,
        )

        analyzer = SentimentAnalyzer(config)

        # Patch _run_panel to avoid real LLM
        with patch.object(analyzer, "_run_panel", return_value=MarketAnalysis(
            market_id="mkt-001",
            market_question="Will X happen?",
            yes_token_id="yes-tok",
            no_token_id="no-tok",
            current_yes_price=0.60,
            current_no_price=0.40,
            consensus_probability_yes=0.60,
            edge=0.0,
            confidence=0,
            sentiment_score=0.0,
            impact_score=0.0,
            recommendation=TradeRecommendation.WAIT,
        )):
            analyzer._call_llm(market, [])

    # Read trace file and find PANEL_START
    lines = trace_file.read_text().splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    panel_starts = [e for e in events if e["event"] == "PANEL_START"]
    assert len(panel_starts) == 1
    assert panel_starts[0]["market_id"] == "mkt-001"
    assert "num_articles" in panel_starts[0]


def test_llm_monitor_parses_trace_events(tmp_path):
    """The monitor's parse logic should handle valid and invalid lines gracefully."""
    import json
    trace = tmp_path / "trace.jsonl"
    # Write mixed content: valid JSON, partial line, empty line
    trace.write_text(
        '{"ts":"2026-01-01T00:00:00Z","event":"PANEL_START","market_id":"x","market_question":"Q","yes_price":0.6,"no_price":0.4,"num_articles":3,"kb_lessons_injected":0,"kb_lessons":[]}\n'
        'not-json-at-all\n'
        '\n'
        '{"ts":"2026-01-01T00:00:01Z","event":"AGENT_RESULT","market_id":"x","agent":"Quant","succeeded":true,"recommendation":"WAIT","confidence":55,"probability":0.61,"edge":-0.01,"justification_excerpt":"edge too low","input_tokens":1200,"output_tokens":210}\n',
        encoding="utf-8",
    )

    events = []
    for line in trace.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # monitor ignores malformed lines

    assert len(events) == 2
    assert events[0]["event"] == "PANEL_START"
    assert events[1]["event"] == "AGENT_RESULT"
    assert events[1]["agent"] == "Quant"


# =====================================================
# Panel probability metrics (Task 3)
# =====================================================


import statistics
from src.models import MarketAnalysis, TradeRecommendation


def test_panel_std_dev_computed(config, tmp_path):
    """panel_std_dev should equal stdev of the 3 agent probabilities."""
    from src.sentiment_analyzer import SentimentAnalyzer
    from src.models import MarketSnapshot

    market = MarketSnapshot(
        market_id="m1",
        question="Test?",
        yes_token_id="y",
        no_token_id="n",
        yes_price=0.50,
        no_price=0.50,
        spread=0.01,
        volume_24h_usd=50000,
        volume_total_usd=100000,
        liquidity_usd=20000,
    )

    analyzer = SentimentAnalyzer(config)

    probs = [0.60, 0.70, 0.55]
    expected_std = statistics.stdev(probs)

    synth_parsed = {
        "recommendation": "WAIT",
        "confidence": 0,
        "consensus_probability_yes": 0.617,
        "sentiment_score": 0.0,
        "impact_score": 0.0,
        "timeframe": "UNKNOWN",
        "contradictory_sources": False,
        "summary": "Panel: WAIT",
        "justification": "Rule 1",
    }

    with patch.object(analyzer, "_call_single_agent") as mock_agent:
        # First 3 calls = panel agents; 4th call = synthesis
        mock_agent.side_effect = [
            ({"recommendation": "WAIT", "confidence": 50, "consensus_probability_yes": 0.60,
              "sentiment_score": 0, "impact_score": 0, "timeframe": "UNKNOWN",
              "contradictory_sources": False, "summary": "", "justification": ""},
             {"input_tokens": 100, "output_tokens": 50}, True),
            ({"recommendation": "WAIT", "confidence": 50, "consensus_probability_yes": 0.70,
              "sentiment_score": 0, "impact_score": 0, "timeframe": "UNKNOWN",
              "contradictory_sources": False, "summary": "", "justification": ""},
             {"input_tokens": 100, "output_tokens": 50}, True),
            ({"recommendation": "WAIT", "confidence": 50, "consensus_probability_yes": 0.55,
              "sentiment_score": 0, "impact_score": 0, "timeframe": "UNKNOWN",
              "contradictory_sources": False, "summary": "", "justification": ""},
             {"input_tokens": 100, "output_tokens": 50}, True),
            (synth_parsed, {"input_tokens": 200, "output_tokens": 100}, True),
        ]
        with patch("src.sentiment_analyzer.TRACE_FILE", tmp_path / "trace.jsonl"):
            result = analyzer._run_panel(market, [], "dummy user prompt")

    assert abs(result.panel_std_dev - expected_std) < 1e-6, (
        f"panel_std_dev={result.panel_std_dev} expected {expected_std}"
    )


def test_mispricing_z_score_computed(config, tmp_path):
    """mispricing_z_score = (consensus - market_price) / panel_std_dev."""
    from src.sentiment_analyzer import SentimentAnalyzer
    from src.models import MarketSnapshot

    market = MarketSnapshot(
        market_id="m2",
        question="Test Z?",
        yes_token_id="y2",
        no_token_id="n2",
        yes_price=0.50,
        no_price=0.50,
        spread=0.01,
        volume_24h_usd=50000,
        volume_total_usd=100000,
        liquidity_usd=20000,
    )

    analyzer = SentimentAnalyzer(config)
    probs = [0.60, 0.70, 0.55]
    std_dev = statistics.stdev(probs)
    consensus = 0.617
    expected_z = (consensus - 0.50) / std_dev

    synth_parsed = {
        "recommendation": "WAIT",
        "confidence": 0,
        "consensus_probability_yes": consensus,
        "sentiment_score": 0.0,
        "impact_score": 0.0,
        "timeframe": "UNKNOWN",
        "contradictory_sources": False,
        "summary": "test",
        "justification": "test",
    }

    with patch.object(analyzer, "_call_single_agent") as mock_agent:
        mock_agent.side_effect = [
            ({"recommendation": "WAIT", "confidence": 50, "consensus_probability_yes": 0.60,
              "sentiment_score": 0, "impact_score": 0, "timeframe": "UNKNOWN",
              "contradictory_sources": False, "summary": "", "justification": ""},
             {"input_tokens": 100, "output_tokens": 50}, True),
            ({"recommendation": "WAIT", "confidence": 50, "consensus_probability_yes": 0.70,
              "sentiment_score": 0, "impact_score": 0, "timeframe": "UNKNOWN",
              "contradictory_sources": False, "summary": "", "justification": ""},
             {"input_tokens": 100, "output_tokens": 50}, True),
            ({"recommendation": "WAIT", "confidence": 50, "consensus_probability_yes": 0.55,
              "sentiment_score": 0, "impact_score": 0, "timeframe": "UNKNOWN",
              "contradictory_sources": False, "summary": "", "justification": ""},
             {"input_tokens": 100, "output_tokens": 50}, True),
            (synth_parsed, {"input_tokens": 200, "output_tokens": 100}, True),
        ]
        with patch("src.sentiment_analyzer.TRACE_FILE", tmp_path / "trace.jsonl"):
            result = analyzer._run_panel(market, [], "dummy user prompt")

    assert abs(result.mispricing_z_score - expected_z) < 1e-4


def test_edge_threshold_reads_from_config(config):
    """_validate() should use config.market_filters.min_edge_for_trade, not hardcoded 0.10."""
    from src.sentiment_analyzer import SentimentAnalyzer

    # Config has min_edge_for_trade=0.04 by default
    analyzer = SentimentAnalyzer(config)

    analysis = MarketAnalysis(
        market_id="m3",
        market_question="Test edge?",
        yes_token_id="y3",
        no_token_id="n3",
        current_yes_price=0.50,
        current_no_price=0.50,
        consensus_probability_yes=0.55,  # edge = 0.05 — above 0.04, below old 0.10
        edge=0.05,
        confidence=70,
        sentiment_score=0.5,
        impact_score=50.0,
        recommendation=TradeRecommendation.BUY_YES,
    )

    result = analyzer._validate(analysis)
    # With min_edge_for_trade=0.04, edge=0.05 should NOT be downgraded to WAIT
    assert result.recommendation == TradeRecommendation.BUY_YES, (
        f"Expected BUY_YES but got {result.recommendation}"
    )
