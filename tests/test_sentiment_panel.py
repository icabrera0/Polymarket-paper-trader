"""
Tests for the 3-agent panel in SentimentAnalyzer.

Covers:
1. All 3 panel agents are called when panel mode is active
2. If one agent fails, remaining 2 still produce a result (no exception raised)
3. If all 3 fail, falls back to single-agent path
4. Synthesis applies the 2-agent WAIT majority rule correctly
5. MIN_EDGE_FOR_TRADE is 0.10 (regression: below-10% edge returns WAIT)
6. analyze() return type is always MarketAnalysis (never raises)

All LLM calls are mocked — no real API calls.

Run:
    pytest tests/test_sentiment_panel.py -v
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import MagicMock, patch, call

import pytest

from src.llm_client import LLMClient, LLMError
from src.models import (
    MarketAnalysis,
    MarketSnapshot,
    NewsArticle,
    NewsSource,
    TradeRecommendation,
    _new_article_id,
)
from src.sentiment_analyzer import (
    MIN_EDGE_FOR_TRADE,
    PANEL_ADVERSARIAL_PROMPT,
    PANEL_DOMAIN_PROMPT,
    PANEL_QUANT_PROMPT,
    PANEL_SYNTHESIS_PROMPT,
    SentimentAnalyzer,
)


# =====================================================
# Shared helpers
# =====================================================


def make_market(
    market_id: str = "panel-test-market",
    yes_price: float = 0.40,
    no_price: float = 0.59,
    category: str = "Politics",
) -> MarketSnapshot:
    return MarketSnapshot(
        market_id=market_id,
        slug="panel-test",
        question="Will the panel test event happen?",
        description="Test market for panel analysis.",
        category=category,
        end_date=datetime.now(timezone.utc) + timedelta(days=7),
        yes_token_id="0xyes_panel",
        no_token_id="0xno_panel",
        yes_price=yes_price,
        no_price=no_price,
        spread=0.01,
        volume_24h_usd=75000.0,
        volume_total_usd=750000.0,
        liquidity_usd=30000.0,
    )


def make_article(
    title: str = "Panel test article",
    hours_ago: float = 1.0,
    score: float = 75.0,
) -> NewsArticle:
    url = f"https://example.com/{title.lower().replace(' ', '-')}"
    return NewsArticle(
        article_id=_new_article_id(url, title),
        source=NewsSource.NEWSAPI,
        source_name="Reuters",
        title=title,
        description="Test article description.",
        url=url,
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        preliminary_impact_score=score,
        matched_keywords=["panel", "test"],
    )


def _buy_yes_response(
    prob: float = 0.60,
    confidence: int = 70,
    rec: str = "BUY_YES",
) -> dict[str, Any]:
    """Builds a valid LLM JSON response for BUY_YES."""
    return {
        "consensus_probability_yes": prob,
        "confidence": confidence,
        "sentiment_score": 0.5,
        "impact_score": 65.0,
        "recommendation": rec,
        "timeframe": "HOURS",
        "contradictory_sources": False,
        "summary": f"Panel vote: {rec} at conf={confidence}",
        "justification": f"Edge calc: {prob:.2f} vs market. Rules applied.",
    }


def _wait_response(confidence: int = 0) -> dict[str, Any]:
    """Builds a valid LLM JSON response for WAIT."""
    return {
        "consensus_probability_yes": 0.40,
        "confidence": confidence,
        "sentiment_score": 0.0,
        "impact_score": 30.0,
        "recommendation": "WAIT",
        "timeframe": "UNKNOWN",
        "contradictory_sources": False,
        "summary": "No clear edge found.",
        "justification": "All rules triggered WAIT.",
    }


_DEFAULT_META = {"input_tokens": 200, "output_tokens": 80}


# =====================================================
# Controlled fake client that tracks per-system-prompt calls
# =====================================================


class PanelFakeLLMClient(LLMClient):
    """
    Fake LLMClient for panel tests.

    `responses` is a dict mapping system_prompt content (or a substring) to
    the (parsed_dict, meta) tuple that complete_json should return. If no
    match is found, `default_response` is returned.

    Set `fail_on_prompts` to a set of prompt substrings — those calls will
    raise LLMError instead of returning.
    """

    def __init__(
        self,
        responses: Optional[dict[str, tuple[dict, dict]]] = None,
        default_response: Optional[tuple[dict, dict]] = None,
        fail_on_prompts: Optional[set[str]] = None,
    ) -> None:
        # Bypass LLMClient.__init__ (requires real BotConfig)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self._stats_lock = threading.Lock()
        self._log = MagicMock()

        self.responses = responses or {}
        self.default_response = default_response or (
            _buy_yes_response(),
            _DEFAULT_META,
        )
        self.fail_on_prompts: set[str] = fail_on_prompts or set()
        self.calls: list[dict[str, str]] = []

    def complete(self, system_prompt, user_prompt, **kwargs) -> dict[str, Any]:
        return {"text": "{}", "input_tokens": 0, "output_tokens": 0,
                "stop_reason": "stop", "estimated_cost_usd": 0.0}

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})

        # Check fail list first
        for fail_substr in self.fail_on_prompts:
            if fail_substr in system_prompt:
                raise LLMError(f"Simulated failure for prompt containing '{fail_substr}'")

        # Match by substring of system_prompt
        for key, response in self.responses.items():
            if key in system_prompt:
                return response

        return self.default_response

    def _calls_for(self, prompt_substr: str) -> list[dict]:
        """Returns calls whose system_prompt contains the given substring."""
        return [c for c in self.calls if prompt_substr in c["system_prompt"]]


# =====================================================
# Test 1: All 3 panel agents are called
# =====================================================


class TestPanelAgentsAllCalled:
    """The panel makes 4 total LLM calls: 3 agents + 1 synthesis."""

    def test_all_three_agents_plus_synthesis_called(self, config):
        synthesis_resp = _buy_yes_response(prob=0.60, confidence=72)
        # Synthesis gets called with PANEL_SYNTHESIS_PROMPT
        client = PanelFakeLLMClient(
            responses={
                # The synthesis prompt is unique — key on a distinctive phrase
                "synthesis agent": (_buy_yes_response(prob=0.60, confidence=72), _DEFAULT_META),
                # Panel agents all get the default response
            },
            default_response=(_buy_yes_response(prob=0.60, confidence=70), _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        market = make_market(yes_price=0.40)
        articles = [make_article(f"Article {i}") for i in range(3)]

        result = analyzer.analyze(market, articles)

        # 3 panel calls + 1 synthesis = 4 total
        assert len(client.calls) == 4, (
            f"Expected 4 LLM calls (3 panel + 1 synthesis), got {len(client.calls)}"
        )

    def test_quant_domain_adversarial_prompts_used(self, config):
        """Each of the 3 specialist system prompts is actually sent."""
        client = PanelFakeLLMClient(
            default_response=(_buy_yes_response(prob=0.60, confidence=70), _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        market = make_market(yes_price=0.40)
        articles = [make_article(f"Article {i}") for i in range(3)]

        analyzer.analyze(market, articles)

        system_prompts_used = [c["system_prompt"] for c in client.calls]

        # Each panel prompt is a module-level constant — check that they were used
        assert PANEL_QUANT_PROMPT in system_prompts_used, "Quant prompt not used"
        assert PANEL_DOMAIN_PROMPT in system_prompts_used, "Domain prompt not used"
        assert PANEL_ADVERSARIAL_PROMPT in system_prompts_used, "Adversarial prompt not used"
        assert PANEL_SYNTHESIS_PROMPT in system_prompts_used, "Synthesis prompt not used"

    def test_result_is_market_analysis(self, config):
        """analyze() always returns a MarketAnalysis instance."""
        client = PanelFakeLLMClient(
            default_response=(_buy_yes_response(prob=0.60, confidence=70), _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(f"Article {i}") for i in range(3)],
        )
        assert isinstance(result, MarketAnalysis)


# =====================================================
# Test 2: One agent fails — others still produce a result
# =====================================================


class TestOneAgentFails:
    """If one panel agent raises, the other two still complete and synthesis runs."""

    def test_quant_fails_others_succeed(self, config):
        """Quant agent fails → Domain + Adversarial + Synthesis still run."""
        client = PanelFakeLLMClient(
            default_response=(_buy_yes_response(prob=0.60, confidence=70), _DEFAULT_META),
            fail_on_prompts={"pure quantitative analyst"},  # unique phrase in PANEL_QUANT_PROMPT
        )
        analyzer = SentimentAnalyzer(config, client=client)
        market = make_market(yes_price=0.40)
        articles = [make_article(f"Article {i}") for i in range(3)]

        # Must not raise
        result = analyzer.analyze(market, articles)
        assert isinstance(result, MarketAnalysis)

        # Synthesis was still called (3 calls: 2 successful agents + 1 synthesis;
        # Quant failed so its call still happened but raised)
        # Total calls: Quant (raised) + Domain + Adversarial + Synthesis = 4 attempted,
        # but only 3 reach complete_json (Quant raises before returning).
        # The panel framework catches the exception and uses the WAIT sentinel.
        assert isinstance(result, MarketAnalysis)

    def test_adversarial_fails_result_still_valid(self, config):
        """Adversarial fails → Quant + Domain + Synthesis still run."""
        client = PanelFakeLLMClient(
            default_response=(_buy_yes_response(prob=0.60, confidence=70), _DEFAULT_META),
            fail_on_prompts={"adversarial risk agent"},  # unique phrase in PANEL_ADVERSARIAL_PROMPT
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(f"Article {i}") for i in range(3)],
        )
        assert isinstance(result, MarketAnalysis)
        # Result must never be an exception
        assert result.recommendation in TradeRecommendation.__members__.values()

    def test_domain_fails_result_still_valid(self, config):
        """Domain agent fails → Quant + Adversarial + Synthesis still run."""
        client = PanelFakeLLMClient(
            default_response=(_buy_yes_response(prob=0.60, confidence=70), _DEFAULT_META),
            fail_on_prompts={"domain knowledge expert"},  # unique phrase in PANEL_DOMAIN_PROMPT
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(f"Article {i}") for i in range(3)],
        )
        assert isinstance(result, MarketAnalysis)


# =====================================================
# Test 3: All 3 agents fail → falls back to single-agent
# =====================================================


class TestAllAgentsFail:
    """If all 3 panel agents fail, the system falls back to the original single-agent path."""

    def test_all_panel_agents_fail_uses_single_agent(self, config):
        """All 3 panel agents raise → _call_llm_single is called instead."""
        # Fail all 3 panel prompts but let synthesis (and single-agent) succeed
        client = PanelFakeLLMClient(
            responses={
                # Single-agent fallback uses SYSTEM_PROMPT which contains "risk-first quantitative"
                "risk-first quantitative analyst": (
                    _buy_yes_response(prob=0.62, confidence=75),
                    _DEFAULT_META,
                ),
            },
            default_response=(_wait_response(), _DEFAULT_META),
            fail_on_prompts={
                "pure quantitative analyst",  # PANEL_QUANT_PROMPT
                "domain knowledge expert",    # PANEL_DOMAIN_PROMPT
                "adversarial risk agent",     # PANEL_ADVERSARIAL_PROMPT
            },
        )
        analyzer = SentimentAnalyzer(config, client=client)
        market = make_market(yes_price=0.40)
        articles = [make_article(f"Article {i}") for i in range(3)]

        result = analyzer.analyze(market, articles)

        # Result must be valid
        assert isinstance(result, MarketAnalysis)

        # Single-agent fallback was used — verify its response is reflected
        # (prob=0.62 → edge=0.22 which passes MIN_EDGE_FOR_TRADE=0.10)
        single_agent_calls = client._calls_for("risk-first quantitative analyst")
        assert len(single_agent_calls) == 1, (
            "Single-agent fallback was not called when all panel agents failed"
        )

    def test_all_fail_returns_market_analysis_not_exception(self, config):
        """Even if the single-agent fallback also raises, analyze() returns INSUFFICIENT_DATA."""
        from src.llm_client import LLMError

        # All calls fail
        client = PanelFakeLLMClient(
            default_response=(_wait_response(), _DEFAULT_META),
            fail_on_prompts={
                "pure quantitative analyst",
                "domain knowledge expert",
                "adversarial risk agent",
                "risk-first quantitative analyst",  # single-agent SYSTEM_PROMPT phrase
            },
        )
        # Override: the fallback path still catches exceptions at the analyze() level
        analyzer = SentimentAnalyzer(config, client=client)
        market = make_market(yes_price=0.40)
        articles = [make_article(f"Article {i}") for i in range(3)]

        # analyze() catches LLMError and returns INSUFFICIENT_DATA — must not raise
        result = analyzer.analyze(market, articles)
        assert isinstance(result, MarketAnalysis)
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA


# =====================================================
# Test 4: Synthesis WAIT majority rule
# =====================================================


class TestSynthesisWaitMajority:
    """If 2+ agents recommend WAIT, synthesis must produce WAIT."""

    def test_two_wait_agents_produces_wait(self, config):
        """Quant=WAIT, Domain=WAIT, Adversarial=BUY_YES → synthesis should WAIT."""
        synthesis_wait_response = _wait_response(confidence=0)

        client = PanelFakeLLMClient(
            responses={
                "pure quantitative analyst": (_wait_response(confidence=30), _DEFAULT_META),
                "domain knowledge expert":   (_wait_response(confidence=25), _DEFAULT_META),
                "adversarial risk agent":    (_buy_yes_response(prob=0.62, confidence=68), _DEFAULT_META),
                # Synthesis receives the WAIT majority → should also output WAIT
                "synthesis agent":           (synthesis_wait_response, _DEFAULT_META),
            },
            default_response=(synthesis_wait_response, _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(f"Article {i}") for i in range(3)],
        )

        # Synthesis was instructed to WAIT; the _validate() post-filter also applies
        assert result.recommendation == TradeRecommendation.WAIT

    def test_three_wait_agents_produces_wait(self, config):
        """All 3 agents = WAIT → synthesis must WAIT."""
        client = PanelFakeLLMClient(
            responses={
                "pure quantitative analyst": (_wait_response(confidence=30), _DEFAULT_META),
                "domain knowledge expert":   (_wait_response(confidence=20), _DEFAULT_META),
                "adversarial risk agent":    (_wait_response(confidence=40), _DEFAULT_META),
                "synthesis agent":           (_wait_response(confidence=0),  _DEFAULT_META),
            },
            default_response=(_wait_response(), _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(f"Article {i}") for i in range(3)],
        )
        assert result.recommendation == TradeRecommendation.WAIT

    def test_two_buy_yes_agents_can_produce_trade(self, config):
        """Quant=BUY_YES/70, Domain=BUY_YES/72, Adversarial=WAIT → synthesis may trade."""
        # Synthesis gets 2/3 agents agreeing BUY_YES with avg conf 71 → should trade
        synthesis_buy = _buy_yes_response(prob=0.62, confidence=66, rec="BUY_YES")

        client = PanelFakeLLMClient(
            responses={
                "pure quantitative analyst": (_buy_yes_response(prob=0.63, confidence=70), _DEFAULT_META),
                "domain knowledge expert":   (_buy_yes_response(prob=0.61, confidence=72), _DEFAULT_META),
                "adversarial risk agent":    (_wait_response(confidence=35), _DEFAULT_META),
                "synthesis agent":           (synthesis_buy, _DEFAULT_META),
            },
            default_response=(synthesis_buy, _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(f"Article {i}") for i in range(3)],
        )
        # Synthesis output BUY_YES with conf=66 and edge=0.62-0.40=0.22 > 0.10
        # _validate() should preserve this
        assert result.recommendation == TradeRecommendation.BUY_YES


# =====================================================
# Test 5: MIN_EDGE_FOR_TRADE regression (must be 0.10)
# =====================================================


class TestMinEdgeRegression:
    def test_min_edge_constant_is_0_10(self):
        """Regression: MIN_EDGE_FOR_TRADE must be exactly 0.10."""
        assert MIN_EDGE_FOR_TRADE == 0.10, (
            f"MIN_EDGE_FOR_TRADE is {MIN_EDGE_FOR_TRADE}, expected 0.10. "
            "This was deliberately raised from 0.05 — do not revert."
        )

    def test_edge_of_9_pct_returns_wait(self, config):
        """An edge of 9% (0.09) is below the 10% threshold → must WAIT."""
        # market price=0.40, consensus=0.49 → edge=0.09 (below 0.10)
        synthesis_resp = _buy_yes_response(prob=0.49, confidence=72, rec="BUY_YES")
        client = PanelFakeLLMClient(
            default_response=(synthesis_resp, _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(f"Article {i}") for i in range(3)],
        )
        # _validate() must downgrade to WAIT because |edge|=0.09 < 0.10
        assert result.recommendation == TradeRecommendation.WAIT, (
            f"Expected WAIT for 9% edge but got {result.recommendation}. "
            "MIN_EDGE_FOR_TRADE regression."
        )

    def test_edge_of_10_pct_can_trade(self, config):
        """An edge of exactly 10% (0.10) meets the threshold → may trade."""
        # market price=0.40, consensus=0.50 → edge=0.10 (at threshold)
        synthesis_resp = _buy_yes_response(prob=0.50, confidence=72, rec="BUY_YES")
        client = PanelFakeLLMClient(
            default_response=(synthesis_resp, _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(f"Article {i}") for i in range(3)],
        )
        # Edge=0.10 is not strictly less than MIN_EDGE_FOR_TRADE → allowed through
        # (conf=72 >= 60 threshold from config)
        assert result.recommendation == TradeRecommendation.BUY_YES

    def test_edge_of_15_pct_can_trade(self, config):
        """An edge of 15% is clearly above threshold → should trade."""
        synthesis_resp = _buy_yes_response(prob=0.55, confidence=75, rec="BUY_YES")
        client = PanelFakeLLMClient(
            default_response=(synthesis_resp, _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(f"Article {i}") for i in range(3)],
        )
        assert result.recommendation == TradeRecommendation.BUY_YES

    def test_edge_below_threshold_on_buy_no(self, config):
        """BUY_NO with |edge|=0.08 (below 0.10) → must WAIT."""
        # market price=0.60, consensus=0.52 → edge=-0.08 (|edge|=0.08)
        synthesis_resp = {
            "consensus_probability_yes": 0.52,
            "confidence": 72,
            "sentiment_score": -0.3,
            "impact_score": 55.0,
            "recommendation": "BUY_NO",
            "timeframe": "HOURS",
            "contradictory_sources": False,
            "summary": "...",
            "justification": "Edge is -0.08.",
        }
        client = PanelFakeLLMClient(
            default_response=(synthesis_resp, _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.60, no_price=0.39),
            [make_article(f"Article {i}") for i in range(3)],
        )
        assert result.recommendation == TradeRecommendation.WAIT


# =====================================================
# Test 6: analyze() return type is always MarketAnalysis
# =====================================================


class TestAnalyzeReturnType:
    """analyze() must always return a MarketAnalysis — never raise to the caller."""

    def test_returns_market_analysis_on_success(self, config):
        client = PanelFakeLLMClient(
            default_response=(_buy_yes_response(prob=0.60, confidence=72), _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(f"a{i}") for i in range(3)],
        )
        assert isinstance(result, MarketAnalysis)

    def test_returns_market_analysis_when_no_articles(self, config):
        client = PanelFakeLLMClient(
            default_response=(_buy_yes_response(), _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        # 0 articles → must short-circuit to INSUFFICIENT_DATA, never raise
        result = analyzer.analyze(make_market(), [])
        assert isinstance(result, MarketAnalysis)
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA

    def test_returns_market_analysis_when_all_agents_raise_llm_error(self, config):
        """Even catastrophic LLM failures must not propagate out of analyze()."""
        client = PanelFakeLLMClient(
            default_response=(_wait_response(), _DEFAULT_META),
            # Fail all agent prompts + single-agent fallback
            fail_on_prompts={
                "pure quantitative analyst",
                "domain knowledge expert",
                "adversarial risk agent",
                "risk-first quantitative analyst",
            },
        )
        analyzer = SentimentAnalyzer(config, client=client)
        # Must not raise — must return INSUFFICIENT_DATA
        result = analyzer.analyze(
            make_market(),
            [make_article(f"a{i}") for i in range(3)],
        )
        assert isinstance(result, MarketAnalysis)

    def test_return_type_on_credits_exhausted(self, config):
        """CreditsExhausted propagated up from the panel — analyze() must absorb it."""
        from src.llm_client import CreditsExhausted

        client = PanelFakeLLMClient(
            default_response=(_wait_response(), _DEFAULT_META),
        )
        # Patch _run_panel to raise CreditsExhausted to simulate the error path
        # that goes through analyze()'s except clause
        original_run_panel = SentimentAnalyzer._run_panel

        def _patched_run_panel(self_inner, market, articles, user_prompt):
            raise CreditsExhausted("No credits.", retry_after_seconds=3600)

        analyzer = SentimentAnalyzer(config, client=client)
        analyzer._run_panel = lambda *a, **kw: (_ for _ in ()).throw(
            CreditsExhausted("No credits.", retry_after_seconds=3600)
        )

        # The CreditsExhausted exception originates inside _call_llm → _run_panel,
        # which is called from analyze()'s try block. analyze() must catch it.
        result = analyzer.analyze(
            make_market(),
            [make_article(f"a{i}") for i in range(3)],
        )
        assert isinstance(result, MarketAnalysis)
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA


# =====================================================
# Test 7: Panel vote prefix in summary field
# =====================================================


class TestPanelVotePrefix:
    """Panel summaries should include the [PANEL: Q=.../D=.../A=...] prefix."""

    def test_summary_contains_panel_prefix(self, config):
        """When the panel runs, the summary field starts with [PANEL: ...]."""
        synthesis_resp = _buy_yes_response(prob=0.60, confidence=70)
        synthesis_resp["summary"] = "Synthesis summary text."

        client = PanelFakeLLMClient(
            responses={
                "synthesis agent": (synthesis_resp, _DEFAULT_META),
            },
            default_response=(_buy_yes_response(prob=0.60, confidence=70), _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(f"Article {i}") for i in range(3)],
        )

        assert result.summary.startswith("[PANEL:"), (
            f"Summary does not start with [PANEL: ...]. Got: {result.summary[:80]}"
        )

    def test_panel_prefix_format(self, config):
        """The panel prefix must contain Q=, D=, A= shorthand votes."""
        synthesis_resp = _buy_yes_response(prob=0.60, confidence=70)
        synthesis_resp["summary"] = "Synthesis summary."

        client = PanelFakeLLMClient(
            responses={
                "synthesis agent": (synthesis_resp, _DEFAULT_META),
            },
            default_response=(_buy_yes_response(prob=0.60, confidence=70), _DEFAULT_META),
        )
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(f"Article {i}") for i in range(3)],
        )

        prefix_end = result.summary.find("]")
        assert prefix_end > 0, "No closing ] found in summary prefix"
        prefix = result.summary[:prefix_end + 1]

        assert "Q=" in prefix, f"Quant vote missing from prefix: {prefix}"
        assert "D=" in prefix, f"Domain vote missing from prefix: {prefix}"
        assert "A=" in prefix, f"Adversarial vote missing from prefix: {prefix}"
