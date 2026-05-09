"""
Live test of the SENTIMENT_ANALYZER with the real Claude API.

Runs the full pipeline:
1. Scans Polymarket markets
2. Selects the TOP N by volume (N=3 by default, configurable)
3. For each market: extracts keywords and searches for news
4. Passes each (market, news) to Claude for quantitative analysis
5. Prints the resulting JSON per market and the tokens consumed

TOKEN CONSUMPTION:
This test does spend tokens from your Anthropic account. Estimate with
claude-sonnet-4-6 at current prices (~$3/M input, $15/M output):
- Input: ~3000 tokens/market × 3 markets = ~9000 tokens (~$0.027)
- Output: ~300 tokens/market × 3 markets = ~900 tokens (~$0.013)
- Total: < $0.05 per run

Run:
    python scripts/test_sentiment_live.py

Requires ANTHROPIC_API_KEY in .env.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402
from src.market_scanner import MarketScanner  # noqa: E402
from src.news_ingestor import NewsIngestor  # noqa: E402
from src.sentiment_analyzer import SentimentAnalyzer  # noqa: E402
from src.models import TradeRecommendation  # noqa: E402

# Reuse the keyword extractor from the other script
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from test_live_integration import extract_keywords  # noqa: E402

# (NUM_MARKETS_TO_ANALYZE now comes from config.decision.markets_to_analyze_per_cycle)


def color_for_recommendation(rec: TradeRecommendation) -> str:
    """Returns an indicative emoji for the terminal."""
    return {
        TradeRecommendation.BUY_YES: "🟢 BUY YES",
        TradeRecommendation.BUY_NO: "🔴 BUY NO",
        TradeRecommendation.WAIT: "🟡 HOLD",
        TradeRecommendation.INSUFFICIENT_DATA: "⚪ NO DATA",
    }.get(rec, str(rec))


def print_section(title: str) -> None:
    print()
    print("═" * 78)
    print(f"  {title}")
    print("═" * 78)


def main() -> None:
    config = load_config()

    # Verify requirements by provider
    if config.llm.provider == "anthropic":
        if not config.anthropic_api_key:
            print("ERROR: provider='anthropic' but ANTHROPIC_API_KEY is missing from .env")
            sys.exit(1)
    elif config.llm.provider == "ollama":
        # Quick server check before wasting time
        from src.llm_client import OllamaClient, OllamaUnavailable
        try:
            OllamaClient(config).verify_setup()
        except OllamaUnavailable as exc:
            print(f"ERROR Ollama: {exc}")
            print("Run first: python scripts/setup_ollama.py")
            sys.exit(1)
    else:
        print(f"ERROR: unknown provider: {config.llm.provider}")
        sys.exit(1)

    print(f"LLM Provider: {config.llm.provider}")
    print(f"Model:        {config.llm.model}")
    print()

    # ---------- 1. Scan ----------
    print_section("STEP 1 — Market scan")
    scanner = MarketScanner(config)
    markets = scanner.scan(force_refresh=True)
    print(f"→ {len(markets)} tradeable markets")

    if not markets:
        print("No markets pass the filters. Aborting.")
        return

    top = scanner.rank_for_analysis(
        markets,
        category_boost=config.decision.category_priority_boost,
        top_n=config.decision.markets_to_analyze_per_cycle,
    )
    print(f"\nAnalyzing TOP {len(top)} (ranking with category boost):")
    for i, m in enumerate(top, 1):
        print(f"  {i:2}. [{m.category or '?':<12}] {m.question[:55]}")
        print(f"      YES={m.yes_price:.3f} | vol24h=${m.volume_24h_usd:,.0f}")

    # ---------- 2. News per market ----------
    print_section("STEP 2 — News ingestion per market")
    ingestor = NewsIngestor(config)

    market_news: dict[str, list] = {}
    fallback_lookback = (
        config.decision.fallback_news_lookback
        if config.decision.enable_fallback_search
        else None
    )
    for m in top:
        kws = extract_keywords(m.question)
        articles = ingestor.fetch(
            kws,
            max_articles=10,
            force_refresh=True,
            fallback_timespan=fallback_lookback,
        )
        market_news[m.market_id] = articles
        print(f"  '{m.question[:50]}...': {len(articles)} articles")

    # ---------- 3. LLM analysis ----------
    provider_label = {
        "anthropic": "Claude",
        "ollama": f"Ollama ({config.llm.model})",
    }.get(config.llm.provider, config.llm.provider)
    print_section(f"STEP 3 — Analysis with {provider_label}")
    analyzer = SentimentAnalyzer(config)

    for i, m in enumerate(top, 1):
        articles = market_news.get(m.market_id, [])
        print(f"\n[{i}/{len(top)}] Analyzing: {m.question[:60]}")
        print(f"        Current YES price: {m.yes_price:.3f}")
        print(f"        Available articles: {len(articles)}")

        analysis = analyzer.analyze(m, articles, force_refresh=True)

        print(f"\n  ┌─ RESULT ─────────────────────────────────────────")
        low_info_tag = " (LOW_INFO)" if analysis.is_low_info else ""
        print(f"  │ Recommendation:       {color_for_recommendation(analysis.recommendation)}{low_info_tag}")
        print(f"  │ YES probability:      {analysis.consensus_probability_yes:.3f}")
        print(f"  │ Edge over price:      {analysis.edge:+.3f} ({analysis.edge*100:+.1f}pp)")
        print(f"  │ Confidence:           {analysis.confidence}/100")
        print(f"  │ Sentiment:            {analysis.sentiment_score:+.2f}")
        print(f"  │ Impact:               {analysis.impact_score:.0f}/100")
        print(f"  │ Timeframe:            {analysis.timeframe.value}")
        print(f"  │ Contradictory sources:{analysis.contradictory_sources}")
        print(f"  │ Tokens (in/out):      {analysis.llm_input_tokens} / {analysis.llm_output_tokens}")
        print(f"  ├─ Summary ──────────────────────────────────────────")
        print(f"  │ {analysis.summary}")
        print(f"  ├─ Justification ────────────────────────────────────")
        print(f"  │ {analysis.justification[:300]}")
        print(f"  └─────────────────────────────────────────────────────")

    # ---------- 4. Summary ----------
    print_section("STEP 4 — Run summary")
    print(f"  Provider:            {config.llm.provider}")
    print(f"  Model:               {config.llm.model}")
    print(f"  LLM calls:           {analyzer.client.total_calls}")
    print(f"  INPUT tokens:        {analyzer.client.total_input_tokens:,}")
    print(f"  OUTPUT tokens:       {analyzer.client.total_output_tokens:,}")

    if config.llm.provider == "anthropic":
        spent = analyzer.client.daily_spend_usd
        limit = config.llm.daily_spend_limit_usd
        print(f"  Cost this run:        ~${spent:.4f} USD")
        if limit > 0:
            remaining = analyzer.client.daily_budget_remaining_usd
            print(f"  Daily budget:         ${limit:.2f} USD (${remaining:.4f} remaining)")
        print()
        print("  Reminder: this cost comes from your pre-loaded credits at")
        print("  console.anthropic.com (NOT from your Claude Pro/Max subscription).")
    else:
        print(f"  Cost:                $0.00 (Ollama is local and free)")
    print()


if __name__ == "__main__":
    main()
