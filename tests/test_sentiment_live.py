"""
Live test of the SENTIMENT_ANALYZER with the real Claude API.

Runs the full pipeline:
1. Scans Polymarket markets
2. Selects the TOP N by volume (N=3 by default, configurable)
3. For each market: extracts keywords and fetches news
4. Passes each (market, news) to Claude for quantitative analysis
5. Prints the result JSON per market and the tokens consumed

TOKEN USAGE:
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

NUM_MARKETS_TO_ANALYZE = 3


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

    if not config.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not configured in .env")
        sys.exit(1)

    # ---------- 1. Scan ----------
    print_section("STEP 1 — Market scan")
    scanner = MarketScanner(config)
    markets = scanner.scan(force_refresh=True)
    print(f"→ {len(markets)} tradeable markets")

    if not markets:
        print("No markets pass the filters. Aborting.")
        return

    top = markets[:NUM_MARKETS_TO_ANALYZE]
    print(f"\nAnalyzing the TOP {len(top)} by volume:")
    for i, m in enumerate(top, 1):
        print(f"  {i}. {m.question[:70]}")
        print(f"     YES={m.yes_price:.3f} | vol24h=${m.volume_24h_usd:,.0f}")

    # ---------- 2. News per market ----------
    print_section("STEP 2 — News ingestion per market")
    ingestor = NewsIngestor(config)

    market_news: dict[str, list] = {}
    for m in top:
        kws = extract_keywords(m.question)
        articles = ingestor.fetch(kws, max_articles=10, force_refresh=True)
        market_news[m.market_id] = articles
        print(f"  '{m.question[:50]}...': {len(articles)} articles")

    # ---------- 3. Analysis with Claude ----------
    print_section("STEP 3 — Analysis with Claude")
    analyzer = SentimentAnalyzer(config)

    for i, m in enumerate(top, 1):
        articles = market_news.get(m.market_id, [])
        print(f"\n[{i}/{len(top)}] Analyzing: {m.question[:60]}")
        print(f"        Current YES price: {m.yes_price:.3f}")
        print(f"        Available articles: {len(articles)}")

        analysis = analyzer.analyze(m, articles, force_refresh=True)

        print(f"\n  ┌─ RESULT ────────────────────────────────────────")
        print(f"  │ Recommendation:       {color_for_recommendation(analysis.recommendation)}")
        print(f"  │ YES probability:      {analysis.consensus_probability_yes:.3f}")
        print(f"  │ Edge over price:      {analysis.edge:+.3f} ({analysis.edge*100:+.1f}pp)")
        print(f"  │ Confidence:           {analysis.confidence}/100")
        print(f"  │ Sentiment:            {analysis.sentiment_score:+.2f}")
        print(f"  │ Impact:               {analysis.impact_score:.0f}/100")
        print(f"  │ Timeframe:            {analysis.timeframe.value}")
        print(f"  │ Contradictory sources: {analysis.contradictory_sources}")
        print(f"  │ Tokens (in/out):      {analysis.llm_input_tokens} / {analysis.llm_output_tokens}")
        print(f"  ├─ Summary ──────────────────────────────────────")
        print(f"  │ {analysis.summary}")
        print(f"  ├─ Justification ────────────────────────────────")
        print(f"  │ {analysis.justification[:300]}")
        print(f"  └─────────────────────────────────────────────────")

    # ---------- 4. Cost summary ----------
    print_section("STEP 4 — Total execution cost")
    print(f"  LLM calls:           {analyzer.client.total_calls}")
    print(f"  INPUT tokens:        {analyzer.client.total_input_tokens:,}")
    print(f"  OUTPUT tokens:       {analyzer.client.total_output_tokens:,}")
    # Rough cost estimate with Sonnet rates
    cost_in = analyzer.client.total_input_tokens / 1_000_000 * 3.0
    cost_out = analyzer.client.total_output_tokens / 1_000_000 * 15.0
    print(f"  Estimated cost:      ~${cost_in + cost_out:.4f} USD")
    print()


if __name__ == "__main__":
    main()
