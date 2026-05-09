"""
Full integration test: MarketScanner + NewsIngestor.

Runs the entire discovery phase pipeline:

1. Connects to Polymarket and fetches the top markets by 24h volume.
2. Automatically extracts significant keywords from the questions.
3. Searches for recent news about those keywords from configured sources.
4. Shows everything correlated: each market with the news that affects it.

Useful for validating the full chain before adding Claude (next module).

Run:
    python scripts/test_live_integration.py

API keys are read from .env. If you don't have any, leave at least GDELT
enabled in config/settings.yaml (no credentials required).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402
from src.market_scanner import MarketScanner  # noqa: E402
from src.news_ingestor import NewsIngestor  # noqa: E402
from src.models import MarketSnapshot, NewsArticle  # noqa: E402


# Minimal stopwords so questions like "Will X happen by Y?" don't
# contaminate the search with irrelevant words.
STOPWORDS = {
    "will", "the", "a", "an", "is", "are", "be", "by", "of", "in", "on",
    "at", "to", "for", "and", "or", "if", "than", "more", "less", "this",
    "that", "before", "after", "any", "all", "with", "from", "into", "as",
    "have", "has", "had", "win", "wins", "won", "do", "does", "did",
    "can", "could", "should", "would", "may", "might", "first", "next",
    "year", "month", "week", "day", "much", "many", "make", "makes",
    "made", "election", "vote",  # too generic in political markets
}


def extract_keywords(question: str, max_kw: int = 4) -> list[str]:
    """Extracts keywords from a market question.

    Simple heuristic:
    - Words with 4+ letters
    - Without stopwords
    - Prioritizes words starting with uppercase (named entities)
    """
    # Keep original capitalization to detect entities
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9'-]{3,}\b", question)
    # Separate entities (Capitalized) from common words
    entities = []
    common = []
    for w in words:
        if w.lower() in STOPWORDS:
            continue
        if w[0].isupper():
            entities.append(w)
        else:
            common.append(w.lower())
    # Maintain order, dedup, entities first
    seen = set()
    result = []
    for w in entities + common:
        key = w.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(w)
        if len(result) >= max_kw:
            break
    return result


def print_section(title: str) -> None:
    print()
    print("═" * 78)
    print(f"  {title}")
    print("═" * 78)


def print_market(idx: int, m: MarketSnapshot) -> None:
    print(f"\n{idx}. {m.question}")
    print(
        f"   YES={m.yes_price:.3f} | NO={m.no_price:.3f} | "
        f"vol24h=${m.volume_24h_usd:>10,.0f} | spread={m.spread:.4f}"
    )
    if m.end_date:
        ttc = m.time_to_close_hours or 0
        print(f"   Closes in {ttc:.0f}h ({m.end_date.strftime('%Y-%m-%d %H:%M UTC')})")


def print_article(art: NewsArticle, indent: str = "      ") -> None:
    title = art.title[:90] + ("..." if len(art.title) > 90 else "")
    print(f"{indent}[{art.preliminary_impact_score:5.1f}] {title}")
    src_name = art.source_name or "(no name)"
    print(f"{indent}        {art.source.value:<8} | {src_name}")
    if art.matched_keywords:
        print(f"{indent}        matched: {art.matched_keywords}")


def main() -> None:
    config = load_config()

    # ---------- Phase 1: markets ----------
    print_section("STEP 1 — Polymarket market scan")
    scanner = MarketScanner(config)
    print("Connecting to Gamma API and filtering tradeable markets...")
    markets = scanner.scan(force_refresh=True)
    print(f"\n→ {len(markets)} markets pass the filters.")

    if not markets:
        f = config.market_filters
        print("\nApplied filters:")
        print(f"  - Min 24h volume: ${f.min_volume_24h_usd:,.0f}")
        print(f"  - Max spread: {f.max_spread_cents}")
        print(f"  - Time to close: {f.min_time_to_close_hours}h - "
              f"{f.max_time_to_close_days} days")
        print("\nTry relaxing the filters in config/settings.yaml.")
        return

    top_markets = markets[:5]
    print(f"\nTop {len(top_markets)} by volume:")
    for i, m in enumerate(top_markets, 1):
        print_market(i, m)

    # ---------- Phase 2: keywords ----------
    print_section("STEP 2 — Keyword extraction")
    market_keywords: dict[str, list[str]] = {}
    all_keywords_set: set[str] = set()
    for m in top_markets:
        kws = extract_keywords(m.question)
        market_keywords[m.market_id] = kws
        all_keywords_set.update(kws)
        print(f"\n  {m.question[:55]}...")
        print(f"  → {kws}")

    all_keywords = sorted(all_keywords_set)
    print(f"\n→ Total unique keywords: {len(all_keywords)}")

    # ---------- Phase 3: global news ----------
    print_section("STEP 3 — News ingestion")
    ingestor = NewsIngestor(config)
    sources_active = []
    if ingestor.newsapi_client is not None: sources_active.append("NewsAPI")
    if ingestor.gdelt_client is not None: sources_active.append("GDELT")
    if ingestor.telegram_client is not None: sources_active.append("Telegram")
    print(f"Active sources: {sources_active or 'NONE'}")

    if not sources_active:
        print("\n⚠ No active sources. Enable at least one in settings.yaml")
        print("  (gdelt is the easiest option — no credentials required)")
        return

    print(f"\nSearching news for {len(all_keywords)} keywords (timespan=24h)...")
    # Override timespan to improve the chance of finding matches
    # in a one-off test. In production it uses the one from config.
    if ingestor.gdelt_client is not None:
        # Force wider timespan for this one-off test
        original_fetch = ingestor.gdelt_client.fetch_articles
        def wrapped_fetch(keywords, **kwargs):
            kwargs.setdefault("timespan", "24h")
            return original_fetch(keywords, **kwargs)
        ingestor.gdelt_client.fetch_articles = wrapped_fetch  # type: ignore

    articles = ingestor.fetch(all_keywords[:15], max_articles=20, force_refresh=True)

    if not articles:
        print("\nNo news found. Possible causes:")
        print("  - The highest-volume markets are about very niche events")
        print("    (sports, esports) that don't appear in mainstream press.")
        print("  - GDELT needs to update its indexes (~15 min delay).")
        print("  - Your keywords are too specific (e.g.: 'Lehecka').")
        print("  - NewsAPI keys are invalid.")
        print("\nTry:")
        print("  - Filtering by 'Politics' category in config (better coverage).")
        print("  - Enabling Telegram with general channels like @bbcbreaking.")
        print("  - Expanding timespan in config to '7d' temporarily.")
        return

    print(f"\n→ {len(articles)} articles after dedup, sorted by score:")
    for art in articles[:10]:
        print_article(art, indent="  ")

    # ---------- Phase 4: correlation ----------
    print_section("STEP 4 — Market ↔ news correlation")
    for m in top_markets:
        kws = market_keywords[m.market_id]
        if not kws:
            continue
        # Filter news that match at least one keyword for this market
        kws_lower = {k.lower() for k in kws}
        relevant = [
            a for a in articles
            if any(mk in kws_lower for mk in a.matched_keywords)
        ]
        if not relevant:
            continue
        print(f"\n  {m.question[:65]}")
        print(f"  ({m.yes_price:.0%} YES — vol24h ${m.volume_24h_usd:,.0f})")
        for art in relevant[:3]:
            print_article(art, indent="    ")

    print()
    print("═" * 78)
    print("  Pipeline complete. The next module (SENTIMENT_ANALYZER)")
    print("  will pass these correlations to Claude for quantitative analysis.")
    print("═" * 78)


if __name__ == "__main__":
    main()
