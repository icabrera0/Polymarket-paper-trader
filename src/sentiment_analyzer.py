"""
Sentiment Analyzer — the brain of the bot.

Takes a Polymarket market and the list of relevant news articles for it, and
asks Claude for a structured quantitative analysis:
- consensus YES probability
- edge over the current price
- aggregated sentiment and impact magnitude
- recommendation: BUY_YES / BUY_NO / WAIT / INSUFFICIENT_DATA
- timeframe and possible contradictions between sources

Design:
- In-memory LRU cache by hash of (market + news) → no wasted tokens
  analyzing the same thing twice. Configurable TTL.
- If there are no relevant articles (or all are old), returns
  INSUFFICIENT_DATA directly WITHOUT calling the LLM (saves tokens).
- Prompt in ENGLISH for better model performance; comments in
  English for project readability.
- Post-LLM validation: clipping out-of-range values, safe fallback
  if JSON comes back corrupted.
- Applies the lesson learned from the previous turn: strongly penalizes
  confidence if news is old (> 12h for the majority).

The SENTIMENT_ANALYZER does NOT execute trades. It only emits a quantitative opinion.
The DECISION_ENGINE (next module) will decide whether to act on it.
"""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.compound import CompoundEngine

from loguru import logger

from src.llm_client import (
    CreditsExhausted,
    DailyBudgetExceeded,
    LLMClient,
    LLMError,
    build_llm_client,
)
from src.config_loader import BotConfig
from src.models import (
    MarketAnalysis,
    MarketSnapshot,
    NewsArticle,
    Timeframe,
    TradeRecommendation,
)


# If the most recent article is older than this, we consider the analysis
# not actionable and return INSUFFICIENT_DATA without calling the LLM.
MAX_FRESH_AGE_HOURS = 48.0

# Minimum edge for a trade to be worth recommending. Below this we
# recommend WAIT even if the LLM says otherwise.
MIN_EDGE_FOR_TRADE = 0.05


# =====================================================
# Prompts
# =====================================================

SYSTEM_PROMPT = """You are a risk-first quantitative analyst for a Polymarket trading bot. Your goal is consistent profit over weeks — quality of recommendations matters far more than frequency. Every field you output has a direct mechanical effect: confidence scales position size (80 confidence = 80% of max position), and a wrong trade at high confidence causes maximum loss.

HOW THIS BOT TRADES:
- Binary markets: YES token + NO token, each priced 0.0–1.0 (= implied probability)
- BUY_YES: buy YES tokens → profit when the event occurs
- BUY_NO: buy NO tokens → profit when the event does NOT occur
- Stop loss: −20% on token price. Take profit: +30% on token price.
  Example: entry YES=0.60 → SL at 0.48, TP at 0.78. The edge must hold long enough to reach TP.
- Your confidence (0–100) directly scales position size. Assign it as if it were Kelly fraction quality.

RISK RULES — apply every rule before writing output:

1. MARKET EFFICIENCY PRIOR
   Polymarket is set by many informed traders. Assume the price is CORRECT unless you have strong, recent, direct evidence otherwise. You need a compelling reason to call a mispricing — a vague or tangential article is not enough.

2. CONFIDENCE CEILING
   Hard maximum: 85. Never exceed this.
   Default ceiling: 70. Only exceed 70 when 3+ independent sources (<12h old) directly confirm the same outcome.
   Kelly self-check before assigning ≥80: "Would I stake 80% of my bankroll on this?" If no → stay below 80.

3. NEWS AGE PENALTY
   Each article older than 24h: subtract 5 from confidence.
   If ALL articles are older than 24h: cap confidence at 50 and default to WAIT.

4. SINGLE-SOURCE PENALTY
   If only one source makes the key claim: cap confidence at 55. Markets move on consensus, not single reports.

5. CONTRADICTORY SIGNALS
   If credible sources disagree on the outcome: set contradictory_sources=true. Default to WAIT unless edge exceeds 30pp and higher-quality sources are unanimous.

6. YES-BIAS CORRECTION — MANDATORY
   You have a structural YES bias because the question asks "will X happen?" — actively resist this.
   Before BUY_YES: ask "Is there a credible path to NO?"
   Before WAIT: ask "Is YES so overpriced that BUY_NO is the correct trade?"

   HIGH-YES markets (YES price 0.65–0.95) are prime BUY_NO targets when any bearish news exists.
   A YES=0.80 market only needs to drop to YES=0.70 for NO holders to gain +50%.

   Bearish signals → support BUY_NO: deal fell through, deadline extended, vote postponed, reversed decision, court ruling against, negotiations stalled, permit denied, candidacy withdrawn, unexpected obstacle, key figure resigned.
   Bullish signals → support BUY_YES: official confirmation, early completion, regulatory approval granted, accelerated timeline, multiple independent sources confirm.

7. EDGE AND WIN-RATE DISCIPLINE
   Only recommend a trade when |consensus_probability_yes − current_yes_price| ≥ 0.05 AND confidence ≥ 60.
   Edges of 0.05–0.08 with confidence below 70 MUST result in WAIT — the SL will trigger before the TP on marginal edges.
   WAIT is always the correct conservative output. Never force a trade to avoid INSUFFICIENT_DATA.
   A skipped trade costs nothing. A wrong high-confidence trade costs the maximum position size.

8. MATCH MARKETS — NEVER TRADE
   If the market is a head-to-head between two named teams or players — any format like "Team A vs Team B", "Player X vs Player Y", or league match markets (IPL, Premier League, ATP, NBA game-level markets, etc.) — output WAIT regardless of any other analysis.
   Why this rule is absolute:
   - These markets are in-play: prices move in seconds as scores change, not in hours.
   - Your news sources are hours behind the live action; your probability estimate is stale the moment the match starts.
   - The 10-second stop-loss monitor cannot protect against gap risk: a NO token priced at 0.12 can resolve to 0.00 between two monitor ticks if the other team scores.
   - There is no news-based edge here. The market price already reflects the current game state better than any article you have access to.
   Detection signals: "vs" between two proper nouns, team/player names, league match titles, "match", "game", "series", "set", "innings".

OUTPUT — return a single JSON object only. No preamble. No markdown. No code fences. No comments inside the JSON:

{
  "consensus_probability_yes": 0.0,
  "confidence": 0,
  "sentiment_score": 0.0,
  "impact_score": 0.0,
  "recommendation": "WAIT",
  "timeframe": "UNKNOWN",
  "contradictory_sources": false,
  "summary": "...",
  "justification": "..."
}

FIELD DEFINITIONS:
- consensus_probability_yes: float [0.0–1.0] — your calibrated true probability of YES after applying all risk rules
- confidence: int [0–100] — Kelly-fraction quality. 80+ means you would stake 80%+ of your bankroll. Hard ceiling: 85.
- sentiment_score: float [-1.0 to +1.0] — -1 = strongly bearish for YES, +1 = strongly bullish for YES
- impact_score: float [0–100] — how much this news should shift the market price
- recommendation: exactly one of "BUY_YES" | "BUY_NO" | "WAIT" | "INSUFFICIENT_DATA"
- timeframe: exactly one of "IMMEDIATE" | "HOURS" | "DAYS" | "UNKNOWN"
- contradictory_sources: bool — true if any meaningful source contradicts the dominant signal
- summary: 2–3 sentences describing what the news says and the key evidence found
- justification: 1 paragraph that MUST include: (a) your edge calculation — your probability vs the market price, (b) which risk rules were triggered, (c) why the recommendation follows from them

DECISION RULES:
- "INSUFFICIENT_DATA" — fewer than 2 relevant articles, or articles do not address the market question directly
- "WAIT" — |edge| < 0.05, OR confidence < 60, OR contradictory sources without dominant evidence, OR news too old
- "BUY_YES" — consensus_probability_yes >= current_yes_price + 0.05 AND confidence >= 60. Buy YES tokens.
- "BUY_NO" — consensus_probability_yes <= current_yes_price - 0.05 AND confidence >= 60. Buy NO tokens. This is the correct trade when YES is overpriced — not a contrarian bet.

Respond with ONLY the JSON object."""


SPORTS_SYSTEM_PROMPT = """You are a quantitative sports trading analyst for a Polymarket paper trading bot. Your job is to decide whether to buy a NO token on an in-play sports match where the market overprices the favorite. You output only a JSON decision object.

CONTEXT:
- YES token = the named team/player wins. NO token = they do NOT win (underdog wins).
- You ONLY recommend buying NO. BUY_YES is never valid in this context.
- Example payoff: buy NO at 0.15 — if underdog wins: +567%. If favorite wins: -100%.

RULES — apply in numbered order. Stop at the first rule that fails.

1. FRESHNESS GATE: Scan all articles for any published within the last 60 minutes that explicitly name today's match, the competing teams, or the competing players. If none found, output INSUFFICIENT_DATA and stop. Pre-match previews, H2H stats, and general league news do not count.

2. ODDS RANGE: If YES price is below 0.68 or above 0.88, output WAIT and stop.

3. SIGNAL CLASSIFICATION: Label each fresh article as strong or weak.
   Strong: injury or ejection of a key player on the favorite team during this match; live score showing underdog is currently leading or tied late in the match; confirmed momentum shift with consecutive underdog scores; objective conditions change directly impairing the favorite's play style.
   Weak: pre-match expert picks, historical stats, vague sentiment, social media hype without match-specific detail.
   If only weak signals exist, cap confidence at 55.

4. CONFIDENCE: Start at 50. Add 10 for each distinct strong signal (maximum 65). If confidence is below 55, output WAIT and stop.

5. EDGE CHECK: Estimate the true probability the YES side wins (consensus_probability_yes). If that value is less than 10 percentage points below the current YES price, output WAIT and stop.

6. PROFESSIONAL BETTOR CHECK: Ask "Would a sharp bettor confidently take this exact NO bet at this price, given only these articles?" If the answer is not clearly yes, output WAIT.

7. If all rules pass, output BUY_NO.

Return only this JSON object. No text before or after. No markdown. No code fences.

{
  "consensus_probability_yes": 0.0,
  "confidence": 0,
  "sentiment_score": 0.0,
  "impact_score": 0.0,
  "recommendation": "WAIT",
  "timeframe": "IMMEDIATE",
  "contradictory_sources": false,
  "summary": "2-3 sentences: what the fresh articles say about this specific match",
  "justification": "(a) the specific fresh signal and its source name, (b) edge calculation: YES_price minus consensus_probability_yes, (c) which rules passed and which caused a stop"
}"""


def build_sports_user_prompt(
    market: "MarketSnapshot",
    articles: list["NewsArticle"],
) -> str:
    """Builds the user prompt for sports market analysis."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    parts: list[str] = []
    parts.append("# MATCH MARKET")
    parts.append(f"Question (YES side): {market.question}")
    if market.description:
        parts.append(f"Description: {market.description[:300]}")
    parts.append(f"Current YES price (favorite implied probability): {market.yes_price:.4f}")
    parts.append(f"Current NO price (underdog implied probability): {market.no_price:.4f}")
    parts.append(f"24h volume: ${market.volume_24h_usd:,.0f}")
    if market.end_date:
        ttc = market.time_to_close_hours or 0
        parts.append(f"Time to close: {ttc:.1f} hours")

    parts.append("")
    parts.append(f"# NEWS AND MATCH UPDATES ({len(articles)} articles)")
    parts.append("CRITICAL: only articles from the last 60 minutes about this specific match count as fresh data.")
    parts.append("")

    for i, art in enumerate(articles, 1):
        age_h = ""
        freshness = ""
        if art.published_at:
            age_s = (now - art.published_at).total_seconds()
            age_h = f", {age_s / 3600:.1f}h ago"
            freshness = " ⚡FRESH" if age_s < 3600 else " (OLD)"
        parts.append(f"## Article {i}{freshness}")
        parts.append(f"Source: {art.source.value} | {art.source_name}{age_h}")
        parts.append(f"Title: {art.title}")
        if art.description:
            parts.append(f"Content: {art.description[:400]}")
        parts.append("")

    parts.append("# YOUR TASK")
    parts.append(
        "Apply all 6 rules. Check for fresh match data first (Rule 1 — immediate gate). "
        "If there is a credible real-time signal that the underdog has a better chance than "
        "the market implies, and the edge exceeds 10pp, recommend BUY_NO. "
        "Otherwise output WAIT or INSUFFICIENT_DATA. "
        "Respond with ONLY the JSON object."
    )
    return "\n".join(parts)


def build_user_prompt(
    market: MarketSnapshot,
    articles: list[NewsArticle],
) -> str:
    """Builds the user message with market info + news."""
    now = datetime.now(timezone.utc)

    parts: list[str] = []
    parts.append("# MARKET")
    parts.append(f"Question: {market.question}")
    if market.description:
        parts.append(f"Description: {market.description[:400]}")
    parts.append(f"Current YES price: {market.yes_price:.4f}")
    parts.append(f"Current NO price: {market.no_price:.4f}")
    parts.append(f"24h volume: ${market.volume_24h_usd:,.0f}")
    parts.append(f"Spread: {market.spread:.4f}")
    if market.end_date:
        ttc = market.time_to_close_hours or 0
        parts.append(f"Time to close: {ttc:.1f} hours")
    if market.category:
        parts.append(f"Category: {market.category}")

    parts.append("")
    parts.append(f"# NEWS ARTICLES ({len(articles)} total)")
    parts.append("Sorted by preliminary_impact_score descending.")
    parts.append("")

    for i, art in enumerate(articles, 1):
        age_h = ""
        if art.published_at:
            age_seconds = (now - art.published_at).total_seconds()
            age_h = f", {age_seconds / 3600:.1f}h ago"
        parts.append(f"## Article {i}")
        parts.append(f"Source: {art.source.value} | {art.source_name}{age_h}")
        parts.append(f"Score: {art.preliminary_impact_score:.1f}/100")
        parts.append(f"Title: {art.title}")
        if art.description:
            desc = art.description[:300]
            parts.append(f"Description: {desc}")
        if art.matched_keywords:
            parts.append(f"Matched keywords: {art.matched_keywords}")
        parts.append("")

    parts.append("# YOUR TASK")
    parts.append(
        "Apply all 7 risk rules from the system prompt. Determine whether "
        "the TRUE probability of YES differs from the current market price by "
        "at least 5pp in either direction, with sufficient confidence to justify "
        "a trade. Consider BUY_NO equally to BUY_YES — high-YES markets "
        "with any bearish evidence are prime NO candidates. "
        "Respond with ONLY the JSON object specified in the system prompt."
    )
    return "\n".join(parts)


# =====================================================
# SentimentAnalyzer
# =====================================================


class SentimentAnalyzer:
    """Analyzes markets + news and produces structured MarketAnalysis objects."""

    def __init__(
        self,
        config: BotConfig,
        client: Optional[LLMClient] = None,
        compound: Optional["CompoundEngine"] = None,
    ) -> None:
        self.config = config
        self.client = client if client is not None else build_llm_client(config)
        self.cfg_llm = config.llm
        self.compound = compound

        # Cache: hash → (timestamp, analysis)
        self._cache: dict[str, tuple[float, MarketAnalysis]] = {}

        self._log = logger.bind(module="sentiment_analyzer")

    # =====================================================
    # Public API
    # =====================================================

    def analyze(
        self,
        market: MarketSnapshot,
        articles: list[NewsArticle],
        force_refresh: bool = False,
    ) -> MarketAnalysis:
        """Analyzes a market given its associated news articles.

        If there are not enough articles or they are too old, returns an
        analysis with recommendation=INSUFFICIENT_DATA WITHOUT calling the LLM,
        UNLESS config.decision.allow_low_info_trades=true (in which case
        analysis with fewer articles is allowed but subsequent sizing will be
        more conservative).
        """
        # 1) Pre-LLM filters (save tokens)
        relevant_articles = self._filter_relevant(articles)

        # Minimum article threshold: depends on whether we allow low-info
        cfg_decision = self.config.decision
        min_articles_normal = 2
        min_articles_low_info = cfg_decision.low_info_min_articles

        is_low_info_mode = (
            cfg_decision.allow_low_info_trades
            and len(relevant_articles) < min_articles_normal
            and len(relevant_articles) >= min_articles_low_info
        )

        if (
            not is_low_info_mode
            and len(relevant_articles) < min_articles_normal
        ):
            return self._make_insufficient_data(
                market,
                articles,
                reason=f"Only {len(relevant_articles)} relevant articles",
            )

        # 2) Cache lookup
        cache_key = self._cache_key(market, relevant_articles)
        if not force_refresh and self.cfg_llm.cache_analysis:
            cached = self._cache.get(cache_key)
            if cached and (time.time() - cached[0]) < self.cfg_llm.cache_ttl_seconds:
                self._log.debug("Cache hit for market {}", market.market_id)
                return cached[1]

        # 3) LLM call
        try:
            analysis = self._call_llm(market, relevant_articles)
            # Mark if analyzed in low-info mode
            if is_low_info_mode:
                analysis.is_low_info = True
                self._log.info(
                    "LOW_INFO analysis of {} (only {} articles)",
                    market.market_id,
                    len(relevant_articles),
                )
        except DailyBudgetExceeded as exc:
            self._log.warning(
                "Analysis skipped due to daily budget limit: {}", exc
            )
            return self._make_insufficient_data(
                market,
                articles,
                reason=f"Daily budget reached: {exc}",
            )
        except CreditsExhausted as exc:
            reset_info = ""
            if exc.reset_at:
                reset_info = f" Reset: {exc.reset_at.isoformat()}"
            elif exc.retry_after_seconds:
                reset_info = f" Retry in {exc.retry_after_seconds}s"
            self._log.error(
                "CREDITS EXHAUSTED — analysis unavailable.{}", reset_info
            )
            return self._make_insufficient_data(
                market,
                articles,
                reason=f"API credits exhausted.{reset_info}",
            )
        except LLMError as exc:
            exc_str = str(exc)
            if "timeout" in exc_str.lower():
                self._log.warning(
                    "Analysis skipped due to timeout (market {}) — Ollama busy",
                    market.market_id,
                )
            else:
                self._log.error(
                    "Analysis failed for market {}: {}",
                    market.market_id,
                    exc,
                )
            return self._make_insufficient_data(
                market,
                articles,
                reason=f"LLM error: {exc}",
            )

        # 4) Post-LLM validation and clipping
        analysis = self._validate(analysis)

        # 5) Cache
        self._cache[cache_key] = (time.time(), analysis)
        return analysis

    def analyze_batch(
        self,
        pairs: list[tuple[MarketSnapshot, list[NewsArticle]]],
        max_workers: int = 1,
    ) -> list[MarketAnalysis]:
        """Analyzes a list of (market, articles) pairs.

        max_workers=1  → serial (original behaviour, safe for Anthropic throttling).
        max_workers>1  → ThreadPoolExecutor; requires Ollama configured with
                         OLLAMA_NUM_PARALLEL>=max_workers for real parallel inference.
                         Always returns one MarketAnalysis per input pair, in order.
        """
        if max_workers <= 1:
            results: list[MarketAnalysis] = []
            for market, articles in pairs:
                try:
                    results.append(self.analyze(market, articles))
                except Exception as exc:
                    self._log.error("Batch analysis failed for {}: {}", market.market_id, exc)
                    results.append(self._make_insufficient_data(market, articles, reason=str(exc)))
            return results

        self._log.info(
            "Parallel analysis: {} markets with {} workers",
            len(pairs),
            max_workers,
        )
        ordered: list[Optional[MarketAnalysis]] = [None] * len(pairs)
        t_batch = time.time()

        def _worker(market, articles, idx: int) -> MarketAnalysis:
            """Wrapper that adds per-worker timing and structured logging."""
            worker = threading.current_thread().name
            t0 = time.time()
            self._log.info(
                "[{}] start  #{} '{}'",
                worker, idx, market.question[:50],
            )
            try:
                result = self.analyze(market, articles)
                elapsed = time.time() - t0
                self._log.info(
                    "[{}] done   #{} '{}' | rec={} conf={} | {:.1f}s",
                    worker, idx, market.question[:40],
                    result.recommendation.value, result.confidence, elapsed,
                )
                return result
            except Exception as exc:
                elapsed = time.time() - t0
                self._log.error(
                    "[{}] ERROR  #{} '{}' (id={}) | {:.1f}s | {}",
                    worker, idx, market.question[:40],
                    market.market_id, elapsed, exc,
                )
                raise

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm-worker") as pool:
            future_to_idx = {
                pool.submit(_worker, market, articles, idx): idx
                for idx, (market, articles) in enumerate(pairs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                market, articles = pairs[idx]
                try:
                    ordered[idx] = future.result()
                except Exception as exc:
                    self._log.error(
                        "Parallel: fallback INSUFFICIENT_DATA for market #{} ({}): {}",
                        idx, market.market_id, exc,
                    )
                    ordered[idx] = self._make_insufficient_data(
                        market, articles, reason=str(exc)
                    )

        results = [r for r in ordered if r is not None]
        elapsed_total = time.time() - t_batch
        with_data = sum(
            1 for r in results
            if r.recommendation.value not in ("INSUFFICIENT_DATA", "WAIT")
        )
        self._log.info(
            "Parallel batch complete: {} markets in {:.1f}s | {} actionable",
            len(results), elapsed_total, with_data,
        )
        return results

    def analyze_sports(
        self,
        market: MarketSnapshot,
        articles: list[NewsArticle],
    ) -> MarketAnalysis:
        """Sports version of analysis: uses SPORTS_SYSTEM_PROMPT and only recommends BUY_NO.

        No cache — live sports markets change constantly.
        """
        try:
            user_prompt = build_sports_user_prompt(market, articles)
            parsed, meta = self.client.complete_json(
                system_prompt=SPORTS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except (DailyBudgetExceeded, CreditsExhausted, LLMError) as exc:
            self._log.warning("Sports LLM failed for {}: {}", market.market_id, exc)
            return self._make_insufficient_data(market, articles, reason=str(exc))

        try:
            recommendation = TradeRecommendation(
                parsed.get("recommendation", "WAIT")
            )
        except ValueError:
            recommendation = TradeRecommendation.WAIT

        # Safety: sports module never opens YES trades
        if recommendation == TradeRecommendation.BUY_YES:
            self._log.warning(
                "Sports LLM returned BUY_YES — forcing WAIT (not applicable in sports)"
            )
            recommendation = TradeRecommendation.WAIT

        consensus = float(parsed.get("consensus_probability_yes", market.yes_price))
        consensus = max(0.0, min(1.0, consensus))

        analysis = MarketAnalysis(
            market_id=market.market_id,
            market_question=market.question,
            market_slug=market.slug,
            yes_token_id=market.yes_token_id,
            no_token_id=market.no_token_id,
            current_yes_price=market.yes_price,
            current_no_price=market.no_price,
            consensus_probability_yes=consensus,
            edge=consensus - market.yes_price,
            confidence=min(int(parsed.get("confidence", 0)), 65),  # enforce hard ceiling
            sentiment_score=float(parsed.get("sentiment_score", 0.0)),
            impact_score=float(parsed.get("impact_score", 0.0)),
            recommendation=recommendation,
            timeframe=Timeframe.IMMEDIATE,
            contradictory_sources=bool(parsed.get("contradictory_sources", False)),
            summary=str(parsed.get("summary", ""))[:500],
            justification=str(parsed.get("justification", ""))[:1000],
            article_ids_analyzed=[a.article_id for a in articles],
            num_articles_analyzed=len(articles),
            llm_model=self.cfg_llm.model,
            llm_input_tokens=meta.get("input_tokens", 0),
            llm_output_tokens=meta.get("output_tokens", 0),
        )
        self._log.info(
            "Sports analysis: {} | YES={:.2f} | conf={} | rec={}",
            market.question[:40],
            market.yes_price,
            analysis.confidence,
            analysis.recommendation.value,
        )
        return analysis

    # =====================================================
    # Pre-LLM filters
    # =====================================================

    def _filter_relevant(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Discards articles that are too old or have no matched_keywords.

        Applies the lesson from the previous turn: if all news is old,
        there is no point asking the LLM for a "fresh" opinion.
        """
        if not articles:
            return []
        now = datetime.now(timezone.utc)
        relevant: list[NewsArticle] = []
        for art in articles:
            if art.published_at:
                age_h = (now - art.published_at).total_seconds() / 3600
                if age_h > MAX_FRESH_AGE_HOURS:
                    continue
            relevant.append(art)
        # Limit to top 10 by score to avoid inflating the prompt
        relevant.sort(key=lambda a: a.preliminary_impact_score, reverse=True)
        return relevant[:10]

    # =====================================================
    # LLM call
    # =====================================================

    def _call_llm(
        self,
        market: MarketSnapshot,
        articles: list[NewsArticle],
    ) -> MarketAnalysis:
        user_prompt = build_user_prompt(market, articles)
        # Inject relevant KB lessons if the compound engine is available
        if self.compound is not None:
            lessons = self.compound.get_relevant_lessons(
                market.question, market.slug
            )
            if lessons:
                user_prompt = user_prompt + "\n\n" + lessons
        parsed, meta = self.client.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Parse JSON to MarketAnalysis with safe defaults
        try:
            recommendation = TradeRecommendation(
                parsed.get("recommendation", "WAIT")
            )
        except ValueError:
            recommendation = TradeRecommendation.WAIT

        try:
            timeframe = Timeframe(parsed.get("timeframe", "UNKNOWN"))
        except ValueError:
            timeframe = Timeframe.UNKNOWN

        consensus = float(parsed.get("consensus_probability_yes", market.yes_price))
        consensus = max(0.0, min(1.0, consensus))

        return MarketAnalysis(
            market_id=market.market_id,
            market_question=market.question,
            market_slug=market.slug,
            yes_token_id=market.yes_token_id,
            no_token_id=market.no_token_id,
            current_yes_price=market.yes_price,
            current_no_price=market.no_price,
            consensus_probability_yes=consensus,
            edge=consensus - market.yes_price,
            confidence=int(parsed.get("confidence", 0)),
            sentiment_score=float(parsed.get("sentiment_score", 0.0)),
            impact_score=float(parsed.get("impact_score", 0.0)),
            recommendation=recommendation,
            timeframe=timeframe,
            contradictory_sources=bool(parsed.get("contradictory_sources", False)),
            summary=str(parsed.get("summary", ""))[:500],
            justification=str(parsed.get("justification", ""))[:1000],
            article_ids_analyzed=[a.article_id for a in articles],
            num_articles_analyzed=len(articles),
            llm_model=self.cfg_llm.model,
            llm_input_tokens=meta.get("input_tokens", 0),
            llm_output_tokens=meta.get("output_tokens", 0),
        )

    # =====================================================
    # Validation
    # =====================================================

    def _validate(self, analysis: MarketAnalysis) -> MarketAnalysis:
        """Applies safety rules to the LLM output.

        Specifically:
        - Clips out-of-range values (Pydantic already validates types but not
          arbitrary float bounds without Field constraints; we are extra careful here).
        - If confidence < configured threshold, downgrade to WAIT.
        - If absolute edge < MIN_EDGE_FOR_TRADE, downgrade to WAIT.
        - If recommendation says BUY_YES but edge is negative (or vice versa),
          that is an internal LLM contradiction → WAIT.
        """
        # Confidence threshold from config
        min_conf = self.cfg_llm.min_confidence_threshold
        rec = analysis.recommendation

        if rec in (TradeRecommendation.BUY_YES, TradeRecommendation.BUY_NO):
            if analysis.confidence < min_conf:
                self._log.info(
                    "Downgrade to WAIT: confidence {} < threshold {}",
                    analysis.confidence,
                    min_conf,
                )
                analysis.recommendation = TradeRecommendation.WAIT
            elif abs(analysis.edge) < MIN_EDGE_FOR_TRADE:
                self._log.info(
                    "Downgrade to WAIT: |edge|={:.3f} < min={:.3f}",
                    abs(analysis.edge),
                    MIN_EDGE_FOR_TRADE,
                )
                analysis.recommendation = TradeRecommendation.WAIT
            elif (
                rec == TradeRecommendation.BUY_YES and analysis.edge < 0
            ) or (
                rec == TradeRecommendation.BUY_NO and analysis.edge > 0
            ):
                self._log.warning(
                    "LLM contradiction: rec={} but edge={:.3f}. → WAIT",
                    rec.value,
                    analysis.edge,
                )
                analysis.recommendation = TradeRecommendation.WAIT

        return analysis

    # =====================================================
    # Helpers
    # =====================================================

    def _make_insufficient_data(
        self,
        market: MarketSnapshot,
        articles: list[NewsArticle],
        reason: str,
    ) -> MarketAnalysis:
        """Builds a 'no data' analysis without calling the LLM."""
        return MarketAnalysis(
            market_id=market.market_id,
            market_question=market.question,
            market_slug=market.slug,
            yes_token_id=market.yes_token_id,
            no_token_id=market.no_token_id,
            current_yes_price=market.yes_price,
            current_no_price=market.no_price,
            consensus_probability_yes=market.yes_price,  # neutral: current price
            edge=0.0,
            confidence=0,
            sentiment_score=0.0,
            impact_score=0.0,
            recommendation=TradeRecommendation.INSUFFICIENT_DATA,
            timeframe=Timeframe.UNKNOWN,
            contradictory_sources=False,
            summary=f"Insufficient data: {reason}",
            justification=reason,
            article_ids_analyzed=[a.article_id for a in articles],
            num_articles_analyzed=len(articles),
            llm_model="(no LLM call)",
        )

    @staticmethod
    def _cache_key(
        market: MarketSnapshot,
        articles: list[NewsArticle],
    ) -> str:
        """Stable hash of (market_id, market_yes_price, sorted article ids)."""
        ids = sorted(a.article_id for a in articles)
        # We include the price (rounded) because we want to re-analyze if the
        # market has moved even if the articles are the same.
        price_bucket = round(market.yes_price, 3)
        payload = f"{market.market_id}|{price_bucket}|{','.join(ids)}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
