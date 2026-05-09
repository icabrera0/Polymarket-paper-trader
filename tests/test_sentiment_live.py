"""
Prueba en vivo del SENTIMENT_ANALYZER con Claude API real.

Ejecuta el pipeline completo:
1. Escanea mercados de Polymarket
2. Selecciona los TOP N por volumen (N=3 por defecto, configurable)
3. Para cada mercado: extrae keywords y busca noticias
4. Pasa cada (mercado, noticias) a Claude para análisis cuantitativo
5. Imprime el JSON resultado por mercado y los tokens consumidos

CONSUMO DE TOKENS:
Esta prueba sí gasta tokens de tu cuenta Anthropic. Estimación con
claude-sonnet-4-6 a precios actuales (~$3/M input, $15/M output):
- Input: ~3000 tokens/mercado × 3 mercados = ~9000 tokens (~$0.027)
- Output: ~300 tokens/mercado × 3 mercados = ~900 tokens (~$0.013)
- Total: < $0.05 por ejecución

Ejecutar:
    python scripts/test_sentiment_live.py

Requiere ANTHROPIC_API_KEY en .env.
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

# Reutilizamos el extractor de keywords del otro script
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from test_live_integration import extract_keywords  # noqa: E402

NUM_MARKETS_TO_ANALYZE = 3


def color_for_recommendation(rec: TradeRecommendation) -> str:
    """Devuelve un emoji indicativo para el terminal."""
    return {
        TradeRecommendation.COMPRAR_YES: "🟢 BUY YES",
        TradeRecommendation.COMPRAR_NO: "🔴 BUY NO",
        TradeRecommendation.ESPERAR: "🟡 HOLD",
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
        print("ERROR: ANTHROPIC_API_KEY no configurada en .env")
        sys.exit(1)

    # ---------- 1. Escaneo ----------
    print_section("PASO 1 — Escaneo de mercados")
    scanner = MarketScanner(config)
    markets = scanner.scan(force_refresh=True)
    print(f"→ {len(markets)} mercados operables")

    if not markets:
        print("No hay mercados que pasen los filtros. Aborto.")
        return

    top = markets[:NUM_MARKETS_TO_ANALYZE]
    print(f"\nAnalizando los TOP {len(top)} por volumen:")
    for i, m in enumerate(top, 1):
        print(f"  {i}. {m.question[:70]}")
        print(f"     YES={m.yes_price:.3f} | vol24h=${m.volume_24h_usd:,.0f}")

    # ---------- 2. Noticias por mercado ----------
    print_section("PASO 2 — Ingesta de noticias por mercado")
    ingestor = NewsIngestor(config)

    market_news: dict[str, list] = {}
    for m in top:
        kws = extract_keywords(m.question)
        articles = ingestor.fetch(kws, max_articles=10, force_refresh=True)
        market_news[m.market_id] = articles
        print(f"  '{m.question[:50]}...': {len(articles)} artículos")

    # ---------- 3. Análisis con Claude ----------
    print_section("PASO 3 — Análisis con Claude")
    analyzer = SentimentAnalyzer(config)

    for i, m in enumerate(top, 1):
        articles = market_news.get(m.market_id, [])
        print(f"\n[{i}/{len(top)}] Analizando: {m.question[:60]}")
        print(f"        Precio actual YES: {m.yes_price:.3f}")
        print(f"        Artículos disponibles: {len(articles)}")

        analysis = analyzer.analyze(m, articles, force_refresh=True)

        print(f"\n  ┌─ RESULTADO ─────────────────────────────────────")
        print(f"  │ Recomendación:        {color_for_recommendation(analysis.recommendation)}")
        print(f"  │ Probabilidad YES:     {analysis.consensus_probability_yes:.3f}")
        print(f"  │ Edge sobre precio:    {analysis.edge:+.3f} ({analysis.edge*100:+.1f}pp)")
        print(f"  │ Confianza:            {analysis.confidence}/100")
        print(f"  │ Sentiment:            {analysis.sentiment_score:+.2f}")
        print(f"  │ Impact:               {analysis.impact_score:.0f}/100")
        print(f"  │ Timeframe:            {analysis.timeframe.value}")
        print(f"  │ Fuentes contradicen:  {analysis.contradictory_sources}")
        print(f"  │ Tokens (in/out):      {analysis.llm_input_tokens} / {analysis.llm_output_tokens}")
        print(f"  ├─ Resumen ──────────────────────────────────────")
        print(f"  │ {analysis.summary}")
        print(f"  ├─ Justificación ────────────────────────────────")
        print(f"  │ {analysis.justification[:300]}")
        print(f"  └─────────────────────────────────────────────────")

    # ---------- 4. Resumen de costes ----------
    print_section("PASO 4 — Coste total de la ejecución")
    print(f"  Llamadas al LLM:     {analyzer.client.total_calls}")
    print(f"  Tokens INPUT:        {analyzer.client.total_input_tokens:,}")
    print(f"  Tokens OUTPUT:       {analyzer.client.total_output_tokens:,}")
    # Estimación grosera de coste con tarifas Sonnet
    cost_in = analyzer.client.total_input_tokens / 1_000_000 * 3.0
    cost_out = analyzer.client.total_output_tokens / 1_000_000 * 15.0
    print(f"  Coste estimado:      ~${cost_in + cost_out:.4f} USD")
    print()


if __name__ == "__main__":
    main()
