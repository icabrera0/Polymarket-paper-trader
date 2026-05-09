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

# (NUM_MARKETS_TO_ANALYZE viene ahora de config.decision.markets_to_analyze_per_cycle)


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

    # Verificar requisitos según provider
    if config.llm.provider == "anthropic":
        if not config.anthropic_api_key:
            print("ERROR: provider='anthropic' pero falta ANTHROPIC_API_KEY en .env")
            sys.exit(1)
    elif config.llm.provider == "ollama":
        # Verificación rápida del servidor antes de gastar tiempo
        from src.llm_client import OllamaClient, OllamaUnavailable
        try:
            OllamaClient(config).verify_setup()
        except OllamaUnavailable as exc:
            print(f"ERROR Ollama: {exc}")
            print("Ejecuta primero: python scripts/setup_ollama.py")
            sys.exit(1)
    else:
        print(f"ERROR: provider desconocido: {config.llm.provider}")
        sys.exit(1)

    print(f"Provider LLM: {config.llm.provider}")
    print(f"Modelo:       {config.llm.model}")
    print()

    # ---------- 1. Escaneo ----------
    print_section("PASO 1 — Escaneo de mercados")
    scanner = MarketScanner(config)
    markets = scanner.scan(force_refresh=True)
    print(f"→ {len(markets)} mercados operables")

    if not markets:
        print("No hay mercados que pasen los filtros. Aborto.")
        return

    top = scanner.rank_for_analysis(
        markets,
        category_boost=config.decision.category_priority_boost,
        top_n=config.decision.markets_to_analyze_per_cycle,
    )
    print(f"\nAnalizando los TOP {len(top)} (ranking con boost de categoría):")
    for i, m in enumerate(top, 1):
        print(f"  {i:2}. [{m.category or '?':<12}] {m.question[:55]}")
        print(f"      YES={m.yes_price:.3f} | vol24h=${m.volume_24h_usd:,.0f}")

    # ---------- 2. Noticias por mercado ----------
    print_section("PASO 2 — Ingesta de noticias por mercado")
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
        print(f"  '{m.question[:50]}...': {len(articles)} artículos")

    # ---------- 3. Análisis con LLM ----------
    provider_label = {
        "anthropic": "Claude",
        "ollama": f"Ollama ({config.llm.model})",
    }.get(config.llm.provider, config.llm.provider)
    print_section(f"PASO 3 — Análisis con {provider_label}")
    analyzer = SentimentAnalyzer(config)

    for i, m in enumerate(top, 1):
        articles = market_news.get(m.market_id, [])
        print(f"\n[{i}/{len(top)}] Analizando: {m.question[:60]}")
        print(f"        Precio actual YES: {m.yes_price:.3f}")
        print(f"        Artículos disponibles: {len(articles)}")

        analysis = analyzer.analyze(m, articles, force_refresh=True)

        print(f"\n  ┌─ RESULTADO ─────────────────────────────────────")
        low_info_tag = " (LOW_INFO)" if analysis.is_low_info else ""
        print(f"  │ Recomendación:        {color_for_recommendation(analysis.recommendation)}{low_info_tag}")
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

    # ---------- 4. Resumen ----------
    print_section("PASO 4 — Resumen de la ejecución")
    print(f"  Provider:            {config.llm.provider}")
    print(f"  Modelo:              {config.llm.model}")
    print(f"  Llamadas al LLM:     {analyzer.client.total_calls}")
    print(f"  Tokens INPUT:        {analyzer.client.total_input_tokens:,}")
    print(f"  Tokens OUTPUT:       {analyzer.client.total_output_tokens:,}")

    if config.llm.provider == "anthropic":
        spent = analyzer.client.daily_spend_usd
        limit = config.llm.daily_spend_limit_usd
        print(f"  Coste esta ejecución: ~${spent:.4f} USD")
        if limit > 0:
            remaining = analyzer.client.daily_budget_remaining_usd
            print(f"  Presupuesto diario:   ${limit:.2f} USD (queda ${remaining:.4f})")
        print()
        print("  Recordatorio: este coste sale de tus créditos pre-cargados en")
        print("  console.anthropic.com (NO de tu suscripción Claude Pro/Max).")
    else:
        print(f"  Coste:               $0.00 (Ollama es local y gratis)")
    print()


if __name__ == "__main__":
    main()
