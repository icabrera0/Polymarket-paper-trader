"""
Prueba de integración completa: MarketScanner + NewsIngestor.

Hace todo el pipeline de la fase de descubrimiento:

1. Conecta a Polymarket y obtiene los top mercados por volumen 24h.
2. Extrae automáticamente keywords significativos de las preguntas.
3. Busca noticias recientes sobre esos keywords en las fuentes configuradas.
4. Muestra todo correlacionado: cada mercado con las noticias que lo afectan.

Útil para validar la cadena completa antes de meter Claude (módulo siguiente).

Ejecutar:
    python scripts/test_live_integration.py

Las API keys se leen de .env. Si no tienes ninguna, deja al menos GDELT
habilitado en config/settings.yaml (no requiere credenciales).
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


# Stopwords mínimos para que las preguntas tipo "Will X happen by Y?" no
# contaminen la búsqueda con palabras irrelevantes.
STOPWORDS = {
    "will", "the", "a", "an", "is", "are", "be", "by", "of", "in", "on",
    "at", "to", "for", "and", "or", "if", "than", "more", "less", "this",
    "that", "before", "after", "any", "all", "with", "from", "into", "as",
    "have", "has", "had", "win", "wins", "won", "do", "does", "did",
    "can", "could", "should", "would", "may", "might", "first", "next",
    "year", "month", "week", "day", "much", "many", "make", "makes",
    "made", "election", "vote",  # demasiado genéricos en mercados políticos
}


def extract_keywords(question: str, max_kw: int = 4) -> list[str]:
    """Saca palabras clave de una pregunta de mercado.

    Heurística simple:
    - Palabras de 4+ letras
    - Sin stopwords
    - Prioriza las que empiezan con mayúscula (entidades nombradas)
    """
    # Mantenemos la capitalización original para detectar entidades
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9'-]{3,}\b", question)
    # Separar entidades (Capitalized) de comunes
    entities = []
    common = []
    for w in words:
        if w.lower() in STOPWORDS:
            continue
        if w[0].isupper():
            entities.append(w)
        else:
            common.append(w.lower())
    # Mantener orden, dedup, entidades primero
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
        print(f"   Cierra en {ttc:.0f}h ({m.end_date.strftime('%Y-%m-%d %H:%M UTC')})")


def print_article(art: NewsArticle, indent: str = "      ") -> None:
    title = art.title[:90] + ("..." if len(art.title) > 90 else "")
    print(f"{indent}[{art.preliminary_impact_score:5.1f}] {title}")
    src_name = art.source_name or "(sin nombre)"
    print(f"{indent}        {art.source.value:<8} | {src_name}")
    if art.matched_keywords:
        print(f"{indent}        matched: {art.matched_keywords}")


def main() -> None:
    config = load_config()

    # ---------- Fase 1: mercados ----------
    print_section("PASO 1 — Escaneo de mercados Polymarket")
    scanner = MarketScanner(config)
    print("Conectando a Gamma API y filtrando mercados operables...")
    markets = scanner.scan(force_refresh=True)
    print(f"\n→ {len(markets)} mercados pasan los filtros.")

    if not markets:
        f = config.market_filters
        print("\nFiltros aplicados:")
        print(f"  - Volumen 24h mínimo: ${f.min_volume_24h_usd:,.0f}")
        print(f"  - Spread máximo: {f.max_spread_cents}")
        print(f"  - Tiempo a cierre: {f.min_time_to_close_hours}h - "
              f"{f.max_time_to_close_days} días")
        print("\nPrueba a relajar los filtros en config/settings.yaml.")
        return

    top_markets = markets[:5]
    print(f"\nTop {len(top_markets)} por volumen:")
    for i, m in enumerate(top_markets, 1):
        print_market(i, m)

    # ---------- Fase 2: keywords ----------
    print_section("PASO 2 — Extracción de keywords")
    market_keywords: dict[str, list[str]] = {}
    all_keywords_set: set[str] = set()
    for m in top_markets:
        kws = extract_keywords(m.question)
        market_keywords[m.market_id] = kws
        all_keywords_set.update(kws)
        print(f"\n  {m.question[:55]}...")
        print(f"  → {kws}")

    all_keywords = sorted(all_keywords_set)
    print(f"\n→ Total keywords únicos: {len(all_keywords)}")

    # ---------- Fase 3: noticias globales ----------
    print_section("PASO 3 — Ingesta de noticias")
    ingestor = NewsIngestor(config)
    sources_active = []
    if ingestor.newsapi_client is not None: sources_active.append("NewsAPI")
    if ingestor.gdelt_client is not None: sources_active.append("GDELT")
    if ingestor.telegram_client is not None: sources_active.append("Telegram")
    print(f"Fuentes activas: {sources_active or 'NINGUNA'}")

    if not sources_active:
        print("\n⚠ Ninguna fuente activa. Activa al menos una en settings.yaml")
        print("  (gdelt es la opción más fácil — no requiere credenciales)")
        return

    print(f"\nBuscando noticias para {len(all_keywords)} keywords (timespan=24h)...")
    # Override del timespan para tener mejor probabilidad de encontrar matches
    # en una prueba puntual. En producción usa el del config.
    if ingestor.gdelt_client is not None:
        # Forzar timespan más amplio para esta prueba puntual
        original_fetch = ingestor.gdelt_client.fetch_articles
        def wrapped_fetch(keywords, **kwargs):
            kwargs.setdefault("timespan", "24h")
            return original_fetch(keywords, **kwargs)
        ingestor.gdelt_client.fetch_articles = wrapped_fetch  # type: ignore

    articles = ingestor.fetch(all_keywords[:15], max_articles=20, force_refresh=True)

    if not articles:
        print("\nNo se encontraron noticias. Posibles causas:")
        print("  - Los mercados con más volumen ahora son sobre eventos muy")
        print("    nicho (deportes, esports) que no salen en prensa generalista.")
        print("  - GDELT necesita actualizar sus indexes (~15 min de delay).")
        print("  - Tus keywords son demasiado específicos (ej: 'Lehecka').")
        print("  - Las API keys de NewsAPI son inválidas.")
        print("\nPrueba a:")
        print("  - Filtrar por categoría 'Politics' en config (más cobertura).")
        print("  - Activar Telegram con canales generalistas como @bbcbreaking.")
        print("  - Ampliar timespan en config a '7d' temporalmente.")
        return

    print(f"\n→ {len(articles)} artículos tras dedup, ordenados por score:")
    for art in articles[:10]:
        print_article(art, indent="  ")

    # ---------- Fase 4: correlación ----------
    print_section("PASO 4 — Correlación mercado ↔ noticias")
    for m in top_markets:
        kws = market_keywords[m.market_id]
        if not kws:
            continue
        # Filtrar noticias que matchean al menos un keyword de este mercado
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
    print("  Pipeline completo. El siguiente módulo (SENTIMENT_ANALYZER)")
    print("  pasará estas correlaciones a Claude para análisis cuantitativo.")
    print("═" * 78)


if __name__ == "__main__":
    main()
