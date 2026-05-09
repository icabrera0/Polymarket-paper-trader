"""
Tests del NewsIngestor.

Cubren:
- Score heurístico de impacto (recencia, reputación, keywords, urgencia).
- Deduplicación por URL exacta y por similitud difusa de títulos.
- Resiliencia: fallo de un cliente no impide que el otro contribuya.
- Cliente deshabilitado / ausente.
- Caché por keywords.

No hace red. Inyecta clientes falsos vía dependency injection.

Ejecutar:
    pytest tests/test_news_ingestor.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.models import NewsArticle, NewsSource, _new_article_id
from src.news_ingestor import NewsIngestor


# =====================================================
# Fakes y helpers
# =====================================================


class FakeNewsClient:
    """Implementa el protocolo de cliente de noticias para tests."""

    def __init__(
        self,
        articles: list[NewsArticle] | None = None,
        raise_on_fetch: Exception | None = None,
    ) -> None:
        self.articles = articles or []
        self.raise_on_fetch = raise_on_fetch
        self.fetch_calls = 0

    def fetch_articles(
        self, keywords: list[str], **kwargs: Any
    ) -> list[NewsArticle]:
        self.fetch_calls += 1
        if self.raise_on_fetch is not None:
            raise self.raise_on_fetch
        return list(self.articles)


def make_article(
    title: str = "Spain wins the World Cup",
    url: str | None = None,
    source_name: str = "Reuters",
    source: NewsSource = NewsSource.NEWSAPI,
    description: str = "",
    hours_ago: float = 1.0,
) -> NewsArticle:
    """Helper para crear NewsArticle de test rápidamente."""
    if url is None:
        url = f"https://example.com/{title.lower().replace(' ', '-')}"
    published_at = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return NewsArticle(
        article_id=_new_article_id(url, title),
        source=source,
        source_name=source_name,
        title=title,
        description=description,
        url=url,
        language="en",
        published_at=published_at,
    )


@pytest.fixture
def ingestor(config) -> NewsIngestor:
    """NewsIngestor con clientes vacíos (no hace red)."""
    return NewsIngestor(
        config,
        newsapi_client=FakeNewsClient(),
        gdelt_client=FakeNewsClient(),
    )


# =====================================================
# Scoring
# =====================================================


class TestScoring:
    def test_articulo_breaking_reciente_reuters_score_alto(self, ingestor):
        art = make_article(
            title="BREAKING: Major event happens",
            source_name="Reuters",
            description="Something big about Spain",
            hours_ago=0.5,
        )
        score = ingestor._score(art, ["Spain"])
        # 30 (recencia) + 30 (Reuters) + 10 (1 keyword) + 10 (BREAKING) ≈ 80
        assert score >= 75

    def test_articulo_viejo_fuente_desconocida_score_bajo(self, ingestor):
        art = make_article(
            title="Some unrelated old story",
            source_name="random-blog.xyz",
            description="",
            hours_ago=48.0,  # 2 días → 0 puntos por recencia
        )
        score = ingestor._score(art, ["Spain"])
        # 0 (recencia) + 10 (no-reputable) + 0 (sin match) + 0 (sin urgencia) = 10
        assert score == pytest.approx(10.0)

    def test_match_keywords_acumula_hasta_30(self, ingestor):
        art = make_article(
            title="Trump and Biden meet to discuss Spain",
            description="",
            source_name="random-blog",
            hours_ago=24.0,  # 0 puntos por recencia
        )
        score = ingestor._score(art, ["Trump", "Biden", "Spain"])
        # 0 (recencia) + 10 (no-reputable) + 30 (3 keywords) = 40
        assert score == pytest.approx(40.0)
        assert set(art.matched_keywords) == {"trump", "biden", "spain"}

    def test_match_keywords_se_capa_en_3(self, ingestor):
        art = make_article(
            title="A B C D E mention",
            description="",
            source_name="random-blog",
            hours_ago=24.0,
        )
        score = ingestor._score(art, ["a", "b", "c", "d", "e"])
        # Solo 3 keywords cuentan → 30 puntos máx en esa categoría
        assert score == pytest.approx(40.0)  # 0 + 10 + 30
        assert len(art.matched_keywords) == 3

    def test_recencia_decae_linealmente(self, ingestor):
        art_now = make_article(hours_ago=0.0, source_name="random-blog")
        art_12h = make_article(hours_ago=12.0, source_name="random-blog")
        art_24h = make_article(hours_ago=24.0, source_name="random-blog")
        s0 = ingestor._score(art_now, [])
        s12 = ingestor._score(art_12h, [])
        s24 = ingestor._score(art_24h, [])
        # Diferencia debe ser ~15 puntos cada 12h
        assert s0 - s12 == pytest.approx(15.0, abs=0.5)
        assert s12 - s24 == pytest.approx(15.0, abs=0.5)

    def test_marcador_urgencia_suma_10(self, ingestor):
        normal = make_article(title="Spain wins the cup", source_name="random-blog", hours_ago=24)
        urgent = make_article(title="JUST IN: Spain wins the cup", source_name="random-blog", hours_ago=24)
        s_norm = ingestor._score(normal, [])
        s_urg = ingestor._score(urgent, [])
        assert s_urg - s_norm == pytest.approx(10.0)

    def test_score_capado_en_100(self, ingestor):
        # Receta para superar 100 si no hay capping
        art = make_article(
            title="BREAKING: Trump Biden Spain Election",
            description="",
            source_name="Reuters",
            hours_ago=0.0,
        )
        score = ingestor._score(art, ["Trump", "Biden", "Spain"])
        # 30 + 30 + 30 + 10 = 100
        assert score == pytest.approx(100.0)

    def test_fuente_alta_reputacion_caso_insensitive(self, ingestor):
        for src in ["REUTERS", "reuters", "Reuters", "Reuters Markets"]:
            art = make_article(source_name=src, hours_ago=24.0)
            assert ingestor._is_high_reputation(art.source_name)

    def test_fuente_no_reputada(self, ingestor):
        for src in ["unknown-blog.com", "twitter.com", "random.xyz", ""]:
            art = make_article(source_name=src, hours_ago=24.0)
            assert not ingestor._is_high_reputation(art.source_name)


# =====================================================
# Deduplicación
# =====================================================


class TestDeduplication:
    def test_dedupe_exacto_por_url(self, ingestor):
        url = "https://reuters.com/article-1"
        a1 = make_article(title="Title v1", url=url, hours_ago=1.0)
        a2 = make_article(title="Title v2", url=url, hours_ago=1.0)
        # Forzar scores diferentes
        a1.preliminary_impact_score = 50
        a2.preliminary_impact_score = 80
        result = ingestor._deduplicate([a1, a2])
        assert len(result) == 1
        assert result[0].preliminary_impact_score == 80

    def test_dedupe_difuso_por_titulo(self, ingestor):
        a1 = make_article(
            title="Spain wins the World Cup final 3-1",
            url="https://a.com/1",
        )
        a2 = make_article(
            title="Spain wins the World Cup final 3 to 1",
            url="https://b.com/1",
            source_name="Bloomberg",
        )
        a1.preliminary_impact_score = 60
        a2.preliminary_impact_score = 75  # mejor → debe ser el ganador
        result = ingestor._deduplicate([a1, a2])
        assert len(result) == 1
        assert result[0].url == "https://b.com/1"

    def test_no_dedupe_titulos_distintos(self, ingestor):
        a1 = make_article(title="Spain wins the World Cup", url="https://a.com")
        a2 = make_article(title="Bitcoin reaches new high", url="https://b.com")
        a1.preliminary_impact_score = 50
        a2.preliminary_impact_score = 50
        result = ingestor._deduplicate([a1, a2])
        assert len(result) == 2

    def test_lista_vacia(self, ingestor):
        assert ingestor._deduplicate([]) == []


# =====================================================
# Fetch — orquestación
# =====================================================


class TestFetch:
    def test_combina_articulos_de_ambas_fuentes(self, config):
        newsapi_articles = [
            make_article(title="Article from NewsAPI", url="https://na.com/1")
        ]
        gdelt_articles = [
            make_article(title="Article from GDELT", url="https://gd.com/1", source=NewsSource.GDELT)
        ]
        ingestor = NewsIngestor(
            config,
            newsapi_client=FakeNewsClient(articles=newsapi_articles),
            gdelt_client=FakeNewsClient(articles=gdelt_articles),
        )
        result = ingestor.fetch(["Spain"])
        assert len(result) == 2

    def test_devuelve_ordenado_por_score(self, config):
        # Bloomberg artículo BREAKING reciente debería ganar a un blog viejo
        good = make_article(
            title="BREAKING: Spain wins",
            url="https://bloomberg.com/1",
            source_name="Bloomberg",
            hours_ago=0.5,
        )
        bad = make_article(
            title="Random story",
            url="https://blog.xyz/1",
            source_name="some-blog",
            hours_ago=20.0,
        )
        ingestor = NewsIngestor(
            config,
            newsapi_client=FakeNewsClient(articles=[bad, good]),
            gdelt_client=FakeNewsClient(),
        )
        result = ingestor.fetch(["Spain"])
        assert result[0].url == "https://bloomberg.com/1"
        assert result[0].preliminary_impact_score > result[-1].preliminary_impact_score

    def test_resiliente_si_un_cliente_falla(self, config):
        good = make_article(title="Good article", url="https://ok.com/1")
        bad_client = FakeNewsClient(raise_on_fetch=RuntimeError("simulated outage"))
        good_client = FakeNewsClient(articles=[good])
        ingestor = NewsIngestor(
            config,
            newsapi_client=bad_client,
            gdelt_client=good_client,
        )
        result = ingestor.fetch(["Spain"])
        assert len(result) == 1
        assert result[0].url == "https://ok.com/1"

    def test_keywords_vacios_devuelve_vacio(self, ingestor):
        assert ingestor.fetch([]) == []
        assert ingestor.fetch(["", " "]) == []

    def test_cache_evita_segunda_llamada(self, config):
        client = FakeNewsClient(articles=[make_article(url="https://a.com/1")])
        ingestor = NewsIngestor(
            config,
            newsapi_client=client,
            gdelt_client=FakeNewsClient(),
            cache_ttl_seconds=60.0,
        )
        ingestor.fetch(["Spain"])
        ingestor.fetch(["Spain"])
        ingestor.fetch(["spain"])  # case insensitive → mismo cache key
        assert client.fetch_calls == 1

    def test_force_refresh_rompe_cache(self, config):
        client = FakeNewsClient(articles=[make_article(url="https://a.com/1")])
        ingestor = NewsIngestor(
            config,
            newsapi_client=client,
            gdelt_client=FakeNewsClient(),
            cache_ttl_seconds=60.0,
        )
        ingestor.fetch(["Spain"])
        ingestor.fetch(["Spain"], force_refresh=True)
        assert client.fetch_calls == 2

    def test_max_articles_recorta_resultado(self, config):
        # Usar titles claramente distintos para que la dedup difusa no los
        # colapse (lección aprendida: "Story 0" vs "Story 1" parecen duplicados
        # para token_set_ratio porque comparten la mayor parte de tokens).
        distinct_titles = [
            "Spain wins football championship final",
            "Bitcoin reaches new all-time price record",
            "Federal Reserve announces quarter point rate cut",
            "European Union proposes major trade reform",
            "OpenAI releases breakthrough language model",
            "Climate summit reaches historic global agreement",
            "Apple unveils next generation augmented reality device",
            "Tesla reports record quarterly vehicle deliveries",
            "Election polls show tight presidential race",
            "Energy prices surge amid Middle East tensions",
            "Healthcare reform bill passes legislative vote",
            "NASA confirms successful Mars rover landing",
            "Quantum computing achieves milestone breakthrough",
            "Auto industry pivots aggressively toward electric",
            "Banking sector posts strong quarterly earnings",
            "Olympic committee announces host city decision",
            "Cybersecurity breach affects financial institutions",
            "Diplomatic talks resume between rival nations",
            "Streaming wars intensify with new platform launch",
            "Pharmaceutical company gains drug approval",
        ]
        articles = [
            make_article(url=f"https://a.com/{i}", title=t)
            for i, t in enumerate(distinct_titles)
        ]
        ingestor = NewsIngestor(
            config,
            newsapi_client=FakeNewsClient(articles=articles),
            gdelt_client=FakeNewsClient(),
        )
        result = ingestor.fetch(["championship"], max_articles=5)
        assert len(result) == 5

    def test_funciona_sin_clientes(self, config):
        ingestor = NewsIngestor(
            config,
            newsapi_client=None,
            gdelt_client=None,
        )
        # NOTA: si la newsapi_key está en config, NewsIngestor podría
        # auto-instanciar el cliente real. En conftest tenemos `newsapi_key="test-key"`
        # así que necesitamos forzar deshabilitado para este test.
        ingestor.newsapi_client = None
        ingestor.gdelt_client = None
        ingestor.telegram_client = None
        result = ingestor.fetch(["anything"])
        assert result == []

    def test_telegram_client_se_integra(self, config):
        """El ingestor debe combinar artículos de Telegram con los demás."""
        from src.models import NewsSource

        tg_articles = [
            make_article(
                title="Telegram breaking news",
                url="https://t.me/testch/123",
                source=NewsSource.TELEGRAM,
                source_name="@testchannel",
            )
        ]
        gdelt_articles = [
            make_article(
                title="GDELT news",
                url="https://example.com/news",
                source=NewsSource.GDELT,
            )
        ]
        ingestor = NewsIngestor(
            config,
            newsapi_client=None,
            gdelt_client=FakeNewsClient(articles=gdelt_articles),
            telegram_client=FakeNewsClient(articles=tg_articles),
        )
        # Bloquea el cliente auto-instanciado por la config para no tocar red
        ingestor.newsapi_client = None
        result = ingestor.fetch(["news"])
        sources = {a.source for a in result}
        assert NewsSource.TELEGRAM in sources
        assert NewsSource.GDELT in sources

    def test_telegram_client_falla_no_rompe_otros(self, config):
        """Si Telegram falla, NewsAPI/GDELT siguen contribuyendo."""
        good_article = make_article(title="Survives", url="https://ok.com/1")
        ingestor = NewsIngestor(
            config,
            newsapi_client=FakeNewsClient(articles=[good_article]),
            gdelt_client=FakeNewsClient(),
            telegram_client=FakeNewsClient(
                raise_on_fetch=RuntimeError("telegram down")
            ),
        )
        result = ingestor.fetch(["Survives"])
        assert len(result) == 1
        assert result[0].url == "https://ok.com/1"
