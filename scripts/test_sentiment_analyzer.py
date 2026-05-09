"""
Tests del SentimentAnalyzer.

Cubren:
- Bypass del LLM cuando no hay noticias suficientes (INSUFFICIENT_DATA).
- Filtrado de noticias viejas (>48h) antes del LLM.
- Llamada al LLM con prompt bien formado.
- Parseo de la respuesta JSON con valores por defecto seguros.
- Validación post-LLM (downgrade a ESPERAR por baja confianza, edge bajo,
  contradicción interna).
- Cache (no llama dos veces por el mismo input).
- Extracción de JSON con prefacio textual.

No usa la API real de Anthropic. FakeAnthropicClient devuelve respuestas
predefinidas.

Ejecutar:
    pytest tests/test_sentiment_analyzer.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest

from src.anthropic_client import AnthropicClient
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
# Fakes y helpers
# =====================================================


class FakeAnthropicClient:
    """Cumple la interfaz mínima de AnthropicClient para tests."""

    def __init__(self, json_response: Optional[dict[str, Any]] = None) -> None:
        self.json_response = json_response or {
            "consensus_probability_yes": 0.55,
            "confidence": 70,
            "sentiment_score": 0.3,
            "impact_score": 60.0,
            "recommendation": "COMPRAR_YES",
            "timeframe": "HORAS",
            "contradictory_sources": False,
            "summary": "Fake summary",
            "justification": "Fake justification",
        }
        self.calls: list[dict[str, Any]] = []
        self.raise_on_call: Optional[Exception] = None

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
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
        }
        return self.json_response, meta


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
    """SentimentAnalyzer con FakeAnthropicClient inyectado."""
    return SentimentAnalyzer(config, client=FakeAnthropicClient())


# =====================================================
# Bypass del LLM (no hay datos)
# =====================================================


class TestInsufficientData:
    def test_sin_noticias_devuelve_insufficient_data(self, analyzer):
        market = make_market()
        result = analyzer.analyze(market, [])
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA
        assert result.confidence == 0
        assert result.consensus_probability_yes == market.yes_price
        # No debe haber llamado al LLM
        assert analyzer.client.calls == []

    def test_una_sola_noticia_devuelve_insufficient_data(self, analyzer):
        market = make_market()
        result = analyzer.analyze(market, [make_article()])
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA
        assert analyzer.client.calls == []

    def test_solo_noticias_viejas_devuelve_insufficient_data(self, analyzer):
        market = make_market()
        old_articles = [make_article(hours_ago=72), make_article(hours_ago=100)]
        result = analyzer.analyze(market, old_articles)
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA
        assert analyzer.client.calls == []


# =====================================================
# Filtrado pre-LLM
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
        # Debe haber llamado al LLM con solo las 3 recientes
        assert len(analyzer.client.calls) == 1
        prompt = analyzer.client.calls[0]["user_prompt"]
        assert "Recent" in prompt
        # Las viejas no deberían aparecer
        assert "Old 0" not in prompt

    def test_top_10_por_score(self, analyzer):
        market = make_market()
        articles = [
            make_article(title=f"News {i}", score=float(i), hours_ago=1.0)
            for i in range(20)
        ]
        analyzer.analyze(market, articles)
        prompt = analyzer.client.calls[0]["user_prompt"]
        # News 19 (mejor score) debe estar; News 0 (peor) NO debe estar
        assert "News 19" in prompt
        assert "News 0" not in prompt


# =====================================================
# Llamada al LLM y parseo
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
            "recommendation": "COMPRAR_YES",
            "timeframe": "INMEDIATO",
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
        assert result.recommendation == TradeRecommendation.COMPRAR_YES
        assert result.timeframe == Timeframe.INMEDIATO
        assert result.summary == "Good news for YES"
        assert result.num_articles_analyzed == 2

    def test_consensus_fuera_de_rango_se_clipa(self, config):
        client = FakeAnthropicClient(json_response={
            "consensus_probability_yes": 1.5,  # fuera de [0, 1]
            "confidence": 70,
            "sentiment_score": 0.3,
            "impact_score": 60,
            "recommendation": "COMPRAR_YES",
            "timeframe": "HORAS",
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
            "recommendation": "INVENTADO",  # no existe
            "timeframe": "HORAS",
            "contradictory_sources": False,
            "summary": "...",
            "justification": "...",
        })
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(), [make_article(), make_article(title="b")]
        )
        assert result.recommendation == TradeRecommendation.ESPERAR


# =====================================================
# Validación post-LLM
# =====================================================


class TestValidation:
    def _llm_with(self, **overrides) -> dict[str, Any]:
        base = {
            "consensus_probability_yes": 0.55,
            "confidence": 70,
            "sentiment_score": 0.3,
            "impact_score": 60.0,
            "recommendation": "COMPRAR_YES",
            "timeframe": "HORAS",
            "contradictory_sources": False,
            "summary": "...",
            "justification": "...",
        }
        base.update(overrides)
        return base

    def test_downgrade_si_confidence_baja(self, config):
        # min_confidence_threshold del conftest es 60
        client = FakeAnthropicClient(self._llm_with(confidence=40))
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.30),
            [make_article(), make_article(title="b")],
        )
        # consensus 0.55, price 0.30, edge 0.25 (suficiente) pero conf 40 baja
        assert result.recommendation == TradeRecommendation.ESPERAR

    def test_downgrade_si_edge_pequeno(self, config):
        # consensus 0.42, price 0.40 → edge 0.02 (< 0.05 mínimo)
        client = FakeAnthropicClient(self._llm_with(
            consensus_probability_yes=0.42, confidence=80,
        ))
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(), make_article(title="b")],
        )
        assert result.recommendation == TradeRecommendation.ESPERAR

    def test_downgrade_si_recomendacion_contradice_edge(self, config):
        # LLM dice COMPRAR_YES pero consensus < precio (edge negativo)
        client = FakeAnthropicClient(self._llm_with(
            consensus_probability_yes=0.20,  # edge = 0.20 - 0.40 = -0.20
            confidence=80,
            recommendation="COMPRAR_YES",
        ))
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(), make_article(title="b")],
        )
        assert result.recommendation == TradeRecommendation.ESPERAR

    def test_compra_yes_se_mantiene_si_todo_ok(self, config):
        client = FakeAnthropicClient(self._llm_with(
            consensus_probability_yes=0.60,  # edge = 0.20 (positivo)
            confidence=85,
            recommendation="COMPRAR_YES",
        ))
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.40),
            [make_article(), make_article(title="b")],
        )
        assert result.recommendation == TradeRecommendation.COMPRAR_YES

    def test_compra_no_se_mantiene_si_todo_ok(self, config):
        client = FakeAnthropicClient(self._llm_with(
            consensus_probability_yes=0.20,  # edge = -0.40 (NO infravalorado)
            confidence=85,
            recommendation="COMPRAR_NO",
        ))
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(yes_price=0.60),
            [make_article(), make_article(title="b")],
        )
        assert result.recommendation == TradeRecommendation.COMPRAR_NO


# =====================================================
# Cache
# =====================================================


class TestCache:
    def test_cache_evita_segunda_llamada(self, analyzer):
        market = make_market()
        articles = [make_article(), make_article(title="b")]
        analyzer.analyze(market, articles)
        analyzer.analyze(market, articles)
        analyzer.analyze(market, articles)
        assert len(analyzer.client.calls) == 1

    def test_force_refresh_rompe_cache(self, analyzer):
        market = make_market()
        articles = [make_article(), make_article(title="b")]
        analyzer.analyze(market, articles)
        analyzer.analyze(market, articles, force_refresh=True)
        assert len(analyzer.client.calls) == 2

    def test_cambio_de_precio_invalida_cache(self, analyzer):
        articles = [make_article(), make_article(title="b")]
        analyzer.analyze(make_market(yes_price=0.40), articles)
        # Mismo mercado pero precio movido > redondeo de 3 decimales
        analyzer.analyze(make_market(yes_price=0.50), articles)
        assert len(analyzer.client.calls) == 2

    def test_cambio_de_articulos_invalida_cache(self, analyzer):
        market = make_market()
        analyzer.analyze(market, [make_article(title="a"), make_article(title="b")])
        analyzer.analyze(market, [make_article(title="a"), make_article(title="c")])
        assert len(analyzer.client.calls) == 2


# =====================================================
# Resiliencia
# =====================================================


class TestResilience:
    def test_error_del_llm_devuelve_insufficient(self, config):
        from src.anthropic_client import AnthropicError

        client = FakeAnthropicClient()
        client.raise_on_call = AnthropicError("API down")
        analyzer = SentimentAnalyzer(config, client=client)
        result = analyzer.analyze(
            make_market(), [make_article(), make_article(title="b")]
        )
        assert result.recommendation == TradeRecommendation.INSUFFICIENT_DATA
        assert "API down" in result.justification


# =====================================================
# JSON extractor (utility)
# =====================================================


class TestJsonExtractor:
    def test_json_puro(self):
        text = '{"a": 1, "b": "x"}'
        assert AnthropicClient.extract_json(text) == {"a": 1, "b": "x"}

    def test_json_con_prefacio(self):
        text = 'Here is the analysis:\n{"a": 1}\nAnything else?'
        assert AnthropicClient.extract_json(text) == {"a": 1}

    def test_json_en_bloque_markdown(self):
        text = "Sure!\n```json\n{\"a\": 1}\n```\n"
        assert AnthropicClient.extract_json(text) == {"a": 1}

    def test_json_anidado_balanceado(self):
        text = 'preface {"outer": {"inner": [1, 2]}} suffix'
        result = AnthropicClient.extract_json(text)
        assert result == {"outer": {"inner": [1, 2]}}

    def test_sin_json_devuelve_none(self):
        assert AnthropicClient.extract_json("no JSON here") is None
        assert AnthropicClient.extract_json("") is None

    def test_json_invalido_devuelve_none(self):
        assert AnthropicClient.extract_json("{not valid json}") is None
