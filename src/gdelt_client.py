"""
Cliente GDELT 2.0 DOC API (https://api.gdeltproject.org/api/v2/doc/doc).

Hablamos directamente con la API HTTP de GDELT en lugar de usar la librería
`gdeltdoc`: es más simple, más predecible y no añade una dependencia que
puede romperse con cambios upstream. La GDELT DOC API:

- Es pública, no necesita API key.
- Devuelve JSON con artículos vistos en los últimos N minutos/horas/días.
- Soporta queries booleanas y filtros por idioma/país/dominio.
- Es muy rápida (~ms) pero no devuelve descripción/contenido, solo URL+título.

Notas operativas (lecciones aprendidas en pruebas):
- GDELT rechaza timespans menores a ~1h con "Timespan is too short". Si
  configuras 15min, lo elevamos automáticamente a 1h.
- GDELT requiere PARÉNTESIS alrededor de queries con OR de 3+ términos:
  CORRECTO: ("trump" OR "biden" OR "spain")
  INCORRECTO: "trump" OR "biden" OR "spain"  (devuelve error textual)
- GDELT tiene rate limits agresivos: lanzamos 429 si pegamos batches sin
  sleep. Esperamos GDELT_BATCH_DELAY_SECONDS entre cada batch.
- GDELT acepta hasta ~5-7 términos OR por query antes de empezar a fallar.
  Hacemos batching cuando hay más keywords.
- Una query sin resultados es una respuesta vacía válida (no un error).

Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config_loader import BotConfig
from src.models import NewsArticle, NewsSource, _new_article_id

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Mínimo aceptado por GDELT en la práctica
MIN_TIMESPAN = "1h"
KEYWORDS_PER_QUERY = 5
GDELT_BATCH_DELAY_SECONDS = 1.5         # Para evitar HTTP 429
GDELT_RATE_LIMIT_BACKOFF = 5.0          # Si nos llega un 429, esperamos esto extra


class GdeltApiError(Exception):
    """Error comunicándose con GDELT."""


class GdeltClient:
    """Cliente GDELT que devuelve NewsArticle."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.cfg = config.news.gdelt
        self.timeout = config.polymarket.request_timeout_seconds  # reutilizamos

        self._log = logger.bind(module="gdelt_client")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": f"PolymarketBot/{config.app.version}",
                "Accept": "application/json",
            }
        )

    # =====================================================
    # API pública
    # =====================================================

    def fetch_articles(
        self,
        keywords: list[str],
        max_results: Optional[int] = None,
        timespan: Optional[str] = None,
    ) -> list[NewsArticle]:
        """Busca artículos vistos por GDELT que mencionen los keywords.

        Si hay más de KEYWORDS_PER_QUERY keywords, hace varias queries en lote
        y une los resultados (deduplicados por URL). Espera entre batches para
        respetar el rate limit de GDELT.
        """
        if not keywords:
            return []

        max_results = max_results or self.cfg.max_records
        timespan = self._normalize_timespan(timespan or self.cfg.timespan)

        clean_kws = [k.strip() for k in keywords if k.strip()]
        if not clean_kws:
            return []

        # Lotes de KEYWORDS_PER_QUERY
        batches = [
            clean_kws[i : i + KEYWORDS_PER_QUERY]
            for i in range(0, len(clean_kws), KEYWORDS_PER_QUERY)
        ]

        all_articles: list[NewsArticle] = []
        seen_urls: set[str] = set()
        for i, batch in enumerate(batches):
            # Sleep entre batches (no antes del primero)
            if i > 0:
                time.sleep(GDELT_BATCH_DELAY_SECONDS)

            # Query con paréntesis: GDELT lo exige cuando hay 3+ términos OR.
            # Con 1 solo término no hace falta pero no estorba.
            quoted = [f'"{k}"' for k in batch]
            if len(quoted) == 1:
                query = quoted[0]
            else:
                query = "(" + " OR ".join(quoted) + ")"

            try:
                raw_articles = self._call(
                    query=query,
                    timespan=timespan,
                    max_records=max_results,
                )
            except GdeltApiError as exc:
                # Si fue 429, esperamos extra y reintentamos UNA vez más
                if "429" in str(exc):
                    self._log.warning(
                        "GDELT rate limited en batch {}, esperando {}s y "
                        "reintentando una vez",
                        i + 1,
                        GDELT_RATE_LIMIT_BACKOFF,
                    )
                    time.sleep(GDELT_RATE_LIMIT_BACKOFF)
                    try:
                        raw_articles = self._call(
                            query=query,
                            timespan=timespan,
                            max_records=max_results,
                        )
                    except GdeltApiError as exc2:
                        self._log.warning(
                            "GDELT batch {} falló tras reintento: {}",
                            i + 1,
                            exc2,
                        )
                        continue
                else:
                    self._log.warning(
                        "GDELT batch {} falló ({}): {}",
                        i + 1,
                        batch,
                        exc,
                    )
                    continue

            for raw in raw_articles:
                art = self._parse_article(raw)
                if art is None or art.url in seen_urls:
                    continue
                seen_urls.add(art.url)
                all_articles.append(art)

        self._log.info(
            "GDELT: {} artículos para {} keywords en {} batches (timespan={})",
            len(all_articles),
            len(clean_kws),
            len(batches),
            timespan,
        )
        return all_articles

    # =====================================================
    # HTTP
    # =====================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _call(
        self,
        query: str,
        timespan: str,
        max_records: int,
    ) -> list[dict[str, Any]]:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "timespan": timespan,
            "maxrecords": max_records,
            "sort": "DateDesc",
        }
        try:
            response = self._session.get(
                GDELT_BASE_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GdeltApiError(str(exc)) from exc

        # GDELT a veces devuelve 200 OK con cuerpo vacío o con un error textual
        # tipo "Timespan is too short.". Detectamos esos casos.
        text = response.text.strip()
        if not text:
            return []
        # Detectar respuesta de error textual
        if not text.startswith("{") and not text.startswith("["):
            self._log.debug("GDELT devolvió texto no-JSON: {}", text[:200])
            # Si es un error de timespan, propagar para que se sepa
            if "timespan" in text.lower():
                raise GdeltApiError(f"GDELT rechazó la query: {text[:200]}")
            return []
        try:
            data = response.json()
        except ValueError:
            return []

        if not isinstance(data, dict):
            return []
        articles = data.get("articles")
        if not isinstance(articles, list):
            return []
        return articles

    # =====================================================
    # Helpers
    # =====================================================

    @staticmethod
    def _normalize_timespan(timespan: str) -> str:
        """Eleva timespans muy cortos al mínimo aceptado por GDELT.

        GDELT rechaza valores como '15min' con "Timespan is too short".
        El mínimo seguro empíricamente es '1h'.
        """
        if not timespan:
            return MIN_TIMESPAN
        # Patrón: número + unidad (min, h, d, w, m)
        match = re.match(r"^(\d+)\s*(min|h|d|w|m)$", timespan.strip().lower())
        if not match:
            return MIN_TIMESPAN
        n, unit = int(match.group(1)), match.group(2)
        if unit == "min" and n < 60:
            return MIN_TIMESPAN
        return timespan

    # =====================================================
    # Parser
    # =====================================================

    def _parse_article(self, raw: dict[str, Any]) -> Optional[NewsArticle]:
        try:
            url = raw.get("url") or ""
            title = raw.get("title") or ""
            if not url or not title:
                return None

            published_at = self._parse_seendate(raw.get("seendate"))
            source_name = raw.get("domain") or ""
            language = (raw.get("language") or "").lower()

            return NewsArticle(
                article_id=_new_article_id(url, title),
                source=NewsSource.GDELT,
                source_name=source_name,
                title=title,
                description="",  # GDELT no incluye descripción
                content="",
                url=url,
                author=None,
                language=language,
                published_at=published_at,
            )
        except (TypeError, ValueError) as exc:
            self._log.debug("Artículo GDELT mal formado: {}", exc)
            return None

    @staticmethod
    def _parse_seendate(value: Any) -> Optional[datetime]:
        """GDELT usa formato YYYYMMDDTHHMMSSZ (ej: 20240315T120000Z)."""
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            return None
