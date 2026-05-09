"""
Cliente HTTP para la Gamma API de Polymarket.

La Gamma API (https://gamma-api.polymarket.com) expone metadata de mercados:
preguntas, fechas, tokens CLOB, precios actuales y volumen. Es el primer punto
de entrada del bot — el MARKET_SCANNER consume estos datos crudos y aplica los
filtros configurados.

Este cliente NO ejecuta órdenes ni accede al order book completo (eso es la
CLOB API, que se integrará en el módulo PAPER_TRADER).

Características:
- Reintentos exponenciales con tenacity (3 intentos, 2-10s de backoff).
- Paginación automática para recoger todos los mercados activos.
- User-Agent identificable y timeout configurables.
- Errores de red se loguean y se relanzan como GammaApiError.
"""

from __future__ import annotations

from typing import Any, Optional

import requests
from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config_loader import BotConfig


class GammaApiError(Exception):
    """Error al comunicarse con la Gamma API."""


class GammaApiClient:
    """Wrapper sincrónico sobre la Gamma API de Polymarket."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.base_url = config.polymarket.gamma_api_url.rstrip("/")
        self.timeout = config.polymarket.request_timeout_seconds

        self._log = logger.bind(module="gamma_client")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": f"PolymarketBot/{config.app.version}",
                "Accept": "application/json",
            }
        )

    # =====================================================
    # GET genérico con reintentos
    # =====================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            self._log.warning("GET {} falló: {}", url, exc)
            raise

    # =====================================================
    # Endpoints de alto nivel
    # =====================================================

    def fetch_markets(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        order: str = "volume24hr",
        ascending: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Devuelve mercados crudos (lista de dicts) ordenados por volumen 24h.

        Args:
            active: solo mercados activos.
            closed: incluir mercados cerrados.
            order: campo de ordenación (por defecto volumen 24h).
            ascending: ascendente o descendente.
            limit: tamaño de página (máx ~500 según API).
            offset: para paginar.
        """
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower(),
            "limit": limit,
            "offset": offset,
        }
        try:
            data = self._get("/markets", params=params)
        except (requests.RequestException, RetryError) as exc:
            raise GammaApiError(f"No se pudieron obtener mercados: {exc}") from exc

        if not isinstance(data, list):
            self._log.warning("Respuesta inesperada (no es lista): {}", type(data))
            return []
        return data

    def fetch_all_active_markets(
        self,
        max_markets: int = 500,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Pagina hasta `max_markets` mercados activos ordenados por volumen.

        Para en cuanto la API devuelve menos resultados que el page_size (señal
        de que hemos llegado al final).
        """
        all_markets: list[dict[str, Any]] = []
        offset = 0
        while len(all_markets) < max_markets:
            remaining = max_markets - len(all_markets)
            batch_limit = min(page_size, remaining)
            batch = self.fetch_markets(limit=batch_limit, offset=offset)
            if not batch:
                break
            all_markets.extend(batch)
            if len(batch) < batch_limit:
                # No hay más resultados
                break
            offset += batch_limit

        self._log.debug(
            "fetch_all_active_markets: {} mercados recogidos en {} páginas",
            len(all_markets),
            (offset // page_size) + 1,
        )
        return all_markets

    def fetch_markets_by_token_ids(
        self, token_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Targeted lookup: returns only the markets that contain the given CLOB
        token IDs. Use this instead of a full 500-market scan when you only
        need prices for a small number of known positions."""
        if not token_ids:
            return []
        params = {
            "clobTokenIds": ",".join(token_ids),
            "active": "true",
            "closed": "false",
        }
        try:
            data = self._get("/markets", params=params)
            return data if isinstance(data, list) else []
        except (requests.RequestException, RetryError) as exc:
            self._log.warning(
                "No se pudieron obtener mercados por token IDs: {}", exc
            )
            return []

    def fetch_market_by_id(self, market_id: str) -> Optional[dict[str, Any]]:
        """Devuelve un único mercado por ID, o None si no existe."""
        try:
            data = self._get(f"/markets/{market_id}")
        except (requests.RequestException, RetryError) as exc:
            self._log.warning("No se pudo obtener mercado {}: {}", market_id, exc)
            return None
        if isinstance(data, dict):
            return data
        return None

    def fetch_market_by_token_id_raw(self, token_id: str) -> Optional[dict[str, Any]]:
        """Devuelve el mercado que contiene este token CLOB, incluyendo resueltos.

        Intenta primero sin filtro, luego con closed=true — la Gamma API puede
        filtrar mercados resueltos por defecto si no se especifica explícitamente.
        """
        for extra in ({}, {"closed": "true"}):
            params = {"clobTokenIds": token_id, **extra}
            try:
                data = self._get("/markets", params=params)
                if isinstance(data, list) and data:
                    return data[0]
            except (requests.RequestException, RetryError) as exc:
                self._log.warning(
                    "fetch_market_by_token_id_raw {}: {}", token_id[:12], exc
                )
                return None  # network error, don't retry with different params
        return None
