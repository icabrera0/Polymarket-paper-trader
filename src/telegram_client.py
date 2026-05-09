"""
Cliente de Telegram para leer canales públicos vía Telethon.

Permite usar Telegram como fuente de noticias sin pagar APIs. Lee mensajes
recientes de los canales públicos configurados en `news.telegram.channels`
(ej: ``["@bloomberg", "@reuters", "@disclosetv"]``).

Setup (una sola vez):
1. Obtén credenciales gratis en https://my.telegram.org → "API development tools".
2. Pon TELEGRAM_API_ID y TELEGRAM_API_HASH en .env.
3. Ejecuta `python scripts/telegram_login.py` para crear la sesión inicial
   (te pedirá tu teléfono y un código SMS).
4. A partir de ahí el bot funciona sin login (usa data/telegram.session).

Decisiones de diseño:
- Usamos Telethon en modo síncrono (telethon.sync).
- Creamos un cliente nuevo por cada llamada a fetch_articles() en lugar de
  cachear el cliente entre llamadas. Esto evita el error "asyncio event loop
  must not change after connection" que ocurre cuando APScheduler ejecuta el
  job en distintos hilos del ThreadPoolExecutor (cada hilo tiene su propio
  event loop). La reconexión es rápida (<1s) porque la sesión ya existe.
- La sesión se persiste en `data/telegram.session`. Renombrable por config.
- Si el cliente no puede conectar o no está autorizado, devuelve [] silenciosamente
  (con un warning en logs). Nunca crashea el bot.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.config_loader import BotConfig
from src.models import NewsArticle, NewsSource, _new_article_id

DEFAULT_SESSION_NAME = "telegram"


class TelegramClient:
    """Lee canales públicos y devuelve NewsArticle."""

    def __init__(
        self,
        config: BotConfig,
        session_name: str = DEFAULT_SESSION_NAME,
    ) -> None:
        self.config = config
        self.cfg = config.news.telegram
        self.session_name = session_name
        self._log = logger.bind(module="telegram_client")

    # =====================================================
    # API pública
    # =====================================================

    def fetch_articles(
        self,
        keywords: list[str],
        max_results: Optional[int] = None,
    ) -> list[NewsArticle]:
        """Lee los mensajes recientes de los canales configurados.

        Crea una conexión Telethon fresca por llamada para evitar el error
        "asyncio event loop must not change after connection" cuando APScheduler
        llama a este método desde distintos hilos del ThreadPoolExecutor.

        Args:
            keywords: keywords solo se usan para matchear y enriquecer los
                NewsArticle. NO filtran a nivel API (los canales mandan todo).
            max_results: tope global; si None, lee todos los canales sin tope.
        """
        if not self.cfg.enabled:
            return []
        if not (self.config.telegram_api_id and self.config.telegram_api_hash):
            self._log.warning(
                "Telegram habilitado pero faltan TELEGRAM_API_ID/HASH en .env"
            )
            return []
        if not self.cfg.channels:
            self._log.debug("Sin canales configurados; nada que leer")
            return []

        try:
            from telethon.sync import TelegramClient as RawClient
        except ImportError:
            self._log.error("Telethon no está instalado. pip install telethon")
            return []

        # Garantizar que este hilo tiene un event loop vivo antes de que
        # Telethon intente usar asyncio internamente.
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("closed")
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        data_dir = Path(__file__).resolve().parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        session_path = str(data_dir / self.session_name)

        client = RawClient(
            session_path,
            self.config.telegram_api_id,
            self.config.telegram_api_hash,
        )
        try:
            client.connect()
            if not client.is_user_authorized():
                self._log.error(
                    "Sesión de Telegram no autorizada. Ejecuta primero: "
                    "python scripts/telegram_login.py"
                )
                return []

            cutoff = datetime.now(timezone.utc) - timedelta(
                minutes=self.cfg.message_lookback_minutes
            )
            articles: list[NewsArticle] = []

            for channel in self.cfg.channels:
                try:
                    articles.extend(self._fetch_channel(client, channel, cutoff))
                except Exception as exc:
                    self._log.warning("Error leyendo canal {}: {}", channel, exc)

            if max_results and len(articles) > max_results:
                articles = articles[:max_results]

            self._log.info(
                "Telegram: {} mensajes en {} canales (lookback={}min)",
                len(articles),
                len(self.cfg.channels),
                self.cfg.message_lookback_minutes,
            )
            return articles

        except Exception as exc:
            self._log.warning("Error conectando a Telegram: {}", exc)
            return []
        finally:
            try:
                client.disconnect()
            except Exception:
                pass

    # =====================================================
    # Internals
    # =====================================================

    def _fetch_channel(
        self,
        client: Any,
        channel: str,
        cutoff: datetime,
    ) -> list[NewsArticle]:
        """Lee mensajes de un canal hasta `cutoff` o hasta `messages_per_channel`."""
        result: list[NewsArticle] = []
        # iter_messages devuelve mensajes del más reciente al más antiguo
        for msg in client.iter_messages(
            channel, limit=self.cfg.messages_per_channel
        ):
            # Normalizar la fecha del mensaje a UTC
            msg_date = msg.date
            if msg_date is None:
                continue
            if msg_date.tzinfo is None:
                msg_date = msg_date.replace(tzinfo=timezone.utc)
            else:
                msg_date = msg_date.astimezone(timezone.utc)

            if msg_date < cutoff:
                # Como vienen ordenados, podemos romper el bucle
                break

            article = self._message_to_article(msg, channel, msg_date)
            if article is not None:
                result.append(article)
        return result

    def _message_to_article(
        self,
        msg: Any,
        channel: str,
        msg_date: datetime,
    ) -> Optional[NewsArticle]:
        """Convierte un Telethon Message en un NewsArticle."""
        text = (msg.text or "").strip()
        if not text:
            return None

        # Title = primera línea (o primeras N palabras), Description = resto
        first_line = text.split("\n", 1)[0]
        if len(first_line) > 200:
            first_line = first_line[:200] + "..."
        description = text[len(first_line):].strip()
        if len(description) > 1000:
            description = description[:1000] + "..."

        # URL "virtual" — Telegram permite linkear a un mensaje específico:
        # https://t.me/CHANNEL/MESSAGE_ID
        clean_channel = channel.lstrip("@")
        url = f"https://t.me/{clean_channel}/{msg.id}"

        return NewsArticle(
            article_id=_new_article_id(url, first_line),
            source=NewsSource.TELEGRAM,
            source_name=channel,           # Ej: "@bloomberg"
            title=first_line,
            description=description,
            content=text,                   # Mensaje completo
            url=url,
            author=None,
            language="",                    # Telegram no marca idioma a nivel mensaje
            published_at=msg_date,
        )

    # =====================================================
    # Cleanup
    # =====================================================

    def disconnect(self) -> None:
        """No-op: las conexiones ahora se abren y cierran por llamada."""
        pass
