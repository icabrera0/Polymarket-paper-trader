"""
Telegram client for reading public channels via Telethon.

Allows using Telegram as a news source without paying for APIs. Reads recent
messages from the public channels configured in `news.telegram.channels`
(e.g.: ``["@bloomberg", "@reuters", "@disclosetv"]``).

Setup (one-time only):
1. Get free credentials at https://my.telegram.org → "API development tools".
2. Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env.
3. Run `python scripts/telegram_login.py` to create the initial session
   (it will prompt for your phone number and an SMS code).
4. After that the bot works without login (uses data/telegram.session).

Design decisions:
- We use Telethon in synchronous mode (telethon.sync).
- We create a new client per fetch_articles() call instead of caching the
  client between calls. This avoids the "asyncio event loop must not change
  after connection" error that occurs when APScheduler runs the job in
  different threads of the ThreadPoolExecutor (each thread has its own event
  loop). Reconnection is fast (<1s) because the session already exists.
- The session is persisted in `data/telegram.session`. Renameable via config.
- If the client cannot connect or is not authorized, it returns [] silently
  (with a warning in logs). It never crashes the bot.
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
    """Reads public channels and returns NewsArticle objects."""

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
    # Public API
    # =====================================================

    def fetch_articles(
        self,
        keywords: list[str],
        max_results: Optional[int] = None,
    ) -> list[NewsArticle]:
        """Reads recent messages from the configured channels.

        Creates a fresh Telethon connection per call to avoid the
        "asyncio event loop must not change after connection" error when
        APScheduler calls this method from different threads of the
        ThreadPoolExecutor.

        Args:
            keywords: keywords are only used to match and enrich the
                NewsArticle objects. They do NOT filter at the API level
                (channels send everything).
            max_results: global cap; if None, reads all channels without a cap.
        """
        if not self.cfg.enabled:
            return []
        if not (self.config.telegram_api_id and self.config.telegram_api_hash):
            self._log.warning(
                "Telegram enabled but TELEGRAM_API_ID/HASH are missing from .env"
            )
            return []
        if not self.cfg.channels:
            self._log.debug("No channels configured; nothing to read")
            return []

        try:
            from telethon.sync import TelegramClient as RawClient
        except ImportError:
            self._log.error("Telethon is not installed. pip install telethon")
            return []

        # Ensure this thread has a live event loop before Telethon
        # tries to use asyncio internally.
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
                    "Telegram session not authorized. Run first: "
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
                    self._log.warning("Error reading channel {}: {}", channel, exc)

            if max_results and len(articles) > max_results:
                articles = articles[:max_results]

            self._log.info(
                "Telegram: {} messages in {} channels (lookback={}min)",
                len(articles),
                len(self.cfg.channels),
                self.cfg.message_lookback_minutes,
            )
            return articles

        except Exception as exc:
            self._log.warning("Error connecting to Telegram: {}", exc)
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
        """Reads messages from a channel up to `cutoff` or `messages_per_channel`."""
        result: list[NewsArticle] = []
        # iter_messages returns messages from newest to oldest
        for msg in client.iter_messages(
            channel, limit=self.cfg.messages_per_channel
        ):
            # Normalize the message date to UTC
            msg_date = msg.date
            if msg_date is None:
                continue
            if msg_date.tzinfo is None:
                msg_date = msg_date.replace(tzinfo=timezone.utc)
            else:
                msg_date = msg_date.astimezone(timezone.utc)

            if msg_date < cutoff:
                # Since they come in order, we can break the loop
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
        """Converts a Telethon Message into a NewsArticle."""
        text = (msg.text or "").strip()
        if not text:
            return None

        # Title = first line (or first N words), Description = rest
        first_line = text.split("\n", 1)[0]
        if len(first_line) > 200:
            first_line = first_line[:200] + "..."
        description = text[len(first_line):].strip()
        if len(description) > 1000:
            description = description[:1000] + "..."

        # "Virtual" URL — Telegram allows linking to a specific message:
        # https://t.me/CHANNEL/MESSAGE_ID
        clean_channel = channel.lstrip("@")
        url = f"https://t.me/{clean_channel}/{msg.id}"

        return NewsArticle(
            article_id=_new_article_id(url, first_line),
            source=NewsSource.TELEGRAM,
            source_name=channel,           # e.g.: "@bloomberg"
            title=first_line,
            description=description,
            content=text,                   # Full message
            url=url,
            author=None,
            language="",                    # Telegram does not mark language at the message level
            published_at=msg_date,
        )

    # =====================================================
    # Cleanup
    # =====================================================

    def disconnect(self) -> None:
        """No-op: connections are now opened and closed per call."""
        pass
