"""
Social Ingestor — scrapes Telegram public channels, Reddit subreddits, and RSS feeds.

Returns NewsArticle objects that plug directly into the existing news pipeline alongside
NewsIngestor. Each source is completely independent — a failure in one never blocks the
others.

Sources:
  - Telegram: public channels via Telethon (async, wrapped with asyncio.run).
              Session file: data/telegram_social.session
              First-run: if no session exists and no credentials, silently skipped.
  - Reddit: public subreddits via PRAW in read-only mode (no user auth needed).
  - RSS: arbitrary feeds via feedparser (handles HTTP internally).

Config section in settings.yaml:
    social:
      enabled: true
      telegram: { enabled, channels, max_posts_per_channel, max_age_hours }
      reddit:   { enabled, subreddits, max_posts_per_subreddit, max_age_hours,
                  client_id, client_secret, user_agent }
      rss:      { enabled, feeds, max_items_per_feed, max_age_hours }

All NewsArticle objects returned have:
  - preliminary_impact_score = 0.6 * (100/100) → stored as 60.0 (scale 0-100, social default)
  - source: NewsSource.TELEGRAM, NewsSource.RSS (Reddit also uses RSS as the closest enum)
  - content truncated to 500 chars
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from loguru import logger

from src.config_loader import BotConfig
from src.models import NewsArticle, NewsSource

# Default relevance score for social sources (0-100 scale used by preliminary_impact_score)
_SOCIAL_DEFAULT_SCORE = 60.0


def _article_id(url: str, title: str) -> str:
    """Stable short hash matching the pattern in models._new_article_id."""
    return hashlib.sha1(f"{url}|{title}".encode("utf-8")).hexdigest()[:16]


def _truncate(text: str, max_chars: int = 500) -> str:
    if not text:
        return ""
    text = text.strip()
    return text[:max_chars] if len(text) <= max_chars else text[:max_chars - 1] + "…"


def _matches_keywords(text: str, keywords: list[str]) -> bool:
    """True if ANY keyword appears (case-insensitive) in text.

    Empty keyword list is treated as "no filter" and always returns True.
    """
    effective = [kw for kw in keywords if kw.strip()]
    if not effective:
        return True
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in effective)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SocialIngestor:
    """Scrapes Telegram channels, Reddit subreddits, and RSS feeds.

    Returns NewsArticle objects compatible with the existing pipeline.
    Each source is independent — failure in one never blocks others.
    """

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._log = logger.bind(module="social_ingestor")
        self._log.info("SocialIngestor initialized")

    # =========================================================================
    # Public API
    # =========================================================================

    def fetch_all(self, keywords: list[str]) -> list[NewsArticle]:
        """Fetch from all enabled social sources and return combined articles.

        Args:
            keywords: Keywords to filter articles — only items where ANY keyword
                      appears in the title or body are included (case-insensitive).

        Returns:
            Deduplicated list of NewsArticle objects sorted by published_at descending.
        """
        cfg = getattr(self.config, "social", None)
        if cfg is None or not cfg.enabled:
            return []

        telegram_arts: list[NewsArticle] = []
        reddit_arts: list[NewsArticle] = []
        rss_arts: list[NewsArticle] = []

        if getattr(cfg.telegram, "enabled", False):
            telegram_arts = self._fetch_telegram(keywords)

        if getattr(cfg.reddit, "enabled", False):
            reddit_arts = self._fetch_reddit(keywords)

        if getattr(cfg.rss, "enabled", False):
            rss_arts = self._fetch_rss(keywords)

        articles = telegram_arts + reddit_arts + rss_arts

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique: list[NewsArticle] = []
        for art in articles:
            if art.url not in seen_urls:
                seen_urls.add(art.url)
                unique.append(art)

        # Sort newest-first
        unique.sort(
            key=lambda a: a.published_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        self._log.info(
            "SocialIngestor: {} total articles from all social sources ({} unique)",
            len(articles),
            len(unique),
        )

        # Write per-source counts to data/social_stats.json for the dashboard
        self._write_stats(
            telegram=len(telegram_arts),
            reddit=len(reddit_arts),
            rss=len(rss_arts),
        )

        return unique

    def _write_stats(self, telegram: int, reddit: int, rss: int) -> None:
        """Write per-source fetch counts to data/social_stats.json."""
        import json
        from pathlib import Path

        stats_path = Path("data") / "social_stats.json"
        try:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(
                json.dumps(
                    {
                        "telegram": telegram,
                        "reddit": reddit,
                        "rss": rss,
                        "updated_at": _utcnow().isoformat(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self._log.debug("social_stats write failed: {}", exc)

    # =========================================================================
    # Telegram
    # =========================================================================

    def _fetch_telegram(self, keywords: list[str]) -> list[NewsArticle]:
        """Fetch messages from public Telegram channels via Telethon.

        Wraps the async Telethon client with asyncio.run() for a synchronous
        interface. On first run, if no session file exists and no API credentials
        are configured, silently skips and logs a warning (never crashes).
        """
        try:
            return asyncio.run(self._fetch_telegram_async(keywords))
        except Exception as exc:
            self._log.warning("Telegram social fetch failed: {}", exc)
            return []

    async def _fetch_telegram_async(self, keywords: list[str]) -> list[NewsArticle]:
        """Async implementation of Telegram fetching."""
        from pathlib import Path

        try:
            from telethon import TelegramClient
            from telethon.errors import SessionPasswordNeededError
        except ImportError:
            self._log.warning(
                "Telegram social: telethon not installed — "
                "run 'pip install telethon' to enable this source"
            )
            return []

        cfg = self.config.social.telegram
        api_id = getattr(self.config, "telegram_api_id", None)
        api_hash = getattr(self.config, "telegram_api_hash", None)

        if not api_id or not api_hash:
            self._log.warning(
                "Telegram social: TELEGRAM_API_ID / TELEGRAM_API_HASH missing in .env "
                "— skipping Telegram social source"
            )
            return []

        session_path = Path("data") / "telegram_social.session"
        session_path.parent.mkdir(parents=True, exist_ok=True)

        if not session_path.exists():
            self._log.warning(
                "Telegram social: no session file at '{}'. "
                "Run the bot interactively once to authenticate.",
                session_path,
            )
            return []

        cutoff = _utcnow() - timedelta(hours=cfg.max_age_hours)
        articles: list[NewsArticle] = []

        client = TelegramClient(str(session_path), api_id, api_hash)
        try:
            await client.connect()
            if not await client.is_user_authorized():
                self._log.warning(
                    "Telegram social: session expired or invalid. "
                    "Delete {} and re-authenticate interactively.",
                    session_path,
                )
                await client.disconnect()
                return []
        except Exception as exc:
            self._log.warning("Telegram social: could not connect ({})", exc)
            return []

        try:
            for channel in cfg.channels:
                try:
                    entity = await client.get_entity(channel)
                    messages = await client.get_messages(
                        entity, limit=cfg.max_posts_per_channel
                    )
                    for msg in messages:
                        if msg.date is None:
                            continue

                        # Ensure timezone-aware
                        pub = msg.date if msg.date.tzinfo else msg.date.replace(tzinfo=timezone.utc)
                        if pub < cutoff:
                            continue

                        text = msg.message or ""
                        if not text.strip():
                            continue

                        # Build a pseudo-title from the first line (max 120 chars)
                        first_line = text.splitlines()[0][:120].strip()
                        title = first_line or f"[{channel}] message {msg.id}"

                        if not _matches_keywords(f"{title} {text}", keywords):
                            continue

                        url = f"https://t.me/{channel.lstrip('@')}/{msg.id}"
                        articles.append(
                            NewsArticle(
                                article_id=_article_id(url, title),
                                source=NewsSource.TELEGRAM,
                                source_name=channel,
                                title=title,
                                content=_truncate(text),
                                url=url,
                                published_at=pub,
                                preliminary_impact_score=_SOCIAL_DEFAULT_SCORE,
                            )
                        )
                except Exception as exc:
                    self._log.warning(
                        "Telegram social: error fetching channel '{}': {}", channel, exc
                    )
                    continue
        finally:
            await client.disconnect()

        self._log.info(
            "Telegram social: {} articles from {} channels",
            len(articles),
            len(cfg.channels),
        )
        return articles

    # =========================================================================
    # Reddit
    # =========================================================================

    def _fetch_reddit(self, keywords: list[str]) -> list[NewsArticle]:
        """Fetch posts from public Reddit subreddits via PRAW (read-only mode).

        PRAW read-only mode requires only client_id, client_secret, and user_agent —
        no user authentication. Empty credentials mean this source is skipped.
        """
        try:
            return self._fetch_reddit_impl(keywords)
        except Exception as exc:
            self._log.warning("Reddit social fetch failed: {}", exc)
            return []

    def _fetch_reddit_impl(self, keywords: list[str]) -> list[NewsArticle]:
        try:
            import praw
        except ImportError:
            self._log.warning(
                "Reddit social: praw not installed — "
                "run 'pip install praw' to enable this source"
            )
            return []

        cfg = self.config.social.reddit
        client_id = cfg.client_id or ""
        client_secret = cfg.client_secret or ""
        user_agent = cfg.user_agent or "polymarket-bot/1.0"

        if not client_id or not client_secret:
            self._log.warning(
                "Reddit social: client_id / client_secret not set in settings.yaml "
                "— skipping Reddit source"
            )
            return []

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            read_only=True,
        )

        cutoff = _utcnow() - timedelta(hours=cfg.max_age_hours)
        articles: list[NewsArticle] = []

        for sub_name in cfg.subreddits:
            try:
                sub = reddit.subreddit(sub_name)
                for post in sub.new(limit=cfg.max_posts_per_subreddit):
                    pub = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                    if pub < cutoff:
                        continue

                    title = post.title or ""
                    body = getattr(post, "selftext", "") or ""
                    combined_text = f"{title} {body}"

                    if not _matches_keywords(combined_text, keywords):
                        continue

                    url = f"https://www.reddit.com{post.permalink}"
                    articles.append(
                        NewsArticle(
                            article_id=_article_id(url, title),
                            source=NewsSource.RSS,   # Closest enum; source_name identifies Reddit
                            source_name=f"reddit/r/{sub_name}",
                            title=title,
                            content=_truncate(body),
                            url=url,
                            published_at=pub,
                            preliminary_impact_score=_SOCIAL_DEFAULT_SCORE,
                        )
                    )
            except Exception as exc:
                self._log.warning(
                    "Reddit social: error fetching r/{}: {}", sub_name, exc
                )
                continue

        self._log.info(
            "Reddit social: {} articles from {} subreddits",
            len(articles),
            len(cfg.subreddits),
        )
        return articles

    # =========================================================================
    # RSS
    # =========================================================================

    def _fetch_rss(self, keywords: list[str]) -> list[NewsArticle]:
        """Fetch items from RSS feeds via feedparser.

        feedparser handles HTTP requests internally — no extra requests library needed.
        """
        try:
            return self._fetch_rss_impl(keywords)
        except Exception as exc:
            self._log.warning("RSS social fetch failed: {}", exc)
            return []

    def _fetch_rss_impl(self, keywords: list[str]) -> list[NewsArticle]:
        try:
            import feedparser
        except ImportError:
            self._log.warning(
                "RSS social: feedparser not installed — "
                "run 'pip install feedparser' to enable this source"
            )
            return []

        cfg = self.config.social.rss
        cutoff = _utcnow() - timedelta(hours=cfg.max_age_hours)
        articles: list[NewsArticle] = []

        for feed_url in cfg.feeds:
            try:
                feed = feedparser.parse(feed_url)
                if feed.bozo and not feed.entries:
                    self._log.warning(
                        "RSS social: feed '{}' parse error: {}",
                        feed_url,
                        feed.bozo_exception,
                    )
                    continue

                feed_title = feed.feed.get("title", feed_url)
                count = 0

                for entry in feed.entries:
                    if count >= cfg.max_items_per_feed:
                        break

                    # Parse published date (feedparser populates published_parsed as time.struct_time)
                    pub: Optional[datetime] = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        import time as time_mod
                        pub = datetime.fromtimestamp(
                            time_mod.mktime(entry.published_parsed), tz=timezone.utc
                        )
                    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                        import time as time_mod
                        pub = datetime.fromtimestamp(
                            time_mod.mktime(entry.updated_parsed), tz=timezone.utc
                        )

                    if pub and pub < cutoff:
                        continue

                    title = entry.get("title", "").strip()
                    summary = entry.get("summary", "") or entry.get("description", "")
                    url = entry.get("link", "")

                    if not title or not url:
                        continue

                    combined_text = f"{title} {summary}"
                    if not _matches_keywords(combined_text, keywords):
                        continue

                    articles.append(
                        NewsArticle(
                            article_id=_article_id(url, title),
                            source=NewsSource.RSS,
                            source_name=feed_title,
                            title=title,
                            content=_truncate(summary),
                            url=url,
                            published_at=pub,
                            preliminary_impact_score=_SOCIAL_DEFAULT_SCORE,
                        )
                    )
                    count += 1

            except Exception as exc:
                self._log.warning(
                    "RSS social: error fetching feed '{}': {}", feed_url, exc
                )
                continue

        self._log.info("RSS social: {} articles from {} feeds", len(articles), len(cfg.feeds))
        return articles
