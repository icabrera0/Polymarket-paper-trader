"""
Bot configuration loading and validation.

Reads `config/settings.yaml` and `.env`, and exposes them as a fully
Pydantic-validated `BotConfig` object. Any invalid value aborts the bot
startup with a clear error, preventing execution with incorrect parameters.

Typical usage:
    from src.config_loader import load_config, validate_secrets
    config = load_config()
    errors = validate_secrets(config)
    if errors:
        raise SystemExit("\\n".join(errors))
    print(config.risk.max_position_size_pct)  # 0.15
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# =====================================================
# Sub-models (one per section of settings.yaml)
# =====================================================


class AppConfig(BaseModel):
    name: str
    version: str
    timezone: str
    base_currency: str


class PaperTradingConfig(BaseModel):
    initial_balance_eur: float = Field(gt=0)
    eur_to_usd_rate: float = Field(gt=0)


class RiskConfig(BaseModel):
    """Risk management parameters. Calibrated for a 150 EUR bankroll."""

    max_position_size_pct: float = Field(gt=0, le=1)
    max_simultaneous_positions: int = Field(gt=0)
    min_trade_size_eur: float = Field(gt=0)
    max_drawdown_pct: float = Field(gt=0, le=1)
    stop_loss_pct: float = Field(gt=0, le=1)
    take_profit_pct: float = Field(gt=0)
    pause_on_drawdown: bool = True
    drawdown_pause_requires_manual_resume: bool
    # Time-based exits: prevent positions from blocking slots indefinitely
    time_tighten_tp_hours: float = Field(default=24.0, gt=0)   # after Nh, use tightened TP
    time_tighten_tp_pct: float = Field(default=0.15, gt=0)     # tightened TP threshold
    time_exit_profit_hours: float = Field(default=48.0, gt=0)  # after Nh, close if P&L >= 0
    time_exit_hard_hours: float = Field(default=72.0, gt=0)    # after Nh, close unconditionally


class MarketFiltersConfig(BaseModel):
    min_volume_24h_usd: float = Field(ge=0)
    max_spread_cents: float = Field(gt=0)
    min_probability_edge: float = Field(gt=0, le=1)
    min_time_to_close_hours: float = Field(ge=0)
    max_time_to_close_days: float = Field(gt=0)
    exclude_categories: list[str] = []


class NewsApiConfig(BaseModel):
    enabled: bool
    poll_interval_seconds: int = Field(gt=0)
    languages: list[str]
    page_size: int = Field(gt=0, le=100)
    sources: list[str] = []


class GdeltConfig(BaseModel):
    enabled: bool
    poll_interval_seconds: int = Field(gt=0)
    timespan: str
    max_records: int = Field(gt=0)


class TelegramConfig(BaseModel):
    """Reading public Telegram channels via Telethon."""

    enabled: bool
    poll_interval_seconds: int = Field(gt=0)
    channels: list[str] = []                  # E.g.: ["@bloomberg", "@reuters"]
    message_lookback_minutes: int = Field(gt=0)
    messages_per_channel: int = Field(gt=0)


class NewsConfig(BaseModel):
    newsapi: NewsApiConfig
    gdelt: GdeltConfig
    telegram: TelegramConfig
    cache_ttl_seconds: int = Field(gt=0)
    dedup_similarity_threshold: float = Field(gt=0, le=1)


class LlmConfig(BaseModel):
    provider: str                                  # "anthropic" | "ollama"
    model: str
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0, le=2)
    retry_attempts: int = Field(ge=0)
    retry_delay_seconds: int = Field(ge=0)
    min_confidence_threshold: int = Field(ge=0, le=100)
    cache_analysis: bool
    cache_ttl_seconds: int = Field(gt=0)
    # Cost protection (applies to Anthropic only)
    daily_spend_limit_usd: float = Field(default=5.0, ge=0)
    dry_run: bool = Field(default=False)
    # Ollama (applies only if provider=="ollama")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_timeout_seconds: int = Field(default=120, gt=0)
    # Parallel LLM workers. >1 requires OLLAMA_NUM_PARALLEL env var set in Ollama.
    llm_parallelism: int = Field(default=1, ge=1, le=8)


class PolymarketConfig(BaseModel):
    gamma_api_url: str
    clob_api_url: str
    scan_interval_seconds: int = Field(gt=0)
    request_timeout_seconds: int = Field(gt=0)
    trading_fee_pct: float = Field(ge=0, le=1)
    simulated_slippage_pct: float = Field(ge=0, le=1)


class DecisionConfig(BaseModel):
    llm_consultation_threshold: float = Field(gt=0, le=1)
    require_news_for_entry: bool
    reevaluate_open_positions_minutes: int = Field(gt=0)
    # Coverage
    markets_to_analyze_per_cycle: int = Field(default=15, gt=0)
    category_priority_boost: dict[str, float] = Field(default_factory=dict)
    # Search fallback
    fallback_news_lookback: str = Field(default="7d")
    enable_fallback_search: bool = Field(default=True)
    # Low-info trades
    allow_low_info_trades: bool = Field(default=True)
    low_info_min_confidence: int = Field(default=75, ge=0, le=100)
    low_info_size_multiplier: float = Field(default=0.5, gt=0, le=1)
    low_info_min_articles: int = Field(default=1, ge=0)
    # NO-opportunity hunt: adds high-YES markets to each analysis cycle
    no_hunt_enabled: bool = Field(default=True)
    no_hunt_min_yes_price: float = Field(default=0.65, ge=0, le=1)
    no_hunt_max_candidates: int = Field(default=3, gt=0)


class SportsInPlayConfig(BaseModel):
    """Optional opt-in module for underdog betting on live matches."""

    enabled: bool = False
    max_positions: int = Field(default=1, gt=0)
    position_size_eur: float = Field(default=3.0, gt=0)
    min_yes_price: float = Field(default=0.68, ge=0, le=1)
    max_yes_price: float = Field(default=0.88, ge=0, le=1)
    max_time_to_close_hours: float = Field(default=4.0, gt=0)
    min_volume_24h_usd: float = Field(default=5000.0, ge=0)
    min_fresh_news_minutes: int = Field(default=60, gt=0)
    min_confidence: int = Field(default=65, ge=0, le=100)
    stop_loss_pct: float = Field(default=0.50, gt=0, le=1)
    take_profit_pct: float = Field(default=0.80, gt=0)
    scan_categories: list[str] = Field(default_factory=lambda: ["Sports", "Esports"])


class ReportsConfig(BaseModel):
    output_directory: str
    filename_format: str
    generation_time: str
    include_charts: bool
    conditional_formatting: bool


class LoggingConfig(BaseModel):
    level: str
    log_directory: str
    rotation_size_mb: int = Field(gt=0)
    retention_days: int = Field(gt=0)
    log_llm_decisions: bool
    log_format: str


class DatabaseConfig(BaseModel):
    type: str
    path: str


class DiscordConfig(BaseModel):
    enabled: bool
    username: str
    avatar_url: str = ""
    notify_on_trade_open: bool
    notify_on_trade_close: bool
    notify_on_stop_loss: bool
    notify_on_take_profit: bool
    notify_on_drawdown_warning: bool
    notify_on_bot_pause: bool
    notify_daily_summary: bool
    daily_summary_time: str
    color_profit: int
    color_loss: int
    color_info: int
    color_warning: int


class NotificationsConfig(BaseModel):
    discord: DiscordConfig


# =====================================================
# Root model
# =====================================================


class BotConfig(BaseModel):
    app: AppConfig
    paper_trading: PaperTradingConfig
    risk: RiskConfig
    market_filters: MarketFiltersConfig
    news: NewsConfig
    llm: LlmConfig
    polymarket: PolymarketConfig
    decision: DecisionConfig
    sports_in_play: SportsInPlayConfig = Field(default_factory=SportsInPlayConfig)
    reports: ReportsConfig
    logging: LoggingConfig
    database: DatabaseConfig
    notifications: NotificationsConfig

    # Secrets loaded from .env
    anthropic_api_key: Optional[str] = None
    newsapi_key: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    telegram_api_id: Optional[int] = None
    telegram_api_hash: Optional[str] = None
    telegram_phone: Optional[str] = None


# =====================================================
# Loader
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def load_config(
    config_path: Optional[Path] = None,
    env_path: Optional[Path] = None,
) -> BotConfig:
    """Loads settings.yaml + .env and returns a validated BotConfig.

    Args:
        config_path: path to the YAML file (default: config/settings.yaml)
        env_path: path to the .env file (default: .env at the project root)

    Raises:
        FileNotFoundError: if the YAML file does not exist
        pydantic.ValidationError: if any value in the YAML is invalid
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    env_path = env_path or DEFAULT_ENV_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    # .env is optional (e.g. in tests)
    if env_path.exists():
        load_dotenv(env_path)

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Inject secrets from the environment
    data["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
    data["newsapi_key"] = os.getenv("NEWSAPI_KEY")
    data["discord_webhook_url"] = os.getenv("DISCORD_WEBHOOK_URL")
    # Telegram: api_id is int; convert defensively
    tg_id = os.getenv("TELEGRAM_API_ID")
    if tg_id and tg_id.strip():
        try:
            data["telegram_api_id"] = int(tg_id)
        except ValueError:
            data["telegram_api_id"] = None
    data["telegram_api_hash"] = os.getenv("TELEGRAM_API_HASH")
    data["telegram_phone"] = os.getenv("TELEGRAM_PHONE")

    return BotConfig(**data)


def validate_secrets(config: BotConfig) -> list[str]:
    """Checks that the required secrets are present.

    Returns a list of error messages (empty if everything is OK).
    Useful for aborting startup with a clear message.
    """
    errors: list[str] = []
    # ANTHROPIC_API_KEY is only required if provider == "anthropic"
    if config.llm.provider.lower() == "anthropic" and not config.anthropic_api_key:
        errors.append(
            "provider='anthropic' but ANTHROPIC_API_KEY is missing from .env"
        )
    if config.news.newsapi.enabled and not config.newsapi_key:
        errors.append("NewsAPI is enabled but NEWSAPI_KEY is missing from .env")
    if config.notifications.discord.enabled and not config.discord_webhook_url:
        errors.append(
            "Discord is enabled but DISCORD_WEBHOOK_URL is missing from .env"
        )
    if config.news.telegram.enabled:
        if not config.telegram_api_id or not config.telegram_api_hash:
            errors.append(
                "Telegram is enabled but TELEGRAM_API_ID and/or "
                "TELEGRAM_API_HASH are missing from .env. Get them for free at "
                "https://my.telegram.org → API development tools."
            )
    # Warning (not error) if ALL news sources are disabled
    if not (
        config.news.newsapi.enabled
        or config.news.gdelt.enabled
        or config.news.telegram.enabled
    ):
        errors.append(
            "No news sources are enabled. Enable at least one in "
            "config/settings.yaml (gdelt does not require credentials)."
        )
    return errors
