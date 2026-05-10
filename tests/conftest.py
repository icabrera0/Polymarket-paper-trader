"""
Fixtures shared by all tests.

Provides `build_test_config()` which builds a valid `BotConfig` without touching the
filesystem. This decouples the tests from `config/settings.yaml`: even if that
file does not exist or its values change, the tests keep passing because
they control their inputs exactly.

Usage from a test:

    def mi_test(config):              # injected fixture
        rm = RiskManager(config)
        ...

    def mi_test_custom(config_factory):
        cfg = config_factory(risk_overrides={"max_position_size_pct": 0.50})
        rm = RiskManager(cfg)
        ...
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from src.config_loader import (
    AppConfig,
    BotConfig,
    DatabaseConfig,
    DecisionConfig,
    DiscordConfig,
    GdeltConfig,
    LlmConfig,
    LoggingConfig,
    MarketFiltersConfig,
    NewsApiConfig,
    NewsConfig,
    NotificationsConfig,
    PaperTradingConfig,
    PolymarketConfig,
    ReportsConfig,
    RiskConfig,
    TelegramConfig,
)


def build_test_config(
    risk_overrides: dict[str, Any] | None = None,
    paper_trading_overrides: dict[str, Any] | None = None,
) -> BotConfig:
    """Builds a test BotConfig with reasonable values.

    Reproduces the calibration for 150€ but allows overriding specific sections
    for specific tests.
    """
    risk_defaults = {
        "max_position_size_pct": 0.15,
        "max_simultaneous_positions": 3,
        "min_trade_size_eur": 5.0,
        "max_drawdown_pct": 0.30,
        "stop_loss_pct": 0.20,
        "take_profit_pct": 0.30,
        "drawdown_pause_requires_manual_resume": True,
        # Permissive VaR limit in tests so existing sizing tests aren't blocked.
        # VaR-specific tests override this via risk_overrides in config_factory.
        "var_daily_limit_pct": 0.20,
    }
    if risk_overrides:
        risk_defaults.update(risk_overrides)

    paper_trading_defaults = {
        "initial_balance_eur": 150.0,
        "eur_to_usd_rate": 1.07,
    }
    if paper_trading_overrides:
        paper_trading_defaults.update(paper_trading_overrides)

    return BotConfig(
        app=AppConfig(
            name="Test Bot",
            version="0.0.1-test",
            timezone="UTC",
            base_currency="EUR",
        ),
        paper_trading=PaperTradingConfig(**paper_trading_defaults),
        risk=RiskConfig(**risk_defaults),
        market_filters=MarketFiltersConfig(
            min_volume_24h_usd=10000,
            max_spread_cents=0.05,
            min_probability_edge=0.10,
            min_time_to_close_hours=2,
            max_time_to_close_days=30,
            exclude_categories=[],
        ),
        news=NewsConfig(
            newsapi=NewsApiConfig(
                enabled=True,
                poll_interval_seconds=300,
                languages=["en", "es"],
                page_size=50,
                sources=[],
            ),
            gdelt=GdeltConfig(
                enabled=True,
                poll_interval_seconds=900,
                timespan="15min",
                max_records=100,
            ),
            telegram=TelegramConfig(
                enabled=False,
                poll_interval_seconds=60,
                channels=[],
                message_lookback_minutes=15,
                messages_per_channel=50,
            ),
            cache_ttl_seconds=3600,
            dedup_similarity_threshold=0.85,
        ),
        llm=LlmConfig(
            provider="anthropic",
            model="claude-haiku-4-5",
            max_tokens=2048,
            temperature=0.2,
            retry_attempts=3,
            retry_delay_seconds=5,
            min_confidence_threshold=60,
            cache_analysis=True,
            cache_ttl_seconds=1800,
            daily_spend_limit_usd=5.0,
            dry_run=False,
            ollama_base_url="http://localhost:11434",
            ollama_timeout_seconds=120,
        ),
        polymarket=PolymarketConfig(
            gamma_api_url="https://gamma-api.polymarket.com",
            clob_api_url="https://clob.polymarket.com",
            scan_interval_seconds=300,
            request_timeout_seconds=30,
            trading_fee_pct=0.0,
            simulated_slippage_pct=0.005,
        ),
        decision=DecisionConfig(
            llm_consultation_threshold=0.08,
            require_news_for_entry=True,
            reevaluate_open_positions_minutes=15,
            markets_to_analyze_per_cycle=15,
            category_priority_boost={
                "Politics": 1.5,
                "Geopolitics": 1.5,
                "Crypto": 1.4,
                "Sports": 0.8,
                "Esports": 0.4,
            },
            fallback_news_lookback="7d",
            enable_fallback_search=True,
            allow_low_info_trades=True,
            low_info_min_confidence=75,
            low_info_size_multiplier=0.5,
            low_info_min_articles=1,
        ),
        reports=ReportsConfig(
            output_directory="reports",
            filename_format="%Y-%m-%d_report.xlsx",
            generation_time="23:55",
            include_charts=True,
            conditional_formatting=True,
        ),
        logging=LoggingConfig(
            level="INFO",
            log_directory="logs",
            rotation_size_mb=50,
            retention_days=30,
            log_llm_decisions=True,
            log_format="{time} | {level} | {message}",
        ),
        database=DatabaseConfig(
            type="sqlite",
            path="data/test.db",
        ),
        notifications=NotificationsConfig(
            discord=DiscordConfig(
                enabled=True,
                username="TestBot",
                avatar_url="",
                notify_on_trade_open=True,
                notify_on_trade_close=True,
                notify_on_stop_loss=True,
                notify_on_take_profit=True,
                notify_on_drawdown_warning=True,
                notify_on_bot_pause=True,
                notify_daily_summary=True,
                daily_summary_time="23:59",
                color_profit=3066993,
                color_loss=15158332,
                color_info=3447003,
                color_warning=16776960,
            ),
        ),
        anthropic_api_key="test-key",
        newsapi_key="test-key",
        discord_webhook_url="https://discord.com/api/webhooks/test/test",
        telegram_api_id=None,
        telegram_api_hash=None,
        telegram_phone=None,
    )


@pytest.fixture
def config() -> BotConfig:
    """Standard test configuration (calibrated for 150€)."""
    return build_test_config()


@pytest.fixture
def config_factory() -> Callable[..., BotConfig]:
    """Factory for building custom configs in specific tests."""
    return build_test_config
