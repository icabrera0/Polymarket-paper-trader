"""
Live test of the MarketScanner against Polymarket.

Run from the project root:
    python scripts/test_scan_live.py

Does NOT require any API key (Polymarket's Gamma API is public).
Does require an Internet connection.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is in sys.path so we can import `src`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402
from src.market_scanner import MarketScanner  # noqa: E402


def main() -> None:
    config = load_config()
    scanner = MarketScanner(config)

    print("Connecting to Polymarket Gamma API...")
    markets = scanner.scan(force_refresh=True)
    print(f"\nFound {len(markets)} tradeable markets after filters\n")

    if not markets:
        print("(No markets pass the current filters)")
        print("Applied filters:")
        f = config.market_filters
        print(f"  - Min 24h volume: ${f.min_volume_24h_usd:,.0f}")
        print(f"  - Max spread: {f.max_spread_cents}")
        print(f"  - Time to close: {f.min_time_to_close_hours}h - "
              f"{f.max_time_to_close_days} days")
        return

    print(f"{'─' * 90}")
    for i, m in enumerate(markets[:10], 1):
        print(f"{i:2}. {m.question[:72]}")
        print(
            f"    YES={m.yes_price:.3f} | NO={m.no_price:.3f} | "
            f"vol24h=${m.volume_24h_usd:>10,.0f} | spread={m.spread:.4f}"
        )
        if m.end_date:
            ttc = m.time_to_close_hours
            print(
                f"    Closes: {m.end_date.strftime('%Y-%m-%d %H:%M UTC')} "
                f"(in {ttc:.0f}h)"
            )
        if m.category:
            print(f"    Category: {m.category}")
        print()

    if len(markets) > 10:
        print(f"... and {len(markets) - 10} more markets.")


if __name__ == "__main__":
    main()
