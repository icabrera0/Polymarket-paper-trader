"""
Manually generates today's Excel report.

Usage:
    python scripts/generate_report.py            # today's report
    python scripts/generate_report.py 2026-04-30 # report for that specific day

The bot also generates it automatically each day at the time configured in
settings.yaml (reports.generation_time) when the orchestrator is running
(module 9). This script is for forcing it manually.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402
from src.database import Database  # noqa: E402
from src.report_generator import ReportGenerator  # noqa: E402


def main() -> None:
    config = load_config()
    target_date: datetime
    if len(sys.argv) > 1:
        try:
            target_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            print(f"ERROR: invalid date '{sys.argv[1]}'. Use format YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = datetime.now(timezone.utc)

    db = Database(config.database.path)
    gen = ReportGenerator(config, db)
    print(f"Generating report for {target_date.strftime('%Y-%m-%d')}...")
    out_path = gen.generate_daily_report(target_date=target_date)
    db.close()
    print(f"\n✓ Report generated: {out_path}")
    print(f"  Size: {out_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
