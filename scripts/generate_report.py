"""
Genera manualmente el reporte Excel del día.

Uso:
    python scripts/generate_report.py            # reporte de hoy
    python scripts/generate_report.py 2026-04-30 # reporte de ese día concreto

El bot también lo genera automáticamente cada día a la hora configurada en
settings.yaml (reports.generation_time) cuando esté en marcha el orquestador
(módulo 9). Este script es para forzarlo manualmente.
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
            print(f"ERROR: fecha inválida '{sys.argv[1]}'. Usa formato YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = datetime.now(timezone.utc)

    db = Database(config.database.path)
    gen = ReportGenerator(config, db)
    print(f"Generando reporte para {target_date.strftime('%Y-%m-%d')}...")
    out_path = gen.generate_daily_report(target_date=target_date)
    db.close()
    print(f"\n✓ Reporte generado: {out_path}")
    print(f"  Tamaño: {out_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
