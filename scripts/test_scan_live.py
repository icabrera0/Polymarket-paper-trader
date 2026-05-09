"""
Prueba en vivo del MarketScanner contra Polymarket.

Ejecutar desde la raíz del proyecto:
    python scripts/test_scan_live.py

NO requiere ninguna API key (la Gamma API de Polymarket es pública).
Sí requiere conexión a Internet.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path para poder importar `src`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402
from src.market_scanner import MarketScanner  # noqa: E402


def main() -> None:
    config = load_config()
    scanner = MarketScanner(config)

    print("Conectando a Polymarket Gamma API...")
    markets = scanner.scan(force_refresh=True)
    print(f"\nEncontrados {len(markets)} mercados operables tras filtros\n")

    if not markets:
        print("(No hay mercados que pasen los filtros actuales)")
        print("Filtros aplicados:")
        f = config.market_filters
        print(f"  - Volumen 24h mínimo: ${f.min_volume_24h_usd:,.0f}")
        print(f"  - Spread máximo: {f.max_spread_cents}")
        print(f"  - Tiempo a cierre: {f.min_time_to_close_hours}h - "
              f"{f.max_time_to_close_days} días")
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
                f"    Cierra: {m.end_date.strftime('%Y-%m-%d %H:%M UTC')} "
                f"(en {ttc:.0f}h)"
            )
        if m.category:
            print(f"    Categoría: {m.category}")
        print()

    if len(markets) > 10:
        print(f"... y {len(markets) - 10} mercados más.")


if __name__ == "__main__":
    main()
