"""
Runs the bot backtesting against resolved Polymarket markets.

Usage:
    python scripts/run_backtest.py                    # 50 markets, current mode
    python scripts/run_backtest.py --mode replay      # Historical news (slower)
    python scripts/run_backtest.py --markets 100      # More markets
    python scripts/run_backtest.py --balance 200      # Custom initial balance
    python scripts/run_backtest.py --excel            # Export results to Excel

IMPORTANT NOTES:
  - "current" mode has look-ahead bias (news is from today, not from the time
    of the market). Useful for calibrating the pipeline, NOT for measuring
    the strategy in a realistic way.
  - "replay" mode is more realistic but depends on GDELT historical coverage,
    which is limited for niche events.
  - The backtest uses a separate DB (data/backtest.db) and does not touch production.
  - If using Ollama, make sure the service is running.
  - Each analyzed market consumes ~1 LLM call. With 50 markets and
    Ollama: ~10-15 minutes. With Anthropic: ~$0.05-0.10.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtester import Backtester  # noqa: E402
from src.config_loader import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtesting of the Polymarket Paper Trading Bot"
    )
    parser.add_argument(
        "--mode",
        choices=["current", "replay"],
        default="current",
        help=(
            "current: today's news (look-ahead bias, fast). "
            "replay: historical GDELT news (more realistic, slower)."
        ),
    )
    parser.add_argument(
        "--markets",
        type=int,
        default=50,
        help="Number of resolved markets to analyze (default: 50).",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=None,
        help="Initial balance in EUR (default: from config, 150€).",
    )
    parser.add_argument(
        "--excel",
        action="store_true",
        help="Export results to Excel in reports/backtest_YYYY-MM-DD.xlsx",
    )
    args = parser.parse_args()

    config = load_config()

    initial_balance = args.balance or config.paper_trading.initial_balance_eur

    print()
    print("═" * 65)
    print("  POLYMARKET PAPER TRADING BOT — BACKTESTING")
    print("═" * 65)
    print(f"  Mode:             {args.mode}")
    print(f"  Markets:          {args.markets}")
    print(f"  Initial balance:  €{initial_balance:.2f}")
    print(f"  LLM:              {config.llm.provider} ({config.llm.model})")
    print()

    if config.llm.provider == "ollama":
        from src.llm_client import OllamaClient, OllamaUnavailable
        try:
            OllamaClient(config).verify_setup()
            print("  ✓ Ollama OK")
        except OllamaUnavailable as exc:
            print(f"  ✗ Ollama not available: {exc}")
            print("    Run 'ollama serve' first.")
            sys.exit(1)

    print(f"  Starting backtesting ({args.markets} markets)...")
    print("  This may take several minutes. Ctrl+C to cancel.")
    print()

    backtester = Backtester(
        config=config,
        mode=args.mode,
        max_markets=args.markets,
        initial_balance=initial_balance,
    )

    try:
        result = backtester.run()
    except KeyboardInterrupt:
        print("\n  Backtesting cancelled.")
        sys.exit(0)

    result.print_summary()

    if args.excel:
        _export_excel(result, config)


def _export_excel(result, config) -> None:
    """Exports results to a basic backtesting Excel file."""
    from datetime import datetime, timezone
    from pathlib import Path as P

    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment

    output_dir = P(config.reports.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"backtest_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M')}.xlsx"
    out_path = output_dir / fname

    wb = openpyxl.Workbook()
    ws_summary = wb.active
    ws_summary.title = "Backtest Summary"

    # Header
    header_font = Font(name="Arial", bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", start_color="1F3864", end_color="1F3864")
    ws_summary["A1"] = "Parameter"
    ws_summary["B1"] = "Value"
    for cell in ["A1", "B1"]:
        ws_summary[cell].font = header_font
        ws_summary[cell].fill = header_fill

    kpis = [
        ("Mode", result.mode),
        ("Markets analyzed", result.markets_analyzed),
        ("Trades executed", result.trades_executed),
        ("Win rate", f"{result.win_rate:.1%}"),
        ("Total P&L", f"€{result.total_pnl_eur:+.2f}"),
        ("Initial balance", f"€{result.initial_balance:.2f}"),
        ("Final balance", f"€{result.final_balance:.2f}"),
        ("Total return",
         f"{(result.final_balance - result.initial_balance) / result.initial_balance:+.2%}"),
        ("Max drawdown", f"{result.max_drawdown_pct:.2%}"),
        ("Sharpe ratio", f"{result.sharpe_ratio:.2f}"),
    ]
    for row, (k, v) in enumerate(kpis, 2):
        ws_summary[f"A{row}"] = k
        ws_summary[f"B{row}"] = v

    ws_summary.column_dimensions["A"].width = 25
    ws_summary.column_dimensions["B"].width = 18

    # Detailed trades sheet
    ws_trades = wb.create_sheet("Detailed Trades")
    trade_headers = [
        "Market", "YES won", "Side", "Entry", "Exit",
        "Size €", "P&L €", "P&L %",
        "Confidence", "Edge", "Articles", "Low info", "LLM Rec.",
    ]
    for col_idx, header in enumerate(trade_headers, 1):
        cell = ws_trades.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    executed = [
        t for t in result.trades
        if t.decision == "OPEN_TRADE"
    ]
    for r_idx, t in enumerate(executed, 2):
        from openpyxl.utils import get_column_letter
        row_values = [
            t.market_question[:60],
            "Yes" if t.resolved_yes else "No",
            t.side.value if hasattr(t.side, "value") else str(t.side),
            t.entry_price_simulated,
            t.exit_price,
            t.size_eur,
            t.pnl_eur,
            t.pnl_pct,
            t.confidence,
            t.edge,
            t.num_articles,
            "Yes" if t.is_low_info else "No",
            t.llm_recommendation,
        ]
        for c_idx, value in enumerate(row_values, 1):
            ws_trades.cell(row=r_idx, column=c_idx, value=value)

        # P&L colors
        pnl_cell = ws_trades.cell(row=r_idx, column=7)
        if (t.pnl_eur or 0) >= 0:
            pnl_cell.fill = PatternFill("solid", start_color="C6EFCE", end_color="C6EFCE")
        else:
            pnl_cell.fill = PatternFill("solid", start_color="FFC7CE", end_color="FFC7CE")

    wb.save(out_path)
    print(f"  Excel exported: {out_path}")


if __name__ == "__main__":
    main()
