"""
Virtual paper trading balance management.

Allows viewing, resetting, or adjusting the balance from the command line.
This same functionality will be available in the Streamlit dashboard (module 11).

Usage:
    python scripts/manage_balance.py status          # View current balance
    python scripts/manage_balance.py reset           # Reset to config value (150€)
    python scripts/manage_balance.py reset 200       # Reset to 200€
    python scripts/manage_balance.py add 50          # Add 50€ to current balance
    python scripts/manage_balance.py subtract 30     # Withdraw 30€ from current balance

WARNING: the bot must be stopped before running any operation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402
from src.database import Database  # noqa: E402


def _get_last_balance(db: Database) -> float:
    history = db.get_balance_history()
    return float(history[-1]["balance_eur"]) if history else 0.0


def _get_last_peak(db: Database) -> float:
    history = db.get_balance_history()
    return float(history[-1]["peak_balance"]) if history else 0.0


def _confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").strip().lower() in ("s", "si", "y", "yes")


def cmd_status(db: Database, config) -> None:
    history = db.get_balance_history()
    balance = _get_last_balance(db)
    peak = _get_last_peak(db)
    open_pos = db.get_open_positions()
    initial = config.paper_trading.initial_balance_eur
    pnl = balance - initial
    pnl_pct = pnl / initial if initial > 0 else 0.0
    drawdown = (peak - balance) / peak if peak > 0 else 0.0

    print("\n" + "═" * 50)
    print("  PAPER TRADING STATUS")
    print("═" * 50)
    print(f"  Current balance:      €{balance:>10.2f}")
    print(f"  Initial balance:      €{initial:>10.2f}  (config)")
    print(f"  Historical peak:      €{peak:>10.2f}")
    print(f"  Current drawdown:     {drawdown:>10.2%}")
    print(f"  Total P&L:            €{pnl:>+10.2f}  ({pnl_pct:+.2%})")
    print(f"  Open positions:       {len(open_pos):>10}")
    print(f"  Snapshots in DB:      {len(history):>10}")
    if open_pos:
        print()
        print("  Open positions:")
        for p in open_pos:
            print(f"    • [{p.side.value}] {p.market_question[:45]}")
            print(f"      Entry €{p.entry_price:.4f} | Size €{p.size_eur:.2f}")
    print("═" * 50 + "\n")


def cmd_reset(db: Database, config, new_balance: float) -> None:
    current = _get_last_balance(db)
    open_pos = db.get_open_positions()
    print(f"\n  Current balance: €{current:.2f}")
    print(f"  New balance:     €{new_balance:.2f}")
    if open_pos:
        print(f"  ⚠  There are {len(open_pos)} open position(s).")
        print("     The reset changes the accounting balance but does NOT close positions.")
    if not _confirm("  Confirm reset?"):
        print("  Cancelled.\n")
        return
    new_peak = max(new_balance, _get_last_peak(db))
    db.log_balance(
        balance_eur=new_balance,
        peak_balance=new_peak,
        drawdown_pct=max(0.0, (new_peak - new_balance) / new_peak if new_peak > 0 else 0.0),
        open_positions=len(open_pos),
        event="MANUAL_RESET",
    )
    print(f"\n  ✓ Balance reset to €{new_balance:.2f}")
    print("    If you also want to change the startup value, edit:")
    print("    config/settings.yaml → paper_trading.initial_balance_eur\n")


def cmd_adjust(db: Database, action: str, amount: float) -> None:
    if amount <= 0:
        print("ERROR: the amount must be positive.")
        return
    current = _get_last_balance(db)
    new_balance = current + amount if action == "add" else current - amount
    if new_balance < 0:
        print(f"ERROR: balance would become negative (€{new_balance:.2f}). Operation cancelled.")
        return
    open_pos = db.get_open_positions()
    verb = "add" if action == "add" else "withdraw"
    print(f"\n  Current balance: €{current:.2f}")
    print(f"  You are about to {verb}: €{amount:.2f}")
    print(f"  New balance:     €{new_balance:.2f}")
    if not _confirm("  Confirm?"):
        print("  Cancelled.\n")
        return
    current_peak = _get_last_peak(db)
    new_peak = max(new_balance, current_peak)
    db.log_balance(
        balance_eur=new_balance,
        peak_balance=new_peak,
        drawdown_pct=max(0.0, (new_peak - new_balance) / new_peak if new_peak > 0 else 0.0),
        open_positions=len(open_pos),
        event=f"MANUAL_{'ADD' if action == 'add' else 'SUBTRACT'}",
    )
    print(f"\n  ✓ Balance updated: €{new_balance:.2f}\n")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    config = load_config()
    db = Database(config.database.path)
    cmd = sys.argv[1].lower()

    try:
        if cmd == "status":
            cmd_status(db, config)

        elif cmd == "reset":
            new_balance = (
                float(sys.argv[2])
                if len(sys.argv) > 2
                else config.paper_trading.initial_balance_eur
            )
            cmd_reset(db, config, new_balance)

        elif cmd in ("add", "subtract"):
            if len(sys.argv) < 3:
                print(f"ERROR: missing amount. Usage: manage_balance.py {cmd} AMOUNT")
                sys.exit(1)
            cmd_adjust(db, cmd, float(sys.argv[2]))

        else:
            print(f"ERROR: unknown command '{cmd}'")
            print("Commands: status | reset [amount] | add amount | subtract amount")
            sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
