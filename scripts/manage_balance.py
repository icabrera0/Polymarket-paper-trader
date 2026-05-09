"""
Gestión del balance virtual de paper trading.

Permite ver, resetear o ajustar el balance desde la línea de comandos.
Esta misma funcionalidad estará en el dashboard Streamlit (módulo 11).

Uso:
    python scripts/manage_balance.py status          # Ver balance actual
    python scripts/manage_balance.py reset           # Reset al valor de config (150€)
    python scripts/manage_balance.py reset 200       # Reset a 200€
    python scripts/manage_balance.py add 50          # Añadir 50€ al balance actual
    python scripts/manage_balance.py subtract 30     # Retirar 30€ del balance actual

ADVERTENCIA: el bot debe estar parado antes de ejecutar cualquier operación.
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
    return input(f"{prompt} [s/N]: ").strip().lower() in ("s", "si", "y", "yes")


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
    print("  ESTADO DEL PAPER TRADING")
    print("═" * 50)
    print(f"  Balance actual:       €{balance:>10.2f}")
    print(f"  Balance inicial:      €{initial:>10.2f}  (config)")
    print(f"  Peak histórico:       €{peak:>10.2f}")
    print(f"  Drawdown actual:      {drawdown:>10.2%}")
    print(f"  P&L total:            €{pnl:>+10.2f}  ({pnl_pct:+.2%})")
    print(f"  Posiciones abiertas:  {len(open_pos):>10}")
    print(f"  Snapshots en DB:      {len(history):>10}")
    if open_pos:
        print()
        print("  Posiciones abiertas:")
        for p in open_pos:
            print(f"    • [{p.side.value}] {p.market_question[:45]}")
            print(f"      Entrada €{p.entry_price:.4f} | Tamaño €{p.size_eur:.2f}")
    print("═" * 50 + "\n")


def cmd_reset(db: Database, config, new_balance: float) -> None:
    current = _get_last_balance(db)
    open_pos = db.get_open_positions()
    print(f"\n  Balance actual: €{current:.2f}")
    print(f"  Nuevo balance:  €{new_balance:.2f}")
    if open_pos:
        print(f"  ⚠  Hay {len(open_pos)} posición/es abierta/s.")
        print("     El reset cambia el balance contable pero NO cierra posiciones.")
    if not _confirm("  ¿Confirmar reset?"):
        print("  Cancelado.\n")
        return
    new_peak = max(new_balance, _get_last_peak(db))
    db.log_balance(
        balance_eur=new_balance,
        peak_balance=new_peak,
        drawdown_pct=max(0.0, (new_peak - new_balance) / new_peak if new_peak > 0 else 0.0),
        open_positions=len(open_pos),
        event="MANUAL_RESET",
    )
    print(f"\n  ✓ Balance restablecido a €{new_balance:.2f}")
    print("    Si quieres que el valor de arranque también cambie, edita:")
    print("    config/settings.yaml → paper_trading.initial_balance_eur\n")


def cmd_adjust(db: Database, action: str, amount: float) -> None:
    if amount <= 0:
        print("ERROR: el importe debe ser positivo.")
        return
    current = _get_last_balance(db)
    new_balance = current + amount if action == "add" else current - amount
    if new_balance < 0:
        print(f"ERROR: el balance quedaría negativo (€{new_balance:.2f}). Operación cancelada.")
        return
    open_pos = db.get_open_positions()
    verb = "añadir" if action == "add" else "retirar"
    print(f"\n  Balance actual: €{current:.2f}")
    print(f"  Vas a {verb}: €{amount:.2f}")
    print(f"  Balance nuevo:  €{new_balance:.2f}")
    if not _confirm("  ¿Confirmar?"):
        print("  Cancelado.\n")
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
    print(f"\n  ✓ Balance actualizado: €{new_balance:.2f}\n")


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
                print(f"ERROR: falta el importe. Uso: manage_balance.py {cmd} IMPORTE")
                sys.exit(1)
            cmd_adjust(db, cmd, float(sys.argv[2]))

        else:
            print(f"ERROR: comando desconocido '{cmd}'")
            print("Comandos: status | reset [importe] | add importe | subtract importe")
            sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
