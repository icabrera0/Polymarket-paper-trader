"""
Diagnosis for the 'NewsSource has no attribute TELEGRAM' problem.

If tests fail saying TELEGRAM doesn't exist but your
src/models.py file DOES contain it, it's because Python is loading a cached
version or a file at a different path.

This script shows:
1. Which file NewsSource is being loaded from.
2. What members it actually has.
3. Whether there are stale .pyc files to clean up.

Run:
    python scripts/diagnose_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    print("─" * 70)
    print("DIAGNOSIS OF src/models.py")
    print("─" * 70)

    # 1) What is on disk?
    models_path = PROJECT_ROOT / "src" / "models.py"
    print(f"\n[1] File on disk: {models_path}")
    print(f"    Exists: {models_path.exists()}")
    if models_path.exists():
        text = models_path.read_text(encoding="utf-8")
        has_telegram_text = 'TELEGRAM = "TELEGRAM"' in text
        print(f"    Contains 'TELEGRAM = \"TELEGRAM\"': {has_telegram_text}")
        if not has_telegram_text:
            print("    ⚠ The file does NOT have the TELEGRAM constant.")
            print("      Make sure to copy the new version of models.py.")

    # 2) What does Python load?
    print("\n[2] Importing NewsSource from src.models...")
    try:
        from src.models import NewsSource
        import src.models as models_module

        loaded_from = getattr(models_module, "__file__", "?")
        print(f"    Loaded from: {loaded_from}")

        members = list(NewsSource)
        print(f"    Members: {members}")
        has_telegram_runtime = hasattr(NewsSource, "TELEGRAM")
        print(f"    Has TELEGRAM at runtime: {has_telegram_runtime}")

        if not has_telegram_runtime:
            print("\n    ⚠ The file on disk and what Python loads DIFFER.")
            print("      Common causes:")
            print("      a) Stale .pyc cache → delete src/__pycache__")
            print("      b) There are two copies of models.py at different paths")
            print("      c) The editor did not save the change")

    except Exception as exc:
        print(f"    ✗ Error importing: {exc}")
        return

    # 3) Are there suspicious caches?
    print("\n[3] Searching for .pyc files of the models module...")
    pyc_files = list(PROJECT_ROOT.rglob("models.cpython-*.pyc"))
    if pyc_files:
        for pyc in pyc_files:
            print(f"    Found: {pyc}")
        print("\n    To clean all caches:")
        print("    rmdir /s /q src\\__pycache__")
        print("    rmdir /s /q tests\\__pycache__")
    else:
        print("    No suspicious caches.")

    # 4) Search for other models.py in the project
    print("\n[4] Searching for OTHER models.py files in the project...")
    other_models = [
        p for p in PROJECT_ROOT.rglob("models.py")
        if "venv" not in str(p) and ".pytest_cache" not in str(p)
    ]
    if len(other_models) > 1:
        print(f"    ⚠ There are {len(other_models)} models.py files:")
        for p in other_models:
            print(f"      {p}")
    else:
        print(f"    Only one found: {other_models[0] if other_models else '(none)'}")

    print("\n" + "─" * 70)


if __name__ == "__main__":
    main()
