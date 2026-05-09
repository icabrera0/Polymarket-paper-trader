"""
dev_runner.py — Hot-reload wrapper for the trading bot.

Monitors src/ and the project root for Python file changes. When you save
any .py file, it signals the bot to finish its current operation and restart
with the new code — no manual Ctrl+C needed.

Usage:
    python dev_runner.py          # instead of: python main.py
    (or launch via start_dev.bat)

How it works:
    1. Starts main.py as a child process.
    2. Watches src/ and root *.py files with watchdog.
    3. On any .py save: writes data/hot_reload.flag.
    4. The orchestrator detects the flag at its next 1-second tick,
       closes the DB cleanly, and exits with code 42.
    5. dev_runner sees exit code 42 → restarts the bot.
    6. Normal Ctrl+C (exit code 0) → dev_runner stops too.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_FLAG = Path("data") / "hot_reload.flag"
_RELOAD_CODE = 42
_WATCHED_DIRS = ["src"]
_WATCHED_ROOT = "."  # catches main.py, dev_runner.py, etc.
# Seconds to ignore file events after each (re)start.
# Python writes __pycache__/*.pyc on import; on Windows this can appear as a
# .py modification event (ReadDirectoryChangesW quirk). 15s covers the 2s
# restart delay plus the time Python takes to import all modules.
_STARTUP_COOLDOWN = 15


def _write_flag() -> None:
    _FLAG.parent.mkdir(exist_ok=True)
    _FLAG.touch()


try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    print("[dev_runner] ERROR: watchdog is not installed.")
    print("[dev_runner] Run:  pip install watchdog")
    sys.exit(1)


class _ChangeHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self._signalled = False
        self._ignore_until: float = 0.0

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        path = str(event.src_path)
        if not path.endswith(".py"):
            return
        if "__pycache__" in path:
            return
        if self._signalled:
            return
        if time.time() < self._ignore_until:
            return  # startup cooldown — suppress spurious __pycache__ events
        self._signalled = True
        print(f"\n[dev_runner] Changed: {path}")
        print("[dev_runner] Signalling bot to reload after current operation...")
        _write_flag()

    def reset(self) -> None:
        self._signalled = False
        # Arm cooldown now so spurious import-time events are suppressed
        # during the 2s restart delay + Python startup.
        self._ignore_until = time.time() + _STARTUP_COOLDOWN


def main() -> None:
    handler = _ChangeHandler()
    observer = Observer()
    for d in _WATCHED_DIRS:
        observer.schedule(handler, d, recursive=True)
    observer.schedule(handler, _WATCHED_ROOT, recursive=False)
    observer.start()

    print("[dev_runner] Hot-reload active. Save any .py file to restart the bot.")
    print("[dev_runner] Press Ctrl+C here to stop everything.\n")

    proc: subprocess.Popen | None = None
    try:
        while True:
            print("[dev_runner] Starting bot (main.py)...")
            proc = subprocess.Popen([sys.executable, "main.py"])

            # Poll until the process exits
            while proc.poll() is None:
                time.sleep(0.5)

            exit_code = proc.returncode
            handler.reset()

            if exit_code == _RELOAD_CODE:
                print("[dev_runner] Bot reloading — restarting in 2s...")
                time.sleep(2)
                # loop continues → bot restarts
            else:
                # Clean shutdown (Ctrl+C in bot window, or error)
                print(f"[dev_runner] Bot exited with code {exit_code}. Stopping watcher.")
                break

    except KeyboardInterrupt:
        print("\n[dev_runner] Stopping bot and watcher...")
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait()
    finally:
        observer.stop()
        observer.join()
        print("[dev_runner] Done.")


if __name__ == "__main__":
    main()
