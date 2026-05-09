"""
Entry point for the Polymarket Paper Trading Bot.

Usage:
    python main.py

The bot starts, runs an initial cycle immediately, and then
loops according to the intervals configured in settings.yaml.

To stop it: Ctrl+C (clean shutdown) or kill -SIGTERM <pid>.
"""

from src.orchestrator import main

if __name__ == "__main__":
    main()