"""
Interactive Telegram login (run ONCE).

Creates data/telegram.session and leaves the client authorized for all
subsequent runs. You only need to do this:
- The first time you configure Telegram.
- If you lose the data/telegram.session file.
- If the session expires in Telegram (rare, usually due to security).

Prerequisites:
1. Have TELEGRAM_API_ID and TELEGRAM_API_HASH in .env (from my.telegram.org).
2. Have TELEGRAM_PHONE in international format (+1...).

Run:
    python scripts/telegram_login.py

It will ask for a code you will receive via the Telegram app (not SMS) and
optionally the 2FA password if you have it enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402


def main() -> None:
    config = load_config()

    if not config.telegram_api_id or not config.telegram_api_hash:
        print("ERROR: Missing TELEGRAM_API_ID and/or TELEGRAM_API_HASH in .env")
        print("Get them at https://my.telegram.org → API development tools")
        sys.exit(1)

    if not config.telegram_phone:
        print("ERROR: Missing TELEGRAM_PHONE in .env (format +1600000000)")
        sys.exit(1)

    try:
        from telethon.sync import TelegramClient
    except ImportError:
        print("ERROR: Telethon is not installed. Run: pip install telethon")
        sys.exit(1)

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    session_path = str(data_dir / "telegram")

    print(f"Starting Telegram session with phone {config.telegram_phone}...")
    print("You will receive a code via the Telegram app (not SMS).")
    print()

    client = TelegramClient(
        session_path,
        config.telegram_api_id,
        config.telegram_api_hash,
    )

    try:
        client.start(phone=config.telegram_phone)
        me = client.get_me()
        print(f"\n✓ Session created successfully.")
        print(f"  User: {me.first_name} (@{me.username or 'no username'})")
        print(f"  ID: {me.id}")
        print(f"  Session saved at: {session_path}.session")
        print()
        print("You can now enable telegram in config/settings.yaml:")
        print("    news.telegram.enabled: true")
        print("And add the channels you want to read.")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
