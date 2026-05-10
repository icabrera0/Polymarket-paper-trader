"""
Interactive Telegram login for the social ingestor (run ONCE).

Creates data/telegram_social.session — a separate session from the news
ingestor so the two clients don't conflict.  Only needed if you want
social_ingestor Telegram to fetch public channel posts.

Prerequisites:
  TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_PHONE in .env

Run:
    python scripts/telegram_social_login.py
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
        print("ERROR: Telethon not installed. Run: pip install telethon")
        sys.exit(1)

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    session_path = str(data_dir / "telegram_social")

    # Delete stale session if it exists so we get a clean auth
    stale = data_dir / "telegram_social.session"
    if stale.exists():
        stale.unlink()
        print(f"Deleted stale session: {stale}")

    print(f"Starting Telegram social session with phone {config.telegram_phone}...")
    print("You will receive a code via the Telegram app (not SMS).")
    print()

    client = TelegramClient(session_path, config.telegram_api_id, config.telegram_api_hash)
    try:
        client.start(phone=config.telegram_phone)
        me = client.get_me()
        print(f"\n✓ Session created successfully.")
        print(f"  User: {me.first_name} (@{me.username or 'no username'})")
        print(f"  Session saved at: {session_path}.session")
        print()
        print("The social ingestor will now use this session automatically.")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
