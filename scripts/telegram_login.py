"""
Login interactivo para Telegram (ejecutar UNA SOLA VEZ).

Crea data/telegram.session y deja al cliente autorizado para todas las
ejecuciones posteriores. Solo necesitas hacer esto:
- La primera vez que configures Telegram.
- Si pierdes el archivo data/telegram.session.
- Si la sesión expira en Telegram (raro, suele ser por seguridad).

Pasos previos:
1. Tener TELEGRAM_API_ID y TELEGRAM_API_HASH en .env (de my.telegram.org).
2. Tener TELEGRAM_PHONE en formato internacional (+34...).

Ejecutar:
    python scripts/telegram_login.py

Te pedirá un código que recibirás por Telegram (no SMS) y opcionalmente la
contraseña 2FA si la tienes activada.
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
        print("ERROR: Falta TELEGRAM_API_ID y/o TELEGRAM_API_HASH en .env")
        print("Consíguelos en https://my.telegram.org → API development tools")
        sys.exit(1)

    if not config.telegram_phone:
        print("ERROR: Falta TELEGRAM_PHONE en .env (formato +34600000000)")
        sys.exit(1)

    try:
        from telethon.sync import TelegramClient
    except ImportError:
        print("ERROR: Telethon no está instalado. Ejecuta: pip install telethon")
        sys.exit(1)

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    session_path = str(data_dir / "telegram")

    print(f"Iniciando sesión en Telegram con teléfono {config.telegram_phone}...")
    print("Vas a recibir un código por la app de Telegram (no SMS).")
    print()

    client = TelegramClient(
        session_path,
        config.telegram_api_id,
        config.telegram_api_hash,
    )

    try:
        client.start(phone=config.telegram_phone)
        me = client.get_me()
        print(f"\n✓ Sesión creada correctamente.")
        print(f"  Usuario: {me.first_name} (@{me.username or 'sin username'})")
        print(f"  ID: {me.id}")
        print(f"  Sesión guardada en: {session_path}.session")
        print()
        print("Ya puedes activar telegram en config/settings.yaml:")
        print("    news.telegram.enabled: true")
        print("Y añadir los canales que quieras leer.")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
