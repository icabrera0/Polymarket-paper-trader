"""
Verificación e inicialización de Ollama para el bot.

Comprueba:
1. Ollama instalado y servicio corriendo en localhost:11434
2. Modelo configurado (config.llm.model) descargado; lo descarga si no
3. Inferencia de prueba para validar que todo funciona

Ejecutar UNA SOLA VEZ tras instalar Ollama:
    python scripts/setup_ollama.py

Si ya tienes Ollama y un modelo descargado, se ejecuta en segundos
y solo verifica. Si el modelo no está, lanza `ollama pull MODELO`
internamente (puede tardar varios minutos).
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402
from src.llm_client import OllamaClient, OllamaUnavailable  # noqa: E402


def main() -> None:
    config = load_config()

    print("─" * 70)
    print("VERIFICACIÓN DE OLLAMA")
    print("─" * 70)

    if config.llm.provider != "ollama":
        print(f"\nERROR: provider configurado es '{config.llm.provider}', no 'ollama'.")
        print("Cambia config/settings.yaml → llm.provider a 'ollama' primero.")
        sys.exit(1)

    print(f"\nServidor:  {config.llm.ollama_base_url}")
    print(f"Modelo:    {config.llm.model}")
    print()

    # 1) Crear cliente y verificar setup
    client = OllamaClient(config)
    try:
        client.verify_setup()
        print("✓ Ollama responde y el modelo está descargado.")
    except OllamaUnavailable as exc:
        msg = str(exc)
        print(f"✗ {msg}")

        # Si es porque el modelo no está, ofrecemos descargarlo
        if "no está descargado" in msg or "not installed" in msg.lower():
            print()
            answer = input(
                f"¿Descargar '{config.llm.model}' ahora con 'ollama pull'? "
                f"(puede tardar varios minutos) [s/N]: "
            ).strip().lower()
            if answer in ("s", "si", "y", "yes"):
                print(f"\nDescargando {config.llm.model}...")
                result = subprocess.run(
                    ["ollama", "pull", config.llm.model],
                    text=True,
                )
                if result.returncode != 0:
                    print("ERROR: 'ollama pull' falló. Verifica que Ollama está instalado.")
                    sys.exit(1)
                # Reintentar verificación
                try:
                    client.verify_setup()
                    print("\n✓ Modelo descargado y disponible.")
                except OllamaUnavailable as exc2:
                    print(f"\n✗ Tras descargar sigue fallando: {exc2}")
                    sys.exit(1)
            else:
                print("Cancelado. Descárgalo manualmente con:")
                print(f"   ollama pull {config.llm.model}")
                sys.exit(1)
        else:
            print()
            print("Ollama no responde. Soluciones:")
            print("  - Instalar: https://ollama.com/download")
            print("  - Arrancar el servicio: 'ollama serve'")
            print("  - Verificar la URL en config/settings.yaml")
            sys.exit(1)

    # 2) Inferencia de prueba (rápida)
    print("\nLanzando inferencia de prueba...")
    try:
        result = client.complete(
            system_prompt="You return JSON only. Nothing else.",
            user_prompt='Reply with this exact JSON: {"status":"ok","value":42}',
            max_tokens=64,
            temperature=0.0,
            force_json=True,
        )
    except Exception as exc:
        print(f"✗ Inferencia falló: {exc}")
        sys.exit(1)

    print(f"  Respuesta cruda: {result['text'][:200]}")
    print(f"  Tokens (in/out): {result['input_tokens']} / {result['output_tokens']}")
    parsed = OllamaClient.extract_json(result["text"])
    if parsed:
        print(f"  JSON parseado:   {parsed}")
    else:
        print("  ⚠ No se pudo parsear como JSON. El modelo puede no respetar bien")
        print("    el formato JSON. Considera otro modelo o ajustar el prompt.")

    print()
    print("─" * 70)
    print("Ollama listo. Ya puedes ejecutar:")
    print("   python scripts/test_sentiment_live.py")
    print("─" * 70)


if __name__ == "__main__":
    main()
