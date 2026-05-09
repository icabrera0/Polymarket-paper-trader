"""
Ollama verification and initialization for the bot.

Checks:
1. Ollama installed and service running on localhost:11434
2. Configured model (config.llm.model) downloaded; downloads it if not present
3. Test inference to validate everything works

Run ONCE after installing Ollama:
    python scripts/setup_ollama.py

If you already have Ollama and a downloaded model, it runs in seconds
and only verifies. If the model is not present, it launches `ollama pull MODEL`
internally (may take several minutes).
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
    print("OLLAMA VERIFICATION")
    print("─" * 70)

    if config.llm.provider != "ollama":
        print(f"\nERROR: configured provider is '{config.llm.provider}', not 'ollama'.")
        print("Change config/settings.yaml → llm.provider to 'ollama' first.")
        sys.exit(1)

    print(f"\nServer:  {config.llm.ollama_base_url}")
    print(f"Model:   {config.llm.model}")
    print()

    # 1) Create client and verify setup
    client = OllamaClient(config)
    try:
        client.verify_setup()
        print("✓ Ollama responds and the model is downloaded.")
    except OllamaUnavailable as exc:
        msg = str(exc)
        print(f"✗ {msg}")

        # If the model is not downloaded, offer to download it
        if "no está descargado" in msg or "not installed" in msg.lower():
            print()
            answer = input(
                f"Download '{config.llm.model}' now with 'ollama pull'? "
                f"(may take several minutes) [y/N]: "
            ).strip().lower()
            if answer in ("s", "si", "y", "yes"):
                print(f"\nDownloading {config.llm.model}...")
                result = subprocess.run(
                    ["ollama", "pull", config.llm.model],
                    text=True,
                )
                if result.returncode != 0:
                    print("ERROR: 'ollama pull' failed. Verify that Ollama is installed.")
                    sys.exit(1)
                # Retry verification
                try:
                    client.verify_setup()
                    print("\n✓ Model downloaded and available.")
                except OllamaUnavailable as exc2:
                    print(f"\n✗ Still failing after download: {exc2}")
                    sys.exit(1)
            else:
                print("Cancelled. Download it manually with:")
                print(f"   ollama pull {config.llm.model}")
                sys.exit(1)
        else:
            print()
            print("Ollama is not responding. Solutions:")
            print("  - Install: https://ollama.com/download")
            print("  - Start the service: 'ollama serve'")
            print("  - Check the URL in config/settings.yaml")
            sys.exit(1)

    # 2) Test inference (quick)
    print("\nRunning test inference...")
    try:
        result = client.complete(
            system_prompt="You return JSON only. Nothing else.",
            user_prompt='Reply with this exact JSON: {"status":"ok","value":42}',
            max_tokens=64,
            temperature=0.0,
            force_json=True,
        )
    except Exception as exc:
        print(f"✗ Inference failed: {exc}")
        sys.exit(1)

    print(f"  Raw response: {result['text'][:200]}")
    print(f"  Tokens (in/out): {result['input_tokens']} / {result['output_tokens']}")
    parsed = OllamaClient.extract_json(result["text"])
    if parsed:
        print(f"  Parsed JSON:   {parsed}")
    else:
        print("  ⚠ Could not parse as JSON. The model may not respect the JSON")
        print("    format well. Consider another model or adjusting the prompt.")

    print()
    print("─" * 70)
    print("Ollama ready. You can now run:")
    print("   python scripts/test_sentiment_live.py")
    print("─" * 70)


if __name__ == "__main__":
    main()
