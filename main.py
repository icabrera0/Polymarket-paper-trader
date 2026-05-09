"""
Punto de entrada del Polymarket Paper Trading Bot.

Uso:
    python main.py

El bot arranca, ejecuta un ciclo inicial inmediatamente y luego
trabaja en bucle según los intervalos configurados en settings.yaml.

Para pararlo: Ctrl+C (cierre limpio) o kill -SIGTERM <pid>.
"""

from src.orchestrator import main

if __name__ == "__main__":
    main()
