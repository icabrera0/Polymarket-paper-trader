"""
Diagnóstico del problema 'NewsSource has no attribute TELEGRAM'.

Si los tests fallan diciendo que TELEGRAM no existe pero tu archivo
src/models.py SÍ lo contiene, es porque Python está cargando una versión
cacheada o un archivo en otra ruta.

Este script muestra:
1. Desde qué archivo se está cargando NewsSource.
2. Qué miembros tiene en realidad.
3. Si hay archivos .pyc obsoletos que limpiar.

Ejecutar:
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
    print("DIAGNÓSTICO DE src/models.py")
    print("─" * 70)

    # 1) ¿Qué hay en disco?
    models_path = PROJECT_ROOT / "src" / "models.py"
    print(f"\n[1] Archivo en disco: {models_path}")
    print(f"    Existe: {models_path.exists()}")
    if models_path.exists():
        text = models_path.read_text(encoding="utf-8")
        has_telegram_text = 'TELEGRAM = "TELEGRAM"' in text
        print(f"    Contiene 'TELEGRAM = \"TELEGRAM\"': {has_telegram_text}")
        if not has_telegram_text:
            print("    ⚠ El archivo NO tiene la constante TELEGRAM.")
            print("      Asegúrate de copiar la versión nueva de models.py.")

    # 2) ¿Qué carga Python?
    print("\n[2] Importando NewsSource desde src.models...")
    try:
        from src.models import NewsSource
        import src.models as models_module

        loaded_from = getattr(models_module, "__file__", "?")
        print(f"    Cargado desde: {loaded_from}")

        members = list(NewsSource)
        print(f"    Miembros: {members}")
        has_telegram_runtime = hasattr(NewsSource, "TELEGRAM")
        print(f"    Tiene TELEGRAM en runtime: {has_telegram_runtime}")

        if not has_telegram_runtime:
            print("\n    ⚠ El archivo en disco y lo que Python carga DIFIEREN.")
            print("      Causas habituales:")
            print("      a) Caché .pyc obsoleta → borra src/__pycache__")
            print("      b) Hay dos copies de models.py en distintas rutas")
            print("      c) El editor no guardó el cambio")

    except Exception as exc:
        print(f"    ✗ Error importando: {exc}")
        return

    # 3) ¿Hay caches sospechosos?
    print("\n[3] Buscando archivos .pyc del módulo models...")
    pyc_files = list(PROJECT_ROOT.rglob("models.cpython-*.pyc"))
    if pyc_files:
        for pyc in pyc_files:
            print(f"    Encontrado: {pyc}")
        print("\n    Para limpiar todos los caches:")
        print("    rmdir /s /q src\\__pycache__")
        print("    rmdir /s /q tests\\__pycache__")
    else:
        print("    Sin caches sospechosos.")

    # 4) Buscar otros models.py en el proyecto
    print("\n[4] Buscando OTROS archivos models.py en el proyecto...")
    other_models = [
        p for p in PROJECT_ROOT.rglob("models.py")
        if "venv" not in str(p) and ".pytest_cache" not in str(p)
    ]
    if len(other_models) > 1:
        print(f"    ⚠ Hay {len(other_models)} archivos models.py:")
        for p in other_models:
            print(f"      {p}")
    else:
        print(f"    Solo hay uno: {other_models[0] if other_models else '(ninguno)'}")

    print("\n" + "─" * 70)


if __name__ == "__main__":
    main()
