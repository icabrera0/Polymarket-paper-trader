"""
Compatibilidad retro: este módulo re-exporta los símbolos del nuevo
`llm_client.py`. Mantenido para no romper imports antiguos.

Los nuevos módulos deben importar directamente de `src.llm_client`.
"""

from src.llm_client import (  # noqa: F401
    AnthropicClient,
    AnthropicError,
    CreditsExhausted,
    DailyBudgetExceeded,
    LLMClient,
    LLMError,
    OllamaClient,
    OllamaUnavailable,
    build_llm_client,
)
