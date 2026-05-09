"""
Unified LLM client: supports Anthropic (Claude) and Ollama (local).

The base class `LLMClient` defines the contract that any provider must
fulfill. The factory `build_llm_client()` selects the implementation based on
`config.llm.provider`:

    provider: "anthropic"  → AnthropicClient (costs money, daily cap)
    provider: "ollama"     → OllamaClient (free, local, no external network)

The SENTIMENT_ANALYZER talks to the interface, not the concrete provider,
so switching from one to the other is just a config change.

Common features:
- `complete_json()` with retries when JSON comes back malformed.
- Tracking of consumed tokens for reporting.
- `extract_json()` with 3 strategies (direct parse, ```json``` blocks, first
  balanced object).

Anthropic-only:
- Daily spend cap in USD (DailyBudgetExceeded).
- Detection of exhausted credits with reset info (CreditsExhausted).
- 0.5s throttling between calls.

Ollama-only:
- Pre-check that the model is downloaded and the server responds.
- Configurable timeout (local models can be slow).
- Native "format=json" when the model supports it to force valid JSON.
"""

from __future__ import annotations

import json
import re
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config_loader import BotConfig


# Anthropic prices in USD per million tokens (input, output)
ANTHROPIC_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5":  (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7":   (5.0, 25.0),
    "claude-opus-4-6":   (5.0, 25.0),
}


# =====================================================
# Exceptions
# =====================================================


class LLMError(Exception):
    """Generic LLM error (non-transient)."""


class DailyBudgetExceeded(LLMError):
    """Daily spend cap reached (Anthropic only)."""


class CreditsExhausted(LLMError):
    """The Anthropic account has no credits."""

    def __init__(
        self,
        message: str,
        reset_at: Optional[datetime] = None,
        retry_after_seconds: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.reset_at = reset_at
        self.retry_after_seconds = retry_after_seconds


class OllamaUnavailable(LLMError):
    """Ollama server is not responding or the model is not downloaded."""


# Backward-compatible aliases with the old anthropic_client module
AnthropicError = LLMError


# =====================================================
# Base interface
# =====================================================


class LLMClient(ABC):
    """Contract that any LLM provider must fulfill."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.cfg = config.llm
        self._log = logger.bind(module=self.__class__.__name__)

        # Accumulated metrics (common to all providers)
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_calls: int = 0
        # Protects counter increments when multiple threads call complete() concurrently
        self._stats_lock = threading.Lock()

    # ---------- Public API (common) ----------

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        force_json: bool = False,
    ) -> dict[str, Any]:
        """Calls the LLM and returns {text, input_tokens, output_tokens, ...}."""

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_attempts: int = 2,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Like complete() but parses the response as JSON.

        If the JSON comes back malformed, retries up to `max_attempts` times with
        a corrective message. This is important for local LLMs that may get the
        format wrong. Anthropic usually gets it right on the first try.
        """
        last_error: Optional[str] = None
        accumulated_meta: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "attempts": 0,
        }

        for attempt in range(1, max_attempts + 1):
            accumulated_meta["attempts"] = attempt
            current_user_prompt = user_prompt
            if last_error:
                current_user_prompt = (
                    f"{user_prompt}\n\n"
                    f"--- Your previous response was not valid JSON ---\n"
                    f"Error: {last_error}\n"
                    f"Return ONLY the JSON, with no text before or after."
                )
            result = self.complete(
                system_prompt=system_prompt,
                user_prompt=current_user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                force_json=True,
            )
            # Accumulate metrics
            accumulated_meta["input_tokens"] += result.get("input_tokens", 0)
            accumulated_meta["output_tokens"] += result.get("output_tokens", 0)
            accumulated_meta["estimated_cost_usd"] += result.get(
                "estimated_cost_usd", 0.0
            )

            parsed = self.extract_json(result["text"])
            if parsed is not None:
                accumulated_meta["stop_reason"] = result.get("stop_reason", "")
                return parsed, accumulated_meta

            last_error = (
                f"could not extract JSON from response of length "
                f"{len(result['text'])}"
            )
            self._log.warning(
                "Invalid JSON on attempt {}/{}. Received text (first 200): {}",
                attempt,
                max_attempts,
                result["text"][:200].replace("\n", " "),
            )

        raise LLMError(
            f"LLM response contains no valid JSON after {max_attempts} attempts. "
            f"Last error: {last_error}"
        )

    # ---------- Common helpers ----------

    @staticmethod
    def extract_json(text: str) -> Optional[dict[str, Any]]:
        """Attempts to extract a JSON object from text, tolerating preamble."""
        if not text:
            return None

        # Strategy 1: direct parse
        text_stripped = text.strip()
        try:
            result = json.loads(text_stripped)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: markdown block ```json ... ```
        markdown_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            re.DOTALL,
        )
        if markdown_match:
            try:
                result = json.loads(markdown_match.group(1))
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3: first { to the matching balanced }
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict):
                            return result
                    except (json.JSONDecodeError, ValueError):
                        return None
        return None


# =====================================================
# Anthropic implementation
# =====================================================


class AnthropicClient(LLMClient):
    """Client for the Anthropic API (Claude). Costs money, daily cap."""

    def __init__(self, config: BotConfig) -> None:
        super().__init__(config)
        if not config.anthropic_api_key:
            raise LLMError(
                "Missing ANTHROPIC_API_KEY in .env. Get it from "
                "https://console.anthropic.com (SEPARATE account from your Claude Pro/Max)."
            )

        try:
            import anthropic
            self._sdk = anthropic.Anthropic(api_key=config.anthropic_api_key)
            self._transient_errors: tuple[type[Exception], ...] = (
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.InternalServerError,
            )
            self._rate_limit_error = anthropic.RateLimitError
        except ImportError as exc:
            raise LLMError(
                "anthropic not installed. pip install anthropic"
            ) from exc

        self._last_call_ts: float = 0.0
        self._min_call_interval: float = 0.5

        # Daily spend tracking
        self._daily_spend_usd: float = 0.0
        self._spend_day_utc: str = self._today_str()

        self._price_in, self._price_out = ANTHROPIC_PRICING_USD_PER_MTOK.get(
            self.cfg.model, (3.0, 15.0)
        )
        if self.cfg.model not in ANTHROPIC_PRICING_USD_PER_MTOK:
            self._log.warning(
                "Model {} has no known pricing; using $3/$15 as default",
                self.cfg.model,
            )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        force_json: bool = False,  # ignored in Anthropic (no native flag)
    ) -> dict[str, Any]:
        max_tokens = max_tokens or self.cfg.max_tokens
        temperature = temperature if temperature is not None else self.cfg.temperature

        self._maybe_reset_daily_spend()

        if (
            self.cfg.daily_spend_limit_usd > 0
            and self._daily_spend_usd >= self.cfg.daily_spend_limit_usd
        ):
            raise DailyBudgetExceeded(
                f"Daily spend ${self._daily_spend_usd:.4f} has reached the "
                f"limit ${self.cfg.daily_spend_limit_usd:.2f} (UTC day "
                f"{self._spend_day_utc}). Resets at 00:00 UTC."
            )

        if self.cfg.dry_run:
            self._log.warning("DRY_RUN active. LLM will not be called.")
            return {
                "text": "{}",
                "input_tokens": 0,
                "output_tokens": 0,
                "stop_reason": "dry_run",
                "estimated_cost_usd": 0.0,
            }

        self._throttle()

        try:
            response = self._call_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except self._rate_limit_error as exc:
            self._handle_rate_limit_error(exc)
            raise
        except Exception as exc:
            self._log.error("Anthropic failed: {}", exc)
            raise LLMError(str(exc)) from exc

        text_parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        text = "".join(text_parts)

        in_tokens = getattr(response.usage, "input_tokens", 0)
        out_tokens = getattr(response.usage, "output_tokens", 0)
        cost = self._estimate_cost(in_tokens, out_tokens)

        with self._stats_lock:
            self.total_input_tokens += in_tokens
            self.total_output_tokens += out_tokens
            self.total_calls += 1
            self._daily_spend_usd += cost

        return {
            "text": text,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "stop_reason": getattr(response, "stop_reason", "unknown"),
            "estimated_cost_usd": cost,
        }

    # ---------- Anthropic internals ----------

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call_ts
        if elapsed < self._min_call_interval:
            time.sleep(self._min_call_interval - elapsed)
        self._last_call_ts = time.time()

    def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Any:
        @retry(
            stop=stop_after_attempt(self.cfg.retry_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=self.cfg.retry_delay_seconds,
                max=30,
            ),
            retry=retry_if_exception_type(self._transient_errors),
            reraise=True,
        )
        def _do_call() -> Any:
            return self._sdk.messages.create(
                model=self.cfg.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

        return _do_call()

    def _estimate_cost(self, in_tokens: int, out_tokens: int) -> float:
        return (
            in_tokens / 1_000_000 * self._price_in
            + out_tokens / 1_000_000 * self._price_out
        )

    @staticmethod
    def _today_str() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _maybe_reset_daily_spend(self) -> None:
        today = self._today_str()
        if today != self._spend_day_utc:
            self._log.info(
                "Daily spend counter reset. Previous day: ${:.4f}",
                self._daily_spend_usd,
            )
            self._daily_spend_usd = 0.0
            self._spend_day_utc = today

    @property
    def daily_spend_usd(self) -> float:
        self._maybe_reset_daily_spend()
        return self._daily_spend_usd

    @property
    def daily_budget_remaining_usd(self) -> float:
        if self.cfg.daily_spend_limit_usd <= 0:
            return float("inf")
        return max(0.0, self.cfg.daily_spend_limit_usd - self.daily_spend_usd)

    def _handle_rate_limit_error(self, exc: Exception) -> None:
        msg = str(exc).lower()
        retry_after: Optional[int] = None
        reset_at: Optional[datetime] = None
        try:
            response = getattr(exc, "response", None)
            if response is not None:
                headers = getattr(response, "headers", {}) or {}
                ra = headers.get("retry-after") or headers.get("Retry-After")
                if ra:
                    retry_after = int(float(ra))
                reset_iso = (
                    headers.get("anthropic-ratelimit-tokens-reset")
                    or headers.get("anthropic-ratelimit-requests-reset")
                )
                if reset_iso:
                    try:
                        reset_at = datetime.fromisoformat(
                            reset_iso.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

        is_credits = any(s in msg for s in (
            "credit balance", "billing", "insufficient credits",
            "payment required", "low balance",
        ))

        wait_msg = ""
        if retry_after:
            wait_msg = f" Retry in {retry_after}s."
        elif reset_at:
            wait_msg = f" Estimated reset: {reset_at.isoformat()}."

        if is_credits:
            self._log.error(
                "CREDITS EXHAUSTED at console.anthropic.com.{} "
                "Top up at https://console.anthropic.com/billing",
                wait_msg,
            )
            raise CreditsExhausted(
                f"Account has no credits.{wait_msg}",
                reset_at=reset_at,
                retry_after_seconds=retry_after,
            ) from exc

        self._log.warning("Rate limit reached.{}", wait_msg)
        raise LLMError(f"Rate limit (HTTP 429).{wait_msg}") from exc


# =====================================================
# Ollama implementation
# =====================================================


class OllamaClient(LLMClient):
    """Client for local Ollama. Free, no cost, no external network."""

    def __init__(self, config: BotConfig) -> None:
        super().__init__(config)
        self.base_url = config.llm.ollama_base_url.rstrip("/")
        self.timeout = config.llm.ollama_timeout_seconds
        # Thread-local sessions: requests.Session is not safe to share across threads.
        # Each worker thread gets its own session created lazily on first use.
        self._local = threading.local()

    @property
    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def verify_setup(self) -> None:
        """Verifies that Ollama is running and the model is downloaded.

        Call at bot startup to fail fast if something is not ready.
        """
        try:
            r = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnavailable(
                f"Ollama not responding at {self.base_url}. "
                f"Make sure the service is running: 'ollama serve'. "
                f"Detail: {exc}"
            ) from exc

        data = r.json()
        installed_models = [m["name"] for m in data.get("models", [])]
        # Accept exact match or config being a prefix (ollama
        # appends ":latest" if no tag is specified)
        target = self.cfg.model
        installed_ok = any(
            m == target or m.startswith(f"{target}:") or m.split(":")[0] == target.split(":")[0]
            for m in installed_models
        )
        if not installed_ok:
            raise OllamaUnavailable(
                f"Model '{target}' is not downloaded. "
                f"Available models: {installed_models}. "
                f"Download it with: ollama pull {target}"
            )
        self._log.info("Ollama OK | model='{}' | server='{}'", target, self.base_url)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        force_json: bool = False,
    ) -> dict[str, Any]:
        max_tokens = max_tokens or self.cfg.max_tokens
        temperature = temperature if temperature is not None else self.cfg.temperature

        if self.cfg.dry_run:
            self._log.warning("DRY_RUN active. Ollama will not be called.")
            return {
                "text": "{}",
                "input_tokens": 0,
                "output_tokens": 0,
                "stop_reason": "dry_run",
                "estimated_cost_usd": 0.0,
            }

        # Ollama uses /api/chat with OpenAI-like format
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        # Force JSON if the model supports it. Ollama accepts format="json" on
        # compatible models (qwen, llama, mistral, etc.). If the model does not
        # support it, it is ignored.
        if force_json:
            payload["format"] = "json"

        try:
            response = self._call_with_retry(payload)
        except requests.Timeout as exc:
            # Timeout is expected on slow hardware — handled upstream, not an error.
            self._log.warning(
                "Ollama timeout ({}s) — market skipped, analysis unavailable",
                self.timeout,
            )
            raise LLMError(f"Ollama timeout after {self.timeout}s") from exc
        except requests.RequestException as exc:
            self._log.error("Ollama failed (network error): {}", exc)
            raise LLMError(f"Ollama HTTP error: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"Ollama returned invalid JSON: {exc}") from exc

        text = data.get("message", {}).get("content", "")
        in_tokens = data.get("prompt_eval_count", 0)
        out_tokens = data.get("eval_count", 0)

        with self._stats_lock:
            self.total_input_tokens += in_tokens
            self.total_output_tokens += out_tokens
            self.total_calls += 1

        return {
            "text": text,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "stop_reason": data.get("done_reason", "stop"),
            "estimated_cost_usd": 0.0,  # free
        }

    # ---------- Ollama internals ----------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        # Only retry on ConnectionError (Ollama process died/restarted).
        # Do NOT retry on Timeout: if Ollama is already processing the request
        # and is slow, retrying queues a second copy behind the first — making
        # things worse. A timeout should fail fast and let the caller skip the market.
        retry=retry_if_exception_type(requests.ConnectionError),
        reraise=True,
    )
    def _call_with_retry(self, payload: dict[str, Any]) -> requests.Response:
        r = self._session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r

    @property
    def daily_spend_usd(self) -> float:
        return 0.0  # Ollama is free

    @property
    def daily_budget_remaining_usd(self) -> float:
        return float("inf")


# =====================================================
# Factory
# =====================================================


def build_llm_client(config: BotConfig) -> LLMClient:
    """Builds the correct client based on `config.llm.provider`."""
    provider = config.llm.provider.lower().strip()
    if provider == "anthropic":
        return AnthropicClient(config)
    if provider == "ollama":
        return OllamaClient(config)
    raise LLMError(
        f"Unknown LLM provider: '{provider}'. "
        f"Valid options: 'anthropic', 'ollama'."
    )
