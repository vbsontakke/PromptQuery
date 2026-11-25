from __future__ import annotations

import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "claude-sonnet-5-5"  # SQL generation
DEFAULT_FAST_MODEL = "claude-haiku-5-5"  # follow-up classification / rewriting
REQUEST_TIMEOUT_S = 60.0


class LLMError(RuntimeError):
    pass


@dataclass
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: "LLMUsage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


Message = dict[str, str]  # {"role": "user" | "assistant", "content": "..."}


class LLMClient(ABC):
    name: str
    model: str = ""

    @abstractmethod
    def generate(self, system: str, user: str) -> str: ...

    def generate_messages(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        cache_system: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Multi-turn call. Clients that only implement ``generate`` get the user turns joined."""
        user = "\n\n".join(m["content"] for m in messages if m.get("role") == "user")
        return self.generate(system, user)


def is_retriable(exc: BaseException) -> bool:
    """Transient API failures worth retrying: rate limits, overload, 5xx, network, timeouts."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status in (408, 409, 425, 429) or status >= 500):
        return True
    return type(exc).__name__ in {
        "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
        "OverloadedError", "Timeout", "ConnectError", "ReadTimeout",
    }


def with_retries(
    fn: Callable[[], Any],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call ``fn`` retrying transient failures with exponential backoff (1s, 2s, 4s...)."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - classified below
            if not is_retriable(e):
                raise
            last = e
            if attempt < attempts - 1:
                sleep(base_delay * (2**attempt))
    raise LLMError(f"LLM request failed after {attempts} attempts: {last}") from last


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = REQUEST_TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("anthropic package not installed") from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        # max_retries=0: retries are done by with_retries so they are bounded and observable.
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=0)
        self.model = model or os.environ.get("PQ_MODEL") or DEFAULT_MODEL
        self.usage = LLMUsage()
        self._sleep = sleep
        self._send_temperature = True

    def generate(self, system: str, user: str) -> str:
        return self.generate_messages(system, [{"role": "user", "content": user}])

    def generate_messages(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        cache_system: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        system_param: Any = system
        if cache_system:
            # The curated schema pack is identical across questions: cache it.
            system_param = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]

        def call() -> Any:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens or 2000,
                "system": system_param,
                "messages": list(messages),
            }
            if self._send_temperature:
                # Determinism is a feature: the same question should yield the same SQL.
                kwargs["temperature"] = 0
            return self._client.messages.create(**kwargs)

        try:
            response = with_retries(call, sleep=self._sleep)
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            # Some newer models reject `temperature`; remember that and retry once without it.
            if self._send_temperature and "temperature" in str(e).lower():
                self._send_temperature = False
                response = with_retries(call, sleep=self._sleep)
            else:
                raise LLMError(f"Anthropic request failed: {e}") from e

        u = getattr(response, "usage", None)
        if u is not None:
            self.usage.add(LLMUsage(
                calls=1,
                input_tokens=getattr(u, "input_tokens", 0) or 0,
                output_tokens=getattr(u, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            ))
        parts: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        try:
            import openai
        except ImportError as e:
            raise LLMError("openai package not installed (pip install promptquery[openai])") from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set")
        self._client = openai.OpenAI(api_key=key, timeout=REQUEST_TIMEOUT_S, max_retries=0)
        self.model = model

    # Reasoning-class OpenAI models (GPT-5.x, o1, o3, o4) accept
    # `max_completion_tokens` and reject the legacy `max_tokens` parameter.
    _REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")

    def generate(self, system: str, user: str) -> str:
        return self.generate_messages(system, [{"role": "user", "content": user}])

    def generate_messages(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        cache_system: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if self.model.startswith(self._REASONING_PREFIXES):
            # Reasoning models reject `temperature`/`seed`; they sample internally.
            kwargs["max_completion_tokens"] = max(4000, max_tokens or 0)
        else:
            kwargs["max_tokens"] = max_tokens or 2000
            # Determinism is a feature: same question -> same SQL. temperature=0
            # plus a fixed seed gives best-effort reproducibility on chat models.
            kwargs["temperature"] = 0
            kwargs["seed"] = 0
        response = with_retries(lambda: self._client.chat.completions.create(**kwargs))
        return response.choices[0].message.content or ""


class CachingLLMClient(LLMClient):
    """Disk cache keyed by (model, system, messages, max_tokens).

    Makes evals deterministic and free to re-run, and lets CI replay recorded responses with no
    network. ``refresh=True`` ignores existing entries and overwrites them.
    """

    name = "caching"

    def __init__(self, inner: LLMClient, cache_dir: str | Path, *, refresh: bool = False):
        self.inner = inner
        self.model = inner.model
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh
        self.hits = 0
        self.misses = 0

    @property
    def usage(self) -> LLMUsage:
        return getattr(self.inner, "usage", LLMUsage())

    def _key(self, system: str, messages: Sequence[Message], max_tokens: int | None) -> str:
        blob = json.dumps(
            [self.inner.model, system, list(messages), max_tokens], sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def generate(self, system: str, user: str) -> str:
        return self.generate_messages(system, [{"role": "user", "content": user}])

    def generate_messages(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        cache_system: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        path = self.dir / f"{self._key(system, messages, max_tokens)}.json"
        if path.exists() and not self.refresh:
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        self.misses += 1
        out = self.inner.generate_messages(
            system, messages, cache_system=cache_system, max_tokens=max_tokens
        )
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"response": out}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic: a crashed run never leaves a half-written entry
        return out


_SQL_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    match = _SQL_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip().rstrip(";").strip()
    return text.strip().rstrip(";").strip()


def make_client(model_spec: str | None = None) -> LLMClient:
    if model_spec:
        provider, _, model = model_spec.partition("/")
        if not model:
            # No explicit provider — guess from model name
            if provider.startswith("claude"):
                return AnthropicClient(model=provider)
            if provider.startswith(("gpt", "o1", "o3", "o4")):
                return OpenAIClient(model=provider)
            raise LLMError(
                f"Cannot infer provider from model {provider!r}. "
                "Use 'anthropic/<model>' or 'openai/<model>'."
            )
        if provider == "anthropic":
            return AnthropicClient(model=model)
        if provider == "openai":
            return OpenAIClient(model=model)
        raise LLMError(f"Unknown provider: {provider!r}")

    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIClient()
    raise LLMError(
        "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
    )
