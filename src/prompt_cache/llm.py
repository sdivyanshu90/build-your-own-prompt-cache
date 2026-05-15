"""Cache-aware wrapper around an arbitrary async LLM completion function."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import time
from typing import Any

from .core import PromptCache
from .types import CompletionResult

CompletionCallable = Callable[..., Awaitable[Any]]


class CachedLLMClient:
    """Wrap a completion callable with prompt cache lookups and write-back."""

    DEFAULT_PRICING_PER_1K_TOKENS = {
        "claude-sonnet-4-20250514": 0.003,
        "claude-opus-4-20250514": 0.015,
        "default": 0.005,
    }

    def __init__(
        self,
        cache: PromptCache,
        completion_callable: CompletionCallable,
        pricing_per_1k_tokens: dict[str, float] | None = None,
    ) -> None:
        self.cache = cache
        self._completion_callable = completion_callable
        self._pricing_per_1k_tokens = pricing_per_1k_tokens or dict(
            self.DEFAULT_PRICING_PER_1K_TOKENS
        )

    async def complete(
        self,
        prompt: str,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system: str | None = None,
        bypass_cache: bool = False,
        cache_ttl: float | None = None,
        **extra_params: Any,
    ) -> CompletionResult:
        """Resolve a completion via exact cache hit, semantic hit, or API miss."""

        started_ns = time.perf_counter_ns()
        llm_params = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system,
            **extra_params,
        }
        cache_key = self.cache.key_builder.build_key(prompt, **llm_params)

        if not bypass_cache:
            cached_entry = await self.cache.get(prompt, **llm_params)
            if cached_entry is not None:
                return CompletionResult(
                    response=cached_entry.value,
                    cached=True,
                    cache_key=cached_entry.key,
                    latency_ms=self._elapsed_ms(started_ns),
                    tokens_used=int(cached_entry.metadata.get("tokens_used", 0)),
                    estimated_cost_usd=0.0,
                    semantic_hit=bool(cached_entry.metadata.get("semantic_hit", False)),
                    metadata=dict(cached_entry.metadata),
                )

        api_started_ns = time.perf_counter_ns()
        try:
            raw_response = await self._completion_callable(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system,
                **extra_params,
            )
        except Exception:
            self.cache.metrics.record_error()
            raise

        api_latency_ms = self._elapsed_ms(api_started_ns)
        self.cache.metrics.record_api_latency(api_latency_ms)
        response_text, tokens_used = self._extract_response_and_usage(raw_response, prompt)
        estimated_cost = self._estimate_cost(model, tokens_used)
        await self.cache.set(
            prompt,
            response_text,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            cache_ttl=cache_ttl,
            tokens_used=tokens_used,
            estimated_cost_usd=estimated_cost,
            **extra_params,
        )
        return CompletionResult(
            response=response_text,
            cached=False,
            cache_key=cache_key,
            latency_ms=self._elapsed_ms(started_ns),
            tokens_used=tokens_used,
            estimated_cost_usd=estimated_cost,
            semantic_hit=False,
            metadata={
                "api_latency_ms": api_latency_ms,
            },
        )

    def _estimate_cost(self, model: str, tokens_used: int) -> float:
        rate = self._pricing_per_1k_tokens.get(
            model, self._pricing_per_1k_tokens["default"]
        )
        return (tokens_used / 1000.0) * rate

    def _extract_response_and_usage(
        self,
        payload: Any,
        prompt: str,
    ) -> tuple[Any, int]:
        if isinstance(payload, str):
            return payload, self._estimate_tokens(prompt + payload)
        if isinstance(payload, dict):
            response = payload.get("response") or payload.get("text") or payload.get("content")
            if isinstance(response, list):
                response = "\n".join(str(item) for item in response)
            usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
            tokens_used = payload.get("tokens_used")
            if tokens_used is None:
                tokens_used = usage.get("total_tokens") or usage.get("output_tokens")
            if tokens_used is None:
                tokens_used = self._estimate_tokens(prompt + str(response))
            return response, int(tokens_used)
        return payload, self._estimate_tokens(prompt + str(payload))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _elapsed_ms(start_ns: int) -> float:
        return (time.perf_counter_ns() - start_ns) / 1_000_000.0
