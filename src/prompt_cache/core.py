"""Prompt cache orchestration layer."""

from __future__ import annotations

import fnmatch
import time
from typing import Any

from .concurrency import AsyncRWLock
from .interfaces import CacheBackend, EvictionPolicy
from .key_builder import CacheKeyBuilder
from .metrics import MetricsCollector
from .semantic import SemanticCacheLayer
from .types import CacheEntry, CacheStats


class PromptCache:
    """Cache orchestrator that coordinates storage, eviction, and observability."""

    _response_metadata_fields = {
        "api_latency_ms",
        "estimated_cost_usd",
        "raw_response",
        "response_metadata",
        "tokens_used",
    }

    def __init__(
        self,
        backend: CacheBackend,
        eviction_policy: EvictionPolicy,
        key_builder: CacheKeyBuilder,
        semantic_layer: SemanticCacheLayer | None = None,
        default_ttl: float = 3600.0,
        max_size_mb: float = 512.0,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        self.backend = backend
        self.eviction_policy = eviction_policy
        self.key_builder = key_builder
        self.semantic_layer = semantic_layer
        self.default_ttl = default_ttl
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.metrics = metrics_collector or MetricsCollector()
        self._lock = AsyncRWLock()

    async def get(self, prompt: str, **llm_params: Any) -> CacheEntry | None:
        """Fetch a cached response using exact match first, semantic match second."""

        start_ns = time.perf_counter_ns()
        key = self.key_builder.build_key(prompt, **llm_params)
        async with self._lock.read_lock():
            entry = await self.backend.get(key)
        if entry is not None:
            await self._mark_access(key, entry, semantic_hit=False)
            self.metrics.record_hit(self._elapsed_ms(start_ns), semantic_hit=False)
            return entry

        if self.semantic_layer is not None:
            try:
                candidate = await self.semantic_layer.find_similar(prompt)
            except Exception:
                self.metrics.record_error()
                candidate = None
            if candidate is not None:
                similar_key, similarity = candidate
                async with self._lock.read_lock():
                    similar_entry = await self.backend.get(similar_key)
                if similar_entry is not None:
                    similar_entry.metadata["semantic_similarity"] = similarity
                    similar_entry.metadata["semantic_hit"] = True
                    await self._mark_access(similar_key, similar_entry, semantic_hit=True)
                    self.metrics.record_hit(self._elapsed_ms(start_ns), semantic_hit=True)
                    return similar_entry

        self.metrics.record_miss(self._elapsed_ms(start_ns))
        return None

    async def set(self, prompt: str, response: Any, **llm_params: Any) -> None:
        """Store a prompt/response pair in the configured backend."""

        llm_params_copy = dict(llm_params)
        ttl = llm_params_copy.pop("cache_ttl", self.default_ttl)
        response_metadata = {
            field: llm_params_copy.pop(field)
            for field in list(llm_params_copy)
            if field in self._response_metadata_fields
        }
        key = self.key_builder.build_key(prompt, **llm_params_copy)
        entry = CacheEntry(
            key=key,
            value=response,
            created_at=time.time(),
            ttl=ttl,
            metadata={
                "prompt": prompt,
                "llm_params": llm_params_copy,
                **response_metadata,
                "semantic_hit": False,
            },
        )
        async with self._lock.write_lock():
            await self.backend.set(key, entry)
            self.eviction_policy.record_insert(key)
            if self.semantic_layer is not None:
                try:
                    await self.semantic_layer.record(key, prompt)
                except Exception:
                    self.metrics.record_error()
            await self._ensure_capacity()
            await self._refresh_cache_gauges()

    async def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching a Unix shell-style glob pattern."""

        async with self._lock.write_lock():
            keys = await self.backend.keys()
            matches = [key for key in keys if fnmatch.fnmatch(key, pattern)]
            removed = 0
            for key in matches:
                if await self.backend.delete(key):
                    removed += 1
                    self.eviction_policy.record_delete(key)
                    if self.semantic_layer is not None:
                        await self.semantic_layer.delete(key)
            await self._refresh_cache_gauges()
            return removed

    async def warm(self, prompt_response_pairs: list[tuple[Any, ...]]) -> None:
        """Warm the cache from prompt/response tuples.

        Tuple forms supported:

        - ``(prompt, response)``
        - ``(prompt, response, llm_params_dict)``
        """

        for item in prompt_response_pairs:
            if len(item) == 2:
                prompt, response = item
                params = {}
            elif len(item) == 3:
                prompt, response, params = item
            else:
                raise ValueError("warm expects tuples of (prompt, response[, llm_params])")
            await self.set(str(prompt), response, **dict(params))

    async def stats(self) -> CacheStats:
        """Return a fresh cache snapshot."""

        await self._refresh_cache_gauges()
        return self.metrics.snapshot()

    async def clear(self) -> None:
        """Clear the cache and reset all policy state."""

        async with self._lock.write_lock():
            keys = await self.backend.keys()
            await self.backend.clear()
            for key in keys:
                self.eviction_policy.record_delete(key)
            if self.semantic_layer is not None:
                await self.semantic_layer.clear()
            await self._refresh_cache_gauges()

    async def list_entries(self) -> list[CacheEntry]:
        """Return all live cache entries."""

        keys = await self.backend.keys()
        entries: list[CacheEntry] = []
        for key in keys:
            entry = await self.backend.get(key)
            if entry is not None:
                entries.append(entry)
        return entries

    async def close(self) -> None:
        """Close external resources used by the cache."""

        await self.backend.close()

    async def _mark_access(self, key: str, entry: CacheEntry, *, semantic_hit: bool) -> None:
        async with self._lock.write_lock():
            current = await self.backend.get(key)
            if current is None:
                return
            current.hit_count += 1
            current.last_accessed = time.time()
            current.metadata["semantic_hit"] = semantic_hit
            await self.backend.set(key, current)
            self.eviction_policy.record_access(key)

    async def _ensure_capacity(self) -> None:
        backend_max_items = getattr(self.backend, "max_items", None)
        backend_max_size_bytes = getattr(self.backend, "max_size_bytes", None)
        while True:
            item_count = await self.backend.item_count()
            size_bytes = await self.backend.size_bytes()
            over_item_limit = backend_max_items is not None and item_count > backend_max_items
            over_size_limit = size_bytes > self.max_size_bytes
            if backend_max_size_bytes is not None:
                over_size_limit = over_size_limit or size_bytes > backend_max_size_bytes
            if not over_item_limit and not over_size_limit:
                break
            victim = self.eviction_policy.select_victim()
            if victim is None:
                break
            deleted = await self.backend.delete(victim)
            if not deleted:
                self.eviction_policy.record_delete(victim)
                continue
            self.eviction_policy.record_delete(victim)
            if self.semantic_layer is not None:
                await self.semantic_layer.delete(victim)
            self.metrics.record_eviction()

    async def _refresh_cache_gauges(self) -> None:
        self.metrics.set_cache_state(
            size_bytes=await self.backend.size_bytes(),
            item_count=await self.backend.item_count(),
        )

    @staticmethod
    def _elapsed_ms(start_ns: int) -> float:
        return (time.perf_counter_ns() - start_ns) / 1_000_000.0
