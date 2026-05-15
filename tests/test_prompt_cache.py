from __future__ import annotations

import asyncio

import pytest

from prompt_cache.backends import InMemoryBackend
from prompt_cache.core import PromptCache
from prompt_cache.eviction import LRUPolicy
from prompt_cache.key_builder import CacheKeyBuilder


def make_cache(max_items: int = 16, max_size_mb: float = 8.0) -> PromptCache:
    return PromptCache(
        backend=InMemoryBackend(
            max_items=max_items,
            max_size_bytes=int(max_size_mb * 1024 * 1024),
        ),
        eviction_policy=LRUPolicy(),
        key_builder=CacheKeyBuilder(namespace="test", version="v1"),
        default_ttl=60.0,
        max_size_mb=max_size_mb,
    )


@pytest.mark.asyncio
async def test_prompt_cache_records_hits_misses_and_evictions() -> None:
    cache = make_cache(max_items=2)
    await cache.set("prompt-1", "response-1")
    await cache.set("prompt-2", "response-2")
    assert await cache.get("unknown") is None
    assert (await cache.get("prompt-1")) is not None
    await cache.set("prompt-3", "response-3")
    stats = await cache.stats()
    assert stats.cache_hits == 1
    assert stats.cache_misses == 1
    assert stats.evictions == 1
    assert stats.cache_item_count <= 2


@pytest.mark.asyncio
async def test_ttl_expiry_with_mocked_time(mocker) -> None:
    cache = make_cache()
    mocker.patch("prompt_cache.core.time.time", return_value=100.0)
    mocker.patch("prompt_cache.types.time.time", return_value=100.0)
    await cache.set("expiring", "soon", cache_ttl=10.0)
    mocker.patch("prompt_cache.types.time.time", return_value=111.0)
    expired = await cache.get("expiring")
    assert expired is None


@pytest.mark.asyncio
async def test_invalidate_pattern_removes_matching_keys() -> None:
    cache = make_cache(max_items=8)
    await cache.set("alpha", "one")
    await cache.set("beta", "two")
    await cache.set("gamma", "three")
    removed = await cache.invalidate("test:v1:*")
    assert removed == 3
    assert (await cache.get("alpha")) is None


@pytest.mark.asyncio
async def test_concurrent_access_patterns() -> None:
    cache = make_cache(max_items=64)
    await asyncio.gather(*(cache.set(f"prompt-{index}", f"response-{index}") for index in range(20)))
    results = await asyncio.gather(*(cache.get(f"prompt-{index % 20}") for index in range(200)))
    assert all(entry is not None for entry in results)
    stats = await cache.stats()
    assert stats.cache_hits >= 200
