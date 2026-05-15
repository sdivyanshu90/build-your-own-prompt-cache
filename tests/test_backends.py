from __future__ import annotations

import time

import pytest

from prompt_cache.backends import DiskBackend, InMemoryBackend, RedisBackend
from prompt_cache.types import CacheEntry


def _entry(key: str, value: str) -> CacheEntry:
    return CacheEntry(
        key=key,
        value=value,
        created_at=time.time(),
        ttl=60.0,
        metadata={"prompt": key},
    )


@pytest.mark.asyncio
async def test_in_memory_backend_round_trip() -> None:
    backend = InMemoryBackend(max_items=16, max_size_bytes=1024 * 1024)
    entry = _entry("alpha", "bravo")
    await backend.set(entry.key, entry)
    loaded = await backend.get(entry.key)
    assert loaded is not None
    assert loaded.value == "bravo"


@pytest.mark.asyncio
async def test_disk_backend_round_trip(tmp_path) -> None:
    backend = DiskBackend(tmp_path / "cache")
    entry = _entry("disk-key", "disk-value")
    await backend.set(entry.key, entry)
    loaded = await backend.get(entry.key)
    assert loaded is not None
    assert loaded.value == "disk-value"


@pytest.mark.asyncio
async def test_redis_backend_with_fakeredis() -> None:
    fakeredis_aioredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis_aioredis.FakeRedis(decode_responses=False)
    backend = RedisBackend(
        "redis://unused",
        client=client,
        fallback_backend=InMemoryBackend(max_items=8, max_size_bytes=1024 * 1024),
    )
    entry = _entry("redis-key", "redis-value")
    await backend.set(entry.key, entry)
    loaded = await backend.get(entry.key)
    assert loaded is not None
    assert loaded.value == "redis-value"
    await backend.close()
