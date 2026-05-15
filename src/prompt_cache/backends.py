"""Storage backend implementations for prompt cache entries."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import time
from typing import Any, Awaitable, Callable, TypeVar

try:
    import aiofiles  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency.
    aiofiles = None

try:
    import redis.asyncio as aioredis  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency.
    aioredis = None

from .interfaces import CacheBackend
from .serde import dumps, loads
from .types import CacheEntry

ResultT = TypeVar("ResultT")


class InMemoryBackend(CacheBackend):
    """Fast in-process backend for low-latency cache lookups.

    ``OrderedDict`` tracks insertion/access order in O(1), which keeps the
    implementation compatible with fallback trimming and quick diagnostics while
    the main orchestrator remains responsible for policy-driven eviction.
    """

    def __init__(
        self,
        max_items: int = 10000,
        max_size_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.max_items = max_items
        self.max_size_bytes = max_size_bytes
        self._entries: dict[str, CacheEntry] = {}
        self._sizes: dict[str, int] = {}
        self._order: OrderedDict[str, None] = OrderedDict()
        self._current_size_bytes = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> CacheEntry | None:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                await self._delete_unlocked(key)
                return None
            self._order.move_to_end(key, last=True)
            return CacheEntry.from_dict(entry.to_dict())

    async def set(self, key: str, entry: CacheEntry) -> None:
        async with self._lock:
            serialized_size = entry.size_bytes()
            previous_size = self._sizes.get(key, 0)
            self._entries[key] = CacheEntry.from_dict(entry.to_dict())
            self._sizes[key] = serialized_size
            self._order[key] = None
            self._order.move_to_end(key, last=True)
            self._current_size_bytes += serialized_size - previous_size

    async def delete(self, key: str) -> bool:
        async with self._lock:
            return await self._delete_unlocked(key)

    async def _delete_unlocked(self, key: str) -> bool:
        if key not in self._entries:
            return False
        self._entries.pop(key, None)
        self._order.pop(key, None)
        self._current_size_bytes -= self._sizes.pop(key, 0)
        return True

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()
            self._sizes.clear()
            self._order.clear()
            self._current_size_bytes = 0

    async def keys(self) -> list[str]:
        async with self._lock:
            return list(self._entries.keys())

    async def size_bytes(self) -> int:
        async with self._lock:
            return self._current_size_bytes

    async def item_count(self) -> int:
        async with self._lock:
            return len(self._entries)


class RedisBackend(CacheBackend):
    """Redis-backed cache with graceful failover to local memory.

    Redis is the right default when prompt cache state must survive process
    restarts or be shared across horizontally scaled application replicas.
    When Redis is unavailable, the backend degrades to an in-memory fallback so
    the application continues serving traffic rather than turning the cache into
    a single point of failure.
    """

    def __init__(
        self,
        redis_url: str | None,
        *,
        pool_size: int = 20,
        socket_timeout_seconds: float = 1.5,
        client: Any | None = None,
        fallback_backend: CacheBackend | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._pool_size = pool_size
        self._socket_timeout_seconds = socket_timeout_seconds
        self._fallback = fallback_backend or InMemoryBackend(max_items=2048)
        self._client = client
        self._available = client is not None or (aioredis is not None and bool(redis_url))
        if self._client is None and self._available:
            self._client = aioredis.from_url(
                redis_url,
                decode_responses=False,
                max_connections=pool_size,
                socket_timeout=socket_timeout_seconds,
            )

    async def _with_failover(
        self,
        *,
        redis_call: Callable[[], Awaitable[ResultT]],
        fallback_call: Callable[[], Awaitable[ResultT]],
    ) -> ResultT:
        if not self._available or self._client is None:
            return await fallback_call()
        try:
            return await redis_call()
        except Exception:
            self._available = False
            return await fallback_call()

    async def get(self, key: str) -> CacheEntry | None:
        async def _redis() -> CacheEntry | None:
            payload = await self._client.get(key)
            if payload is None:
                return None
            entry = CacheEntry.from_dict(loads(payload))
            if entry.is_expired():
                await self._client.delete(key)
                return None
            return entry

        return await self._with_failover(redis_call=_redis, fallback_call=lambda: self._fallback.get(key))

    async def set(self, key: str, entry: CacheEntry) -> None:
        async def _redis() -> None:
            payload = dumps(entry.to_dict())
            if entry.ttl is not None:
                remaining = entry.expires_at - time.time() if entry.expires_at else entry.ttl
                if remaining <= 0:
                    await self._client.delete(key)
                    return
                await self._client.setex(key, max(1, int(remaining)), payload)
            else:
                await self._client.set(key, payload)

        await self._with_failover(redis_call=_redis, fallback_call=lambda: self._fallback.set(key, entry))

    async def delete(self, key: str) -> bool:
        async def _redis() -> bool:
            return bool(await self._client.delete(key))

        return await self._with_failover(redis_call=_redis, fallback_call=lambda: self._fallback.delete(key))

    async def clear(self) -> None:
        async def _redis() -> None:
            keys = [key async for key in self._client.scan_iter(match="*")]
            if keys:
                await self._client.delete(*keys)

        await self._with_failover(redis_call=_redis, fallback_call=self._fallback.clear)

    async def keys(self) -> list[str]:
        async def _redis() -> list[str]:
            keys: list[str] = []
            async for key in self._client.scan_iter(match="*"):
                keys.append(key.decode("utf-8") if isinstance(key, bytes) else str(key))
            return keys

        return await self._with_failover(redis_call=_redis, fallback_call=self._fallback.keys)

    async def size_bytes(self) -> int:
        async def _redis() -> int:
            total = 0
            async for key in self._client.scan_iter(match="*"):
                usage = await self._client.memory_usage(key)
                if usage is None:
                    usage = await self._client.strlen(key)
                total += int(usage or 0)
            return total

        return await self._with_failover(redis_call=_redis, fallback_call=self._fallback.size_bytes)

    async def item_count(self) -> int:
        async def _redis() -> int:
            count = 0
            async for _ in self._client.scan_iter(match="*"):
                count += 1
            return count

        return await self._with_failover(redis_call=_redis, fallback_call=self._fallback.item_count)

    async def close(self) -> None:
        if self._client is not None:
            if hasattr(self._client, "aclose"):
                await self._client.aclose()
            else:  # pragma: no cover - depends on redis client version.
                await self._client.close()
        await self._fallback.close()


class DiskBackend(CacheBackend):
    """Compressed filesystem-backed cache with atomic writes.

    Files are sharded by the first two hex characters of the key hash so a busy
    deployment does not accumulate hundreds of thousands of files in a single
    directory. Writes follow the temp-file-then-rename pattern because rename is
    atomic on POSIX filesystems; this prevents partially written cache entries
    from appearing after process crashes.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._lock = asyncio.Lock()

    def _digest(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _path_for_key(self, key: str) -> Path:
        digest = self._digest(key)
        return self.cache_dir / digest[:2] / f"{digest}.json.gz"

    async def _read_bytes(self, path: Path) -> bytes:
        if aiofiles is not None:
            async with aiofiles.open(path, "rb") as handle:
                return await handle.read()
        return await asyncio.to_thread(path.read_bytes)

    async def _write_bytes(self, path: Path, payload: bytes) -> None:
        if aiofiles is not None:
            async with aiofiles.open(path, "wb") as handle:
                await handle.write(payload)
            return
        await asyncio.to_thread(path.write_bytes, payload)

    async def get(self, key: str) -> CacheEntry | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        async with self._lock:
            if not path.exists():
                return None
            compressed = await self._read_bytes(path)
            payload = await asyncio.to_thread(gzip.decompress, compressed)
            entry = CacheEntry.from_dict(loads(payload))
            if entry.is_expired():
                await asyncio.to_thread(path.unlink)
                return None
            return entry

    async def set(self, key: str, entry: CacheEntry) -> None:
        path = self._path_for_key(key)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        async with self._lock:
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            compressed = await asyncio.to_thread(gzip.compress, dumps(entry.to_dict()))
            await self._write_bytes(temp_path, compressed)
            await asyncio.to_thread(os.replace, temp_path, path)

    async def delete(self, key: str) -> bool:
        path = self._path_for_key(key)
        async with self._lock:
            if not path.exists():
                return False
            await asyncio.to_thread(path.unlink)
            return True

    async def clear(self) -> None:
        async with self._lock:
            await asyncio.to_thread(shutil.rmtree, self.cache_dir, True)
            await asyncio.to_thread(self.cache_dir.mkdir, parents=True, exist_ok=True)

    async def keys(self) -> list[str]:
        async with self._lock:
            if not self.cache_dir.exists():
                return []
            paths = await asyncio.to_thread(
                lambda: [path for path in self.cache_dir.rglob("*.json.gz") if path.is_file()]
            )
        keys: list[str] = []
        for path in paths:
            compressed = await self._read_bytes(path)
            payload = await asyncio.to_thread(gzip.decompress, compressed)
            entry = CacheEntry.from_dict(loads(payload))
            if entry.is_expired():
                await self.delete(entry.key)
                continue
            keys.append(entry.key)
        return keys

    async def size_bytes(self) -> int:
        if not self.cache_dir.exists():
            return 0
        return await asyncio.to_thread(
            lambda: sum(path.stat().st_size for path in self.cache_dir.rglob("*.json.gz") if path.is_file())
        )

    async def item_count(self) -> int:
        if not self.cache_dir.exists():
            return 0
        return await asyncio.to_thread(
            lambda: sum(1 for path in self.cache_dir.rglob("*.json.gz") if path.is_file())
        )
