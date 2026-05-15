"""Concurrency primitives used by the cache orchestrator."""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
from typing import AsyncIterator


class AsyncRWLock:
    """Fair async read-write lock.

    Reads may proceed concurrently until a writer arrives. Once a writer is
    waiting, new readers are paused so the writer is not starved by an endless
    stream of cache lookups.
    """

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active_readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    @asynccontextmanager
    async def read_lock(self) -> AsyncIterator[None]:
        """Acquire a shared read lock."""

        async with self._condition:
            while self._writer_active or self._writers_waiting > 0:
                await self._condition.wait()
            self._active_readers += 1
        try:
            yield
        finally:
            async with self._condition:
                self._active_readers -= 1
                if self._active_readers == 0:
                    self._condition.notify_all()

    @asynccontextmanager
    async def write_lock(self) -> AsyncIterator[None]:
        """Acquire an exclusive write lock."""

        async with self._condition:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._active_readers > 0:
                    await self._condition.wait()
                self._writer_active = True
            finally:
                self._writers_waiting -= 1
        try:
            yield
        finally:
            async with self._condition:
                self._writer_active = False
                self._condition.notify_all()
