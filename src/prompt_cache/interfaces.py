"""Abstract contracts for storage, eviction, and semantic lookup."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .types import CacheEntry

EmbeddingFunction = Callable[[str], Sequence[float] | Awaitable[Sequence[float]]]


class CacheBackend(ABC):
    """Persistence contract implemented by all storage backends."""

    @abstractmethod
    async def get(self, key: str) -> CacheEntry | None:
        """Fetch an entry by key."""

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> None:
        """Persist an entry."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete an entry and return ``True`` when something was removed."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove all cached entries."""

    @abstractmethod
    async def keys(self) -> list[str]:
        """Return the list of cache keys known to the backend."""

    @abstractmethod
    async def size_bytes(self) -> int:
        """Return the backend's estimated size in bytes."""

    @abstractmethod
    async def item_count(self) -> int:
        """Return the number of entries stored in the backend."""

    async def close(self) -> None:
        """Release resources held by the backend."""


class EvictionPolicy(ABC):
    """Policy contract used by ``PromptCache`` to select eviction victims."""

    @abstractmethod
    def record_insert(self, key: str) -> None:
        """Notify the policy that a new key was stored."""

    @abstractmethod
    def record_access(self, key: str) -> None:
        """Notify the policy that a key was accessed."""

    @abstractmethod
    def record_delete(self, key: str) -> None:
        """Notify the policy that a key was removed."""

    @abstractmethod
    def select_victim(self) -> str | None:
        """Return the next key to evict, if any."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of keys tracked by the policy."""


class SemanticMatcher(ABC):
    """Optional semantic matching layer consulted on exact-cache misses."""

    @abstractmethod
    async def record(self, key: str, prompt: str) -> None:
        """Insert or update the embedding for a cacheable prompt."""

    @abstractmethod
    async def find_similar(self, prompt: str) -> tuple[str, float] | None:
        """Return the most similar cached key and similarity score."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a semantic index entry."""

    @abstractmethod
    async def clear(self) -> None:
        """Reset the semantic index."""
