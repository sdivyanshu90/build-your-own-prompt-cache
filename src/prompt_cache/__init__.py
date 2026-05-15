"""Prompt caching reference implementation.

The package is intentionally modular so applications can swap storage,
eviction, semantic matching, and observability components independently.
"""

from .backends import DiskBackend, InMemoryBackend, RedisBackend
from .config import CacheConfig
from .core import PromptCache
from .factory import build_cache
from .interfaces import CacheBackend, EvictionPolicy
from .key_builder import CacheKeyBuilder
from .llm import CachedLLMClient
from .metrics import MetricsCollector
from .semantic import SemanticCacheLayer
from .types import CacheEntry, CacheStats, CompletionResult, WarmingResult
from .warm import dump_cache_to_file, warm_cache_from_file

__all__ = [
    "CacheConfig",
    "CacheBackend",
    "CacheEntry",
    "CacheStats",
    "CachedLLMClient",
    "CompletionResult",
    "DiskBackend",
    "EvictionPolicy",
    "InMemoryBackend",
    "MetricsCollector",
    "PromptCache",
    "RedisBackend",
    "SemanticCacheLayer",
    "WarmingResult",
    "CacheKeyBuilder",
    "build_cache",
    "dump_cache_to_file",
    "warm_cache_from_file",
]
