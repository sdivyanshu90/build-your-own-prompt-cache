"""Factory helpers for constructing configured cache components."""

from __future__ import annotations

from .backends import DiskBackend, InMemoryBackend, RedisBackend
from .config import CacheConfig
from .core import PromptCache
from .eviction import LFUPolicy, LRUPolicy, SLRUPolicy
from .key_builder import CacheKeyBuilder
from .metrics import MetricsCollector
from .semantic import SemanticCacheLayer, hashing_embedding


def build_backend(config: CacheConfig):
    max_size_bytes = int(config.max_size_mb * 1024 * 1024)
    if config.backend == "memory":
        return InMemoryBackend(max_items=config.max_items, max_size_bytes=max_size_bytes)
    if config.backend == "redis":
        return RedisBackend(
            config.redis_url,
            pool_size=config.redis_pool_size,
            socket_timeout_seconds=config.redis_socket_timeout_seconds,
            fallback_backend=InMemoryBackend(
                max_items=max(256, config.max_items // 10),
                max_size_bytes=max(16 * 1024 * 1024, max_size_bytes // 10),
            ),
        )
    return DiskBackend(config.disk_cache_dir)


def build_eviction_policy(config: CacheConfig):
    if config.eviction_policy == "lru":
        return LRUPolicy()
    if config.eviction_policy == "lfu":
        return LFUPolicy()
    protected_capacity = max(1, int(config.max_items * config.protected_segment_ratio))
    probationary_capacity = max(1, config.max_items - protected_capacity)
    return SLRUPolicy(
        probationary_capacity=probationary_capacity,
        protected_capacity=protected_capacity,
    )


def build_semantic_layer(config: CacheConfig) -> SemanticCacheLayer | None:
    if not config.semantic_cache_enabled:
        return None
    return SemanticCacheLayer(
        embedding_function=hashing_embedding,
        similarity_threshold=config.semantic_similarity_threshold,
    )


def build_cache(config: CacheConfig) -> PromptCache:
    metrics = MetricsCollector(buffer_size=config.metrics_buffer_size)
    return PromptCache(
        backend=build_backend(config),
        eviction_policy=build_eviction_policy(config),
        key_builder=CacheKeyBuilder(namespace=config.namespace, version=config.version),
        semantic_layer=build_semantic_layer(config),
        default_ttl=config.default_ttl_seconds,
        max_size_mb=config.max_size_mb,
        metrics_collector=metrics,
    )
