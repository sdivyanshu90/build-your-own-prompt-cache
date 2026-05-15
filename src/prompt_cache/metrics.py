"""In-memory metrics collection and Prometheus export helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean
import threading
import time
from typing import Iterable

from .serde import dumps_text
from .types import CacheStats


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


@dataclass(slots=True)
class _Observation:
    timestamp: float
    cached: bool
    semantic_hit: bool
    latency_ms: float


class MetricsCollector:
    """Thread-safe metrics collector with rolling windows and text export."""

    def __init__(self, buffer_size: int = 10000) -> None:
        self._buffer_size = buffer_size
        self._lock = threading.RLock()
        self._request_events: deque[_Observation] = deque(maxlen=buffer_size)
        self._hit_latency_ms: deque[float] = deque(maxlen=buffer_size)
        self._miss_latency_ms: deque[float] = deque(maxlen=buffer_size)
        self._api_latency_ms: deque[float] = deque(maxlen=buffer_size)
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.semantic_hits = 0
        self.evictions = 0
        self.errors = 0
        self.cache_size_bytes = 0
        self.cache_item_count = 0

    def record_hit(self, latency_ms: float, semantic_hit: bool = False) -> None:
        """Record a cache hit and its latency."""

        with self._lock:
            self.total_requests += 1
            self.cache_hits += 1
            if semantic_hit:
                self.semantic_hits += 1
            self._hit_latency_ms.append(latency_ms)
            self._request_events.append(
                _Observation(
                    timestamp=time.time(),
                    cached=True,
                    semantic_hit=semantic_hit,
                    latency_ms=latency_ms,
                )
            )

    def record_miss(self, latency_ms: float) -> None:
        """Record a cache miss and its latency."""

        with self._lock:
            self.total_requests += 1
            self.cache_misses += 1
            self._miss_latency_ms.append(latency_ms)
            self._request_events.append(
                _Observation(
                    timestamp=time.time(),
                    cached=False,
                    semantic_hit=False,
                    latency_ms=latency_ms,
                )
            )

    def record_api_latency(self, latency_ms: float) -> None:
        """Record upstream API latency."""

        with self._lock:
            self._api_latency_ms.append(latency_ms)

    def record_eviction(self, count: int = 1) -> None:
        """Increment the eviction counter."""

        with self._lock:
            self.evictions += count

    def record_error(self, count: int = 1) -> None:
        """Increment the error counter."""

        with self._lock:
            self.errors += count

    def set_cache_state(self, *, size_bytes: int, item_count: int) -> None:
        """Update cache size gauges."""

        with self._lock:
            self.cache_size_bytes = size_bytes
            self.cache_item_count = item_count

    def _window_hit_rate(self, seconds: float, now: float) -> float:
        relevant = [
            event
            for event in self._request_events
            if event.timestamp >= now - seconds
        ]
        if not relevant:
            return 0.0
        hits = sum(1 for event in relevant if event.cached)
        return hits / len(relevant)

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        values_list = list(values)
        return mean(values_list) if values_list else 0.0

    def snapshot(self) -> CacheStats:
        """Return a structured snapshot of counters, gauges, and histograms."""

        with self._lock:
            now = time.time()
            hit_rate = (
                self.cache_hits / self.total_requests if self.total_requests else 0.0
            )
            gauges = {
                "hit_rate_1m": self._window_hit_rate(60.0, now),
                "hit_rate_5m": self._window_hit_rate(300.0, now),
                "hit_latency_p50_ms": _percentile(list(self._hit_latency_ms), 0.50),
                "hit_latency_p95_ms": _percentile(list(self._hit_latency_ms), 0.95),
                "hit_latency_p99_ms": _percentile(list(self._hit_latency_ms), 0.99),
                "miss_latency_p50_ms": _percentile(list(self._miss_latency_ms), 0.50),
                "miss_latency_p95_ms": _percentile(list(self._miss_latency_ms), 0.95),
                "miss_latency_p99_ms": _percentile(list(self._miss_latency_ms), 0.99),
                "api_latency_p50_ms": _percentile(list(self._api_latency_ms), 0.50),
                "api_latency_p95_ms": _percentile(list(self._api_latency_ms), 0.95),
                "api_latency_p99_ms": _percentile(list(self._api_latency_ms), 0.99),
                "semantic_hit_rate": (
                    self.semantic_hits / self.total_requests if self.total_requests else 0.0
                ),
            }
            return CacheStats(
                total_requests=self.total_requests,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                evictions=self.evictions,
                errors=self.errors,
                cache_size_bytes=self.cache_size_bytes,
                cache_item_count=self.cache_item_count,
                hit_rate=hit_rate,
                avg_hit_latency_ms=self._mean(self._hit_latency_ms),
                avg_miss_latency_ms=self._mean(self._miss_latency_ms),
                avg_api_latency_ms=self._mean(self._api_latency_ms),
                gauges=gauges,
            )

    def export_prometheus(self) -> str:
        """Render metrics in Prometheus text exposition format."""

        snapshot = self.snapshot()
        lines = [
            "# HELP prompt_cache_total_requests Total cache lookup attempts",
            "# TYPE prompt_cache_total_requests counter",
            f"prompt_cache_total_requests {snapshot.total_requests}",
            "# HELP prompt_cache_hits Total exact or semantic cache hits",
            "# TYPE prompt_cache_hits counter",
            f"prompt_cache_hits {snapshot.cache_hits}",
            "# HELP prompt_cache_misses Total cache misses",
            "# TYPE prompt_cache_misses counter",
            f"prompt_cache_misses {snapshot.cache_misses}",
            "# HELP prompt_cache_evictions Total evictions",
            "# TYPE prompt_cache_evictions counter",
            f"prompt_cache_evictions {snapshot.evictions}",
            "# HELP prompt_cache_errors Total cache-related errors",
            "# TYPE prompt_cache_errors counter",
            f"prompt_cache_errors {snapshot.errors}",
            "# HELP prompt_cache_size_bytes Approximate cache size in bytes",
            "# TYPE prompt_cache_size_bytes gauge",
            f"prompt_cache_size_bytes {snapshot.cache_size_bytes}",
            "# HELP prompt_cache_item_count Number of items in the cache",
            "# TYPE prompt_cache_item_count gauge",
            f"prompt_cache_item_count {snapshot.cache_item_count}",
            "# HELP prompt_cache_hit_rate Overall cache hit rate",
            "# TYPE prompt_cache_hit_rate gauge",
            f"prompt_cache_hit_rate {snapshot.hit_rate:.6f}",
        ]
        for metric_name, value in snapshot.gauges.items():
            lines.extend(
                [
                    f"# HELP prompt_cache_{metric_name} Derived cache metric",
                    f"# TYPE prompt_cache_{metric_name} gauge",
                    f"prompt_cache_{metric_name} {value:.6f}",
                ]
            )
        return "\n".join(lines) + "\n"

    def export_json(self) -> str:
        """Render metrics as structured JSON text."""

        snapshot = self.snapshot()
        return dumps_text(
            {
                "counters": {
                    "total_requests": snapshot.total_requests,
                    "cache_hits": snapshot.cache_hits,
                    "cache_misses": snapshot.cache_misses,
                    "evictions": snapshot.evictions,
                    "errors": snapshot.errors,
                },
                "gauges": {
                    "cache_size_bytes": snapshot.cache_size_bytes,
                    "cache_item_count": snapshot.cache_item_count,
                    "hit_rate": snapshot.hit_rate,
                    **snapshot.gauges,
                },
                "averages": {
                    "hit_latency_ms": snapshot.avg_hit_latency_ms,
                    "miss_latency_ms": snapshot.avg_miss_latency_ms,
                    "api_latency_ms": snapshot.avg_api_latency_ms,
                },
            }
        )
