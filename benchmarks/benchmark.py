from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompt_cache.backends import InMemoryBackend
from prompt_cache.core import PromptCache
from prompt_cache.eviction import LFUPolicy, LRUPolicy, SLRUPolicy
from prompt_cache.key_builder import CacheKeyBuilder
from prompt_cache.llm import CachedLLMClient
from prompt_cache.semantic import SemanticCacheLayer, hashing_embedding


@dataclass(slots=True)
class ScenarioResult:
    name: str
    avg_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    hit_rate: float
    memory_per_entry_bytes: float


async def mock_completion(**kwargs):
    await asyncio.sleep(0.085)
    prompt = kwargs["prompt"]
    return {
        "response": f"mock completion for {prompt}",
        "tokens_used": max(1, len(prompt) // 4),
    }


def build_cache(policy_name: str) -> PromptCache:
    if policy_name == "lru":
        policy = LRUPolicy()
    elif policy_name == "lfu":
        policy = LFUPolicy()
    else:
        policy = SLRUPolicy(probationary_capacity=200, protected_capacity=800)
    return PromptCache(
        backend=InMemoryBackend(max_items=1000, max_size_bytes=64 * 1024 * 1024),
        eviction_policy=policy,
        key_builder=CacheKeyBuilder(namespace="bench", version="v1"),
        semantic_layer=SemanticCacheLayer(hashing_embedding, similarity_threshold=0.88),
        default_ttl=300.0,
        max_size_mb=64.0,
    )


async def run_no_cache_scenario(
    *,
    requests: int = 400,
    concurrency: int = 50,
) -> ScenarioResult:
    latencies: list[float] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(index: int) -> None:
        async with semaphore:
            started_ns = time.perf_counter_ns()
            await mock_completion(prompt=f"no-cache-{index}")
            latencies.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)

    started_ns = time.perf_counter_ns()
    await asyncio.gather(*(worker(index) for index in range(requests)))
    duration_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
    return ScenarioResult(
        name="No cache",
        avg_latency_ms=statistics.fmean(latencies),
        p99_latency_ms=sorted(latencies)[max(0, int(len(latencies) * 0.99) - 1)],
        throughput_rps=requests / duration_s,
        hit_rate=0.0,
        memory_per_entry_bytes=0.0,
    )


async def run_latency_scenario(
    name: str,
    target_hit_rate: float,
    *,
    requests: int = 400,
    concurrency: int = 50,
    policy_name: str = "lru",
) -> ScenarioResult:
    cache = build_cache(policy_name)
    client = CachedLLMClient(cache, mock_completion)
    rng = random.Random(7)
    hot_prompts = [f"hot-prompt-{index}" for index in range(25)]
    for prompt in hot_prompts:
        await client.complete(prompt)

    latencies: list[float] = []
    cached_count = 0
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(index: int) -> None:
        nonlocal cached_count
        async with semaphore:
            if rng.random() <= target_hit_rate:
                prompt = hot_prompts[index % len(hot_prompts)]
            else:
                prompt = f"cold-{index}-{rng.random():.8f}"
            result = await client.complete(prompt)
            latencies.append(result.latency_ms)
            cached_count += int(result.cached)

    started_ns = time.perf_counter_ns()
    await asyncio.gather(*(worker(index) for index in range(requests)))
    duration_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
    stats = await cache.stats()
    memory_per_entry = (
        stats.cache_size_bytes / max(stats.cache_item_count, 1)
        if stats.cache_item_count
        else 0.0
    )
    return ScenarioResult(
        name=name,
        avg_latency_ms=statistics.fmean(latencies),
        p99_latency_ms=sorted(latencies)[max(0, int(len(latencies) * 0.99) - 1)],
        throughput_rps=requests / duration_s,
        hit_rate=cached_count / requests,
        memory_per_entry_bytes=memory_per_entry,
    )


async def run_semantic_scenario(
    *,
    requests: int = 400,
    concurrency: int = 50,
    target_semantic_rate: float = 0.85,
) -> ScenarioResult:
    cache = build_cache("lru")
    client = CachedLLMClient(cache, mock_completion)
    rng = random.Random(19)
    canonical_prompts = [
        f"explain prompt caching strategy number {index}" for index in range(25)
    ]
    semantic_variants = [
        f"describe prompt caching strategy number {index}" for index in range(25)
    ]
    for prompt in canonical_prompts:
        await client.complete(prompt)

    latencies: list[float] = []
    cached_count = 0
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(index: int) -> None:
        nonlocal cached_count
        async with semaphore:
            if rng.random() <= target_semantic_rate:
                prompt = semantic_variants[index % len(semantic_variants)]
            else:
                prompt = f"semantic-cold-{index}-{rng.random():.8f}"
            result = await client.complete(prompt)
            latencies.append(result.latency_ms)
            cached_count += int(result.cached)

    started_ns = time.perf_counter_ns()
    await asyncio.gather(*(worker(index) for index in range(requests)))
    duration_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
    stats = await cache.stats()
    memory_per_entry = (
        stats.cache_size_bytes / max(stats.cache_item_count, 1)
        if stats.cache_item_count
        else 0.0
    )
    return ScenarioResult(
        name="Semantic cache 85%",
        avg_latency_ms=statistics.fmean(latencies),
        p99_latency_ms=sorted(latencies)[max(0, int(len(latencies) * 0.99) - 1)],
        throughput_rps=requests / duration_s,
        hit_rate=cached_count / requests,
        memory_per_entry_bytes=memory_per_entry,
    )


def benchmark_key_builder(iterations: int = 10000) -> float:
    builder = CacheKeyBuilder(namespace="bench", version="v1")
    started_ns = time.perf_counter_ns()
    for index in range(iterations):
        builder.build_key(
            f"Explain prompt caching example {index % 20}",
            model="claude-sonnet-4-20250514",
            temperature=0.0,
            max_tokens=1024,
            system="You are a terse systems engineer.",
        )
    elapsed_ns = time.perf_counter_ns() - started_ns
    return elapsed_ns / iterations / 1_000_000.0


def benchmark_eviction_policies(trace_length: int = 50000) -> dict[str, float]:
    rng = random.Random(11)
    trace = [f"key-{rng.randint(0, 500)}" for _ in range(trace_length)]
    results: dict[str, float] = {}
    for name, policy in {
        "lru": LRUPolicy(),
        "lfu": LFUPolicy(),
        "slru": SLRUPolicy(probationary_capacity=128, protected_capacity=512),
    }.items():
        started_ns = time.perf_counter_ns()
        for key in trace:
            if policy.select_victim() is None or len(policy) < 512:
                policy.record_insert(key)
            else:
                policy.record_access(key)
        results[name] = (time.perf_counter_ns() - started_ns) / trace_length / 1_000.0
    return results


async def main() -> None:
    scenarios = [
        await run_no_cache_scenario(),
        await run_latency_scenario("50% hit rate", 0.50),
        await run_latency_scenario("90% hit rate", 0.90),
        await run_latency_scenario("99% hit rate", 0.99),
        await run_semantic_scenario(),
    ]
    print("Scenario,Avg Latency (ms),P99 Latency (ms),Throughput (req/s),Observed Hit Rate,Memory Per Entry (bytes)")
    for scenario in scenarios:
        print(
            f"{scenario.name},{scenario.avg_latency_ms:.2f},{scenario.p99_latency_ms:.2f},"
            f"{scenario.throughput_rps:.2f},{scenario.hit_rate:.2%},{scenario.memory_per_entry_bytes:.2f}"
        )
    print(f"Average key build time (ms): {benchmark_key_builder():.6f}")
    for policy_name, microseconds in benchmark_eviction_policies().items():
        print(f"Policy {policy_name} average bookkeeping time (us/op): {microseconds:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
