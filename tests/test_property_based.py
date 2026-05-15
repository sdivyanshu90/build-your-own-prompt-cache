from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from prompt_cache.backends import InMemoryBackend
from prompt_cache.core import PromptCache
from prompt_cache.eviction import LRUPolicy
from prompt_cache.key_builder import CacheKeyBuilder


@given(
    prompt=st.text(min_size=0, max_size=128),
    model=st.text(min_size=1, max_size=32),
)
def test_same_prompt_always_produces_same_key(prompt: str, model: str) -> None:
    builder = CacheKeyBuilder()
    left = builder.build_key(prompt, model=model, temperature=0.0)
    right = builder.build_key(prompt, model=model, temperature=0.0)
    assert left == right


@pytest.mark.asyncio
@settings(deadline=None, max_examples=30)
@given(
    prompt_a=st.text(min_size=1, max_size=64),
    prompt_b=st.text(min_size=1, max_size=64),
)
async def test_cache_never_returns_a_different_prompt(prompt_a: str, prompt_b: str) -> None:
    builder = CacheKeyBuilder(namespace="property", version="v1")
    assume(builder.normalize_text(prompt_a) != builder.normalize_text(prompt_b))
    cache = PromptCache(
        backend=InMemoryBackend(max_items=16, max_size_bytes=1024 * 1024),
        eviction_policy=LRUPolicy(),
        key_builder=builder,
        default_ttl=60.0,
        max_size_mb=1.0,
    )
    await cache.set(prompt_a, "response-a")
    assert await cache.get(prompt_b) is None


@pytest.mark.asyncio
@settings(deadline=None, max_examples=30)
@given(prompts=st.lists(st.text(min_size=1, max_size=32), min_size=1, max_size=20))
async def test_eviction_never_exceeds_max_items(prompts: list[str]) -> None:
    cache = PromptCache(
        backend=InMemoryBackend(max_items=5, max_size_bytes=1024 * 1024),
        eviction_policy=LRUPolicy(),
        key_builder=CacheKeyBuilder(namespace="property", version="v1"),
        default_ttl=60.0,
        max_size_mb=1.0,
    )
    for index, prompt in enumerate(prompts):
        await cache.set(f"{prompt}-{index}", f"response-{index}")
    stats = await cache.stats()
    assert stats.cache_item_count <= 5
