from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("pytest_mock")

from prompt_cache.backends import InMemoryBackend
from prompt_cache.core import PromptCache
from prompt_cache.eviction import LRUPolicy
from prompt_cache.key_builder import CacheKeyBuilder
from prompt_cache.llm import CachedLLMClient


def _cache() -> PromptCache:
    return PromptCache(
        backend=InMemoryBackend(max_items=16, max_size_bytes=4 * 1024 * 1024),
        eviction_policy=LRUPolicy(),
        key_builder=CacheKeyBuilder(namespace="integration", version="v1"),
        default_ttl=60.0,
        max_size_mb=4.0,
    )


@pytest.mark.asyncio
async def test_cached_llm_client_reuses_cached_response(mocker) -> None:
    llm_callable = mocker.AsyncMock(
        return_value={"response": "42", "tokens_used": 128}
    )
    client = CachedLLMClient(_cache(), llm_callable)
    first = await client.complete("What is 6 * 7?")
    second = await client.complete("What is 6 * 7?")
    assert first.cached is False
    assert second.cached is True
    assert second.response == "42"
    assert llm_callable.await_count == 1


@pytest.mark.asyncio
async def test_cached_llm_client_handles_string_payload() -> None:
    llm_callable = AsyncMock(return_value="cached answer")
    client = CachedLLMClient(_cache(), llm_callable)
    first = await client.complete("What is prompt caching?")
    second = await client.complete("What is prompt caching?")
    assert first.cached is False
    assert second.cached is True
    assert second.tokens_used >= 1
