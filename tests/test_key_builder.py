from __future__ import annotations

from prompt_cache.key_builder import CacheKeyBuilder


def test_identical_prompts_with_whitespace_noise_share_key() -> None:
    builder = CacheKeyBuilder(namespace="prod", version="v2")
    left = builder.build_key("Hello   world\n", model="claude", temperature=0.0)
    right = builder.build_key(" hello world ", model="claude", temperature=0.0, trace_id="abc")
    assert left == right


def test_system_prompt_hash_changes_key() -> None:
    builder = CacheKeyBuilder()
    key_a = builder.build_key("Explain caching", model="claude", system="You are brief.")
    key_b = builder.build_key("Explain caching", model="claude", system="You are verbose.")
    assert key_a != key_b


def test_unicode_normalization_is_stable() -> None:
    builder = CacheKeyBuilder()
    composed = builder.build_key("Café", model="claude")
    decomposed = builder.build_key("Cafe\u0301", model="claude")
    assert composed == decomposed


def test_long_prompt_generates_namespaced_digest() -> None:
    builder = CacheKeyBuilder(namespace="staging", version="v9")
    prompt = "cache me " * 10000
    key = builder.build_key(prompt, model="claude-sonnet")
    assert key.startswith("staging:v9:")
    assert len(key.split(":")) == 3
