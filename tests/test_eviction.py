from __future__ import annotations

from prompt_cache.eviction import LFUPolicy, LRUPolicy, SLRUPolicy


def test_lru_policy_returns_least_recently_used_key() -> None:
    policy = LRUPolicy()
    for key in ("a", "b", "c"):
        policy.record_insert(key)
    policy.record_access("a")
    assert policy.select_victim() == "b"


def test_lfu_policy_prefers_lowest_frequency_then_oldest() -> None:
    policy = LFUPolicy()
    for key in ("a", "b", "c"):
        policy.record_insert(key)
    policy.record_access("a")
    policy.record_access("a")
    policy.record_access("b")
    assert policy.select_victim() == "c"


def test_slru_promotes_on_second_access() -> None:
    policy = SLRUPolicy(probationary_capacity=2, protected_capacity=2)
    policy.record_insert("a")
    policy.record_insert("b")
    policy.record_access("a")
    policy.record_insert("c")
    assert policy.select_victim() == "b"
