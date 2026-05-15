"""Eviction strategy implementations used by ``PromptCache``."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

from .interfaces import EvictionPolicy


@dataclass(slots=True)
class _Node:
    key: str
    prev: "_Node | None" = None
    next: "_Node | None" = None


class _LRUChain:
    """Minimal doubly linked list with O(1) access via a side map.

    The data structure mirrors the classic textbook LRU layout:

    - a hash map gives O(1) access to nodes by key
    - a doubly linked list orders nodes from least recently used to most recent

    Moving a node after a successful ``get`` is therefore constant time because
    the map avoids linear search and the linked list avoids array shuffling.
    """

    def __init__(self) -> None:
        self._head = _Node("__head__")
        self._tail = _Node("__tail__")
        self._head.next = self._tail
        self._tail.prev = self._head
        self._nodes: dict[str, _Node] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def append_mru(self, key: str) -> None:
        if key in self._nodes:
            self.move_to_mru(key)
            return
        node = _Node(key=key)
        self._nodes[key] = node
        self._insert_before(self._tail, node)

    def move_to_mru(self, key: str) -> None:
        node = self._nodes.get(key)
        if node is None:
            return
        self._unlink(node)
        self._insert_before(self._tail, node)

    def remove(self, key: str) -> None:
        node = self._nodes.pop(key, None)
        if node is None:
            return
        self._unlink(node)

    def lru_key(self) -> str | None:
        node = self._head.next
        if node is None or node is self._tail:
            return None
        return node.key

    def pop_lru(self) -> str | None:
        key = self.lru_key()
        if key is None:
            return None
        self.remove(key)
        return key

    def _unlink(self, node: _Node) -> None:
        prev_node = node.prev
        next_node = node.next
        if prev_node is not None:
            prev_node.next = next_node
        if next_node is not None:
            next_node.prev = prev_node
        node.prev = None
        node.next = None

    def _insert_before(self, anchor: _Node, node: _Node) -> None:
        prev_node = anchor.prev
        node.prev = prev_node
        node.next = anchor
        if prev_node is not None:
            prev_node.next = node
        anchor.prev = node


class LRUPolicy(EvictionPolicy):
    """Least Recently Used eviction with O(1) access bookkeeping."""

    def __init__(self) -> None:
        self._chain = _LRUChain()

    def record_insert(self, key: str) -> None:
        self._chain.append_mru(key)

    def record_access(self, key: str) -> None:
        self._chain.move_to_mru(key)

    def record_delete(self, key: str) -> None:
        self._chain.remove(key)

    def select_victim(self) -> str | None:
        return self._chain.lru_key()

    def __len__(self) -> int:
        return len(self._chain)


class LFUPolicy(EvictionPolicy):
    """Least Frequently Used eviction with O(log n) updates.

    Frequency is tracked in a min-heap keyed by ``(count, recency, key)``. The
    heap may hold stale records, so victim selection discards obsolete tuples
    until the top of the heap matches the current state map.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, str]] = []
        self._state: dict[str, tuple[int, int]] = {}
        self._clock = 0

    def record_insert(self, key: str) -> None:
        self._clock += 1
        self._state[key] = (1, self._clock)
        heapq.heappush(self._heap, (1, self._clock, key))

    def record_access(self, key: str) -> None:
        current = self._state.get(key)
        if current is None:
            self.record_insert(key)
            return
        frequency, _ = current
        self._clock += 1
        updated = (frequency + 1, self._clock)
        self._state[key] = updated
        heapq.heappush(self._heap, (updated[0], updated[1], key))

    def record_delete(self, key: str) -> None:
        self._state.pop(key, None)

    def select_victim(self) -> str | None:
        while self._heap:
            frequency, clock, key = self._heap[0]
            current = self._state.get(key)
            if current is None or current != (frequency, clock):
                heapq.heappop(self._heap)
                continue
            return key
        return None

    def __len__(self) -> int:
        return len(self._state)


class SLRUPolicy(EvictionPolicy):
    """Segmented LRU with probationary and protected segments.

    Newly inserted items start in the probationary segment. The second access
    promotes them to the protected segment, which prevents one-off scans from
    displacing repeatedly used prompts. This mirrors CPU cache designs that
    separate recently touched lines from genuinely hot working sets.
    """

    def __init__(
        self,
        probationary_capacity: int = 1024,
        protected_capacity: int = 4096,
    ) -> None:
        if probationary_capacity <= 0 or protected_capacity <= 0:
            raise ValueError("SLRU capacities must be positive")
        self._probationary = _LRUChain()
        self._protected = _LRUChain()
        self._probationary_capacity = probationary_capacity
        self._protected_capacity = protected_capacity

    def record_insert(self, key: str) -> None:
        if key in self._protected:
            self._protected.move_to_mru(key)
            return
        if key in self._probationary:
            self._probationary.move_to_mru(key)
            return
        self._probationary.append_mru(key)

    def record_access(self, key: str) -> None:
        if key in self._protected:
            self._protected.move_to_mru(key)
            return
        if key in self._probationary:
            self._probationary.remove(key)
            self._protected.append_mru(key)
            if len(self._protected) > self._protected_capacity:
                demoted = self._protected.pop_lru()
                if demoted is not None:
                    self._probationary.append_mru(demoted)
            return
        self._probationary.append_mru(key)

    def record_delete(self, key: str) -> None:
        self._probationary.remove(key)
        self._protected.remove(key)

    def select_victim(self) -> str | None:
        if len(self._probationary) > self._probationary_capacity:
            return self._probationary.lru_key()
        return self._probationary.lru_key() or self._protected.lru_key()

    def __len__(self) -> int:
        return len(self._probationary) + len(self._protected)
