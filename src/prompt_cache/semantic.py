"""Semantic similarity cache used for fuzzy prompt matching."""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Sequence

try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency.
    NearestNeighbors = None

from .interfaces import EmbeddingFunction, SemanticMatcher


def _normalize_vector(values: Sequence[float]) -> list[float]:
    magnitude = math.sqrt(sum(component * component for component in values))
    if magnitude == 0:
        return [0.0 for _ in values]
    return [component / magnitude for component in values]


def hashing_embedding(text: str, dimensions: int = 256) -> list[float]:
    """Deterministic fallback embedding for local development and tests.

    This is intentionally simple: tokens are hashed into a fixed-width vector and
    signed to reduce collision bias. It is not a replacement for model-quality
    embeddings, but it keeps the reference implementation runnable without a GPU
    or third-party embedding service.
    """

    vector = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return _normalize_vector(vector)


async def _resolve_embedding(
    embedding_function: EmbeddingFunction,
    prompt: str,
) -> list[float]:
    candidate = embedding_function(prompt)
    if asyncio.iscoroutine(candidate):
        resolved = await candidate
    else:
        resolved = candidate
    return _normalize_vector(list(resolved))


class SemanticCacheLayer(SemanticMatcher):
    """Fuzzy matching layer based on cosine similarity over prompt embeddings."""

    def __init__(
        self,
        embedding_function: EmbeddingFunction = hashing_embedding,
        similarity_threshold: float = 0.92,
    ) -> None:
        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold
        self._lock = asyncio.Lock()
        self._vectors: dict[str, list[float]] = {}
        self._index_keys: list[str] = []
        self._index_matrix: list[list[float]] = []
        self._nn_model = None
        self._dirty = True

    async def record(self, key: str, prompt: str) -> None:
        vector = await _resolve_embedding(self.embedding_function, prompt)
        async with self._lock:
            self._vectors[key] = vector
            self._dirty = True

    async def find_similar(self, prompt: str) -> tuple[str, float] | None:
        vector = await _resolve_embedding(self.embedding_function, prompt)
        async with self._lock:
            if not self._vectors:
                return None
            await self._ensure_index()
            if self._nn_model is not None:
                distance, indices = self._nn_model.kneighbors([vector], n_neighbors=1)
                best_index = int(indices[0][0])
                similarity = 1.0 - float(distance[0][0])
                if similarity >= self.similarity_threshold:
                    return self._index_keys[best_index], similarity
                return None

            best_key = ""
            best_similarity = -1.0
            for key, candidate in self._vectors.items():
                similarity = sum(left * right for left, right in zip(vector, candidate))
                if similarity > best_similarity:
                    best_key = key
                    best_similarity = similarity
            if best_similarity >= self.similarity_threshold:
                return best_key, best_similarity
            return None

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._vectors.pop(key, None)
            self._dirty = True

    async def clear(self) -> None:
        async with self._lock:
            self._vectors.clear()
            self._index_keys.clear()
            self._index_matrix.clear()
            self._nn_model = None
            self._dirty = False

    async def _ensure_index(self) -> None:
        if not self._dirty:
            return
        self._index_keys = list(self._vectors.keys())
        self._index_matrix = [self._vectors[key] for key in self._index_keys]
        if NearestNeighbors is not None and len(self._index_matrix) >= 4:
            self._nn_model = NearestNeighbors(metric="cosine")
            self._nn_model.fit(self._index_matrix)
        else:
            self._nn_model = None
        self._dirty = False
