"""Deterministic cache key generation for prompt/parameter tuples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import re
from typing import Any
import unicodedata

from .serde import dumps_text


class CacheKeyBuilder:
    """Build stable cache keys from prompts and LLM request parameters.

    The key builder is intentionally opinionated because prompt caching fails in
    subtle ways when logically equivalent requests produce different keys. The
    algorithm performs four steps:

    1. Normalize prompt text with Unicode NFKC normalization, lowercasing, and
       whitespace collapsing. This strips formatting noise that should not create
       distinct cache entries.
    2. Canonicalize request metadata into JSON-safe primitives with sorted map
       keys and deterministic ordering for sets and tuples.
    3. Replace bulky or sensitive fields such as the system prompt with a stable
       hash so keys do not leak large prompt bodies into Redis or filesystem
       paths.
    4. Hash the canonical JSON payload with SHA-256 and prepend the namespace
       and version. The namespace separates environments such as ``dev`` and
       ``prod``; the version allows safe invalidation after schema changes.

    Exclusions exist because some request parameters are operational rather than
    semantic. For example, ``bypass_cache`` or ``trace_id`` should not fragment
    the keyspace. The class makes those exclusions explicit instead of relying on
    call-site discipline.
    """

    _whitespace_pattern = re.compile(r"\s+")

    def __init__(
        self,
        namespace: str = "default",
        version: str = "v1",
        exclude_params: set[str] | None = None,
    ) -> None:
        self.namespace = namespace.strip() or "default"
        self.version = version.strip() or "v1"
        self.exclude_params = {
            "bypass_cache",
            "cache_ttl",
            "request_id",
            "trace_id",
        }
        if exclude_params:
            self.exclude_params.update(exclude_params)

    def normalize_text(self, value: str) -> str:
        """Normalize text so inconsequential formatting does not fragment keys."""

        normalized = unicodedata.normalize("NFKC", value)
        normalized = normalized.strip().lower()
        return self._whitespace_pattern.sub(" ", normalized)

    def _canonicalize(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self.normalize_text(value)
        if isinstance(value, bytes):
            return hashlib.sha256(value).hexdigest()
        if isinstance(value, Mapping):
            return {
                str(key): self._canonicalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, set):
            return sorted(self._canonicalize(item) for item in value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._canonicalize(item) for item in value]
        return self.normalize_text(str(value))

    def build_components(self, prompt: str, **llm_params: Any) -> dict[str, Any]:
        """Return the canonical key material before hashing."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        normalized_prompt = self.normalize_text(prompt)
        metadata: dict[str, Any] = {}
        for key, value in llm_params.items():
            if key in self.exclude_params:
                continue
            if key == "system" and value is not None:
                metadata["system_hash"] = hashlib.sha256(
                    self.normalize_text(str(value)).encode("utf-8")
                ).hexdigest()
                continue
            metadata[key] = self._canonicalize(value)

        return {
            "namespace": self.namespace,
            "version": self.version,
            "prompt": normalized_prompt,
            "metadata": self._canonicalize(metadata),
        }

    def canonical_json(self, prompt: str, **llm_params: Any) -> str:
        """Return the canonical JSON used as hash input."""

        return dumps_text(self.build_components(prompt, **llm_params))

    def build_key(self, prompt: str, **llm_params: Any) -> str:
        """Build the final namespaced cache key."""

        canonical_json = self.canonical_json(prompt, **llm_params)
        digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        return f"{self.namespace}:{self.version}:{digest}"
