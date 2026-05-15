"""JSON serialization helpers with optional ``orjson`` acceleration."""

from __future__ import annotations

import json
from typing import Any

try:
    import orjson  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised when optional deps are absent.
    orjson = None


def dumps(payload: Any) -> bytes:
    """Serialize a payload to stable JSON bytes.

    Stable ordering matters for cache key generation and for predictable test
    fixtures. ``orjson`` is used when available because it is materially faster
    under high request rates, but the fallback preserves identical ordering.
    """

    if orjson is not None:
        return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return json.dumps(
        payload,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def dumps_text(payload: Any) -> str:
    """Serialize a payload to a UTF-8 JSON string."""

    return dumps(payload).decode("utf-8")


def loads(payload: bytes | bytearray | memoryview | str) -> Any:
    """Deserialize a JSON payload from bytes or text."""

    if orjson is not None:
        if isinstance(payload, str):
            return orjson.loads(payload.encode("utf-8"))
        return orjson.loads(payload)
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return json.loads(bytes(payload).decode("utf-8"))
    return json.loads(payload)
