"""Cache warm-up and persistence helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time

try:
    import aiofiles  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency.
    aiofiles = None

from .core import PromptCache
from .serde import dumps_text, loads
from .types import WarmingResult


async def warm_cache_from_file(cache: PromptCache, filepath: str) -> WarmingResult:
    """Load a JSONL file of prompt/response pairs and populate the cache."""

    start_ns = time.perf_counter_ns()
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(filepath)
    loaded_entries = 0
    skipped_entries = 0
    errors: list[str] = []

    if aiofiles is not None:
        async with aiofiles.open(path, "r", encoding="utf-8") as handle:
            async for line in handle:
                loaded_entries, skipped_entries = await _process_warm_line(
                    cache,
                    line,
                    loaded_entries,
                    skipped_entries,
                    errors,
                )
    else:
        content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        for line in content.splitlines():
            loaded_entries, skipped_entries = await _process_warm_line(
                cache,
                line,
                loaded_entries,
                skipped_entries,
                errors,
            )

    return WarmingResult(
        loaded_entries=loaded_entries,
        skipped_entries=skipped_entries,
        duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000.0,
        output_path=str(path),
        errors=errors,
    )


async def _process_warm_line(
    cache: PromptCache,
    line: str,
    loaded_entries: int,
    skipped_entries: int,
    errors: list[str],
) -> tuple[int, int]:
    stripped = line.strip()
    if not stripped:
        return loaded_entries, skipped_entries + 1
    try:
        payload = loads(stripped)
        prompt = payload["prompt"]
        response = payload["response"]
        llm_params = dict(payload.get("llm_params", {}))
        if "ttl" in payload:
            llm_params["cache_ttl"] = payload["ttl"]
        await cache.set(prompt, response, **llm_params)
        return loaded_entries + 1, skipped_entries
    except Exception as exc:
        errors.append(str(exc))
        return loaded_entries, skipped_entries + 1


async def dump_cache_to_file(cache: PromptCache, filepath: str) -> WarmingResult:
    """Persist live cache entries to a JSONL file for warm restart workflows."""

    start_ns = time.perf_counter_ns()
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = await cache.list_entries()
    lines = []
    skipped_entries = 0
    for entry in entries:
        prompt = entry.metadata.get("prompt")
        llm_params = dict(entry.metadata.get("llm_params", {}))
        if not prompt:
            skipped_entries += 1
            continue
        lines.append(
            dumps_text(
                {
                    "prompt": prompt,
                    "response": entry.value,
                    "llm_params": llm_params,
                    "ttl": entry.ttl,
                }
            )
        )
    payload = "\n".join(lines) + ("\n" if lines else "")
    if aiofiles is not None:
        async with aiofiles.open(path, "w", encoding="utf-8") as handle:
            await handle.write(payload)
    else:
        await asyncio.to_thread(path.write_text, payload, encoding="utf-8")

    return WarmingResult(
        loaded_entries=len(lines),
        skipped_entries=skipped_entries,
        duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000.0,
        output_path=str(path),
    )
