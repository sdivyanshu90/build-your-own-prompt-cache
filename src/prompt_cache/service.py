"""Minimal HTTP service for health checks and metrics export."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import os
from typing import AsyncIterator

from .config import CacheConfig
from .factory import build_cache

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import PlainTextResponse
except ImportError:  # pragma: no cover - optional dependency.
    FastAPI = None
    HTTPException = RuntimeError
    PlainTextResponse = None


def _require_fastapi() -> None:
    if FastAPI is None or PlainTextResponse is None:
        raise RuntimeError(
            "FastAPI is not installed. Install the 'service' extra to run the HTTP service."
        )


def create_app(config: CacheConfig):
    """Create a FastAPI app exposing cache health and metrics endpoints."""

    _require_fastapi()
    cache = build_cache(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await cache.close()

    app = FastAPI(title="Prompt Cache", version=config.version, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, object]:
        stats = await cache.stats()
        return {
            "status": "ok",
            "backend": config.backend,
            "eviction_policy": config.eviction_policy,
            "namespace": config.namespace,
            "stats": asdict(stats),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        if not config.enable_metrics:
            raise HTTPException(status_code=404, detail="Metrics endpoint disabled")
        return cache.metrics.export_prometheus()

    return app


def create_default_app():
    """Factory entrypoint used by ``uvicorn --factory``."""

    config_path = os.getenv("PROMPT_CACHE_CONFIG")
    config = CacheConfig.from_yaml(config_path)
    return create_app(config)
