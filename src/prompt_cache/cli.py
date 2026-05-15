"""Command-line interface for prompt cache operations."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .config import CacheConfig
from .factory import build_cache
from .service import create_app
from .warm import dump_cache_to_file, warm_cache_from_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt cache reference CLI")
    parser.add_argument("--config", help="Path to a YAML config file", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("stats", help="Print cache stats as JSON")
    subparsers.add_parser("clear", help="Clear all cache entries")
    subparsers.add_parser("health", help="Print a compact health summary")

    warm_parser = subparsers.add_parser("warm", help="Warm the cache from JSONL")
    warm_parser.add_argument("filepath")

    dump_parser = subparsers.add_parser("dump", help="Dump live cache entries to JSONL")
    dump_parser.add_argument("filepath")

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI health service")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    return parser


async def _run_async(args: argparse.Namespace) -> int:
    config = CacheConfig.from_yaml(args.config)
    cache = build_cache(config)
    try:
        if args.command == "stats":
            await cache.stats()
            print(cache.metrics.export_json())
            return 0
        if args.command == "health":
            stats = await cache.stats()
            print(
                {
                    "backend": config.backend,
                    "eviction_policy": config.eviction_policy,
                    "namespace": config.namespace,
                    "hit_rate": stats.hit_rate,
                    "items": stats.cache_item_count,
                    "size_bytes": stats.cache_size_bytes,
                }
            )
            return 0
        if args.command == "clear":
            await cache.clear()
            print("cache cleared")
            return 0
        if args.command == "warm":
            result = await warm_cache_from_file(cache, args.filepath)
            print(result)
            return 0
        if args.command == "dump":
            result = await dump_cache_to_file(cache, args.filepath)
            print(result)
            return 0
        raise ValueError(f"Unsupported command: {args.command}")
    finally:
        await cache.close()


def main() -> None:
    """CLI entrypoint exposed via ``python -m`` and console scripts."""

    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError(
                "uvicorn is not installed. Install the 'service' extra to serve HTTP endpoints."
            ) from exc
        config = CacheConfig.from_yaml(args.config)
        app = create_app(config)
        uvicorn.run(
            app,
            host=args.host or config.service_host,
            port=args.port or config.service_port,
        )
        return
    asyncio.run(_run_async(args))
