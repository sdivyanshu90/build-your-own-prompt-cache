"""Configuration models and YAML loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class CacheConfig(BaseSettings):
    """Typed configuration for the prompt cache service.

    Environment variables use the ``PROMPT_CACHE_`` prefix. YAML configuration
    can be layered underneath environment variables so production rollouts can
    keep immutable defaults in source control while allowing overrides through
    deployment manifests.
    """

    backend: Literal["memory", "redis", "disk"] = "memory"
    eviction_policy: Literal["lru", "lfu", "slru"] = "lru"
    max_size_mb: float = Field(default=512.0, gt=0)
    max_items: int = Field(default=10000, gt=0)
    default_ttl_seconds: float = Field(default=3600.0, gt=0)
    redis_url: str | None = None
    redis_pool_size: int = Field(default=20, gt=0)
    redis_socket_timeout_seconds: float = Field(default=1.5, gt=0)
    disk_cache_dir: str = "/tmp/prompt_cache"
    semantic_cache_enabled: bool = False
    semantic_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    namespace: str = "default"
    version: str = "v1"
    enable_metrics: bool = True
    log_level: str = "INFO"
    service_host: str = "0.0.0.0"
    service_port: int = Field(default=8000, gt=0)
    protected_segment_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    embedding_backend: str = "builtin"
    metrics_buffer_size: int = Field(default=10000, gt=10)

    model_config = SettingsConfigDict(
        env_prefix="PROMPT_CACHE_",
        env_file=".env",
        extra="ignore",
    )

    @classmethod
    def from_yaml(cls, filepath: str | Path | None = None) -> "CacheConfig":
        """Load configuration from YAML and then apply environment overrides."""

        if filepath is None:
            return cls()
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(**payload)
