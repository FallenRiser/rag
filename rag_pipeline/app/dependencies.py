"""
FastAPI dependency injection.

Provides singleton instances of AppSettings and PipelineRegistry
via ``Depends()``.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from config.settings import AppSettings, get_settings
from utils.pipeline_registry import PipelineRegistry, registry as _global_registry


# ---------------------------------------------------------------------------
# Settings dependency
# ---------------------------------------------------------------------------


def get_app_settings() -> AppSettings:
    """Return the cached AppSettings instance."""
    return get_settings()


# ---------------------------------------------------------------------------
# Registry dependency
# ---------------------------------------------------------------------------


def get_registry() -> PipelineRegistry:
    """Return the module-level PipelineRegistry singleton."""
    return _global_registry
