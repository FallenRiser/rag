"""
PipelineRegistry
================

Thread-safe registry that lazily builds and caches Haystack pipelines.

- ``get_indexing(name)``  → cached indexing pipeline
- ``get_query(name)``     → cached query pipeline (stub; implemented in Phase 5)
- ``reload(name)``        → drop cache for named config, rebuilt on next access
- ``register(name, cfg)`` → register a named AppSettings for a pipeline variant

Supports multiple named pipeline configs (e.g. "legal_docs", "financial"
with different chunking / enrichment settings) that can be selected
per-request via the API.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haystack import Pipeline

    from config.settings import AppSettings

logger = logging.getLogger(__name__)

_DEFAULT = "default"


class PipelineRegistry:
    """
    Lazily builds and caches Haystack indexing and query pipelines.

    Thread-safe: a lock guards both build and cache access so concurrent
    requests won't trigger duplicate (expensive) pipeline constructions.
    """

    def __init__(self) -> None:
        self._settings: dict[str, "AppSettings"] = {}
        self._indexing: dict[str, "Pipeline"] = {}
        self._query: dict[str, "Pipeline"] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, settings: "AppSettings", name: str = _DEFAULT) -> None:
        """Register an AppSettings instance under a pipeline name."""
        with self._lock:
            if name in self._settings:
                logger.warning(
                    "PipelineRegistry: overwriting existing settings for '%s'.", name
                )
            self._settings[name] = settings
            # Drop any stale cached pipelines for this name.
            self._indexing.pop(name, None)
            self._query.pop(name, None)
        logger.info("PipelineRegistry: registered pipeline config '%s'.", name)

    # ------------------------------------------------------------------
    # Getters (lazy build)
    # ------------------------------------------------------------------

    def get_indexing(self, name: str = _DEFAULT) -> "Pipeline":
        """Return (and lazily build) the indexing pipeline for ``name``."""
        with self._lock:
            if name not in self._indexing:
                self._indexing[name] = self._build_indexing(name)
        return self._indexing[name]

    def get_query(self, name: str = _DEFAULT) -> "Pipeline":
        """Return (and lazily build) the query pipeline for ``name``."""
        with self._lock:
            if name not in self._query:
                self._query[name] = self._build_query(name)
        return self._query[name]

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reload(self, name: str = _DEFAULT) -> None:
        """
        Drop cached pipelines for ``name``.
        They will be rebuilt on the next ``get_*`` call.
        Useful after a config hot-reload without restarting the server.
        """
        with self._lock:
            dropped = []
            if name in self._indexing:
                del self._indexing[name]
                dropped.append("indexing")
            if name in self._query:
                del self._query[name]
                dropped.append("query")
        logger.info(
            "PipelineRegistry: evicted %s pipeline(s) for '%s'.", dropped, name
        )

    def reload_all(self) -> None:
        """Drop all cached pipelines."""
        with self._lock:
            self._indexing.clear()
            self._query.clear()
        logger.info("PipelineRegistry: evicted all cached pipelines.")

    def registered_names(self) -> list[str]:
        """Return names of all registered pipeline configs."""
        with self._lock:
            return list(self._settings.keys())

    def is_built(self, name: str = _DEFAULT) -> dict[str, bool]:
        """Return build status for each pipeline type."""
        with self._lock:
            return {
                "indexing": name in self._indexing,
                "query": name in self._query,
            }

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _settings_for(self, name: str) -> "AppSettings":
        if name not in self._settings:
            raise KeyError(
                f"No pipeline config registered under '{name}'. "
                f"Available names: {list(self._settings.keys())}. "
                f"Call registry.register(settings, name='{name}') first."
            )
        return self._settings[name]

    def _build_indexing(self, name: str) -> "Pipeline":
        from utils.indexing_pipeline import IndexingPipelineBuilder

        logger.info("PipelineRegistry: building indexing pipeline '%s' …", name)
        settings = self._settings_for(name)
        pipeline = IndexingPipelineBuilder(settings).build()
        logger.info("PipelineRegistry: indexing pipeline '%s' ready.", name)
        return pipeline

    def _build_query(self, name: str) -> "Pipeline":
        from utils.indexing_pipeline import IndexingPipelineBuilder
        from utils.query_pipeline import QueryPipelineBuilder

        logger.info("PipelineRegistry: building query pipeline '%s' …", name)
        settings = self._settings_for(name)

        # Reuse the document store from the indexing pipeline so both pipelines
        # share the same in-memory / remote store instance.
        idx_pipeline = self.get_indexing(name)
        writer = idx_pipeline.get_component("writer")
        document_store = writer.document_store

        pipeline = QueryPipelineBuilder(settings).build(document_store)
        logger.info("PipelineRegistry: query pipeline '%s' ready.", name)
        return pipeline


# ---------------------------------------------------------------------------
# Module-level singleton — import and use directly.
# ---------------------------------------------------------------------------
registry = PipelineRegistry()
