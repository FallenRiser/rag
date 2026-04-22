"""
RAG Pipeline — FastAPI application factory.

Startup sequence
----------------
1. Load + validate YAML configs (AppSettings).
2. Set up structured logging + optional OpenTelemetry tracing.
3. Register default pipeline config with PipelineRegistry (lazy build on
   first request).
4. Rebuild DocumentVersionRegistry from existing store metadata so a server
   restart doesn't lose version history.

OpenAPI / Swagger UI fix
------------------------
Pydantic v2 generates ``contentMediaType: application/octet-stream`` for
``UploadFile`` fields instead of the ``format: binary`` that Swagger UI
requires to render a file-picker widget. The ``custom_openapi()`` function
patches this post-generation so the Swagger UI shows correct file pickers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan hook: startup → yield → shutdown."""
    # ── Startup ─────────────────────────────────────────────────────────
    from config.settings import get_settings
    from utils.pipeline_registry import registry
    from utils.tracing import setup_tracing

    settings = get_settings()
    setup_tracing(settings)

    registry.register(settings, name="default")
    logger.info(
        "RAG Pipeline started | env=%s | chunking=%s | embedding=%s | store=%s",
        settings.env,
        settings.chunking.strategy.value,
        settings.embedding.provider.value,
        settings.document_store.backend.value,
    )

    # Rebuild version history from store so restarts don't lose tracking.
    try:
        from utils.document_version_registry import version_registry
        idx_pipeline = registry.get_indexing("default")
        store = idx_pipeline.get_component("writer").document_store
        n = version_registry.rebuild_from_store(store)
        logger.info("Version registry: %d record(s) restored from store.", n)
    except Exception as exc:
        logger.warning(
            "Could not rebuild version registry from store: %s — "
            "history will repopulate on new ingests.",
            exc,
        )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("RAG Pipeline shutting down.")


def _patch_upload_schema(node: object) -> None:
    """
    Recursively replace ``contentMediaType: application/octet-stream`` with
    ``format: binary`` so Swagger UI renders file-picker buttons.

    Root cause: Pydantic v2 changed the JSON schema representation of
    UploadFile.  FastAPI has not yet aligned its schema generation.
    """
    if isinstance(node, dict):
        if node.get("contentMediaType") == "application/octet-stream":
            node.pop("contentMediaType")
            node["format"] = "binary"
        for v in node.values():
            _patch_upload_schema(v)
    elif isinstance(node, list):
        for item in node:
            _patch_upload_schema(item)


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Pipeline API",
        description=(
            "Production-grade document extraction and RAG ingestion pipeline "
            "built on Haystack 2.x + Docling. "
            "PDF · DOCX · HTML · PPTX · images | OCR | table extraction | "
            "LLM image captioning | 12 chunking strategies | "
            "user isolation | document versioning."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    from app.routes.v1 import v1_router
    app.include_router(v1_router)

    # ── Custom OpenAPI: fix UploadFile schema for Swagger file pickers ───
    from fastapi.openapi.utils import get_openapi

    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        _patch_upload_schema(schema)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    # ── Root redirect ─────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"service": "rag-pipeline", "docs": "/docs", "health": "/v1/health"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
