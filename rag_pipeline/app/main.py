"""
RAG Pipeline — FastAPI application entry point.

Lifecycle
---------
startup:
  1. Load and validate all YAML configs via AppSettings.
  2. Initialise OpenTelemetry tracing.
  3. Register the default pipeline config with PipelineRegistry.
     (Pipelines are built lazily on first request.)
shutdown:
  - Graceful cleanup (nothing heavyweight needed for now).
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
    """FastAPI lifespan: startup → yield → shutdown."""
    # ── Startup ──────────────────────────────────────────────────────────
    from config.settings import get_settings
    from utils.pipeline_registry import registry
    from utils.tracing import setup_tracing

    settings = get_settings()

    # Structured logging + optional OTel tracing.
    setup_tracing(settings)

    # Register the default pipeline so /v1/ready returns 200.
    registry.register(settings, name="default")

    logger.info(
        "RAG Pipeline started | env=%s | chunking=%s | embedding=%s | store=%s",
        settings.env,
        settings.chunking.strategy.value,
        settings.embedding.provider.value,
        settings.document_store.backend.value,
    )

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("RAG Pipeline shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Pipeline API",
        description=(
            "Production-grade document extraction and RAG ingestion pipeline "
            "built on Haystack 2.x + Docling. "
            "Supports PDF, DOCX, HTML, PPTX, images with OCR fallback, "
            "table extraction, and LLM-based image captioning."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    from app.routes.v1 import v1_router

    app.include_router(v1_router)

    # ── Root ──────────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "service": "rag-pipeline",
            "docs": "/docs",
            "health": "/v1/health",
        }

    return app


# ---------------------------------------------------------------------------
# ASGI entry point
# ---------------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
