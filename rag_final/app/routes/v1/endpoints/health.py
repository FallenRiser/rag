"""
GET /v1/health  — Component health check.
GET /v1/ready   — Kubernetes readiness probe.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Response, status

from app.dependencies import get_app_settings, get_registry
from app.schemas.pipeline import ComponentStatus, HealthResponse, ReadyResponse
from config.settings import AppSettings
from utils.pipeline_registry import PipelineRegistry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["observability"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Component health check",
)
async def health(
    settings: AppSettings = Depends(get_app_settings),
    registry: PipelineRegistry = Depends(get_registry),
) -> HealthResponse:
    components: dict[str, ComponentStatus] = {}

    # ── Registry ──────────────────────────────────────────────────────────
    registered = registry.registered_names()
    components["registry"] = ComponentStatus(
        name="pipeline_registry",
        type="PipelineRegistry",
        status="ok" if registered else "degraded",
        details={"registered_pipelines": registered},
    )

    # ── Document store (via default indexing pipeline if built) ───────────
    store_status = "unavailable"
    store_details: dict = {}
    if "default" in registered and registry.is_built("default")["indexing"]:
        try:
            pipeline = registry.get_indexing("default")
            writer = pipeline.get_component("writer")
            count = writer.document_store.count_documents()
            store_status = "ok"
            store_details = {
                "backend": settings.document_store.backend.value,
                "document_count": count,
            }
        except Exception as exc:
            store_status = "degraded"
            store_details = {"error": str(exc)}

    components["document_store"] = ComponentStatus(
        name="document_store",
        type=settings.document_store.backend.value,
        status=store_status,
        details=store_details,
    )

    # ── Embedding provider ────────────────────────────────────────────────
    components["embedder"] = ComponentStatus(
        name="embedder",
        type=settings.embedding.provider.value,
        status="ok",
        details={"provider": settings.embedding.provider.value},
    )

    overall = (
        "ok"
        if all(c.status == "ok" for c in components.values())
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        env=settings.env,
        components=components,
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Kubernetes readiness probe",
)
async def ready(
    response: Response,
    registry: PipelineRegistry = Depends(get_registry),
) -> ReadyResponse:
    """
    Returns 200 when the default pipeline config is registered and
    at least partially initialised.  Returns 503 otherwise.
    """
    if "default" not in registry.registered_names():
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(ready=False, message="Default pipeline not registered.")

    return ReadyResponse(ready=True, message="Ready.")
