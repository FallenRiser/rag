"""
GET  /v1/pipelines          — List all registered pipeline configs.
GET  /v1/pipelines/{name}   — Get detail for a named pipeline.
POST /v1/pipelines/{name}/reload — Hot-reload pipeline (drop cache).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_app_settings, get_registry
from app.schemas.pipeline import PipelineInfo
from config.settings import AppSettings
from utils.pipeline_registry import PipelineRegistry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.get(
    "",
    response_model=list[PipelineInfo],
    summary="List registered pipeline configs",
)
async def list_pipelines(
    registry: PipelineRegistry = Depends(get_registry),
    settings: AppSettings = Depends(get_app_settings),
) -> list[PipelineInfo]:
    return [
        _pipeline_info(name, registry, settings)
        for name in registry.registered_names()
    ]


@router.get(
    "/{name}",
    response_model=PipelineInfo,
    summary="Get pipeline config detail",
)
async def get_pipeline(
    name: str,
    registry: PipelineRegistry = Depends(get_registry),
    settings: AppSettings = Depends(get_app_settings),
) -> PipelineInfo:
    if name not in registry.registered_names():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline '{name}' not found.",
        )
    return _pipeline_info(name, registry, settings)


@router.post(
    "/{name}/reload",
    summary="Hot-reload pipeline (drop cache)",
    status_code=status.HTTP_200_OK,
)
async def reload_pipeline(
    name: str,
    registry: PipelineRegistry = Depends(get_registry),
) -> dict:
    if name not in registry.registered_names():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline '{name}' not found.",
        )
    registry.reload(name)
    return {"message": f"Pipeline '{name}' cache evicted. Will rebuild on next request."}


def _pipeline_info(
    name: str,
    registry: PipelineRegistry,
    settings: AppSettings,
) -> PipelineInfo:
    built = registry.is_built(name)
    return PipelineInfo(
        name=name,
        indexing_built=built["indexing"],
        query_built=built["query"],
        chunking_strategy=settings.chunking.strategy.value,
        embedding_provider=settings.embedding.provider.value,
        document_store_backend=settings.document_store.backend.value,
        export_type=settings.docling.export.type.value,
    )
