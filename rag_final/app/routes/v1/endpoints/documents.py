"""
GET    /v1/documents                           — List all source documents for the user.
GET    /v1/documents/{source_name}/versions    — List all versions of one source.
GET    /v1/documents/{source_name}/versions/{version}/chunks — Fetch chunks of a version.
DELETE /v1/documents/{source_name}             — Delete all versions.
DELETE /v1/documents/{source_name}/versions/{version} — Delete one version.

All endpoints are scoped to the authenticated user via X-User-ID header.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_registry
from app.schemas.documents import (
    ChunkDetail,
    DocumentSourceSummary,
    ListChunksResponse,
    ListSourcesResponse,
    ListVersionsResponse,
    VersionDetail,
)
from app.schemas.ingest import DeleteDocumentResponse
from utils.document_version_registry import version_registry
from utils.pipeline_registry import PipelineRegistry
from utils.user_context import UserContext, require_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# List all sources owned by the user
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=ListSourcesResponse,
    summary="List all ingested source documents for the user",
)
async def list_sources(
    ctx: UserContext = Depends(require_user),
) -> ListSourcesResponse:
    """
    Returns one summary entry per distinct source document (de-duplicated by
    source_name). Only shows the latest version metadata.
    """
    latest_records = version_registry.list_sources(ctx.user_id)

    summaries = [
        DocumentSourceSummary(
            source_name=rec.source_name,
            latest_version=rec.version,
            version_count=version_registry.version_count(ctx.user_id, rec.source_name),
            source_hash=rec.source_hash,
            document_count=rec.document_count,
            chunk_count=len(rec.chunk_ids),
            ingested_at=rec.ingested_at,
            version_note=rec.version_note,
        )
        for rec in latest_records
    ]

    return ListSourcesResponse(
        user_id=ctx.user_id,
        source_count=len(summaries),
        sources=summaries,
    )


# ---------------------------------------------------------------------------
# List all versions of one source
# ---------------------------------------------------------------------------


@router.get(
    "/{source_name}/versions",
    response_model=ListVersionsResponse,
    summary="List all versions of a source document",
)
async def list_versions(
    source_name: str,
    ctx: UserContext = Depends(require_user),
) -> ListVersionsResponse:
    records = version_registry.list_versions(ctx.user_id, source_name)

    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source '{source_name}' not found for user '{ctx.user_id}'.",
        )

    versions = [
        VersionDetail(
            source_name=rec.source_name,
            version=rec.version,
            is_latest=rec.is_latest,
            source_hash=rec.source_hash,
            document_count=rec.document_count,
            chunk_count=len(rec.chunk_ids),
            ingested_at=rec.ingested_at,
            version_note=rec.version_note,
        )
        for rec in records
    ]

    return ListVersionsResponse(
        user_id=ctx.user_id,
        source_name=source_name,
        version_count=len(versions),
        versions=versions,
    )


# ---------------------------------------------------------------------------
# Retrieve chunks for a specific version
# ---------------------------------------------------------------------------


@router.get(
    "/{source_name}/versions/{version}/chunks",
    response_model=ListChunksResponse,
    summary="Retrieve all chunks of a specific document version",
)
async def get_version_chunks(
    source_name: str,
    version: int,
    ctx: UserContext = Depends(require_user),
    registry: PipelineRegistry = Depends(get_registry),
) -> ListChunksResponse:
    record = version_registry.get_version(ctx.user_id, source_name, version)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Version {version} of '{source_name}' not found "
                f"for user '{ctx.user_id}'."
            ),
        )

    # Fetch chunks from the store using the version record's chunk IDs.
    chunks: list[ChunkDetail] = []
    try:
        store = registry.get_indexing().get_component("writer").document_store
        if record.chunk_ids:
            # Filter by explicit IDs for precision.
            docs = store.filter_documents(
                filters={
                    "operator": "AND",
                    "conditions": [
                        {
                            "field": "meta.user_id",
                            "operator": "==",
                            "value": ctx.user_id,
                        },
                        {
                            "field": "meta.source_name",
                            "operator": "==",
                            "value": source_name,
                        },
                        {
                            "field": "meta.version",
                            "operator": "==",
                            "value": version,
                        },
                    ],
                }
            )
        else:
            docs = []

        for doc in docs:
            meta = doc.meta or {}
            chunks.append(
                ChunkDetail(
                    document_id=doc.id or "",
                    content=doc.content or "",
                    chunk_index=meta.get("chunk_index"),
                    page_number=meta.get("page_number"),
                    headings=meta.get("headings", []),
                    is_table=meta.get("is_table", False),
                    is_picture=meta.get("is_picture", False),
                    picture_caption=meta.get("picture_caption"),
                    version=meta.get("version"),
                    is_latest=meta.get("is_latest", False),
                )
            )
    except Exception as exc:
        logger.error("get_version_chunks: store query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunks: {exc}",
        )

    return ListChunksResponse(
        user_id=ctx.user_id,
        source_name=source_name,
        version=version,
        is_latest=record.is_latest,
        chunk_count=len(chunks),
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# Delete all versions of a source
# ---------------------------------------------------------------------------


@router.delete(
    "/{source_name}",
    response_model=DeleteDocumentResponse,
    summary="Delete all versions of a source document",
)
async def delete_source(
    source_name: str,
    ctx: UserContext = Depends(require_user),
    registry: PipelineRegistry = Depends(get_registry),
) -> DeleteDocumentResponse:
    chunk_ids = version_registry.delete_all_versions(ctx.user_id, source_name)

    if not chunk_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source '{source_name}' not found for user '{ctx.user_id}'.",
        )

    removed = _delete_chunks_from_store(chunk_ids, registry)

    return DeleteDocumentResponse(
        source_name=source_name,
        version=None,
        deleted=True,
        chunks_removed=removed,
        message=f"All versions of '{source_name}' deleted ({removed} chunks removed).",
    )


# ---------------------------------------------------------------------------
# Delete a specific version
# ---------------------------------------------------------------------------


@router.delete(
    "/{source_name}/versions/{version}",
    response_model=DeleteDocumentResponse,
    summary="Delete a specific version of a source document",
)
async def delete_version(
    source_name: str,
    version: int,
    ctx: UserContext = Depends(require_user),
    registry: PipelineRegistry = Depends(get_registry),
) -> DeleteDocumentResponse:
    chunk_ids = version_registry.delete_version(ctx.user_id, source_name, version)

    if not chunk_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Version {version} of '{source_name}' not found "
                f"for user '{ctx.user_id}'."
            ),
        )

    removed = _delete_chunks_from_store(chunk_ids, registry)

    # If the previous version was promoted to is_latest, reflect that in the store.
    new_latest = version_registry.get_latest(ctx.user_id, source_name)
    if new_latest:
        try:
            store = registry.get_indexing().get_component("writer").document_store
            version_registry.update_latest_flag_in_store(
                document_store=store,
                user_id=ctx.user_id,
                source_name=source_name,
                new_latest_version=new_latest.version,
            )
        except Exception as exc:
            logger.warning(
                "delete_version: could not update is_latest flags: %s", exc
            )

    return DeleteDocumentResponse(
        source_name=source_name,
        version=version,
        deleted=True,
        chunks_removed=removed,
        message=(
            f"Version {version} of '{source_name}' deleted ({removed} chunks removed)."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _delete_chunks_from_store(
    chunk_ids: list[str], registry: PipelineRegistry
) -> int:
    """Delete a list of chunk IDs from the document store. Returns count deleted."""
    if not chunk_ids:
        return 0
    try:
        store = registry.get_indexing().get_component("writer").document_store
        store.delete_documents(chunk_ids)
        return len(chunk_ids)
    except Exception as exc:
        logger.error("_delete_chunks_from_store failed: %s", exc, exc_info=True)
        return 0
