"""
POST /v1/ingest   — Upload files and trigger the Docling extraction + chunking pipeline.
GET  /v1/ingest/{job_id} — Poll async job status.
DELETE /v1/documents/{document_id} — Remove a document from the store.

Design decisions
----------------
- Files are written to a temp directory; their paths are passed to DoclingConverter.
- Conversion runs in a BackgroundTask so the HTTP response is immediate.
- A simple in-process dict tracks job state (replace with Redis for multi-worker).
- Per-file errors are captured and surfaced in the job detail without aborting others.
- Idempotency: content-hash checked before writing to avoid re-embedding duplicates.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status

from app.dependencies import get_app_settings, get_registry
from app.schemas.ingest import (
    DeleteDocumentResponse,
    FileIngestResult,
    IngestJobDetail,
    IngestResponse,
    JobStatus,
)
from config.settings import AppSettings
from utils.pipeline_registry import PipelineRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# ---------------------------------------------------------------------------
# In-process job store  (swap for Redis / DB in production multi-worker setup)
# ---------------------------------------------------------------------------

_jobs: dict[str, IngestJobDetail] = {}


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_ingestion(
    job_id: str,
    tmp_dir: str,
    file_paths: list[Path],
    pipeline_name: str,
    extra_meta: dict,
    registry: PipelineRegistry,
) -> None:
    """
    Execute Docling extraction → chunk → embed → write for each file.
    Updates the shared job record as files complete.
    """
    job = _jobs[job_id]
    job.status = JobStatus.RUNNING

    indexing_pipeline = None
    try:
        indexing_pipeline = registry.get_indexing(pipeline_name)
    except Exception as exc:
        logger.error("Failed to build indexing pipeline for job %s: %s", job_id, exc)
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.completed_at = datetime.now(tz=timezone.utc)
        return

    results: list[FileIngestResult] = []
    total_chunks = 0
    total_written = 0
    any_failed = False

    for file_path in file_paths:
        fname = file_path.name
        logger.info("Job %s: processing file '%s'", job_id, fname)
        try:
            output = indexing_pipeline.run(
                {
                    "converter": {
                        "sources": [str(file_path)],
                        "meta": {"source_file": fname, **extra_meta},
                    }
                }
            )
            written = output.get("writer", {}).get("documents_written", 0)
            # Chunk count is the number of docs entering the writer.
            chunks = output.get("embedder", {}).get("meta", {}).get("batch_size", written)

            total_written += written
            total_chunks += chunks

            results.append(
                FileIngestResult(
                    filename=fname,
                    status=JobStatus.COMPLETED,
                    documents_written=written,
                    chunks_created=chunks,
                )
            )
            logger.info(
                "Job %s: '%s' → %d chunks, %d written", job_id, fname, chunks, written
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Job %s: failed on '%s': %s", job_id, fname, exc, exc_info=True)
            results.append(
                FileIngestResult(
                    filename=fname,
                    status=JobStatus.FAILED,
                    error=str(exc),
                )
            )
            any_failed = True

        # Update running totals after each file.
        job.files_processed += 1
        job.total_chunks = total_chunks
        job.total_documents_written = total_written
        job.results = results

    # Cleanup temp files.
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    job.status = JobStatus.PARTIAL if any_failed else JobStatus.COMPLETED
    job.completed_at = datetime.now(tz=timezone.utc)
    logger.info(
        "Job %s finished: status=%s chunks=%d written=%d",
        job_id,
        job.status,
        total_chunks,
        total_written,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest documents",
    description=(
        "Upload one or more documents (PDF, DOCX, HTML, PPTX, images) and trigger "
        "the Docling extraction + chunking + embedding pipeline asynchronously. "
        "Poll GET /v1/ingest/{job_id} to check completion."
    ),
)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File(description="Documents to ingest")],
    pipeline_name: Annotated[str, Form()] = "default",
    registry: PipelineRegistry = Depends(get_registry),
    settings: AppSettings = Depends(get_app_settings),
) -> IngestResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file must be provided.",
        )

    # Validate pipeline name is registered.
    if pipeline_name not in registry.registered_names():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Pipeline '{pipeline_name}' is not registered. "
                f"Available: {registry.registered_names()}"
            ),
        )

    job_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix=f"rag_ingest_{job_id}_")
    saved_paths: list[Path] = []

    # Save uploaded files to temp dir synchronously before launching background task.
    for upload in files:
        safe_name = Path(upload.filename or "upload").name
        dest = Path(tmp_dir) / safe_name
        with dest.open("wb") as f:
            content = await upload.read()
            f.write(content)
        saved_paths.append(dest)

    # Create job record.
    job = IngestJobDetail(
        job_id=job_id,
        status=JobStatus.PENDING,
        pipeline_name=pipeline_name,
        created_at=datetime.now(tz=timezone.utc),
        files_received=len(saved_paths),
    )
    _jobs[job_id] = job

    # Launch background processing.
    background_tasks.add_task(
        _run_ingestion,
        job_id=job_id,
        tmp_dir=tmp_dir,
        file_paths=saved_paths,
        pipeline_name=pipeline_name,
        extra_meta={},
        registry=registry,
    )

    logger.info(
        "Ingestion job %s created | files=%d | pipeline=%s",
        job_id,
        len(saved_paths),
        pipeline_name,
    )

    return IngestResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        pipeline_name=pipeline_name,
        files_received=len(saved_paths),
        message=f"Job {job_id} queued. Poll GET /v1/ingest/{job_id} for status.",
    )


@router.get(
    "/{job_id}",
    response_model=IngestJobDetail,
    summary="Get ingestion job status",
)
async def get_job_status(job_id: str) -> IngestJobDetail:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return job


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document from the store",
)
async def delete_document(
    document_id: str,
    pipeline_name: str = "default",
    registry: PipelineRegistry = Depends(get_registry),
) -> DeleteDocumentResponse:
    try:
        pipeline = registry.get_indexing(pipeline_name)
        # Access the writer's document store.
        writer = pipeline.get_component("writer")
        store = writer.document_store
        store.delete_documents([document_id])
        return DeleteDocumentResponse(
            document_id=document_id,
            deleted=True,
            message=f"Document '{document_id}' deleted from store.",
        )
    except Exception as exc:
        logger.error("Failed to delete document %s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
