"""
POST /v1/ingest           — Upload files (multipart/form-data).
GET  /v1/ingest/jobs/{id} — Poll async job status.

File upload
-----------
Send files as standard multipart/form-data — exactly like a browser
<input type="file"> form. Do NOT base64-encode.

  curl -X POST http://localhost:8000/v1/ingest \\
       -H "X-User-ID: alice-001" \\
       -F "files=@report.pdf" \\
       -F "files=@slides.pptx" \\
       -F "pipeline_name=default" \\
       -F "version_note=Q3 reports"

Pipeline flow per file
----------------------
1. SHA-256 dedup check  →  skip if identical content already ingested
2. Save to temp dir
3. Run Haystack indexing pipeline
   converter  →  cleaner  →  chunker  →  meta_enricher  →  embedder  →  writer
   (meta_enricher receives user_id, source_name, version, is_latest per file)
4. Retire old is_latest flags in document store
5. Commit version record to DocumentVersionRegistry
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.dependencies import get_app_settings, get_registry
from app.schemas.ingest import (
    FileIngestResult,
    IngestJobDetail,
    IngestResponse,
    JobStatus,
)
from config.settings import AppSettings
from utils.document_version_registry import version_registry
from utils.pipeline_registry import PipelineRegistry
from utils.user_context import UserContext, require_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# In-process job store — swap for Redis / DB in multi-process deployments.
_jobs: dict[str, IngestJobDetail] = {}


# ---------------------------------------------------------------------------
# Background ingestion worker
# ---------------------------------------------------------------------------


def _run_ingestion(
    job_id: str,
    user_ctx: UserContext,
    tmp_dir: str,
    file_entries: list[dict],
    pipeline_name: str,
    version_note: str,
    registry: PipelineRegistry,
) -> None:
    """
    Runs per-job in a FastAPI background task (same process, different thread).
    """
    job = _jobs[job_id]
    job.status = JobStatus.RUNNING
    now_iso = datetime.now(tz=timezone.utc).isoformat()

    # Build (or fetch cached) pipeline.
    try:
        idx_pipeline = registry.get_indexing(pipeline_name)
    except Exception as exc:
        logger.error("Job %s: pipeline build failed: %s", job_id, exc)
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.completed_at = datetime.now(tz=timezone.utc)
        return

    # Grab document store reference for is_latest flag updates.
    try:
        store = idx_pipeline.get_component("writer").document_store
    except Exception:
        store = None

    results: list[FileIngestResult] = []
    total_chunks = 0
    total_written = 0
    any_failed = False

    for entry in file_entries:
        file_path: Path = entry["path"]
        source_name: str = entry["name"]
        file_bytes: bytes = entry["raw_bytes"]

        logger.info("Job %s | user=%s | file=%s", job_id, user_ctx.user_id, source_name)

        # ── Step 1: Dedup + version number assignment ──────────────────────
        check = version_registry.check_and_prepare(
            user_id=user_ctx.user_id,
            source_name=source_name,
            file_bytes=file_bytes,
            version_note=version_note,
        )

        if check.is_duplicate:
            logger.info(
                "Job %s | %s is duplicate (v%d) — skipping.",
                job_id, source_name, check.version,
            )
            results.append(FileIngestResult(
                filename=source_name,
                status=JobStatus.COMPLETED,
                documents_written=0,
                chunks_created=0,
                version=check.version,
                is_duplicate=True,
                message="Identical content already ingested as latest version.",
            ))
            job.files_processed += 1
            job.results = results
            continue

        # ── Step 2: Run the Haystack pipeline ─────────────────────────────
        try:
            output = idx_pipeline.run({
                "converter": {
                    "sources": [str(file_path)],
                    "meta": {
                        "source_file": str(file_path),
                        "source_name": source_name,
                    },
                },
                "meta_enricher": {
                    "user_id":      user_ctx.user_id,
                    "source_name":  source_name,
                    "source_hash":  check.source_hash,
                    "version":      check.version,
                    "is_latest":    True,
                    "ingested_at":  now_iso,
                    "version_note": version_note,
                },
            })

            written = output.get("writer", {}).get("documents_written", 0)
            total_written += written
            total_chunks  += written

            # Collect chunk IDs for the version record
            try:
                chunk_ids = [
                    d.id
                    for d in output.get("embedder", {}).get("documents", [])
                    if d.id
                ]
            except Exception:
                chunk_ids = []

            # ── Step 3: Retire old is_latest flags ─────────────────────────
            if store is not None and check.previous_version is not None:
                version_registry.update_latest_flag_in_store(
                    document_store=store,
                    user_id=user_ctx.user_id,
                    source_name=source_name,
                    new_latest_version=check.version,
                )

            # ── Step 4: Commit version record ──────────────────────────────
            record = version_registry.commit_version(
                user_id=user_ctx.user_id,
                source_name=source_name,
                source_hash=check.source_hash,
                version=check.version,
                chunk_ids=chunk_ids,
                document_count=written,
                version_note=version_note,
            )

            results.append(FileIngestResult(
                filename=source_name,
                status=JobStatus.COMPLETED,
                documents_written=written,
                chunks_created=written,
                version=record.version,
                is_duplicate=False,
                message=f"Ingested as version {record.version}.",
            ))
            logger.info(
                "Job %s | %s → v%d | %d chunks written",
                job_id, source_name, record.version, written,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Job %s | failed on %s: %s", job_id, source_name, exc, exc_info=True
            )
            results.append(FileIngestResult(
                filename=source_name,
                status=JobStatus.FAILED,
                error=str(exc),
                version=check.version,
            ))
            any_failed = True

        job.files_processed += 1
        job.total_chunks = total_chunks
        job.total_documents_written = total_written
        job.results = results

    shutil.rmtree(tmp_dir, ignore_errors=True)
    job.status = JobStatus.PARTIAL if any_failed else JobStatus.COMPLETED
    job.completed_at = datetime.now(tz=timezone.utc)
    logger.info("Job %s done | status=%s | chunks=%d", job_id, job.status, total_chunks)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest documents",
    description=(
        "Upload one or more documents as multipart/form-data. "
        "Accepts PDF, DOCX, HTML, PPTX, and image files. "
        "Identical content (SHA-256 dedup) is skipped automatically. "
        "Each changed re-upload creates a new version."
    ),
)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    # UploadFile list MUST use List[UploadFile] = File(...) syntax.
    # Annotated[list[UploadFile], File(...)] breaks Swagger UI (renders
    # array<string> instead of file pickers) due to a Pydantic v2 schema
    # generation bug with UploadFile in Annotated context.
    files: List[UploadFile] = File(
        description="Documents to ingest — PDF, DOCX, HTML, PPTX, images"
    ),
    pipeline_name: str = Form(default="default"),
    version_note: str = Form(
        default="",
        description="Optional human-readable note stored with this version",
    ),
    ctx: UserContext = Depends(require_user),
    registry: PipelineRegistry = Depends(get_registry),
    settings: AppSettings = Depends(get_app_settings),
) -> IngestResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file must be provided.",
        )
    if pipeline_name not in registry.registered_names():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Pipeline '{pipeline_name}' not registered. "
                f"Available: {registry.registered_names()}"
            ),
        )

    job_id  = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix=f"rag_{job_id}_")
    file_entries: list[dict] = []

    for upload in files:
        raw_bytes   = await upload.read()
        source_name = Path(upload.filename or "upload").name
        dest        = Path(tmp_dir) / source_name
        dest.write_bytes(raw_bytes)
        file_entries.append({"path": dest, "name": source_name, "raw_bytes": raw_bytes})

    job = IngestJobDetail(
        job_id=job_id,
        status=JobStatus.PENDING,
        pipeline_name=pipeline_name,
        user_id=ctx.user_id,
        created_at=datetime.now(tz=timezone.utc),
        files_received=len(file_entries),
    )
    _jobs[job_id] = job

    background_tasks.add_task(
        _run_ingestion,
        job_id=job_id,
        user_ctx=ctx,
        tmp_dir=tmp_dir,
        file_entries=file_entries,
        pipeline_name=pipeline_name,
        version_note=version_note,
        registry=registry,
    )

    logger.info(
        "Job %s queued | user=%s | files=%d | pipeline=%s",
        job_id, ctx.user_id, len(file_entries), pipeline_name,
    )

    return IngestResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        pipeline_name=pipeline_name,
        files_received=len(file_entries),
        message=f"Job {job_id} queued. Poll GET /v1/ingest/jobs/{job_id}.",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=IngestJobDetail,
    summary="Poll ingestion job status",
)
async def get_job_status(
    job_id: str,
    ctx: UserContext = Depends(require_user),
) -> IngestJobDetail:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.user_id != ctx.user_id:
        raise HTTPException(status_code=403, detail="Access denied.")
    return job
