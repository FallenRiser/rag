"""Pydantic v2 schemas for the /v1/ingest endpoint family."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    PARTIAL   = "partial"


class FileIngestResult(BaseModel):
    """Per-file result within an ingest job."""

    filename: str
    status: JobStatus
    documents_written: int = 0
    chunks_created: int = 0
    version: int | None = None           # version number assigned
    is_duplicate: bool = False           # True when identical content already latest
    error: str | None = None
    message: str = ""


class IngestResponse(BaseModel):
    """Immediate response from POST /v1/ingest."""

    job_id: str
    status: JobStatus
    pipeline_name: str
    files_received: int
    message: str

    class Config:
        use_enum_values = True


class IngestJobDetail(BaseModel):
    """Full status returned from GET /v1/ingest/jobs/{job_id}."""

    job_id: str
    status: JobStatus
    pipeline_name: str
    user_id: str
    created_at: datetime
    completed_at: datetime | None = None
    files_received: int
    files_processed: int = 0
    total_chunks: int = 0
    total_documents_written: int = 0
    results: list[FileIngestResult] = Field(default_factory=list)
    error: str | None = None

    class Config:
        use_enum_values = True


class DeleteDocumentResponse(BaseModel):
    source_name: str
    version: int | None = None   # None = all versions
    deleted: bool
    chunks_removed: int
    message: str
