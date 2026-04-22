"""
Pydantic v2 schemas for the /v1/ingest endpoint family.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator


class JobStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    PARTIAL    = "partial"   # some files succeeded, some failed


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class UrlIngestRequest(BaseModel):
    """Ingest one or more URLs (web pages, remote PDFs, …)."""

    urls: list[HttpUrl] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of URLs to fetch and ingest.",
    )
    pipeline_name: str = Field(
        default="default",
        description="Named pipeline config to use (registered in PipelineRegistry).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata attached to every document produced from these URLs.",
    )

    @field_validator("urls")
    @classmethod
    def urls_not_empty(cls, v: list[HttpUrl]) -> list[HttpUrl]:
        if not v:
            raise ValueError("urls must contain at least one item.")
        return v


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class FileIngestResult(BaseModel):
    """Per-file result within an ingest job."""

    filename: str
    status: JobStatus
    documents_written: int = 0
    chunks_created: int = 0
    error: str | None = None


class IngestResponse(BaseModel):
    """Response returned immediately from POST /v1/ingest (async job)."""

    job_id: str
    status: JobStatus
    pipeline_name: str
    files_received: int
    message: str

    class Config:
        use_enum_values = True


class IngestJobDetail(BaseModel):
    """Full job status returned from GET /v1/ingest/{job_id}."""

    job_id: str
    status: JobStatus
    pipeline_name: str
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
    document_id: str
    deleted: bool
    message: str
