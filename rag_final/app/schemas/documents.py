"""Pydantic v2 schemas for the /v1/documents endpoint family."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentSourceSummary(BaseModel):
    """Latest-version summary for one source document owned by the user."""

    source_name: str
    latest_version: int
    version_count: int
    source_hash: str
    document_count: int
    chunk_count: int
    ingested_at: str
    version_note: str = ""


class VersionDetail(BaseModel):
    """Full detail for one version of a source document."""

    source_name: str
    version: int
    is_latest: bool
    source_hash: str
    document_count: int
    chunk_count: int
    ingested_at: str
    version_note: str = ""


class ChunkDetail(BaseModel):
    """A single retrieved chunk from a versioned document."""

    document_id: str
    content: str
    chunk_index: int | None = None
    page_number: int | None = None
    headings: list[str] = Field(default_factory=list)
    is_table: bool = False
    is_picture: bool = False
    picture_caption: str | None = None
    version: int | None = None
    is_latest: bool = True


class ListSourcesResponse(BaseModel):
    user_id: str
    source_count: int
    sources: list[DocumentSourceSummary]


class ListVersionsResponse(BaseModel):
    user_id: str
    source_name: str
    version_count: int
    versions: list[VersionDetail]


class ListChunksResponse(BaseModel):
    user_id: str
    source_name: str
    version: int
    is_latest: bool
    chunk_count: int
    chunks: list[ChunkDetail]
