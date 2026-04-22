"""
Pydantic v2 schemas for the /v1/query and /v1/search endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MetadataFilter(BaseModel):
    """Haystack-compatible metadata filter expression."""

    field: str
    operator: str = Field(
        description="==, !=, >, >=, <, <=, in, not in, AND, OR, NOT"
    )
    value: Any


# ---------------------------------------------------------------------------
# Query request/response
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    pipeline_name: str = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional Haystack metadata filter dict passed to the retriever.",
    )
    search_type: str | None = Field(
        default=None,
        description="Override config search_type: embedding | bm25 | hybrid",
    )
    version: int | None = Field(
        default=None,
        description=(
            "Query a specific historical version of documents. "
            "Omit to query the latest version only (default)."
        ),
    )
    include_sources: bool = Field(default=True)
    stream: bool = Field(
        default=False,
        description="Stream the answer token-by-token via Server-Sent Events.",
    )


class SourceDocument(BaseModel):
    """A single retrieved source document included with a query answer."""

    document_id: str
    content: str
    score: float | None = None
    source_file: str | None = None
    page_number: int | None = None
    headings: list[str] = Field(default_factory=list)
    is_table: bool = False
    is_picture: bool = False
    picture_caption: str | None = None
    chunk_index: int | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    pipeline_name: str
    sources: list[SourceDocument] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Search (retrieval-only) request/response
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    pipeline_name: str = Field(default="default")
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] | None = None
    search_type: str = Field(
        default="embedding",
        description="embedding | bm25 | hybrid",
    )


class SearchResponse(BaseModel):
    query: str
    results: list[SourceDocument]
    total: int
    search_type: str
