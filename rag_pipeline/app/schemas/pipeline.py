"""Schemas for /v1/pipelines and /v1/health endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ComponentStatus(BaseModel):
    name: str
    type: str
    status: str   # "ok" | "degraded" | "unavailable"
    details: dict[str, Any] = {}


class PipelineInfo(BaseModel):
    name: str
    indexing_built: bool
    query_built: bool
    chunking_strategy: str
    embedding_provider: str
    document_store_backend: str
    export_type: str


class HealthResponse(BaseModel):
    status: str   # "ok" | "degraded" | "unavailable"
    env: str
    components: dict[str, ComponentStatus] = {}


class ReadyResponse(BaseModel):
    ready: bool
    message: str
