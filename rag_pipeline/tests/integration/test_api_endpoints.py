"""
Integration tests for FastAPI endpoints.

Uses httpx.AsyncClient with ASGITransport (no live server needed).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest


def _requires_httpx():
    try:
        import httpx  # noqa: F401
    except ImportError:
        pytest.skip("httpx not installed")


def _requires_haystack():
    try:
        import haystack  # noqa: F401
    except ImportError:
        pytest.skip("haystack-ai not installed")


@pytest.fixture
async def client(settings, monkeypatch):
    """Async test client with app wired to test config."""
    _requires_httpx()
    _requires_haystack()

    import httpx

    # Patch get_settings to return our test settings.
    from config import settings as settings_module

    settings_module.get_settings.cache_clear()
    monkeypatch.setattr(settings_module, "get_settings", lambda **_: settings)

    from app.main import create_app

    test_app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app), base_url="http://test"
    ) as c:
        yield c
    settings_module.get_settings.cache_clear()


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "env" in body

    @pytest.mark.asyncio
    async def test_ready_returns_200_after_startup(self, client):
        resp = await client.get("/v1/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ready"] is True

    @pytest.mark.asyncio
    async def test_root_redirect(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "service" in body


class TestPipelinesEndpoints:
    @pytest.mark.asyncio
    async def test_list_pipelines(self, client):
        resp = await client.get("/v1/pipelines")
        assert resp.status_code == 200
        pipelines = resp.json()
        assert isinstance(pipelines, list)
        assert len(pipelines) >= 1

    @pytest.mark.asyncio
    async def test_get_default_pipeline(self, client):
        resp = await client.get("/v1/pipelines/default")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "default"
        assert "chunking_strategy" in body
        assert "embedding_provider" in body

    @pytest.mark.asyncio
    async def test_get_unknown_pipeline_returns_404(self, client):
        resp = await client.get("/v1/pipelines/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_reload_pipeline(self, client):
        resp = await client.post("/v1/pipelines/default/reload")
        assert resp.status_code == 200
        assert "evicted" in resp.json()["message"].lower()


class TestIngestEndpoints:
    @pytest.mark.asyncio
    async def test_ingest_empty_files_returns_422(self, client):
        resp = await client.post("/v1/ingest")
        # No files = 422 Unprocessable Entity
        assert resp.status_code in (422, 400)

    @pytest.mark.asyncio
    async def test_ingest_unknown_pipeline_returns_404(self, client):
        dummy = io.BytesIO(b"Hello world test content for ingest")
        resp = await client.post(
            "/v1/ingest",
            files={"files": ("test.txt", dummy, "text/plain")},
            data={"pipeline_name": "nonexistent_pipeline_xyz"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_ingest_returns_job_id(self, client):
        """Uploading a file should return a job_id immediately."""
        dummy = io.BytesIO(b"Sample document content for testing ingestion pipeline.")
        resp = await client.post(
            "/v1/ingest",
            files={"files": ("sample.txt", dummy, "text/plain")},
            data={"pipeline_name": "default"},
        )
        # 202 Accepted or 404 if docling isn't installed.
        if resp.status_code == 404:
            pytest.skip("Docling pipeline not available in test env.")
        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["files_received"] == 1

    @pytest.mark.asyncio
    async def test_get_job_status_unknown_id_returns_404(self, client):
        resp = await client.get("/v1/ingest/nonexistent-job-id-xyz")
        assert resp.status_code == 404
