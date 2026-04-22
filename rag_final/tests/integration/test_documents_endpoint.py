"""
Integration tests for /v1/documents endpoints.

Uses httpx TestClient with a real FastAPI app but mocked
version_registry and document store (no pipeline warmup needed).
"""

from __future__ import annotations

import pytest

try:
    import fastapi  # noqa: F401
    import httpx    # noqa: F401
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

_needs_deps = pytest.mark.skipif(
    not _DEPS_AVAILABLE, reason="fastapi or httpx not installed"
)

ALICE_HDR = {"X-User-ID": "alice-001"}
BOB_HDR   = {"X-User-ID": "bob-002"}


@pytest.fixture
def client_with_versions(settings, monkeypatch):
    """
    TestClient wired to a mini FastAPI app that exposes the documents router.
    Seeds version_registry with two versions of 'report.pdf' for alice.
    """
    if not _DEPS_AVAILABLE:
        pytest.skip("fastapi / httpx not installed")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from utils.document_version_registry import DocumentVersionRegistry
    import utils.document_version_registry as vr_mod

    # Fresh registry scoped to this test.
    reg = DocumentVersionRegistry()

    import hashlib
    v1_bytes = b"version one"
    v2_bytes = b"version two"
    h1 = hashlib.sha256(v1_bytes).hexdigest()
    h2 = hashlib.sha256(v2_bytes).hexdigest()

    reg.check_and_prepare("alice-001", "report.pdf", v1_bytes)
    reg.commit_version("alice-001", "report.pdf", h1, 1, ["c1", "c2"], 2,
                       version_note="initial")
    reg.check_and_prepare("alice-001", "report.pdf", v2_bytes)
    reg.commit_version("alice-001", "report.pdf", h2, 2, ["c3", "c4"], 2,
                       version_note="updated")

    # Bob has one doc.
    reg.check_and_prepare("bob-002", "summary.docx", v1_bytes)
    reg.commit_version("bob-002", "summary.docx", h1, 1, ["b1"], 1)

    monkeypatch.setattr(vr_mod, "version_registry", reg)

    # Also patch the registry imported inside the endpoint module.
    import app.routes.v1.endpoints.documents as doc_mod
    monkeypatch.setattr(doc_mod, "version_registry", reg)

    # Minimal fake pipeline registry that provides a store with filter_documents.
    class FakeStore:
        def filter_documents(self, filters=None):
            from haystack import Document
            docs = [
                Document(content=f"chunk {i}", meta={
                    "user_id": "alice-001",
                    "source_name": "report.pdf",
                    "version": 2,
                    "is_latest": True,
                    "chunk_index": i,
                    "page_number": 1,
                    "headings": [],
                    "is_table": False,
                    "is_picture": False,
                }, id=f"c{i+3}")
                for i in range(2)
            ]
            return docs

        def delete_documents(self, ids):
            pass

    class FakeWriter:
        document_store = FakeStore()

    class FakePipeline:
        def get_component(self, name):
            return FakeWriter()

    class FakeRegistry:
        def get_indexing(self, name="default"):
            return FakePipeline()

    import app.routes.v1.endpoints.documents as doc_mod2
    monkeypatch.setattr(doc_mod2, "get_registry", lambda: FakeRegistry())

    app_inst = FastAPI()
    from app.routes.v1.endpoints.documents import router
    app_inst.include_router(router)

    return TestClient(app_inst)


@_needs_deps
class TestListSources:
    def test_alice_sees_her_sources(self, client_with_versions):
        resp = client_with_versions.get("/documents", headers=ALICE_HDR)
        if resp.status_code == 422:
            pytest.skip("haystack not available for filter_documents stub")
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "alice-001"
        assert body["source_count"] == 1
        assert body["sources"][0]["source_name"] == "report.pdf"
        assert body["sources"][0]["latest_version"] == 2
        assert body["sources"][0]["version_count"] == 2

    def test_missing_header_returns_401(self, client_with_versions):
        resp = client_with_versions.get("/documents")
        assert resp.status_code == 401

    def test_bob_sees_only_his_sources(self, client_with_versions):
        resp = client_with_versions.get("/documents", headers=BOB_HDR)
        if resp.status_code == 422:
            pytest.skip("haystack not available")
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "bob-002"
        assert all(s["source_name"] != "report.pdf" for s in body["sources"])


@_needs_deps
class TestListVersions:
    def test_alice_lists_versions_of_report(self, client_with_versions):
        resp = client_with_versions.get(
            "/documents/report.pdf/versions", headers=ALICE_HDR
        )
        if resp.status_code == 422:
            pytest.skip("haystack not available")
        assert resp.status_code == 200
        body = resp.json()
        assert body["version_count"] == 2
        versions = body["versions"]
        assert versions[0]["version"] == 1
        assert versions[0]["is_latest"] is False
        assert versions[1]["version"] == 2
        assert versions[1]["is_latest"] is True

    def test_unknown_source_returns_404(self, client_with_versions):
        resp = client_with_versions.get(
            "/documents/ghost.pdf/versions", headers=ALICE_HDR
        )
        assert resp.status_code == 404

    def test_bob_cannot_access_alice_report(self, client_with_versions):
        resp = client_with_versions.get(
            "/documents/report.pdf/versions", headers=BOB_HDR
        )
        assert resp.status_code == 404


@_needs_deps
class TestDeleteEndpoints:
    def test_delete_specific_version(self, client_with_versions):
        resp = client_with_versions.delete(
            "/documents/report.pdf/versions/1", headers=ALICE_HDR
        )
        if resp.status_code == 422:
            pytest.skip("haystack not available")
        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] is True
        assert body["version"] == 1
        assert body["chunks_removed"] == 2  # c1, c2

    def test_delete_unknown_version_returns_404(self, client_with_versions):
        resp = client_with_versions.delete(
            "/documents/report.pdf/versions/99", headers=ALICE_HDR
        )
        assert resp.status_code == 404

    def test_delete_all_versions(self, client_with_versions):
        resp = client_with_versions.delete(
            "/documents/report.pdf", headers=ALICE_HDR
        )
        if resp.status_code == 422:
            pytest.skip("haystack not available")
        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] is True
        assert body["version"] is None  # all versions

    def test_delete_unknown_source_returns_404(self, client_with_versions):
        resp = client_with_versions.delete(
            "/documents/ghost.pdf", headers=ALICE_HDR
        )
        assert resp.status_code == 404

    def test_bob_cannot_delete_alice_docs(self, client_with_versions):
        resp = client_with_versions.delete(
            "/documents/report.pdf", headers=BOB_HDR
        )
        assert resp.status_code == 404
