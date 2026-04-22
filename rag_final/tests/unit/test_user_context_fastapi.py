"""
Tests for the FastAPI user context dependencies (require_user, optional_user).

Uses pytest-asyncio + httpx in-process transport so no live server needed.
All tests are dependency-injection unit tests — no Haystack pipeline involved.
"""

from __future__ import annotations

import pytest

try:
    import fastapi  # noqa: F401
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

_needs_fastapi = pytest.mark.skipif(
    not _FASTAPI_AVAILABLE, reason="fastapi not installed"
)


@pytest.fixture
def mini_app():
    """Minimal FastAPI app exposing both user dependencies for testing."""
    if not _FASTAPI_AVAILABLE:
        pytest.skip("fastapi not installed")

    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient
    from utils.user_context import UserContext, optional_user, require_user

    app = FastAPI()

    @app.get("/required")
    def required_endpoint(ctx: UserContext = Depends(require_user)):
        return {"user_id": ctx.user_id, "is_anonymous": ctx.is_anonymous}

    @app.get("/optional")
    def optional_endpoint(ctx: UserContext = Depends(optional_user)):
        return {"user_id": ctx.user_id, "is_anonymous": ctx.is_anonymous}

    return TestClient(app)


@_needs_fastapi
class TestRequireUser:
    def test_valid_header_returns_200(self, mini_app):
        resp = mini_app.get("/required", headers={"X-User-ID": "alice-123"})
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "alice-123"
        assert resp.json()["is_anonymous"] is False

    def test_missing_header_returns_401(self, mini_app):
        resp = mini_app.get("/required")
        assert resp.status_code == 401

    def test_blank_header_returns_401(self, mini_app):
        resp = mini_app.get("/required", headers={"X-User-ID": "   "})
        assert resp.status_code == 401

    def test_header_is_stripped(self, mini_app):
        resp = mini_app.get("/required", headers={"X-User-ID": "  alice  "})
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "alice"

    def test_different_users_get_their_own_id(self, mini_app):
        for uid in ["alice", "bob", "charlie"]:
            resp = mini_app.get("/required", headers={"X-User-ID": uid})
            assert resp.json()["user_id"] == uid


@_needs_fastapi
class TestOptionalUser:
    def test_with_header_returns_user(self, mini_app):
        resp = mini_app.get("/optional", headers={"X-User-ID": "alice"})
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "alice"
        assert resp.json()["is_anonymous"] is False

    def test_without_header_returns_anonymous(self, mini_app):
        resp = mini_app.get("/optional")
        assert resp.status_code == 200
        assert resp.json()["is_anonymous"] is True
        assert resp.json()["user_id"] == "anonymous"
