"""
UserContext
===========

Per-request user identity carrier for document isolation and version scoping.

The ``X-User-ID`` header carries a pre-authenticated user identifier injected
by the upstream API gateway (Kong, AWS API GW, nginx+JWT, etc.) after it has
validated the token.

FastAPI dependencies
--------------------
``require_user``  — raises 401 when header is missing or blank.
``optional_user`` — falls back to "anonymous" when header absent (local dev).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ANONYMOUS_USER = "anonymous"


@dataclass(frozen=True)
class UserContext:
    """Immutable per-request identity carrier."""

    user_id: str
    is_anonymous: bool = False

    # ------------------------------------------------------------------
    # Haystack metadata filter helpers
    # ------------------------------------------------------------------

    def as_metadata_filter(self) -> dict[str, Any]:
        return {"field": "meta.user_id", "operator": "==", "value": self.user_id}

    def latest_version_filter(self) -> dict[str, Any]:
        return {
            "operator": "AND",
            "conditions": [
                self.as_metadata_filter(),
                {"field": "meta.is_latest", "operator": "==", "value": True},
            ],
        }

    def version_filter(self, version: int) -> dict[str, Any]:
        return {
            "operator": "AND",
            "conditions": [
                self.as_metadata_filter(),
                {"field": "meta.version", "operator": "==", "value": version},
            ],
        }

    def source_filter(self, source_name: str, latest_only: bool = True) -> dict[str, Any]:
        conditions: list[dict[str, Any]] = [
            self.as_metadata_filter(),
            {"field": "meta.source_name", "operator": "==", "value": source_name},
        ]
        if latest_only:
            conditions.append(
                {"field": "meta.is_latest", "operator": "==", "value": True}
            )
        return {"operator": "AND", "conditions": conditions}


# ---------------------------------------------------------------------------
# FastAPI dependencies — only defined when fastapi is importable
# ---------------------------------------------------------------------------

try:
    from fastapi import Header, HTTPException, status
    from typing import Annotated

    def require_user(
        x_user_id: Annotated[Optional[str], Header(alias="X-User-ID")] = None,
    ) -> UserContext:
        """
        FastAPI dependency that requires ``X-User-ID`` header.
        Raises 401 if missing or blank.
        """
        if not x_user_id or not x_user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Missing or empty 'X-User-ID' header. "
                    "Set it to your user identifier."
                ),
                headers={"WWW-Authenticate": "X-User-ID"},
            )
        return UserContext(user_id=x_user_id.strip())

    def optional_user(
        x_user_id: Annotated[Optional[str], Header(alias="X-User-ID")] = None,
    ) -> UserContext:
        """
        FastAPI dependency — returns anonymous context when header absent.
        """
        if x_user_id and x_user_id.strip():
            return UserContext(user_id=x_user_id.strip())
        return UserContext(user_id=_ANONYMOUS_USER, is_anonymous=True)

except ImportError:
    # FastAPI not installed — stubs for test environments that only import
    # UserContext itself (the dataclass).
    def require_user(*args, **kwargs) -> UserContext:  # type: ignore[misc]
        raise ImportError("fastapi is required for require_user dependency.")

    def optional_user(*args, **kwargs) -> UserContext:  # type: ignore[misc]
        return UserContext(user_id=_ANONYMOUS_USER, is_anonymous=True)
