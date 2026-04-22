"""
UserIsolationFilter
===================

Helper that composes Haystack metadata filters so every retrieval
operation is scoped to a specific user (and optionally a specific version).

Haystack filter grammar (v2)
-----------------------------
Atomic:   {"field": "meta.user_id", "operator": "==", "value": "alice"}
Compound: {"operator": "AND", "conditions": [...]}

Operators: == != > >= < <= in not in AND OR NOT

All document metadata is stored under the ``meta`` namespace in Haystack,
so filters use ``meta.<field_name>`` as the field path.
"""

from __future__ import annotations

from typing import Any

from utils.user_context import UserContext


# ---------------------------------------------------------------------------
# Core filter builder
# ---------------------------------------------------------------------------


def user_filter(ctx: UserContext) -> dict[str, Any]:
    """
    Returns a bare user_id equality filter.

    Use this when you want ALL versions (e.g. listing historical chunks).
    """
    return {"field": "meta.user_id", "operator": "==", "value": ctx.user_id}


def latest_filter(ctx: UserContext) -> dict[str, Any]:
    """
    Returns user_id AND is_latest == True.

    Use this for normal query flows — users only see current versions.
    """
    return {
        "operator": "AND",
        "conditions": [
            user_filter(ctx),
            {"field": "meta.is_latest", "operator": "==", "value": True},
        ],
    }


def version_filter(ctx: UserContext, version: int) -> dict[str, Any]:
    """
    Returns user_id AND version == <version>.

    Use this when the user explicitly requests a historical version.
    """
    return {
        "operator": "AND",
        "conditions": [
            user_filter(ctx),
            {"field": "meta.version", "operator": "==", "value": version},
        ],
    }


def source_and_version_filter(
    ctx: UserContext,
    source_name: str,
    version: int | None = None,
    latest_only: bool = True,
) -> dict[str, Any]:
    """
    Scope retrieval to a specific source document (and optionally version).

    Parameters
    ----------
    ctx:          Calling user's context.
    source_name:  Normalised filename / URL identifier.
    version:      If set, restrict to this exact version number.
    latest_only:  If True (default), add ``is_latest == True`` constraint.
                  Ignored when ``version`` is set.
    """
    conditions: list[dict[str, Any]] = [
        user_filter(ctx),
        {"field": "meta.source_name", "operator": "==", "value": source_name},
    ]
    if version is not None:
        conditions.append(
            {"field": "meta.version", "operator": "==", "value": version}
        )
    elif latest_only:
        conditions.append(
            {"field": "meta.is_latest", "operator": "==", "value": True}
        )

    return {"operator": "AND", "conditions": conditions}


# ---------------------------------------------------------------------------
# Filter merger
# ---------------------------------------------------------------------------


def merge_with_user_filter(
    ctx: UserContext,
    caller_filters: dict[str, Any] | None,
    latest_only: bool = True,
) -> dict[str, Any]:
    """
    Merge a caller-supplied filter with the mandatory user-isolation filter.

    The user-isolation filter is always applied — even if the caller passes
    ``filters=None`` or an empty dict.  This makes it impossible for a
    misbehaving client to escape their own data silo.

    Parameters
    ----------
    ctx:            User context.
    caller_filters: Optional filter from the API request body.
    latest_only:    Include ``is_latest == True`` in the mandatory filter.

    Returns
    -------
    A single compound filter ready to pass to ``retriever.run(filters=...)``.
    """
    base = latest_filter(ctx) if latest_only else user_filter(ctx)

    if not caller_filters:
        return base

    # Wrap both in an AND.
    return {
        "operator": "AND",
        "conditions": [base, caller_filters],
    }
