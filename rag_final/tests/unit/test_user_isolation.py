"""
Unit tests for user isolation and document versioning.

These tests cover:
- UserContext creation and filter generation
- UserIsolationFilter composition
- DocumentVersionRegistry: version assignment, dedup, commit, retire,
  delete (single version + all), rebuild from store stub
- All pure-Python logic; no haystack-ai or docling required.
"""

from __future__ import annotations

import hashlib
import threading
import time

import pytest

from utils.user_context import UserContext
from utils.user_isolation import (
    latest_filter,
    merge_with_user_filter,
    source_and_version_filter,
    user_filter,
    version_filter,
)
from utils.document_version_registry import DocumentVersionRegistry, VersionRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALICE = "user-alice-001"
BOB   = "user-bob-002"
DOC_A = "report.pdf"
DOC_B = "slides.pptx"

_CONTENT_V1 = b"This is version one content."
_CONTENT_V2 = b"This is version two content, updated."
_HASH_V1 = hashlib.sha256(_CONTENT_V1).hexdigest()
_HASH_V2 = hashlib.sha256(_CONTENT_V2).hexdigest()


@pytest.fixture
def registry() -> DocumentVersionRegistry:
    """Fresh registry instance per test."""
    return DocumentVersionRegistry()


@pytest.fixture
def alice() -> UserContext:
    return UserContext(user_id=ALICE)


@pytest.fixture
def bob() -> UserContext:
    return UserContext(user_id=BOB)


# ===========================================================================
# UserContext
# ===========================================================================


class TestUserContext:
    def test_frozen(self, alice):
        with pytest.raises((AttributeError, TypeError)):
            alice.user_id = "hacked"  # type: ignore[misc]

    def test_as_metadata_filter(self, alice):
        f = alice.as_metadata_filter()
        assert f["field"] == "meta.user_id"
        assert f["operator"] == "=="
        assert f["value"] == ALICE

    def test_latest_version_filter_structure(self, alice):
        f = alice.latest_version_filter()
        assert f["operator"] == "AND"
        fields = {c["field"] for c in f["conditions"]}
        assert "meta.user_id" in fields
        assert "meta.is_latest" in fields

    def test_version_filter_specific(self, alice):
        f = alice.version_filter(3)
        assert f["operator"] == "AND"
        ver_cond = next(c for c in f["conditions"] if c.get("field") == "meta.version")
        assert ver_cond["value"] == 3

    def test_source_filter_latest_only(self, alice):
        f = alice.source_filter(DOC_A, latest_only=True)
        fields = {c["field"] for c in f["conditions"]}
        assert "meta.source_name" in fields
        assert "meta.is_latest" in fields

    def test_source_filter_all_versions(self, alice):
        f = alice.source_filter(DOC_A, latest_only=False)
        fields = {c["field"] for c in f["conditions"]}
        assert "meta.source_name" in fields
        assert "meta.is_latest" not in fields

    def test_anonymous_user(self):
        ctx = UserContext(user_id="anonymous", is_anonymous=True)
        assert ctx.is_anonymous is True
        assert ctx.user_id == "anonymous"


# ===========================================================================
# UserIsolationFilter helpers
# ===========================================================================


class TestUserIsolationFilter:
    def test_user_filter_shape(self, alice):
        f = user_filter(alice)
        assert f == {"field": "meta.user_id", "operator": "==", "value": ALICE}

    def test_latest_filter_requires_is_latest(self, alice):
        f = latest_filter(alice)
        conditions = f["conditions"]
        has_latest = any(
            c.get("field") == "meta.is_latest" and c.get("value") is True
            for c in conditions
        )
        assert has_latest, "latest_filter must include is_latest==True condition"

    def test_version_filter_pins_version(self, alice):
        f = version_filter(alice, 5)
        ver_cond = next(
            (c for c in f["conditions"] if c.get("field") == "meta.version"), None
        )
        assert ver_cond is not None
        assert ver_cond["value"] == 5

    def test_source_and_version_filter_all_fields(self, alice):
        f = source_and_version_filter(alice, DOC_A, version=2)
        fields = {c["field"] for c in f["conditions"]}
        assert "meta.user_id" in fields
        assert "meta.source_name" in fields
        assert "meta.version" in fields

    def test_source_and_version_filter_latest_when_no_version(self, alice):
        f = source_and_version_filter(alice, DOC_A)
        fields = {c["field"] for c in f["conditions"]}
        assert "meta.is_latest" in fields
        assert "meta.version" not in fields

    def test_merge_no_caller_filter_returns_latest(self, alice):
        f = merge_with_user_filter(alice, caller_filters=None, latest_only=True)
        # Should be the same as latest_filter
        assert f["operator"] == "AND"
        fields = {c["field"] for c in f["conditions"]}
        assert "meta.is_latest" in fields

    def test_merge_with_caller_filter_wraps_both(self, alice):
        caller = {"field": "meta.source_name", "operator": "==", "value": DOC_A}
        f = merge_with_user_filter(alice, caller_filters=caller, latest_only=True)
        # Top-level AND wrapping isolation + caller
        assert f["operator"] == "AND"
        assert len(f["conditions"]) == 2

    def test_merge_without_latest_excludes_is_latest(self, alice):
        f = merge_with_user_filter(alice, caller_filters=None, latest_only=False)
        # Flatten all fields
        def _fields(node):
            if "field" in node:
                return {node["field"]}
            return {field for c in node.get("conditions", []) for field in _fields(c)}

        assert "meta.is_latest" not in _fields(f)

    def test_user_isolation_is_mandatory(self, alice):
        """Even with caller_filters=None, user_id must always appear in the filter."""
        f = merge_with_user_filter(alice, caller_filters=None)

        def _has_user_id(node) -> bool:
            if node.get("field") == "meta.user_id":
                return True
            return any(_has_user_id(c) for c in node.get("conditions", []))

        assert _has_user_id(f), "user_id filter must always be present"


# ===========================================================================
# DocumentVersionRegistry
# ===========================================================================


class TestVersionRegistryCheckAndPrepare:
    def test_first_upload_is_new_source(self, registry):
        result = registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V1)
        assert result.is_new_source is True
        assert result.is_duplicate is False
        assert result.version == 1
        assert result.previous_version is None

    def test_second_upload_different_content(self, registry):
        registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V1)
        registry.commit_version(ALICE, DOC_A, _HASH_V1, 1, [], 1)

        result = registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V2)
        assert result.is_duplicate is False
        assert result.version == 2
        assert result.previous_version == 1

    def test_identical_content_is_duplicate(self, registry):
        registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V1)
        registry.commit_version(ALICE, DOC_A, _HASH_V1, 1, [], 1)

        result = registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V1)
        assert result.is_duplicate is True
        assert result.version == 1

    def test_different_users_same_doc_independent(self, registry):
        """Alice and Bob's versions of the same filename are independent."""
        r_alice = registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V1)
        r_bob   = registry.check_and_prepare(BOB,   DOC_A, _CONTENT_V1)

        assert r_alice.version == 1
        assert r_bob.version == 1
        # Neither sees the other as a duplicate.
        assert r_alice.is_duplicate is False
        assert r_bob.is_duplicate is False

    def test_hash_is_sha256(self, registry):
        result = registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V1)
        assert result.source_hash == _HASH_V1


class TestVersionRegistryCommit:
    def test_commit_marks_is_latest_true(self, registry):
        registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V1)
        rec = registry.commit_version(ALICE, DOC_A, _HASH_V1, 1, ["c1", "c2"], 2)
        assert rec.is_latest is True
        assert rec.version == 1
        assert rec.chunk_ids == ["c1", "c2"]

    def test_commit_v2_retires_v1(self, registry):
        registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V1)
        registry.commit_version(ALICE, DOC_A, _HASH_V1, 1, ["c1"], 1)

        registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V2)
        registry.commit_version(ALICE, DOC_A, _HASH_V2, 2, ["c2"], 1)

        v1 = registry.get_version(ALICE, DOC_A, 1)
        v2 = registry.get_version(ALICE, DOC_A, 2)

        assert v1.is_latest is False
        assert v2.is_latest is True

    def test_commit_records_version_note(self, registry):
        registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V1)
        rec = registry.commit_version(
            ALICE, DOC_A, _HASH_V1, 1, [], 1, version_note="Initial draft"
        )
        assert rec.version_note == "Initial draft"

    def test_multiple_sources_independent(self, registry):
        registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V1)
        registry.commit_version(ALICE, DOC_A, _HASH_V1, 1, ["a1"], 1)

        registry.check_and_prepare(ALICE, DOC_B, _CONTENT_V1)
        registry.commit_version(ALICE, DOC_B, _HASH_V1, 1, ["b1"], 1)

        assert registry.version_count(ALICE, DOC_A) == 1
        assert registry.version_count(ALICE, DOC_B) == 1


class TestVersionRegistryRead:
    def _seed(self, registry):
        """Seed two versions of DOC_A for ALICE."""
        registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V1)
        registry.commit_version(ALICE, DOC_A, _HASH_V1, 1, ["c1"], 1)
        registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V2)
        registry.commit_version(ALICE, DOC_A, _HASH_V2, 2, ["c2"], 1)

    def test_list_sources_returns_latest_per_source(self, registry):
        self._seed(registry)
        sources = registry.list_sources(ALICE)
        assert len(sources) == 1
        assert sources[0].version == 2
        assert sources[0].is_latest is True

    def test_list_sources_excludes_other_user(self, registry):
        self._seed(registry)
        registry.check_and_prepare(BOB, DOC_B, _CONTENT_V1)
        registry.commit_version(BOB, DOC_B, _HASH_V1, 1, [], 1)

        alice_sources = registry.list_sources(ALICE)
        assert all(s.source_name == DOC_A for s in alice_sources)

    def test_list_versions_returns_all_ascending(self, registry):
        self._seed(registry)
        versions = registry.list_versions(ALICE, DOC_A)
        assert len(versions) == 2
        assert versions[0].version == 1
        assert versions[1].version == 2

    def test_list_versions_unknown_source_returns_empty(self, registry):
        assert registry.list_versions(ALICE, "nonexistent.pdf") == []

    def test_get_version_exact(self, registry):
        self._seed(registry)
        rec = registry.get_version(ALICE, DOC_A, 1)
        assert rec is not None
        assert rec.version == 1
        assert rec.source_hash == _HASH_V1

    def test_get_version_not_found_returns_none(self, registry):
        self._seed(registry)
        assert registry.get_version(ALICE, DOC_A, 99) is None

    def test_get_latest(self, registry):
        self._seed(registry)
        latest = registry.get_latest(ALICE, DOC_A)
        assert latest.version == 2

    def test_source_count(self, registry):
        self._seed(registry)
        registry.check_and_prepare(ALICE, DOC_B, _CONTENT_V1)
        registry.commit_version(ALICE, DOC_B, _HASH_V1, 1, [], 1)
        assert registry.source_count(ALICE) == 2

    def test_version_count(self, registry):
        self._seed(registry)
        assert registry.version_count(ALICE, DOC_A) == 2


class TestVersionRegistryDelete:
    def _seed(self, registry):
        registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V1)
        registry.commit_version(ALICE, DOC_A, _HASH_V1, 1, ["c1", "c2"], 2)
        registry.check_and_prepare(ALICE, DOC_A, _CONTENT_V2)
        registry.commit_version(ALICE, DOC_A, _HASH_V2, 2, ["c3", "c4"], 2)

    def test_delete_version_returns_chunk_ids(self, registry):
        self._seed(registry)
        ids = registry.delete_version(ALICE, DOC_A, 1)
        assert sorted(ids) == ["c1", "c2"]

    def test_delete_latest_promotes_previous(self, registry):
        self._seed(registry)
        registry.delete_version(ALICE, DOC_A, 2)

        v1 = registry.get_version(ALICE, DOC_A, 1)
        assert v1 is not None
        assert v1.is_latest is True

    def test_delete_old_version_leaves_latest_unchanged(self, registry):
        self._seed(registry)
        registry.delete_version(ALICE, DOC_A, 1)

        v2 = registry.get_version(ALICE, DOC_A, 2)
        assert v2.is_latest is True

    def test_delete_unknown_version_returns_empty_list(self, registry):
        self._seed(registry)
        ids = registry.delete_version(ALICE, DOC_A, 99)
        assert ids == []

    def test_delete_all_versions_returns_all_chunk_ids(self, registry):
        self._seed(registry)
        ids = registry.delete_all_versions(ALICE, DOC_A)
        assert sorted(ids) == ["c1", "c2", "c3", "c4"]

    def test_delete_all_removes_source_from_registry(self, registry):
        self._seed(registry)
        registry.delete_all_versions(ALICE, DOC_A)
        assert registry.list_versions(ALICE, DOC_A) == []
        assert registry.source_count(ALICE) == 0

    def test_delete_all_unknown_source_returns_empty(self, registry):
        ids = registry.delete_all_versions(ALICE, "ghost.pdf")
        assert ids == []

    def test_delete_all_only_affects_target_user(self, registry):
        self._seed(registry)
        registry.check_and_prepare(BOB, DOC_A, _CONTENT_V1)
        registry.commit_version(BOB, DOC_A, _HASH_V1, 1, ["bob_c1"], 1)

        registry.delete_all_versions(ALICE, DOC_A)

        # Bob's version is untouched
        bob_versions = registry.list_versions(BOB, DOC_A)
        assert len(bob_versions) == 1


class TestVersionRegistryThreadSafety:
    def test_concurrent_commits_no_data_race(self):
        """Hammer the registry with concurrent writes from multiple threads."""
        reg = DocumentVersionRegistry()
        errors: list[Exception] = []

        def worker(user_id: str, doc: str, content: bytes):
            try:
                check = reg.check_and_prepare(user_id, doc, content)
                if not check.is_duplicate:
                    reg.commit_version(
                        user_id, doc, check.source_hash,
                        check.version, [], 1
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(
                target=worker,
                args=(f"user-{i}", "shared.pdf", f"content {i}".encode()),
            )
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"


class TestVersionRegistryRebuild:
    def test_rebuild_from_store_stub(self, registry):
        """Rebuild from a stub store that returns documents with version metadata."""

        class StubDoc:
            def __init__(self, uid, sname, ver, is_latest, hash_):
                self.id = f"{uid}-{sname}-{ver}"
                self.content = "content"
                self.meta = {
                    "user_id": uid,
                    "source_name": sname,
                    "version": ver,
                    "is_latest": is_latest,
                    "source_hash": hash_,
                    "ingested_at": "2024-01-01T00:00:00+00:00",
                    "version_note": "",
                }

        class StubStore:
            def filter_documents(self, filters):
                return [
                    StubDoc(ALICE, DOC_A, 1, False, _HASH_V1),
                    StubDoc(ALICE, DOC_A, 2, True,  _HASH_V2),
                    StubDoc(BOB,   DOC_B, 1, True,  _HASH_V1),
                ]

        restored = registry.rebuild_from_store(StubStore())
        assert restored == 3

        # Alice has 2 versions of DOC_A
        assert registry.version_count(ALICE, DOC_A) == 2
        # Latest is v2
        assert registry.get_latest(ALICE, DOC_A).version == 2
        assert registry.get_latest(ALICE, DOC_A).is_latest is True
        # v1 is retired
        assert registry.get_version(ALICE, DOC_A, 1).is_latest is False

        # Bob has 1 version
        assert registry.version_count(BOB, DOC_B) == 1

    def test_rebuild_tolerates_store_error(self, registry):
        """rebuild_from_store must not raise when the store throws."""

        class BrokenStore:
            def filter_documents(self, filters):
                raise RuntimeError("store unavailable")

        # Should log a warning and return 0, not raise.
        restored = registry.rebuild_from_store(BrokenStore())
        assert restored == 0

    def test_rebuild_skips_docs_without_version_meta(self, registry):
        """Documents missing user_id/source_name/version are silently skipped."""

        class PartialStore:
            def filter_documents(self, filters):
                class D:
                    id = "x"
                    content = "c"
                    meta = {}   # missing required fields
                return [D()]

        restored = registry.rebuild_from_store(PartialStore())
        assert restored == 0


class TestVersionRecordSerialization:
    def test_to_dict_contains_all_fields(self):
        rec = VersionRecord(
            user_id=ALICE,
            source_name=DOC_A,
            source_hash=_HASH_V1,
            version=1,
            is_latest=True,
            ingested_at="2024-01-01T00:00:00+00:00",
            chunk_ids=["c1", "c2"],
            document_count=2,
            version_note="first upload",
        )
        d = rec.to_dict()
        assert d["user_id"] == ALICE
        assert d["version"] == 1
        assert d["is_latest"] is True
        assert d["chunk_ids"] == ["c1", "c2"]
        assert d["version_note"] == "first upload"
