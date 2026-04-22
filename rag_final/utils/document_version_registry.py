"""
DocumentVersionRegistry
========================

Thread-safe, in-process registry that tracks every (user_id, source_name)
→ [VersionRecord] mapping.

Responsibilities
----------------
1. Assign monotonically increasing version numbers per user+source.
2. Detect unchanged content via SHA-256 hash so re-uploads of identical
   files don't create a new version (idempotent ingest).
3. Maintain ``is_latest = True`` on exactly one version per source per user
   and flip older versions to ``is_latest = False`` in the document store.
4. Allow querying historical versions and deleting specific ones.
5. Persist its state by re-reading metadata from the document store on
   startup (``rebuild_from_store``), so a server restart doesn't lose
   version history.

Data model
----------
Every chunk in the document store carries these metadata fields (set here
or in MetadataEnricher):

    user_id       : str   — owner of this document
    source_name   : str   — normalised filename / URL (no path prefix)
    source_hash   : str   — SHA-256 of the raw file bytes
    version       : int   — 1-based version counter per (user, source)
    is_latest     : bool  — True only on the most recent version
    ingested_at   : str   — ISO 8601 UTC timestamp
    version_note  : str   — optional human note supplied at ingest time

Concurrency
-----------
A ``threading.Lock`` serialises all writes.  The registry is safe to use
from multiple FastAPI worker threads within the same process.  For
multi-process / multi-pod deployments, migrate ``_versions`` to Redis or
a lightweight SQL table.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VersionRecord:
    """Metadata about one ingested version of a source document."""

    user_id: str
    source_name: str        # normalised: basename, no leading path
    source_hash: str        # SHA-256 hex of raw file bytes
    version: int            # 1-based, monotonically increasing per (user, source)
    is_latest: bool
    ingested_at: str        # ISO 8601 UTC
    chunk_ids: list[str] = field(default_factory=list)   # IDs of all chunks
    document_count: int = 0
    version_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "source_name": self.source_name,
            "source_hash": self.source_hash,
            "version": self.version,
            "is_latest": self.is_latest,
            "ingested_at": self.ingested_at,
            "chunk_ids": self.chunk_ids,
            "document_count": self.document_count,
            "version_note": self.version_note,
        }


@dataclass
class VersionCheckResult:
    """Outcome of ``check_and_prepare`` — tells the caller what to do next."""

    is_duplicate: bool         # identical hash already ingested as latest
    is_new_source: bool        # first time this (user, source) is seen
    version: int               # version number to assign (even if duplicate)
    previous_version: int | None   # previous latest, or None
    source_hash: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class DocumentVersionRegistry:
    """
    Singleton (use module-level ``version_registry``) that tracks versions
    of every document per user.
    """

    def __init__(self) -> None:
        # _versions[(user_id, source_name)] = sorted list of VersionRecord (asc)
        self._versions: dict[tuple[str, str], list[VersionRecord]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def check_and_prepare(
        self,
        user_id: str,
        source_name: str,
        file_bytes: bytes,
        version_note: str = "",
    ) -> VersionCheckResult:
        """
        Call this BEFORE ingesting a file.

        Computes the file hash, checks for duplicates, and determines the
        version number to assign.  Does NOT write any record yet — call
        ``commit_version`` after the pipeline run succeeds.

        Parameters
        ----------
        user_id:       Owner of this document.
        source_name:   Normalised filename (``Path(p).name``).
        file_bytes:    Raw bytes of the file being ingested.
        version_note:  Optional human-readable note for this version.

        Returns
        -------
        VersionCheckResult with all the information needed to decide
        whether to proceed with ingestion.
        """
        source_hash = hashlib.sha256(file_bytes).hexdigest()
        key = (user_id, source_name)

        with self._lock:
            records = self._versions.get(key, [])
            is_new_source = len(records) == 0

            # Check if this exact content is already the latest version.
            if records:
                latest = records[-1]
                if latest.source_hash == source_hash and latest.is_latest:
                    logger.info(
                        "Version check: duplicate detected for user=%s source=%s "
                        "(hash=%s…, version=%d) — skipping re-ingest.",
                        user_id,
                        source_name,
                        source_hash[:8],
                        latest.version,
                    )
                    return VersionCheckResult(
                        is_duplicate=True,
                        is_new_source=False,
                        version=latest.version,
                        previous_version=latest.version - 1 if latest.version > 1 else None,
                        source_hash=source_hash,
                    )

            next_version = (records[-1].version + 1) if records else 1
            prev_version = records[-1].version if records else None

            logger.info(
                "Version check: user=%s source=%s → version %d "
                "(prev=%s, hash=%s…)",
                user_id,
                source_name,
                next_version,
                prev_version,
                source_hash[:8],
            )

            return VersionCheckResult(
                is_duplicate=False,
                is_new_source=is_new_source,
                version=next_version,
                previous_version=prev_version,
                source_hash=source_hash,
            )

    def commit_version(
        self,
        user_id: str,
        source_name: str,
        source_hash: str,
        version: int,
        chunk_ids: list[str],
        document_count: int,
        version_note: str = "",
    ) -> VersionRecord:
        """
        Record a successfully ingested version.

        Marks all previous versions of this source as ``is_latest = False``
        and inserts the new record as ``is_latest = True``.

        Call this only after the Haystack pipeline run has finished without error.
        """
        key = (user_id, source_name)
        now = datetime.now(tz=timezone.utc).isoformat()

        record = VersionRecord(
            user_id=user_id,
            source_name=source_name,
            source_hash=source_hash,
            version=version,
            is_latest=True,
            ingested_at=now,
            chunk_ids=chunk_ids,
            document_count=document_count,
            version_note=version_note,
        )

        with self._lock:
            records = self._versions.get(key, [])

            # Retire all previous versions in the registry.
            for rec in records:
                rec.is_latest = False

            records.append(record)
            self._versions[key] = records

        logger.info(
            "Version committed: user=%s source=%s version=%d "
            "chunks=%d doc_count=%d",
            user_id,
            source_name,
            version,
            len(chunk_ids),
            document_count,
        )
        return record

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def list_sources(self, user_id: str) -> list[VersionRecord]:
        """
        Return the latest version record for each source owned by user_id.
        """
        with self._lock:
            result = []
            for (uid, _sname), records in self._versions.items():
                if uid == user_id and records:
                    result.append(records[-1])  # last is always latest
            return sorted(result, key=lambda r: r.ingested_at, reverse=True)

    def list_versions(
        self, user_id: str, source_name: str
    ) -> list[VersionRecord]:
        """
        Return all versions of a source for a user (oldest first).
        Returns empty list if source is unknown.
        """
        key = (user_id, source_name)
        with self._lock:
            return list(self._versions.get(key, []))

    def get_version(
        self, user_id: str, source_name: str, version: int
    ) -> VersionRecord | None:
        """Return a specific version record, or None if not found."""
        key = (user_id, source_name)
        with self._lock:
            for rec in self._versions.get(key, []):
                if rec.version == version:
                    return rec
        return None

    def get_latest(self, user_id: str, source_name: str) -> VersionRecord | None:
        """Return the latest version record for a source, or None."""
        key = (user_id, source_name)
        with self._lock:
            records = self._versions.get(key, [])
            return records[-1] if records else None

    def delete_version(
        self, user_id: str, source_name: str, version: int
    ) -> list[str]:
        """
        Remove a specific version record from the registry.

        Returns the ``chunk_ids`` that the caller should delete from the
        document store.  If the deleted version was ``is_latest``, the
        previous version (if any) is promoted to ``is_latest = True``.
        """
        key = (user_id, source_name)
        with self._lock:
            records = self._versions.get(key, [])
            target = next((r for r in records if r.version == version), None)
            if target is None:
                return []

            chunk_ids = list(target.chunk_ids)
            records.remove(target)

            # Promote previous if we deleted the latest.
            if target.is_latest and records:
                records[-1].is_latest = True

            if records:
                self._versions[key] = records
            else:
                del self._versions[key]

        logger.info(
            "Version deleted: user=%s source=%s version=%d chunks=%d",
            user_id,
            source_name,
            version,
            len(chunk_ids),
        )
        return chunk_ids

    def delete_all_versions(self, user_id: str, source_name: str) -> list[str]:
        """
        Remove all versions of a source.  Returns all chunk_ids to delete.
        """
        key = (user_id, source_name)
        with self._lock:
            records = self._versions.pop(key, [])

        chunk_ids = [cid for rec in records for cid in rec.chunk_ids]
        logger.info(
            "All versions deleted: user=%s source=%s versions=%d chunks=%d",
            user_id,
            source_name,
            len(records),
            len(chunk_ids),
        )
        return chunk_ids

    def source_count(self, user_id: str) -> int:
        """Number of distinct sources owned by user_id."""
        with self._lock:
            return sum(1 for (uid, _) in self._versions if uid == user_id)

    def version_count(self, user_id: str, source_name: str) -> int:
        """Total number of stored versions for (user, source)."""
        key = (user_id, source_name)
        with self._lock:
            return len(self._versions.get(key, []))

    # ------------------------------------------------------------------
    # Store synchronisation
    # ------------------------------------------------------------------

    def rebuild_from_store(self, document_store: Any) -> int:
        """
        Reconstruct version history by scanning metadata of all documents
        in the store.  Call this once at startup after the store is ready.

        Returns the number of (user, source) pairs restored.
        """
        logger.info("VersionRegistry: rebuilding from document store …")

        try:
            all_docs = document_store.filter_documents(filters={})
        except Exception as exc:
            logger.warning(
                "VersionRegistry: could not read from store during rebuild: %s", exc
            )
            return 0

        # Group by (user_id, source_name, version).
        groups: dict[tuple[str, str, int], list[Any]] = {}
        for doc in all_docs:
            meta = doc.meta or {}
            uid = meta.get("user_id")
            sname = meta.get("source_name")
            ver = meta.get("version")
            if not (uid and sname and ver is not None):
                continue
            key = (uid, sname, int(ver))
            groups.setdefault(key, []).append(doc)

        restored = 0
        with self._lock:
            for (uid, sname, ver), docs in sorted(groups.items()):
                rkey = (uid, sname)
                records = self._versions.setdefault(rkey, [])
                # Skip if already present (e.g. from a concurrent ingest).
                if any(r.version == ver for r in records):
                    continue

                sample_meta = docs[0].meta or {}
                record = VersionRecord(
                    user_id=uid,
                    source_name=sname,
                    source_hash=sample_meta.get("source_hash", ""),
                    version=ver,
                    is_latest=sample_meta.get("is_latest", False),
                    ingested_at=sample_meta.get("ingested_at", ""),
                    chunk_ids=[d.id for d in docs if d.id],
                    document_count=len(docs),
                    version_note=sample_meta.get("version_note", ""),
                )
                records.append(record)
                restored += 1

            # Re-sort each source's records and enforce is_latest.
            for rkey, records in self._versions.items():
                records.sort(key=lambda r: r.version)
                for r in records:
                    r.is_latest = False
                if records:
                    records[-1].is_latest = True

        logger.info(
            "VersionRegistry: restored %d version record(s) from store.", restored
        )
        return restored

    def update_latest_flag_in_store(
        self,
        document_store: Any,
        user_id: str,
        source_name: str,
        new_latest_version: int,
    ) -> int:
        """
        Flip ``is_latest`` in the document store for all chunks belonging to
        (user_id, source_name):
          - ``is_latest = True``  for chunks where version == new_latest_version
          - ``is_latest = False`` for all other versions

        Returns the number of documents updated.

        Note: Not all store backends support in-place metadata updates.
        InMemoryDocumentStore does.  For Qdrant/OpenSearch/PgVector, use the
        store's native update API via a backend-specific helper.
        """
        try:
            all_docs = document_store.filter_documents(
                filters={
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.user_id", "operator": "==", "value": user_id},
                        {
                            "field": "meta.source_name",
                            "operator": "==",
                            "value": source_name,
                        },
                    ],
                }
            )
        except Exception as exc:
            logger.warning(
                "update_latest_flag_in_store: filter_documents failed: %s", exc
            )
            return 0

        updated = 0
        docs_to_write = []
        for doc in all_docs:
            meta = dict(doc.meta or {})
            expected = meta.get("version") == new_latest_version
            if meta.get("is_latest") != expected:
                meta["is_latest"] = expected
                try:
                    from haystack import Document

                    docs_to_write.append(
                        Document(content=doc.content, meta=meta, id=doc.id)
                    )
                    updated += 1
                except ImportError:
                    pass

        if docs_to_write:
            try:
                document_store.write_documents(
                    docs_to_write, policy="overwrite"
                )
            except Exception as exc:
                logger.warning(
                    "update_latest_flag_in_store: write_documents failed: %s", exc
                )

        return updated


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
version_registry = DocumentVersionRegistry()
