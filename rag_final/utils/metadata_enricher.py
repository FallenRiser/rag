"""
MetadataEnricher
================

Custom Haystack 2.x ``@component`` that stamps user-isolation and versioning
fields onto every chunk produced by the chunking stage.

Fields added to every chunk
---------------------------
  chunk_index      int    — position in current batch
  chunk_id         str    — SHA-256[:16] of content (stable, content-addressed)
  source_file      str    — normalised file path / URL
  source_name      str    — basename only (used as version key)
  page_number      int|None
  headings         list[str] — heading breadcrumb from Docling dl_meta
  bbox             dict|None — bounding box from Docling
  is_table         bool
  is_picture       bool
  picture_caption  str|None — caption from VLM enrichment
  char_count       int
  word_count       int

User-isolation fields (stamped per-file at ingest time)
-------------------------------------------------------
  user_id          str    — owner; drives retrieval isolation
  source_hash      str    — SHA-256 of raw file bytes (for dedup)
  version          int    — monotonically increasing per (user, source_name)
  is_latest        bool   — True only on the most recent version
  ingested_at      str    — ISO 8601 UTC
  version_note     str    — optional human note
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helper functions (no Haystack dependency — fully testable standalone)
# ---------------------------------------------------------------------------


def _extract_source(meta: dict[str, Any]) -> str:
    for key in ("file_path", "source", "url"):
        val = meta.get(key)
        if val:
            return str(val)
    origin = meta.get("origin")
    if isinstance(origin, dict) and origin.get("filename"):
        return origin["filename"]
    return "unknown"


def _extract_source_name(meta: dict[str, Any]) -> str:
    """Basename of the source file (used as version registry key)."""
    sf = meta.get("source_name") or meta.get("source_file") or _extract_source(meta)
    return sf.replace("\\", "/").split("/")[-1] or "unknown"


def _extract_page_number(meta: dict[str, Any]) -> int | None:
    for key in ("page_no", "page_number", "page"):
        val = meta.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    dl_meta = meta.get("dl_meta") or {}
    if isinstance(dl_meta, dict):
        origin = dl_meta.get("origin") or {}
        if isinstance(origin, dict):
            pn = origin.get("page_no")
            if pn is not None:
                return int(pn)
    return None


def _extract_docling_fields(meta: dict[str, Any], dl_meta: dict[str, Any]) -> None:
    """Populate headings, origin, bbox, is_table, is_picture, picture_caption."""
    headings = dl_meta.get("headings") or []
    meta["headings"] = headings if isinstance(headings, list) else []

    origin = dl_meta.get("origin") or {}
    if origin:
        meta.setdefault("origin", origin)
        if "page_no" in origin:
            meta.setdefault("page_number", int(origin["page_no"]))
        if "filename" in origin and meta.get("source_file") in (None, "unknown"):
            meta["source_file"] = origin["filename"]

    bbox = dl_meta.get("bbox") or (origin.get("bbox") if isinstance(origin, dict) else None)
    if bbox:
        meta["bbox"] = bbox

    for item in dl_meta.get("doc_items") or []:
        label = (
            item.get("label", "") if isinstance(item, dict) else getattr(item, "label", "")
        ).lower()

        if "table" in label:
            meta["is_table"] = True
        if "picture" in label or "figure" in label:
            meta["is_picture"] = True
            caption = (
                item.get("caption") or item.get("text")
                if isinstance(item, dict)
                else getattr(item, "caption", None) or getattr(item, "text", None)
            )
            if caption:
                meta["picture_caption"] = str(caption)


def enrich_document_meta(
    doc_meta: dict[str, Any],
    doc_content: str | None,
    idx: int,
    *,
    user_id: str | None = None,
    source_name: str | None = None,
    source_hash: str | None = None,
    version: int | None = None,
    is_latest: bool = True,
    ingested_at: str | None = None,
    version_note: str = "",
) -> dict[str, Any]:
    """
    Pure-function core of MetadataEnricher.
    Takes and returns a plain dict — no Haystack types required.
    """
    meta = dict(doc_meta or {})
    content = doc_content or ""

    # ── Chunk identity ────────────────────────────────────────────────────
    meta["chunk_index"] = idx
    meta["chunk_id"] = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    # ── Source provenance ─────────────────────────────────────────────────
    meta.setdefault("source_file", _extract_source(meta))
    meta.setdefault("source_name", source_name or _extract_source_name(meta))
    meta.setdefault("page_number", _extract_page_number(meta))

    # ── Docling structured metadata ───────────────────────────────────────
    dl_meta = meta.get("dl_meta") or {}
    if dl_meta:
        _extract_docling_fields(meta, dl_meta)

    # ── Content type flags (defaults) ─────────────────────────────────────
    meta.setdefault("headings", [])
    meta.setdefault("is_table", False)
    meta.setdefault("is_picture", False)
    meta.setdefault("picture_caption", None)

    # ── Text statistics ───────────────────────────────────────────────────
    meta["char_count"] = len(content)
    meta["word_count"] = len(content.split())

    # ── User isolation ────────────────────────────────────────────────────
    if user_id is not None:
        meta["user_id"] = user_id

    # ── Document versioning ───────────────────────────────────────────────
    if source_name is not None:
        meta["source_name"] = source_name
    if source_hash is not None:
        meta["source_hash"] = source_hash
    if version is not None:
        meta["version"] = version
    meta["is_latest"] = is_latest
    if ingested_at is not None:
        meta["ingested_at"] = ingested_at
    if version_note:
        meta["version_note"] = version_note

    return meta


# ---------------------------------------------------------------------------
# Haystack @component wrapper
# ---------------------------------------------------------------------------

try:
    from haystack import Document, component

    @component
    class MetadataEnricher:
        """
        Haystack component that enriches chunk metadata with user-isolation
        and versioning fields.

        All user/version params default to None so the component can be used
        without them (existing tests, non-user-scoped pipelines).
        Per-file values are injected at pipeline.run() time via the
        ``meta_enricher`` key in the input dict.
        """

        @component.output_types(documents=list[Document])
        def run(
            self,
            documents: list[Document],
            user_id: str | None = None,
            source_name: str | None = None,
            source_hash: str | None = None,
            version: int | None = None,
            is_latest: bool = True,
            ingested_at: str | None = None,
            version_note: str = "",
        ) -> dict[str, list[Document]]:
            enriched: list[Document] = []
            for idx, doc in enumerate(documents):
                try:
                    new_meta = enrich_document_meta(
                        doc.meta or {},
                        doc.content,
                        idx,
                        user_id=user_id,
                        source_name=source_name,
                        source_hash=source_hash,
                        version=version,
                        is_latest=is_latest,
                        ingested_at=ingested_at,
                        version_note=version_note,
                    )
                    enriched.append(
                        Document(content=doc.content, meta=new_meta, id=doc.id)
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "MetadataEnricher: failed on doc id=%s: %s — passing unchanged.",
                        doc.id,
                        exc,
                    )
                    enriched.append(doc)

            logger.debug("MetadataEnricher: enriched %d documents.", len(enriched))
            return {"documents": enriched}

except ImportError:
    # Stub so the module imports cleanly without haystack-ai installed.
    class MetadataEnricher:  # type: ignore[no-redef]
        def run(self, documents: list, **kwargs) -> dict:
            raise ImportError("haystack-ai is required. Run: pip install haystack-ai")
