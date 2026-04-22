"""
MetadataEnricher
================

A custom Haystack 2.x ``@component`` that normalises and enriches the
metadata on every ``Document`` coming out of the chunking stage.

Fields injected / normalised
-----------------------------
- chunk_index       Sequential index within the current batch.
- chunk_id          Stable content-addressed ID (SHA256 prefix).
- source_file       Normalised file path / URL.
- page_number       Extracted from Docling's ``dl_meta`` if present.
- headings          Breadcrumb list of section headings from Docling.
- bbox              Bounding box dict {left, top, right, bottom, page}.
- is_table          True when the chunk originates from a table.
- is_picture        True when the chunk originates from a picture / figure.
- picture_caption   Caption text produced by VLM enrichment (if any).
- origin            Docling origin dict (filename, page_no, …).
- word_count        Approximate word count of the chunk.
- char_count        Character count of the chunk.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from haystack import Document, component

    @component
    class MetadataEnricher:
        """
        Normalise and enrich metadata on a list of Haystack Documents.

    This component is idempotent — running a document through it twice
    produces the same metadata (chunk_id is content-addressed).

    Usage in a Haystack Pipeline::

        from utils.metadata_enricher import MetadataEnricher

        p.add_component("meta_enricher", MetadataEnricher())
        p.connect("chunker.documents", "meta_enricher.documents")
    """

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Enrich metadata for each document in the list.

        Parameters
        ----------
        documents:
            List of Haystack Document objects (post-chunking).

        Returns
        -------
        dict with key "documents" containing the enriched list.
        """
        enriched: list[Document] = []

        for idx, doc in enumerate(documents):
            try:
                enriched_doc = self._enrich(doc, idx)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MetadataEnricher: failed to enrich doc id=%s: %s. "
                    "Passing through unchanged.",
                    doc.id,
                    exc,
                )
                enriched_doc = doc

            enriched.append(enriched_doc)

        logger.debug("MetadataEnricher: enriched %d documents.", len(enriched))
        return {"documents": enriched}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enrich(self, doc: Document, idx: int) -> Document:
        meta = dict(doc.meta or {})

        # Chunk index in the current batch.
        meta["chunk_index"] = idx

        # Stable content-addressed chunk ID.
        content_bytes = (doc.content or "").encode("utf-8")
        meta["chunk_id"] = hashlib.sha256(content_bytes).hexdigest()[:16]

        # Source file / URL normalisation.
        meta.setdefault("source_file", self._extract_source(meta))

        # Page number — Docling stores this as page_no inside dl_meta.
        meta.setdefault("page_number", self._extract_page_number(meta))

        # Docling-specific metadata extraction.
        dl_meta: dict[str, Any] = meta.get("dl_meta") or {}
        if dl_meta:
            self._extract_docling_fields(meta, dl_meta)

        # Chunk type flags.
        meta.setdefault("is_table", False)
        meta.setdefault("is_picture", False)
        meta.setdefault("picture_caption", None)

        # Text statistics.
        content = doc.content or ""
        meta["char_count"] = len(content)
        meta["word_count"] = len(content.split())

        return Document(content=doc.content, meta=meta, id=doc.id)

    @staticmethod
    def _extract_source(meta: dict[str, Any]) -> str:
        """Pull the source identifier from various Docling/Haystack metadata keys."""
        for key in ("file_path", "source", "origin", "url"):
            val = meta.get(key)
            if val:
                if isinstance(val, dict):
                    return val.get("filename", str(val))
                return str(val)
        return "unknown"

    @staticmethod
    def _extract_page_number(meta: dict[str, Any]) -> int | None:
        """Extract page number from various possible metadata keys."""
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
                return origin.get("page_no")
        return None

    @staticmethod
    def _extract_docling_fields(meta: dict[str, Any], dl_meta: dict[str, Any]) -> None:
        """
        Extract and normalise fields from Docling's ``dl_meta`` structure.

        Docling populates ``dl_meta`` with:
        - headings:  list of heading strings (section breadcrumb)
        - origin:    dict with filename, page_no, bounding box
        - doc_items: list of DocItem objects or dicts
        """
        # Section heading breadcrumb.
        headings = dl_meta.get("headings") or []
        if isinstance(headings, list):
            meta["headings"] = headings
        else:
            meta["headings"] = []

        # Origin / provenance.
        origin = dl_meta.get("origin") or {}
        if origin:
            meta["origin"] = origin
            if "page_no" in origin:
                meta["page_number"] = origin["page_no"]
            if "filename" in origin and meta.get("source_file") == "unknown":
                meta["source_file"] = origin["filename"]

        # Bounding box.
        bbox = dl_meta.get("bbox") or origin.get("bbox")
        if bbox:
            meta["bbox"] = bbox

        # Doc item types — detect tables and pictures.
        doc_items = dl_meta.get("doc_items") or []
        for item in doc_items:
            if isinstance(item, dict):
                label = item.get("label", "").lower()
            else:
                label = getattr(item, "label", "").lower()

            if "table" in label:
                meta["is_table"] = True
            if "picture" in label or "figure" in label:
                meta["is_picture"] = True
                # Docling may have attached a VLM-generated caption.
                caption = None
                if isinstance(item, dict):
                    caption = item.get("caption") or item.get("text")
                else:
                    caption = getattr(item, "caption", None) or getattr(
                        item, "text", None
                    )
                if caption:
                    meta["picture_caption"] = caption

except ImportError:
    # Haystack not installed — stub so the module can be imported without error.
    class MetadataEnricher:  # type: ignore[no-redef]
        """Stub used when haystack-ai is not installed."""

        def run(self, documents: list) -> dict:
            raise ImportError(
                "haystack-ai is required for MetadataEnricher. "
                "Run: pip install haystack-ai"
            )
