"""
IndexingPipelineBuilder
========================

Wires DoclingConverter → DocumentCleaner → [Chunker] → MetadataEnricher
→ DocumentEmbedder → DocumentWriter into a single Haystack Pipeline.

document_aware special case
----------------------------
When ``chunking.strategy = "document_aware"``, ``ChunkingFactory.build()``
returns ``None``. In that case:

  - DoclingConverter is built with HybridChunker embedded.
  - No cleaner or splitter component is added to the pipeline.
  - converter.documents connects directly to meta_enricher.documents.

All other strategies
--------------------
  converter → cleaner → chunker → meta_enricher → embedder → writer
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from config.models import ChunkingStrategy
from config.settings import AppSettings
from utils.chunking import ChunkingFactory
from utils.docling_pipeline import DoclingPipelineBuilder
from utils.embedding import EmbeddingFactory
from utils.metadata_enricher import MetadataEnricher

if TYPE_CHECKING:
    from haystack import Pipeline

logger = logging.getLogger(__name__)


class IndexingPipelineBuilder:
    """
    Build the Haystack ingestion pipeline from AppSettings.

    Usage::

        pipeline = IndexingPipelineBuilder(settings).build()
        result   = pipeline.run({
            "converter":    {"sources": ["report.pdf"]},
            "meta_enricher": {
                "user_id": "alice", "source_name": "report.pdf",
                "source_hash": "abc…", "version": 1,
                "is_latest": True, "ingested_at": "2024-01-01T00:00:00+00:00",
            },
        })
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def build(self) -> "Pipeline":
        from haystack import Pipeline

        cfg_chunk   = self.settings.chunking
        cfg_docling = self.settings.docling
        is_doc_aware = cfg_chunk.strategy == ChunkingStrategy.DOCUMENT_AWARE

        logger.info(
            "Building indexing pipeline | strategy=%s | export=%s | embedding=%s",
            cfg_chunk.strategy.value,
            cfg_docling.export.type.value,
            self.settings.embedding.provider.value,
        )

        pipeline = Pipeline()

        # ── 1. DoclingConverter ───────────────────────────────────────────
        if is_doc_aware:
            da = cfg_chunk.document_aware
            converter = DoclingPipelineBuilder(cfg_docling).build_with_hybrid_chunker(
                tokenizer=da.tokenizer,
                max_tokens=da.max_tokens,
                merge_peers=da.merge_peers,
            )
        else:
            converter = DoclingPipelineBuilder(cfg_docling).build()

        pipeline.add_component("converter", converter)

        # ── 2 & 3. DocumentCleaner + Chunker (skipped for document_aware) ──
        factory = ChunkingFactory(cfg_chunk, self.settings.embedding)
        chunker: Any = factory.build()   # None when document_aware

        if not is_doc_aware:
            cleaner = factory.build_cleaner()
            pipeline.add_component("cleaner", cleaner)

            pipeline.add_component("chunker", chunker)

            pipeline.connect("converter.documents", "cleaner.documents")
            pipeline.connect("cleaner.documents",   "chunker.documents")
            last_output = "chunker.documents"
        else:
            last_output = "converter.documents"

        # ── 4. MetadataEnricher ───────────────────────────────────────────
        pipeline.add_component("meta_enricher", MetadataEnricher())
        pipeline.connect(last_output, "meta_enricher.documents")

        # ── 5. DocumentEmbedder ───────────────────────────────────────────
        embedder = EmbeddingFactory(self.settings.embedding).build_document_embedder()
        pipeline.add_component("embedder", embedder)
        pipeline.connect("meta_enricher.documents", "embedder.documents")

        # ── 6. DocumentWriter ─────────────────────────────────────────────
        from utils.document_store import StoreFactory
        from haystack.components.writers import DocumentWriter

        store  = StoreFactory(self.settings.document_store).build()
        writer = DocumentWriter(document_store=store)
        pipeline.add_component("writer", writer)
        pipeline.connect("embedder.documents", "writer.documents")

        logger.info("Indexing pipeline built OK.")
        return pipeline
