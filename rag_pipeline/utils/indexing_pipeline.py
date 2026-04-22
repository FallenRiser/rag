"""
IndexingPipelineBuilder
========================

Wires together all extraction, chunking, embedding, and storage
components into a single Haystack 2.x ``Pipeline`` for document ingestion.

Full pipeline graph
-------------------
    [file paths / URLs]
           │
           ▼
    DoclingConverter           ← utils/docling_pipeline.py
    (extraction + enrichments)
           │
           ▼  List[Document]
    DocumentCleaner
           │
           ▼
    <Chunker>                  ← utils/chunking.py  (strategy from config)
           │
           ▼
    MetadataEnricher           ← utils/metadata_enricher.py
           │
           ▼
    DocumentEmbedder           ← utils/embedding.py  (provider from config)
           │
           ▼
    DocumentWriter             ← utils/document_store.py  (backend from config)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
    Builds and returns a Haystack ``Pipeline`` for document ingestion.

    Usage::

        from config.settings import get_settings
        from utils.indexing_pipeline import IndexingPipelineBuilder

        settings = get_settings()
        builder  = IndexingPipelineBuilder(settings)
        pipeline = builder.build()

        result = pipeline.run({
            "converter": {"sources": ["path/to/doc.pdf", "https://example.com/doc.html"]}
        })
        print(result["writer"]["documents_written"])
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def build(self) -> "Pipeline":
        from haystack import Pipeline

        cfg_chunk = self.settings.chunking
        cfg_docling = self.settings.docling
        is_document_aware = cfg_chunk.strategy == ChunkingStrategy.DOCUMENT_AWARE

        logger.info(
            "Building indexing pipeline | chunking=%s | export=%s | embedding=%s",
            cfg_chunk.strategy.value,
            cfg_docling.export.type.value,
            self.settings.embedding.provider.value,
        )

        pipeline = Pipeline()

        # ----------------------------------------------------------------
        # 1. DoclingConverter  —  extraction + optional enrichments
        # ----------------------------------------------------------------
        if is_document_aware:
            # Build with HybridChunker embedded; downstream splitter is a no-op.
            da_cfg = cfg_chunk.document_aware
            converter = DoclingPipelineBuilder(cfg_docling).build_with_hybrid_chunker(
                tokenizer=da_cfg.tokenizer,
                max_tokens=da_cfg.max_tokens,
                merge_peers=da_cfg.merge_peers,
            )
        else:
            converter = DoclingPipelineBuilder(cfg_docling).build()

        pipeline.add_component("converter", converter)

        # ----------------------------------------------------------------
        # 2. DocumentCleaner
        # ----------------------------------------------------------------
        cleaner = ChunkingFactory(cfg_chunk, self.settings.embedding).build_cleaner()
        pipeline.add_component("cleaner", cleaner)

        # ----------------------------------------------------------------
        # 3. Chunker
        # ----------------------------------------------------------------
        chunker = ChunkingFactory(cfg_chunk, self.settings.embedding).build()
        pipeline.add_component("chunker", chunker)

        # ----------------------------------------------------------------
        # 4. MetadataEnricher
        # ----------------------------------------------------------------
        pipeline.add_component("meta_enricher", MetadataEnricher())

        # ----------------------------------------------------------------
        # 5. DocumentEmbedder
        # ----------------------------------------------------------------
        embedder = EmbeddingFactory(self.settings.embedding).build_document_embedder()
        pipeline.add_component("embedder", embedder)

        # ----------------------------------------------------------------
        # 6. DocumentWriter
        # ----------------------------------------------------------------
        from utils.document_store import StoreFactory

        store = StoreFactory(self.settings.document_store).build()

        from haystack.components.writers import DocumentWriter

        writer = DocumentWriter(document_store=store)
        pipeline.add_component("writer", writer)

        # ----------------------------------------------------------------
        # Wire connections
        # ----------------------------------------------------------------
        pipeline.connect("converter.documents", "cleaner.documents")
        pipeline.connect("cleaner.documents", "chunker.documents")
        pipeline.connect("chunker.documents", "meta_enricher.documents")
        pipeline.connect("meta_enricher.documents", "embedder.documents")
        pipeline.connect("embedder.documents", "writer.documents")

        logger.info("Indexing pipeline built successfully.")
        return pipeline
