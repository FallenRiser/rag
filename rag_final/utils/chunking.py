"""
ChunkingFactory
===============

Returns a configured Haystack 2.x splitter from ``ChunkingConfig``.

All 12 strategies are verified against Haystack 2.26:
  word / passage / page / line     → DocumentSplitter
  sentence                         → DocumentSplitter (needs NLTK punkt_tab)
  hierarchical                     → HierarchicalDocumentSplitter
  recursive / character            → RecursiveDocumentSplitter (no NLTK needed
                                     when 'sentence' omitted from separators)
  token                            → RecursiveDocumentSplitter (needs NLTK)
  markdown_header                  → MarkdownHeaderSplitter
  semantic                         → EmbeddingBasedDocumentSplitter
  document_aware                   → chunking done inside DoclingConverter;
                                     IndexingPipelineBuilder skips splitter
"""

from __future__ import annotations

import logging
from typing import Any

from config.models import ChunkingConfig, ChunkingStrategy, EmbeddingConfig

logger = logging.getLogger(__name__)


class ChunkingFactory:
    """
    Build a Haystack splitter from ``ChunkingConfig``.

    Usage::

        factory = ChunkingFactory(settings.chunking, settings.embedding)
        splitter = factory.build()           # None for document_aware
        cleaner  = factory.build_cleaner()
    """

    def __init__(
        self,
        cfg: ChunkingConfig,
        embedding_cfg: EmbeddingConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.embedding_cfg = embedding_cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> Any | None:
        """
        Return the splitter for the configured strategy.

        Returns ``None`` for ``document_aware`` — callers must skip adding
        a chunker component in that case; chunking is done by DoclingConverter.
        """
        strategy = self.cfg.strategy
        logger.info("Building chunker | strategy=%s", strategy.value)

        builder_map: dict[ChunkingStrategy, Any] = {
            ChunkingStrategy.DOCUMENT_AWARE: self._build_document_aware,
            ChunkingStrategy.HIERARCHICAL:   self._build_hierarchical,
            ChunkingStrategy.RECURSIVE:      self._build_recursive,
            ChunkingStrategy.CHARACTER:      self._build_character,
            ChunkingStrategy.TOKEN:          self._build_token,
            ChunkingStrategy.SENTENCE:       self._build_sentence,
            ChunkingStrategy.WORD:           self._build_word,
            ChunkingStrategy.PASSAGE:        self._build_passage,
            ChunkingStrategy.PAGE:           self._build_page,
            ChunkingStrategy.LINE:           self._build_line,
            ChunkingStrategy.MARKDOWN_HEADER: self._build_markdown_header,
            ChunkingStrategy.SEMANTIC:       self._build_semantic,
        }

        builder = builder_map.get(strategy)
        if builder is None:
            raise ValueError(
                f"Unknown chunking strategy: '{strategy}'. "
                f"Valid: {[s.value for s in ChunkingStrategy]}"
            )

        component = builder()
        if component is None:
            logger.info(
                "Strategy 'document_aware': chunking is handled inside "
                "DoclingConverter — no downstream splitter will be added."
            )
        else:
            logger.info("Chunker built: %s", type(component).__name__)
        return component

    def build_cleaner(self) -> Any:
        """Return a configured ``DocumentCleaner``."""
        from haystack.components.preprocessors import DocumentCleaner

        c = self.cfg.cleaner
        return DocumentCleaner(
            remove_empty_lines=c.remove_empty_lines,
            remove_extra_whitespaces=c.remove_extra_whitespaces,
            remove_repeated_substrings=c.remove_repeated_substrings,
        )

    # ------------------------------------------------------------------
    # Strategy builders
    # ------------------------------------------------------------------

    def _build_document_aware(self) -> None:
        """
        document_aware: HybridChunker runs INSIDE DoclingConverter.
        Return None so IndexingPipelineBuilder skips the splitter component.
        """
        return None

    def _build_hierarchical(self) -> Any:
        """
        HierarchicalDocumentSplitter — parent/child block tree.
        block_sizes must be in descending order.
        Output key: ``documents`` (contains all levels).
        Pair with AutoMergingRetriever at query time.
        """
        from haystack.components.preprocessors import HierarchicalDocumentSplitter

        h = self.cfg.hierarchical
        return HierarchicalDocumentSplitter(
            block_sizes=set(h.block_sizes),
            split_by=h.split_by,
            split_overlap=h.split_overlap,
        )

    def _build_recursive(self) -> Any:
        """
        RecursiveDocumentSplitter — cascading separator strategy.
        NOTE: Do NOT include "sentence" in separators unless NLTK is installed.
              Use ['\n\n', '\n', ' '] for a safe default.
        """
        from haystack.components.preprocessors import RecursiveDocumentSplitter

        r = self.cfg.recursive
        return RecursiveDocumentSplitter(
            split_length=r.split_length,
            split_overlap=r.split_overlap,
            split_unit=r.split_unit,
            separators=r.separators,
        )

    def _build_character(self) -> Any:
        """RecursiveDocumentSplitter with split_unit='char'."""
        from haystack.components.preprocessors import RecursiveDocumentSplitter

        c = self.cfg.character
        return RecursiveDocumentSplitter(
            split_length=c.split_length,
            split_overlap=c.split_overlap,
            split_unit="char",
            separators=c.separators,
        )

    def _build_token(self) -> Any:
        """
        RecursiveDocumentSplitter with split_unit='token'.
        Requires NLTK punkt_tab: python -m nltk.downloader punkt_tab
        """
        from haystack.components.preprocessors import RecursiveDocumentSplitter

        t = self.cfg.token
        return RecursiveDocumentSplitter(
            split_length=t.split_length,
            split_overlap=t.split_overlap,
            split_unit="token",
            separators=t.separators,
        )

    def _build_sentence(self) -> Any:
        """
        DocumentSplitter(split_by='sentence') using NLTK sentence tokenizer.
        Requires NLTK punkt_tab: python -m nltk.downloader punkt_tab
        """
        return self._doc_splitter("sentence", self.cfg.sentence)

    def _build_word(self) -> Any:
        return self._doc_splitter("word", self.cfg.word)

    def _build_passage(self) -> Any:
        return self._doc_splitter("passage", self.cfg.passage)

    def _build_page(self) -> Any:
        return self._doc_splitter("page", self.cfg.page)

    def _build_line(self) -> Any:
        return self._doc_splitter("line", self.cfg.line)

    def _build_markdown_header(self) -> Any:
        """
        MarkdownHeaderSplitter — splits on #/## heading hierarchy.
        Param notes from Haystack 2.26:
          page_break_character — default \\f (form feed)
          header_split_levels  — list of heading levels to split on, or None (all)
          secondary_split      — word | passage | period | line | None
        """
        from haystack.components.preprocessors import MarkdownHeaderSplitter

        m = self.cfg.markdown_header
        kwargs: dict[str, Any] = dict(
            keep_headers=m.keep_headers,
            split_length=m.split_length,
            split_overlap=m.split_overlap,
            split_threshold=m.split_threshold,
            skip_empty_documents=m.skip_empty_documents,
        )
        if m.secondary_split is not None:
            kwargs["secondary_split"] = m.secondary_split
        # header_split_levels is optional; omit when None to use all levels
        if hasattr(m, "header_split_levels") and m.header_split_levels is not None:
            kwargs["header_split_levels"] = m.header_split_levels

        return MarkdownHeaderSplitter(**kwargs)

    def _build_semantic(self) -> Any:
        """
        EmbeddingBasedDocumentSplitter — splits on cosine-distance threshold.
        Uses the provider in embedding_cfg (falls back to SentenceTransformers).
        """
        if self.embedding_cfg is None:
            raise ValueError(
                "Semantic chunking requires embedding_cfg. "
                "Pass EmbeddingConfig to ChunkingFactory."
            )

        from haystack.components.preprocessors import EmbeddingBasedDocumentSplitter

        s = self.cfg.semantic
        embedder = self._build_semantic_embedder()

        return EmbeddingBasedDocumentSplitter(
            document_embedder=embedder,
            sentences_per_group=s.sentences_per_group,
            percentile=s.percentile,
            min_length=s.min_length,
            max_length=s.max_length,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _doc_splitter(self, split_by: str, splitter_cfg: Any) -> Any:
        from haystack.components.preprocessors import DocumentSplitter

        logger.debug(
            "DocumentSplitter | split_by=%s | length=%d | overlap=%d | threshold=%d",
            split_by,
            splitter_cfg.split_length,
            splitter_cfg.split_overlap,
            splitter_cfg.split_threshold,
        )
        return DocumentSplitter(
            split_by=split_by,
            split_length=splitter_cfg.split_length,
            split_overlap=splitter_cfg.split_overlap,
            split_threshold=splitter_cfg.split_threshold,
        )

    def _build_semantic_embedder(self) -> Any:
        from config.models import EmbeddingProvider

        provider = self.embedding_cfg.provider  # type: ignore[union-attr]

        if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            from haystack.components.embedders import SentenceTransformersDocumentEmbedder

            st = self.embedding_cfg.sentence_transformers  # type: ignore[union-attr]
            return SentenceTransformersDocumentEmbedder(
                model=st.model,
                batch_size=st.batch_size,
                normalize_embeddings=st.normalize_embeddings,
            )
        elif provider == EmbeddingProvider.OLLAMA:
            try:
                from haystack_integrations.components.embedders.ollama import (
                    OllamaDocumentEmbedder,
                )
            except ImportError as exc:
                raise ImportError(
                    "ollama-haystack not installed. Run: pip install ollama-haystack"
                ) from exc
            ol = self.embedding_cfg.ollama  # type: ignore[union-attr]
            return OllamaDocumentEmbedder(
                model=ol.model,
                url=f"{ol.base_url.rstrip('/')}/api/embeddings",
                batch_size=ol.batch_size,
            )
        else:
            # For API providers (Azure, OpenAI), fall back to a local model to
            # avoid API costs during chunking.
            logger.warning(
                "Using SentenceTransformers for semantic chunking boundary detection "
                "(provider '%s' would incur API costs per chunk). "
                "Set embedding.provider=sentence_transformers to silence this warning.",
                provider.value,
            )
            from haystack.components.embedders import SentenceTransformersDocumentEmbedder

            return SentenceTransformersDocumentEmbedder(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
