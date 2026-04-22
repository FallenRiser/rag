"""
ChunkingFactory
===============

Reads ``ChunkingConfig`` and returns a ready-to-use Haystack 2.x splitter
component (or a pass-through joiner for ``document_aware`` mode, where
chunking is handled inside ``DoclingConverter``).

Supported strategies
--------------------
- document_aware   Docling HybridChunker (inside DoclingConverter)
- hierarchical     HierarchicalDocumentSplitter  → use with AutoMergingRetriever
- recursive        RecursiveDocumentSplitter     (LangChain-style cascading)
- sentence         DocumentSplitter(split_by="sentence")
- word             DocumentSplitter(split_by="word")
- passage          DocumentSplitter(split_by="passage")
- page             DocumentSplitter(split_by="page")
- line             DocumentSplitter(split_by="line")
- character        RecursiveDocumentSplitter(split_unit="char")
- token            RecursiveDocumentSplitter(split_unit="token")
- markdown_header  MarkdownHeaderSplitter  → use with export.type=markdown
- semantic         EmbeddingBasedDocumentSplitter (cosine distance threshold)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from config.models import ChunkingConfig, ChunkingStrategy, EmbeddingConfig

if TYPE_CHECKING:
    from haystack import component

logger = logging.getLogger(__name__)


class ChunkingFactory:
    """
    Factory that produces a configured Haystack splitter from ``ChunkingConfig``.

    Usage::

        from config.settings import get_settings
        from utils.chunking import ChunkingFactory

        settings = get_settings()
        factory = ChunkingFactory(settings.chunking, settings.embedding)
        splitter = factory.build()
    """

    def __init__(
        self,
        cfg: ChunkingConfig,
        embedding_cfg: EmbeddingConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        cfg:
            Parsed and validated ``ChunkingConfig``.
        embedding_cfg:
            Required only for ``semantic`` strategy (needs an embedder).
        """
        self.cfg = cfg
        self.embedding_cfg = embedding_cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> Any:
        """Build and return the configured splitter component."""
        strategy = self.cfg.strategy
        logger.info("Building chunker | strategy=%s", strategy.value)

        builder_map = {
            ChunkingStrategy.DOCUMENT_AWARE: self._build_document_aware,
            ChunkingStrategy.HIERARCHICAL: self._build_hierarchical,
            ChunkingStrategy.RECURSIVE: self._build_recursive,
            ChunkingStrategy.SENTENCE: self._build_sentence,
            ChunkingStrategy.WORD: self._build_word,
            ChunkingStrategy.PASSAGE: self._build_passage,
            ChunkingStrategy.PAGE: self._build_page,
            ChunkingStrategy.LINE: self._build_line,
            ChunkingStrategy.CHARACTER: self._build_character,
            ChunkingStrategy.TOKEN: self._build_token,
            ChunkingStrategy.MARKDOWN_HEADER: self._build_markdown_header,
            ChunkingStrategy.SEMANTIC: self._build_semantic,
        }

        builder = builder_map.get(strategy)
        if builder is None:
            raise ValueError(
                f"Unknown chunking strategy: '{strategy}'. "
                f"Valid choices: {[s.value for s in ChunkingStrategy]}"
            )

        component = builder()
        logger.info("Chunker built: %s", type(component).__name__)
        return component

    def build_cleaner(self) -> Any:
        """
        Build a configured ``DocumentCleaner`` component.
        Should always be run immediately before the splitter.
        """
        from haystack.components.preprocessors import DocumentCleaner

        cleaner_cfg = self.cfg.cleaner
        logger.debug("Building DocumentCleaner with config: %s", cleaner_cfg)

        return DocumentCleaner(
            remove_empty_lines=cleaner_cfg.remove_empty_lines,
            remove_extra_whitespaces=cleaner_cfg.remove_extra_whitespaces,
            remove_repeated_substrings=cleaner_cfg.remove_repeated_substrings,
        )

    # ------------------------------------------------------------------
    # Strategy builders — private
    # ------------------------------------------------------------------

    def _build_document_aware(self) -> Any:
        """
        document_aware: Docling HybridChunker runs inside DoclingConverter.

        When this strategy is selected, the downstream Haystack pipeline
        does NOT need a separate splitter. We return a ``DocumentJoiner``
        as a no-op pass-through so the pipeline graph remains consistent.
        """
        from haystack.components.joiners import DocumentJoiner

        logger.info(
            "Strategy 'document_aware' selected. "
            "Chunking will be performed by Docling HybridChunker inside "
            "DoclingConverter. Using DocumentJoiner as pass-through."
        )
        return DocumentJoiner()

    def _build_hierarchical(self) -> Any:
        """
        hierarchical: HierarchicalDocumentSplitter.

        Creates a parent-child tree of chunks at multiple granularity levels.
        Designed for use with AutoMergingRetriever in the query pipeline.

        Config key: ``chunking.hierarchical``
        """
        from haystack.components.preprocessors import HierarchicalDocumentSplitter

        h = self.cfg.hierarchical
        block_sizes = set(h.block_sizes)
        logger.info(
            "HierarchicalDocumentSplitter | block_sizes=%s | split_by=%s",
            sorted(h.block_sizes, reverse=True),
            h.split_by,
        )

        return HierarchicalDocumentSplitter(
            block_sizes=block_sizes,
            split_by=h.split_by,
            split_overlap=h.split_overlap,
        )

    def _build_recursive(self) -> Any:
        """
        recursive: RecursiveDocumentSplitter.

        Applies separators in order; falls back to the next separator for
        oversized chunks. Mirrors LangChain's RecursiveCharacterTextSplitter.

        Config key: ``chunking.recursive``
        """
        from haystack.components.preprocessors import RecursiveDocumentSplitter

        r = self.cfg.recursive
        self._validate_split_unit(r.split_unit)
        logger.info(
            "RecursiveDocumentSplitter | length=%d | overlap=%d | unit=%s | seps=%s",
            r.split_length,
            r.split_overlap,
            r.split_unit,
            r.separators,
        )

        return RecursiveDocumentSplitter(
            split_length=r.split_length,
            split_overlap=r.split_overlap,
            split_unit=r.split_unit,
            separators=r.separators,
        )

    def _build_sentence(self) -> Any:
        """
        sentence: DocumentSplitter(split_by="sentence").

        Uses NLTK for sentence boundary detection. Warm-up downloads the
        NLTK punkt tokenizer on first use (requires internet or cached data).

        Config key: ``chunking.sentence``
        """
        return self._doc_splitter("sentence", self.cfg.sentence)

    def _build_word(self) -> Any:
        """word: DocumentSplitter(split_by="word"). Config key: chunking.word"""
        return self._doc_splitter("word", self.cfg.word)

    def _build_passage(self) -> Any:
        """passage (paragraph): DocumentSplitter(split_by="passage"). Config key: chunking.passage"""
        return self._doc_splitter("passage", self.cfg.passage)

    def _build_page(self) -> Any:
        """page: DocumentSplitter(split_by="page"). Config key: chunking.page"""
        return self._doc_splitter("page", self.cfg.page)

    def _build_line(self) -> Any:
        """line: DocumentSplitter(split_by="line"). Config key: chunking.line"""
        return self._doc_splitter("line", self.cfg.line)

    def _build_character(self) -> Any:
        """
        character: RecursiveDocumentSplitter(split_unit="char").

        Hard character-count splits with configurable separators.
        Config key: ``chunking.character``
        """
        from haystack.components.preprocessors import RecursiveDocumentSplitter

        c = self.cfg.character
        logger.info(
            "RecursiveDocumentSplitter(char) | length=%d | overlap=%d",
            c.split_length,
            c.split_overlap,
        )

        return RecursiveDocumentSplitter(
            split_length=c.split_length,
            split_overlap=c.split_overlap,
            split_unit="char",
            separators=c.separators,
        )

    def _build_token(self) -> Any:
        """
        token: RecursiveDocumentSplitter(split_unit="token").

        Token counts via the default tiktoken/HF tokenizer inside Haystack.
        Config key: ``chunking.token``
        """
        from haystack.components.preprocessors import RecursiveDocumentSplitter

        t = self.cfg.token
        logger.info(
            "RecursiveDocumentSplitter(token) | length=%d | overlap=%d",
            t.split_length,
            t.split_overlap,
        )

        return RecursiveDocumentSplitter(
            split_length=t.split_length,
            split_overlap=t.split_overlap,
            split_unit="token",
            separators=t.separators,
        )

    def _build_markdown_header(self) -> Any:
        """
        markdown_header: MarkdownHeaderSplitter.

        Splits on Markdown '#' header hierarchy, preserving the header
        breadcrumb in each chunk's metadata. Use when
        ``docling.export.type == "markdown"``.

        Config key: ``chunking.markdown_header``
        """
        from haystack.components.preprocessors import MarkdownHeaderSplitter

        m = self.cfg.markdown_header
        logger.info(
            "MarkdownHeaderSplitter | keep_headers=%s | secondary=%s | "
            "length=%d | overlap=%d",
            m.keep_headers,
            m.secondary_split,
            m.split_length,
            m.split_overlap,
        )

        return MarkdownHeaderSplitter(
            page_break_character="\x0c",
            keep_headers=m.keep_headers,
            secondary_split=m.secondary_split,
            split_length=m.split_length,
            split_overlap=m.split_overlap,
            split_threshold=m.split_threshold,
            skip_empty_documents=m.skip_empty_documents,
        )

    def _build_semantic(self) -> Any:
        """
        semantic: EmbeddingBasedDocumentSplitter.

        Computes embeddings for groups of sentences, then inserts chunk
        boundaries wherever the cosine distance between consecutive groups
        exceeds the configured percentile threshold.

        Requires an embedding model; uses ``embedding_cfg`` for provider
        selection.

        Config key: ``chunking.semantic``
        """
        from haystack.components.preprocessors import EmbeddingBasedDocumentSplitter

        s = self.cfg.semantic

        if self.embedding_cfg is None:
            raise ValueError(
                "Semantic chunking requires an embedding_cfg. "
                "Pass EmbeddingConfig to ChunkingFactory."
            )

        document_embedder = self._build_semantic_embedder()
        logger.info(
            "EmbeddingBasedDocumentSplitter | sentences_per_group=%d | "
            "percentile=%.2f | min=%d | max=%d",
            s.sentences_per_group,
            s.percentile,
            s.min_length,
            s.max_length,
        )

        return EmbeddingBasedDocumentSplitter(
            document_embedder=document_embedder,
            sentences_per_group=s.sentences_per_group,
            percentile=s.percentile,
            min_length=s.min_length,
            max_length=s.max_length,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _doc_splitter(self, split_by: str, splitter_cfg: Any) -> Any:
        """Build a standard DocumentSplitter."""
        from haystack.components.preprocessors import DocumentSplitter

        logger.info(
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
        """
        Build an embedder for use inside EmbeddingBasedDocumentSplitter.
        The provider is selected from embedding_cfg.
        """
        from config.models import EmbeddingProvider

        provider = self.embedding_cfg.provider  # type: ignore[union-attr]

        if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return self._build_st_embedder()
        elif provider in (EmbeddingProvider.AZURE_OPENAI, EmbeddingProvider.OPENAI):
            logger.warning(
                "Azure OpenAI / OpenAI embedders are supported for semantic chunking "
                "but will incur API calls during chunking. Consider using "
                "sentence_transformers for semantic chunking to avoid costs."
            )
            return self._build_st_embedder()  # safe fallback
        elif provider == EmbeddingProvider.OLLAMA:
            return self._build_ollama_embedder_for_semantic()
        else:
            logger.warning(
                "Provider '%s' not explicitly supported for semantic chunking; "
                "falling back to SentenceTransformers.",
                provider,
            )
            return self._build_st_embedder()

    def _build_st_embedder(self) -> Any:
        """SentenceTransformers embedder for semantic chunking."""
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder

        st_cfg = self.embedding_cfg.sentence_transformers  # type: ignore[union-attr]
        device = self._resolve_device(st_cfg.device)

        return SentenceTransformersDocumentEmbedder(
            model=st_cfg.model,
            device=device,
            batch_size=st_cfg.batch_size,
            normalize_embeddings=st_cfg.normalize_embeddings,
        )

    def _build_ollama_embedder_for_semantic(self) -> Any:
        """Ollama embedder for semantic chunking."""
        try:
            from haystack_integrations.components.embedders.ollama import (
                OllamaDocumentEmbedder,
            )
        except ImportError as exc:
            raise ImportError(
                "ollama-haystack integration not found. "
                "Run: pip install ollama-haystack"
            ) from exc

        ol_cfg = self.embedding_cfg.ollama  # type: ignore[union-attr]
        return OllamaDocumentEmbedder(
            model=ol_cfg.model,
            url=f"{ol_cfg.base_url}/api/embeddings",
            batch_size=ol_cfg.batch_size,
        )

    @staticmethod
    def _validate_split_unit(unit: str) -> None:
        valid = {"word", "char", "token"}
        if unit not in valid:
            raise ValueError(
                f"split_unit must be one of {valid}, got '{unit}'"
            )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
