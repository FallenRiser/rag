"""
Integration tests for the full extraction + chunking + indexing pipeline.

These tests exercise the real Haystack pipeline components end-to-end
using InMemoryDocumentStore and SentenceTransformers embeddings (no API keys
needed). They are skipped if the required packages are not installed.

Run with:
    pytest tests/integration/ -v
    pytest tests/integration/ -v -m "not slow"
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    content = textwrap.dedent("""
        Introduction to Machine Learning

        Machine learning is a subfield of artificial intelligence that gives systems
        the ability to automatically learn and improve from experience.

        Types of Machine Learning

        There are three main types: supervised learning, unsupervised learning, and
        reinforcement learning.

        Supervised learning involves training a model on labelled data. The algorithm
        learns a mapping from inputs to outputs.

        Unsupervised learning works with unlabelled data. The algorithm tries to find
        hidden patterns or intrinsic structures in the input data.

        Reinforcement learning involves an agent that learns by interacting with an
        environment and receiving rewards or penalties.

        Applications

        Machine learning is used in image recognition, natural language processing,
        recommendation systems, fraud detection, and many other domains.
    """).strip()

    f = tmp_path / "ml_intro.txt"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def markdown_file(tmp_path: Path) -> Path:
    content = textwrap.dedent("""
        # Deep Learning Overview

        Deep learning uses artificial neural networks with multiple layers.

        ## Convolutional Neural Networks

        CNNs are particularly effective for image recognition tasks. They use
        convolutional filters to detect spatial features.

        ## Recurrent Neural Networks

        RNNs process sequential data. LSTM and GRU variants address the vanishing
        gradient problem.

        ## Transformers

        The transformer architecture uses self-attention mechanisms and has become
        the dominant paradigm in NLP and increasingly in vision tasks.
    """).strip()

    f = tmp_path / "deep_learning.md"
    f.write_text(content, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _requires_haystack():
    try:
        import haystack  # noqa: F401
    except ImportError:
        pytest.skip("haystack-ai not installed")


def _requires_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pytest.skip("sentence-transformers not installed")


# ---------------------------------------------------------------------------
# ChunkingFactory integration tests (no Docling needed)
# ---------------------------------------------------------------------------


class TestChunkingPipelineIntegration:
    """Test chunking on pre-made Document objects (no Docling required)."""

    def test_word_strategy_full_run(self, settings, sample_documents):
        _requires_haystack()
        import yaml

        from config.models import ChunkingConfig
        from utils.chunking import ChunkingFactory
        from utils.metadata_enricher import MetadataEnricher

        # Override strategy to word for this test.
        cfg = ChunkingConfig.model_validate({
            **settings.chunking.model_dump(),
            "strategy": "word",
            "word": {"split_length": 20, "split_overlap": 3, "split_threshold": 1},
        })
        factory = ChunkingFactory(cfg)
        cleaner = factory.build_cleaner()
        splitter = factory.build()
        enricher = MetadataEnricher()

        cleaned = cleaner.run(documents=sample_documents)["documents"]
        chunks = splitter.run(documents=cleaned)["documents"]
        enriched = enricher.run(documents=chunks)["documents"]

        assert len(enriched) >= len(sample_documents)
        for doc in enriched:
            assert doc.meta["chunk_id"] is not None
            assert doc.meta["word_count"] >= 0

    def test_sentence_strategy_chunk_boundaries(self, settings, sample_documents):
        _requires_haystack()

        from config.models import ChunkingConfig
        from utils.chunking import ChunkingFactory

        cfg = ChunkingConfig.model_validate({
            **settings.chunking.model_dump(),
            "strategy": "sentence",
            "sentence": {"split_length": 2, "split_overlap": 0, "split_threshold": 0},
        })
        factory = ChunkingFactory(cfg)
        splitter = factory.build()

        # Use only text documents (sentence splitter requires text).
        text_docs = [d for d in sample_documents if d.content and len(d.content) > 50]
        if not text_docs:
            pytest.skip("No suitable text documents in fixture.")

        chunks = splitter.run(documents=text_docs)["documents"]
        assert len(chunks) >= len(text_docs)

    @pytest.mark.parametrize("strategy", [
        "word", "passage", "line", "recursive", "character",
    ])
    def test_all_non_model_strategies_run(self, settings, sample_documents, strategy):
        """All strategies that don't require a model should run without errors."""
        _requires_haystack()

        from config.models import ChunkingConfig
        from utils.chunking import ChunkingFactory

        cfg = ChunkingConfig.model_validate({
            **settings.chunking.model_dump(),
            "strategy": strategy,
        })
        factory = ChunkingFactory(cfg)
        splitter = factory.build()
        result = splitter.run(documents=sample_documents[:1])
        docs = result.get("documents", result.get("merged_documents", []))
        assert len(docs) >= 1

    def test_hierarchical_strategy_creates_parent_child(self, settings, sample_documents):
        """HierarchicalDocumentSplitter should add parent_id metadata to child chunks."""
        _requires_haystack()

        from config.models import ChunkingConfig
        from utils.chunking import ChunkingFactory

        cfg = ChunkingConfig.model_validate({
            **settings.chunking.model_dump(),
            "strategy": "hierarchical",
            "hierarchical": {
                "block_sizes": [80, 20, 5],
                "split_by": "word",
                "split_overlap": 0,
            },
        })
        factory = ChunkingFactory(cfg)
        splitter = factory.build()

        # Use the longer document.
        long_docs = [d for d in sample_documents if d.content and len(d.content) > 200]
        if not long_docs:
            pytest.skip("Need longer documents for hierarchical test.")

        result = splitter.run(documents=long_docs)
        # HierarchicalDocumentSplitter outputs root + child documents.
        all_docs = result.get("documents", [])
        assert len(all_docs) >= 2

    def test_markdown_header_strategy(self, markdown_file, settings):
        """MarkdownHeaderSplitter should produce one chunk per heading section."""
        _requires_haystack()

        from haystack import Document

        from config.models import ChunkingConfig
        from utils.chunking import ChunkingFactory

        md_content = markdown_file.read_text(encoding="utf-8")
        docs = [Document(content=md_content, meta={"source_file": "deep_learning.md"})]

        cfg = ChunkingConfig.model_validate({
            **settings.chunking.model_dump(),
            "strategy": "markdown_header",
            "markdown_header": {
                "keep_headers": True,
                "secondary_split": None,
                "split_length": 200,
                "split_overlap": 0,
                "split_threshold": 0,
                "skip_empty_documents": True,
            },
        })
        factory = ChunkingFactory(cfg)
        splitter = factory.build()
        chunks = splitter.run(documents=docs)["documents"]

        # Markdown has 3 ## headers + 1 # header = at least 3 sections.
        assert len(chunks) >= 3


# ---------------------------------------------------------------------------
# MetadataEnricher integration
# ---------------------------------------------------------------------------


class TestMetadataEnricherIntegration:
    def test_pipeline_connector_runs(self, settings, sample_documents):
        """Verify cleaner → splitter → enricher runs as a mini-pipeline."""
        _requires_haystack()

        from haystack import Pipeline

        from utils.chunking import ChunkingFactory
        from utils.metadata_enricher import MetadataEnricher

        p = Pipeline()
        factory = ChunkingFactory(settings.chunking)
        p.add_component("cleaner", factory.build_cleaner())
        p.add_component("splitter", factory.build())
        p.add_component("enricher", MetadataEnricher())

        p.connect("cleaner.documents", "splitter.documents")
        p.connect("splitter.documents", "enricher.documents")

        result = p.run({"cleaner": {"documents": sample_documents}})
        enriched = result["enricher"]["documents"]

        assert len(enriched) >= 1
        for doc in enriched:
            assert "chunk_id" in doc.meta
            assert "chunk_index" in doc.meta
            assert "word_count" in doc.meta
            assert doc.meta["word_count"] >= 0

    def test_is_table_flag_from_meta(self):
        """is_table flag should be preserved when set on input document."""
        from haystack import Document

        from utils.metadata_enricher import MetadataEnricher

        enricher = MetadataEnricher()
        doc = Document(
            content="col1 col2\nval1 val2",
            meta={"is_table": True},
        )
        result = enricher.run(documents=[doc])["documents"][0]
        assert result.meta["is_table"] is True


# ---------------------------------------------------------------------------
# Full indexing pipeline smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIndexingPipelineSmokeTest:
    """
    Smoke test: extract text file → chunk → embed → write to InMemoryDocumentStore.
    Requires sentence-transformers.
    Skipped if heavy dependencies are missing.
    """

    def test_txt_file_ingested(self, settings, text_file):
        _requires_haystack()
        _requires_sentence_transformers()

        try:
            from docling_haystack.converter import DoclingConverter  # noqa: F401
        except ImportError:
            pytest.skip("docling-haystack not installed")

        from utils.indexing_pipeline import IndexingPipelineBuilder

        pipeline = IndexingPipelineBuilder(settings).build()
        result = pipeline.run({
            "converter": {"sources": [str(text_file)]}
        })

        written = result.get("writer", {}).get("documents_written", 0)
        assert written >= 1, (
            f"Expected at least 1 document written, got {written}. "
            f"Full result: {result}"
        )
