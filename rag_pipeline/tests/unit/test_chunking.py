"""
Unit tests for ChunkingFactory and MetadataEnricher.

Tests that require haystack-ai skip automatically when the package
is not installed — they produce a clean SKIP, not a FAIL.
Tests covering pure Python logic (config validation, unknown strategy)
always run.
"""

from __future__ import annotations

import importlib.util
import textwrap

import pytest

_HAYSTACK_AVAILABLE = importlib.util.find_spec("haystack") is not None
needs_haystack = pytest.mark.skipif(
    not _HAYSTACK_AVAILABLE, reason="haystack-ai not installed"
)


# ---------------------------------------------------------------------------
# Config/factory helpers
# ---------------------------------------------------------------------------


def _make_config(strategy: str, **overrides):
    from config.models import ChunkingConfig

    data = {
        "strategy": strategy,
        "word":     {"split_length": 20, "split_overlap": 2,  "split_threshold": 1},
        "sentence": {"split_length": 3,  "split_overlap": 1,  "split_threshold": 1},
        "passage":  {"split_length": 2,  "split_overlap": 0,  "split_threshold": 0},
        "page":     {"split_length": 1,  "split_overlap": 0,  "split_threshold": 0},
        "line":     {"split_length": 5,  "split_overlap": 1,  "split_threshold": 1},
        "hierarchical": {
            "block_sizes": [60, 20, 5], "split_by": "word", "split_overlap": 0,
        },
        "recursive": {
            "split_length": 60, "split_overlap": 5,
            "split_unit": "word", "separators": ["\n\n", "sentence", "\n", " "],
        },
        "character": {
            "split_length": 200, "split_overlap": 20,
            "split_unit": "char", "separators": ["\n\n", "\n", " ", ""],
        },
        "token": {
            "split_length": 64, "split_overlap": 8,
            "split_unit": "token", "separators": ["\n\n", "\n", " "],
        },
        "markdown_header": {
            "keep_headers": True, "secondary_split": None,
            "split_length": 100, "split_overlap": 0,
            "split_threshold": 0, "skip_empty_documents": True,
        },
        "semantic": {
            "sentences_per_group": 2, "percentile": 0.90,
            "min_length": 10, "max_length": 500,
        },
        "document_aware": {
            "tokenizer": "BAAI/bge-small-en-v1.5", "max_tokens": 64,
            "merge_peers": True, "include_headings_in_metadata": True,
        },
        "cleaner": {
            "remove_empty_lines": True, "remove_extra_whitespaces": True,
            "remove_repeated_substrings": False, "min_content_length": 5,
        },
    }
    data.update(overrides)
    return ChunkingConfig.model_validate(data)


def _run_splitter(splitter, documents):
    result = splitter.run(documents=documents)
    return result.get("documents", result.get("merged_documents", []))


_LOREM = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
) * 4


# ---------------------------------------------------------------------------
# Pure-Python tests (no haystack needed)
# ---------------------------------------------------------------------------


def test_unknown_strategy_raises():
    """Invalid strategy value is rejected at Pydantic model validation time."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _make_config("totally_invalid_strategy_xyz")


def test_hierarchical_config_descending_required():
    """block_sizes must be in descending order."""
    from config.models import ChunkingConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="descending"):
        ChunkingConfig.model_validate({"hierarchical": {"block_sizes": [5, 20, 60]}})


def test_semantic_config_percentile_bounds():
    from config.models import ChunkingConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChunkingConfig.model_validate({"semantic": {"percentile": 1.5}})


# ---------------------------------------------------------------------------
# Haystack-dependent tests
# ---------------------------------------------------------------------------


@needs_haystack
@pytest.mark.parametrize(
    "strategy",
    ["word", "sentence", "passage", "line", "recursive", "character", "token"],
)
def test_strategy_returns_documents(strategy):
    from haystack import Document
    from utils.chunking import ChunkingFactory

    cfg = _make_config(strategy)
    splitter = ChunkingFactory(cfg).build()
    docs = [Document(content=_LOREM, meta={"source_file": "test.txt"})]
    out = _run_splitter(splitter, docs)
    assert len(out) >= 1
    assert all(d.content for d in out)


@needs_haystack
def test_word_splits_produce_overlap():
    from haystack import Document
    from utils.chunking import ChunkingFactory

    cfg = _make_config(
        "word", word={"split_length": 10, "split_overlap": 3, "split_threshold": 0}
    )
    chunks = _run_splitter(ChunkingFactory(cfg).build(), [Document(content=_LOREM)])
    assert len(chunks) > 1
    words_0 = set(chunks[0].content.split())
    words_1 = set(chunks[1].content.split())
    assert words_0 & words_1, "Expected overlapping words between consecutive chunks."


@needs_haystack
def test_source_id_metadata_propagated():
    from haystack import Document
    from utils.chunking import ChunkingFactory

    cfg = _make_config("word")
    chunks = _run_splitter(
        ChunkingFactory(cfg).build(),
        [Document(content=_LOREM, meta={"source_file": "report.pdf"})],
    )
    for chunk in chunks:
        assert chunk.meta is not None
        assert "source_id" in chunk.meta


@needs_haystack
def test_hierarchical_builds_without_error():
    from utils.chunking import ChunkingFactory

    splitter = ChunkingFactory(_make_config("hierarchical")).build()
    assert splitter is not None
    assert "Hierarchical" in type(splitter).__name__


@needs_haystack
def test_hierarchical_block_sizes_applied():
    from utils.chunking import ChunkingFactory

    cfg = _make_config(
        "hierarchical",
        hierarchical={"block_sizes": [100, 25, 5], "split_by": "word", "split_overlap": 0},
    )
    splitter = ChunkingFactory(cfg).build()
    assert 100 in splitter.block_sizes
    assert 25 in splitter.block_sizes
    assert 5 in splitter.block_sizes


@needs_haystack
def test_recursive_correct_unit_word():
    from utils.chunking import ChunkingFactory

    splitter = ChunkingFactory(
        _make_config(
            "recursive",
            recursive={"split_length": 50, "split_overlap": 5, "split_unit": "word",
                       "separators": ["\n\n", "sentence", "\n", " "]},
        )
    ).build()
    assert splitter is not None


@needs_haystack
def test_recursive_correct_unit_char():
    from haystack import Document
    from utils.chunking import ChunkingFactory

    cfg = _make_config(
        "character",
        character={"split_length": 150, "split_overlap": 10, "split_unit": "char",
                   "separators": ["\n\n", "\n", " ", ""]},
    )
    chunks = _run_splitter(ChunkingFactory(cfg).build(), [Document(content=_LOREM)])
    for chunk in chunks[:-1]:
        assert len(chunk.content) <= 200, f"Chunk exceeded char limit: {len(chunk.content)}"


@needs_haystack
def test_markdown_header_splitter():
    from haystack import Document
    from utils.chunking import ChunkingFactory

    md_content = textwrap.dedent("""
        # Introduction

        This is the introduction section with some content about the topic.

        ## Background

        Here we discuss background information in detail.

        ## Methods

        The methods section describes our approach in full detail here.

        # Results

        Results are presented here with supporting data from the analysis.
    """)

    splitter = ChunkingFactory(_make_config("markdown_header")).build()
    chunks = _run_splitter(splitter, [Document(content=md_content)])
    assert len(chunks) >= 2, "Markdown splitter should produce at least 2 chunks."


@needs_haystack
def test_document_aware_returns_joiner():
    from haystack.components.joiners import DocumentJoiner
    from utils.chunking import ChunkingFactory

    component = ChunkingFactory(_make_config("document_aware")).build()
    assert isinstance(component, DocumentJoiner)


@needs_haystack
def test_cleaner_removes_extra_whitespace():
    from haystack import Document
    from utils.chunking import ChunkingFactory

    cleaner = ChunkingFactory(_make_config("word")).build_cleaner()
    docs = [Document(content="  Hello   world   this   is   a   test.  ")]
    result = cleaner.run(documents=docs)
    cleaned = result["documents"]
    assert len(cleaned) == 1
    assert "   " not in cleaned[0].content


@needs_haystack
def test_semantic_requires_embedding_config():
    from utils.chunking import ChunkingFactory

    cfg = _make_config("semantic")
    with pytest.raises(ValueError, match="embedding_cfg"):
        ChunkingFactory(cfg, embedding_cfg=None).build()


# ---------------------------------------------------------------------------
# MetadataEnricher tests (require haystack)
# ---------------------------------------------------------------------------


class TestMetadataEnricher:
    @pytest.fixture(autouse=True)
    def _require_haystack(self):
        if not _HAYSTACK_AVAILABLE:
            pytest.skip("haystack-ai not installed")

    def test_injects_required_fields(self):
        from haystack import Document
        from utils.metadata_enricher import MetadataEnricher

        enricher = MetadataEnricher()
        docs = [Document(content="Hello world this is test content.")]
        result = enricher.run(documents=docs)["documents"]

        assert len(result) == 1
        meta = result[0].meta
        assert "chunk_index" in meta
        assert "chunk_id" in meta
        assert "source_file" in meta
        assert "word_count" in meta
        assert "char_count" in meta
        assert meta["char_count"] == len("Hello world this is test content.")

    def test_chunk_id_is_deterministic(self):
        from haystack import Document
        from utils.metadata_enricher import MetadataEnricher

        enricher = MetadataEnricher()
        doc = Document(content="Deterministic content string.")
        r1 = enricher.run(documents=[doc])["documents"][0].meta["chunk_id"]
        r2 = enricher.run(documents=[doc])["documents"][0].meta["chunk_id"]
        assert r1 == r2

    def test_chunk_index_is_sequential(self):
        from haystack import Document
        from utils.metadata_enricher import MetadataEnricher

        enricher = MetadataEnricher()
        docs = [Document(content=f"Doc {i}") for i in range(5)]
        result = enricher.run(documents=docs)["documents"]
        for i, doc in enumerate(result):
            assert doc.meta["chunk_index"] == i

    def test_preserves_existing_metadata(self):
        from haystack import Document
        from utils.metadata_enricher import MetadataEnricher

        enricher = MetadataEnricher()
        doc = Document(
            content="test", meta={"source_file": "my_file.pdf", "page_number": 3}
        )
        result = enricher.run(documents=[doc])["documents"][0]
        assert result.meta["source_file"] == "my_file.pdf"
        assert result.meta["page_number"] == 3

    def test_is_table_flag_default(self):
        from haystack import Document
        from utils.metadata_enricher import MetadataEnricher

        enricher = MetadataEnricher()
        doc = Document(content="plain text")
        result = enricher.run(documents=[doc])["documents"][0]
        assert result.meta["is_table"] is False
        assert result.meta["is_picture"] is False

    def test_empty_document_list(self):
        from utils.metadata_enricher import MetadataEnricher

        enricher = MetadataEnricher()
        result = enricher.run(documents=[])["documents"]
        assert result == []

    def test_handles_none_content_gracefully(self):
        from haystack import Document
        from utils.metadata_enricher import MetadataEnricher

        enricher = MetadataEnricher()
        doc = Document(content=None)
        result = enricher.run(documents=[doc])["documents"][0]
        assert result.meta["word_count"] == 0
