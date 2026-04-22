"""
Unit tests for ChunkingFactory and MetadataEnricher.

- Pure-Python tests (config, metadata helpers) always run.
- Haystack-dependent tests skip gracefully when haystack-ai is not installed.
"""

from __future__ import annotations

import importlib.util
import textwrap

import pytest

_HAYSTACK = importlib.util.find_spec("haystack") is not None
needs_hs = pytest.mark.skipif(not _HAYSTACK, reason="haystack-ai not installed")

LOREM = "The quick brown fox jumps over the lazy dog. " * 15


def _cfg(strategy: str, **overrides):
    from config.models import ChunkingConfig
    base = {
        "strategy": strategy,
        "word":      {"split_length": 20, "split_overlap": 2,  "split_threshold": 1},
        "passage":   {"split_length": 3,  "split_overlap": 0,  "split_threshold": 0},
        "page":      {"split_length": 1,  "split_overlap": 0,  "split_threshold": 0},
        "line":      {"split_length": 10, "split_overlap": 1,  "split_threshold": 1},
        "character": {"split_length": 200, "split_overlap": 20, "split_unit": "char",
                      "separators": ["\n\n", "\n", " ", ""]},
        "recursive": {"split_length": 60,  "split_overlap": 5,  "split_unit": "word",
                      "separators": ["\n\n", "\n", " "]},
        "hierarchical": {"block_sizes": [80, 20, 5], "split_by": "word", "split_overlap": 0},
        "markdown_header": {
            "keep_headers": True, "secondary_split": None,
            "split_length": 100, "split_overlap": 0,
            "split_threshold": 0, "skip_empty_documents": True,
        },
        "semantic":  {"sentences_per_group": 2, "percentile": 0.90,
                      "min_length": 10, "max_length": 300},
        "document_aware": {"tokenizer": "BAAI/bge-small-en-v1.5",
                           "max_tokens": 64, "merge_peers": True,
                           "include_headings_in_metadata": True},
        "cleaner": {"remove_empty_lines": True, "remove_extra_whitespaces": True,
                    "remove_repeated_substrings": False, "min_content_length": 5},
    }
    base.update(overrides)
    return ChunkingConfig.model_validate(base)


def _run(splitter, docs):
    result = splitter.run(documents=docs)
    return result.get("documents", result.get("merged_documents", []))


# ===========================================================================
# Config validation (no external deps)
# ===========================================================================

class TestChunkingConfigValidation:
    def test_all_strategies_parse(self):
        from config.models import ChunkingStrategy
        for s in ChunkingStrategy:
            assert _cfg(s.value).strategy == s

    def test_hierarchical_must_be_descending(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="descending"):
            _cfg("hierarchical", hierarchical={"block_sizes": [32, 128], "split_by": "word", "split_overlap": 0})

    def test_hierarchical_needs_two_levels(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _cfg("hierarchical", hierarchical={"block_sizes": [512], "split_by": "word", "split_overlap": 0})

    def test_recursive_invalid_unit(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _cfg("recursive", recursive={"split_length": 100, "split_overlap": 0,
                                          "split_unit": "banana", "separators": []})

    def test_invalid_strategy_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _cfg("nonexistent_xyz")

    def test_semantic_percentile_out_of_range(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _cfg("semantic", semantic={"sentences_per_group": 2, "percentile": 1.5,
                                        "min_length": 10, "max_length": 200})


# ===========================================================================
# document_aware returns None
# ===========================================================================

def test_document_aware_build_returns_none():
    from utils.chunking import ChunkingFactory
    assert ChunkingFactory(_cfg("document_aware")).build() is None


# ===========================================================================
# Haystack strategy tests
# ===========================================================================

@needs_hs
@pytest.mark.parametrize("strategy", ["word", "passage", "page", "line"])
def test_doc_splitter_strategy_runs(strategy):
    from haystack import Document
    from utils.chunking import ChunkingFactory
    splitter = ChunkingFactory(_cfg(strategy)).build()
    assert splitter is not None
    out = _run(splitter, [Document(content=LOREM)])
    assert len(out) >= 1
    assert all(d.content for d in out)


@needs_hs
def test_word_splitter_creates_overlap():
    from haystack import Document
    from utils.chunking import ChunkingFactory
    cfg = _cfg("word", word={"split_length": 10, "split_overlap": 4, "split_threshold": 0})
    chunks = _run(ChunkingFactory(cfg).build(), [Document(content=LOREM)])
    assert len(chunks) > 1
    w0 = set(chunks[0].content.split())
    w1 = set(chunks[1].content.split())
    assert w0 & w1, "Adjacent chunks must share words (overlap)"


@needs_hs
def test_word_splitter_adds_source_id():
    from haystack import Document
    from utils.chunking import ChunkingFactory
    chunks = _run(ChunkingFactory(_cfg("word")).build(),
                  [Document(content=LOREM, meta={"source_file": "r.pdf"})])
    for c in chunks:
        assert "source_id" in c.meta


@needs_hs
def test_recursive_no_nltk_dependency():
    """Recursive splitter with no 'sentence' separator must not need NLTK."""
    from haystack import Document
    from utils.chunking import ChunkingFactory
    cfg = _cfg("recursive", recursive={
        "split_length": 50, "split_overlap": 5,
        "split_unit": "word", "separators": ["\n\n", "\n", " "],
    })
    out = _run(ChunkingFactory(cfg).build(), [Document(content=LOREM)])
    assert len(out) >= 1


@needs_hs
def test_character_splitter_respects_limit():
    from haystack import Document
    from utils.chunking import ChunkingFactory
    cfg = _cfg("character", character={
        "split_length": 100, "split_overlap": 10,
        "split_unit": "char", "separators": ["\n\n", "\n", " ", ""],
    })
    chunks = _run(ChunkingFactory(cfg).build(), [Document(content=LOREM)])
    for c in chunks[:-1]:
        assert len(c.content) <= 130


@needs_hs
def test_hierarchical_produces_multiple_levels():
    from haystack import Document
    from utils.chunking import ChunkingFactory
    cfg = _cfg("hierarchical", hierarchical={
        "block_sizes": [80, 20, 5], "split_by": "word", "split_overlap": 0,
    })
    result = ChunkingFactory(cfg).build().run(documents=[Document(content=LOREM)])
    assert len(result.get("documents", [])) >= 2


@needs_hs
def test_hierarchical_block_sizes_on_component():
    from utils.chunking import ChunkingFactory
    cfg = _cfg("hierarchical", hierarchical={
        "block_sizes": [100, 25, 5], "split_by": "word", "split_overlap": 0,
    })
    splitter = ChunkingFactory(cfg).build()
    # Haystack may store block_sizes as list or set depending on version
    assert set(splitter.block_sizes) == {100, 25, 5}


@needs_hs
def test_markdown_header_splits_on_headings():
    from haystack import Document
    from utils.chunking import ChunkingFactory
    md = textwrap.dedent("""
        # Chapter One
        Content for chapter one goes here with enough words.

        ## Section 1.1
        Section content with enough text to be meaningful.

        # Chapter Two
        Content for chapter two is separate from chapter one.
    """).strip()
    out = _run(ChunkingFactory(_cfg("markdown_header")).build(),
               [Document(content=md)])
    assert len(out) >= 2


@needs_hs
def test_semantic_requires_embedding_cfg():
    from utils.chunking import ChunkingFactory
    with pytest.raises(ValueError, match="embedding_cfg"):
        ChunkingFactory(_cfg("semantic"), embedding_cfg=None).build()


@needs_hs
def test_cleaner_strips_whitespace():
    from haystack import Document
    from utils.chunking import ChunkingFactory
    cleaner = ChunkingFactory(_cfg("word")).build_cleaner()
    out = cleaner.run(documents=[Document(content="  a   b   c  ")])["documents"]
    assert "   " not in out[0].content


@needs_hs
def test_full_pipeline_word_strategy():
    """Integration: cleaner → splitter → meta_enricher → writer."""
    from haystack import Pipeline, Document
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.writers import DocumentWriter
    from utils.chunking import ChunkingFactory
    from utils.metadata_enricher import MetadataEnricher

    cfg = _cfg("word", word={"split_length": 20, "split_overlap": 3, "split_threshold": 1})
    factory = ChunkingFactory(cfg)
    store = InMemoryDocumentStore()

    p = Pipeline()
    p.add_component("cleaner",  factory.build_cleaner())
    p.add_component("chunker",  factory.build())
    p.add_component("enricher", MetadataEnricher())
    p.add_component("writer",   DocumentWriter(document_store=store))
    p.connect("cleaner.documents",  "chunker.documents")
    p.connect("chunker.documents",  "enricher.documents")
    p.connect("enricher.documents", "writer.documents")

    result = p.run({
        "cleaner":  {"documents": [Document(content=LOREM)]},
        "enricher": {"user_id": "alice", "source_name": "test.txt",
                     "version": 1, "is_latest": True,
                     "source_hash": "abc123", "ingested_at": "2024-01-01T00:00:00Z"},
    })

    assert result["writer"]["documents_written"] >= 1
    stored = store.filter_documents()
    m = stored[0].meta
    assert m["user_id"] == "alice"
    assert m["version"] == 1
    assert m["is_latest"] is True
    assert "chunk_id" in m
    assert "chunk_index" in m
    assert m["source_name"] == "test.txt"


# ===========================================================================
# MetadataEnricher pure-function tests (no haystack)
# ===========================================================================

class TestMetadataHelpers:
    def test_extract_source_from_file_path(self):
        from utils.metadata_enricher import _extract_source
        assert _extract_source({"file_path": "/a/b/doc.pdf"}) == "/a/b/doc.pdf"

    def test_extract_source_from_origin_dict(self):
        from utils.metadata_enricher import _extract_source
        assert _extract_source({"origin": {"filename": "doc.pdf"}}) == "doc.pdf"

    def test_extract_source_fallback(self):
        from utils.metadata_enricher import _extract_source
        assert _extract_source({}) == "unknown"

    def test_extract_source_name_basename(self):
        from utils.metadata_enricher import _extract_source_name
        assert _extract_source_name({"source_file": "/a/b/report.pdf"}) == "report.pdf"
        assert _extract_source_name({"source_file": "C:\\docs\\file.docx"}) == "file.docx"

    def test_page_number_from_page_no(self):
        from utils.metadata_enricher import _extract_page_number
        assert _extract_page_number({"page_no": "5"}) == 5

    def test_page_number_from_dl_meta(self):
        from utils.metadata_enricher import _extract_page_number
        assert _extract_page_number({"dl_meta": {"origin": {"page_no": 3}}}) == 3

    def test_page_number_none_when_missing(self):
        from utils.metadata_enricher import _extract_page_number
        assert _extract_page_number({}) is None

    def test_docling_fields_headings(self):
        from utils.metadata_enricher import _extract_docling_fields
        meta: dict = {}
        _extract_docling_fields(meta, {"headings": ["Intro", "Methods"]})
        assert meta["headings"] == ["Intro", "Methods"]

    def test_docling_fields_table(self):
        from utils.metadata_enricher import _extract_docling_fields
        meta: dict = {}
        _extract_docling_fields(meta, {"doc_items": [{"label": "table", "text": "x"}]})
        assert meta["is_table"] is True

    def test_docling_fields_picture_caption(self):
        from utils.metadata_enricher import _extract_docling_fields
        meta: dict = {}
        _extract_docling_fields(meta, {"doc_items": [{"label": "picture", "caption": "A fig."}]})
        assert meta["is_picture"] is True
        assert meta["picture_caption"] == "A fig."

    def test_enrich_document_meta_stamps_all_fields(self):
        from utils.metadata_enricher import enrich_document_meta
        meta = enrich_document_meta(
            {}, "Hello world content here", idx=2,
            user_id="alice", source_name="doc.pdf",
            source_hash="deadbeef", version=3,
            is_latest=True, ingested_at="2024-01-01T00:00:00+00:00",
            version_note="first",
        )
        assert meta["chunk_index"] == 2
        assert len(meta["chunk_id"]) == 16
        assert meta["user_id"] == "alice"
        assert meta["version"] == 3
        assert meta["is_latest"] is True
        assert meta["char_count"] == len("Hello world content here")
        assert meta["word_count"] == 4

    def test_chunk_id_deterministic(self):
        from utils.metadata_enricher import enrich_document_meta
        assert (enrich_document_meta({}, "same", 0)["chunk_id"] ==
                enrich_document_meta({}, "same", 0)["chunk_id"])

    def test_chunk_id_differs_for_different_content(self):
        from utils.metadata_enricher import enrich_document_meta
        assert (enrich_document_meta({}, "A", 0)["chunk_id"] !=
                enrich_document_meta({}, "B", 0)["chunk_id"])

    def test_none_content_no_crash(self):
        from utils.metadata_enricher import enrich_document_meta
        meta = enrich_document_meta({}, None, 0)
        assert meta["word_count"] == 0
        assert meta["char_count"] == 0


@needs_hs
class TestMetadataEnricherComponent:
    def test_stamps_user_and_version(self):
        from haystack import Document
        from utils.metadata_enricher import MetadataEnricher
        out = MetadataEnricher().run(
            documents=[Document(content="Hello")],
            user_id="bob", source_name="notes.md",
            source_hash="xyz", version=2, is_latest=True,
        )["documents"]
        m = out[0].meta
        assert m["user_id"] == "bob"
        assert m["version"] == 2
        assert m["source_name"] == "notes.md"
        assert len(m["chunk_id"]) == 16

    def test_empty_list(self):
        from utils.metadata_enricher import MetadataEnricher
        assert MetadataEnricher().run(documents=[])["documents"] == []

    def test_none_content_no_crash(self):
        from haystack import Document
        from utils.metadata_enricher import MetadataEnricher
        out = MetadataEnricher().run(documents=[Document(content=None)])["documents"]
        assert out[0].meta["word_count"] == 0

    def test_preserves_existing_meta(self):
        from haystack import Document
        from utils.metadata_enricher import MetadataEnricher
        doc = Document(content="test", meta={"source_file": "kept.pdf", "page_number": 7})
        out = MetadataEnricher().run(documents=[doc])["documents"]
        assert out[0].meta["source_file"] == "kept.pdf"
        assert out[0].meta["page_number"] == 7
