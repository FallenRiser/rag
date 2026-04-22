"""
Shared pytest fixtures for the RAG pipeline test suite.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """
    Write minimal but complete config YAML files into a temp directory
    and return the path. Tests can override individual keys after this.
    """
    docling_cfg = {
        "docling": {
            "pdf": {
                "do_ocr": False,  # fast in tests; no OCR engine required
                "ocr_engine": "easyocr",
                "do_table_structure": True,
                "table_structure_options": {"mode": "fast"},
                "generate_page_images": False,
                "generate_picture_images": False,
                "images_scale": 1.0,
            },
            "enrichments": {
                "picture_description": {"enabled": False},
                "picture_classification": {"enabled": False},
                "code_understanding": {"enabled": False},
                "formula_understanding": {"enabled": False},
            },
            "export": {"type": "doc_chunks"},
            "performance": {
                "device": "cpu",
                "batch_size": 1,
                "timeout_per_document_seconds": 60,
            },
        }
    }

    chunking_cfg = {
        "chunking": {
            "strategy": "word",
            "split_length": 50,
            "split_overlap": 5,
            "split_threshold": 2,
            "word": {"split_length": 50, "split_overlap": 5, "split_threshold": 2},
            "sentence": {"split_length": 5, "split_overlap": 1, "split_threshold": 1},
            "passage": {"split_length": 2, "split_overlap": 0, "split_threshold": 0},
            "page": {"split_length": 1, "split_overlap": 0, "split_threshold": 0},
            "line": {"split_length": 10, "split_overlap": 1, "split_threshold": 1},
            "hierarchical": {
                "block_sizes": [100, 30, 10],
                "split_by": "word",
                "split_overlap": 0,
            },
            "recursive": {
                "split_length": 100,
                "split_overlap": 10,
                "split_unit": "word",
                "separators": ["\n\n", "sentence", "\n", " "],
            },
            "character": {
                "split_length": 500,
                "split_overlap": 50,
                "split_unit": "char",
                "separators": ["\n\n", "\n", " ", ""],
            },
            "token": {
                "split_length": 128,
                "split_overlap": 16,
                "split_unit": "token",
                "separators": ["\n\n", "\n", " "],
            },
            "markdown_header": {
                "keep_headers": True,
                "secondary_split": None,
                "split_length": 100,
                "split_overlap": 0,
                "split_threshold": 0,
                "skip_empty_documents": True,
            },
            "semantic": {
                "sentences_per_group": 2,
                "percentile": 0.90,
                "min_length": 20,
                "max_length": 500,
            },
            "document_aware": {
                "tokenizer": "BAAI/bge-small-en-v1.5",
                "max_tokens": 128,
                "merge_peers": True,
                "include_headings_in_metadata": True,
            },
            "cleaner": {
                "remove_empty_lines": True,
                "remove_extra_whitespaces": True,
                "remove_repeated_substrings": False,
                "min_content_length": 5,
            },
        }
    }

    embedding_cfg = {
        "embedding": {
            "provider": "sentence_transformers",
            "sentence_transformers": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
                "batch_size": 4,
                "normalize_embeddings": True,
                "cache_dir": None,
                "query_instruction": "",
                "document_instruction": "",
            },
        }
    }

    store_cfg = {
        "document_store": {
            "backend": "in_memory",
            "in_memory": {"bm25_retrieval": True},
        }
    }

    query_cfg = {
        "query": {
            "retrieval": {"top_k": 3, "search_type": "embedding"},
            "auto_merging": {"enabled": False},
            "reranker": {"strategy": "none"},
            "prompt": {
                "system_message": "Answer from context.",
                "template": "Context:\n{% for doc in documents %}{{ doc.content }}\n{% endfor %}\nQ: {{ query }}\nA:",
                "max_context_docs": 3,
                "max_chars_per_doc": 500,
            },
            "generator": {
                "backend": "ollama",
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "llama3.1:8b",
                    "temperature": 0.0,
                    "context_window": 4096,
                    "streaming": False,
                },
            },
            "answer": {"include_sources": True, "max_sources": 3, "include_scores": True},
        }
    }

    for fname, data in [
        ("docling_config.yaml", docling_cfg),
        ("chunking_config.yaml", chunking_cfg),
        ("embedding_config.yaml", embedding_cfg),
        ("store_config.yaml", store_cfg),
        ("query_config.yaml", query_cfg),
    ]:
        (tmp_path / fname).write_text(yaml.dump(data), encoding="utf-8")

    return tmp_path


@pytest.fixture
def settings(config_dir: Path):
    """Return a test AppSettings pointing at the temp config directory."""
    # Clear the lru_cache so each test gets a fresh instance.
    from config.settings import get_settings

    get_settings.cache_clear()
    s = get_settings(config_dir=str(config_dir))
    yield s
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_documents():
    """Return a list of Haystack Document objects for unit testing."""
    from haystack import Document

    lorem = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
        "The five boxing wizards jump quickly. "
    )

    return [
        Document(
            content=lorem * 3,
            meta={"source_file": "test.pdf", "page_number": 1},
        ),
        Document(
            content="# Introduction\n\nThis is a markdown document.\n\n## Section 1\n\nContent here.",
            meta={"source_file": "test.md", "page_number": 1},
        ),
        Document(
            content="Row1Col1\tRow1Col2\nRow2Col1\tRow2Col2",
            meta={"source_file": "table.pdf", "is_table": True, "page_number": 2},
        ),
    ]
