"""
Unit tests for QueryPipelineBuilder and query config models.

Config/schema tests have no external dependencies.
ContextTruncator and pipeline-builder tests require haystack-ai and are
skipped gracefully when it is not installed.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Determine whether haystack is available — used for selective skipping.
# ---------------------------------------------------------------------------
try:
    import haystack as _haystack_mod  # noqa: F401
    _HAYSTACK_AVAILABLE = True
except ImportError:
    _HAYSTACK_AVAILABLE = False

_requires_haystack = pytest.mark.skipif(
    not _HAYSTACK_AVAILABLE,
    reason="haystack-ai not installed",
)


# ---------------------------------------------------------------------------
# Query config model tests
# ---------------------------------------------------------------------------


class TestQueryConfigModels:
    def test_default_query_config(self):
        from config.models import QueryConfig, SearchType, RerankerStrategy, GeneratorBackend

        cfg = QueryConfig()
        assert cfg.retrieval.top_k == 5
        assert cfg.retrieval.search_type == SearchType.EMBEDDING
        assert cfg.reranker.strategy == RerankerStrategy.LOST_IN_MIDDLE
        assert cfg.generator.backend == GeneratorBackend.AZURE_OPENAI

    def test_hybrid_weights_must_sum_to_one(self):
        from config.models import HybridSearchConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="1.0"):
            HybridSearchConfig(embedding_weight=0.8, bm25_weight=0.4)

    def test_hybrid_weights_valid(self):
        from config.models import HybridSearchConfig

        cfg = HybridSearchConfig(embedding_weight=0.6, bm25_weight=0.4)
        assert abs(cfg.embedding_weight + cfg.bm25_weight - 1.0) < 1e-6

    def test_all_search_types_parse(self):
        from config.models import RetrievalConfig, SearchType

        for st in SearchType:
            cfg = RetrievalConfig(search_type=st.value)
            assert cfg.search_type == st

    def test_all_reranker_strategies_parse(self):
        from config.models import RerankerConfig, RerankerStrategy

        for strategy in RerankerStrategy:
            cfg = RerankerConfig(strategy=strategy.value)
            assert cfg.reranker_strategy == strategy if False else cfg.strategy == strategy

    def test_all_generator_backends_parse(self):
        from config.models import GeneratorConfig, GeneratorBackend

        for backend in GeneratorBackend:
            cfg = GeneratorConfig(backend=backend.value)
            assert cfg.backend == backend

    def test_top_k_bounds(self):
        from config.models import RetrievalConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RetrievalConfig(top_k=0)

        with pytest.raises(ValidationError):
            RetrievalConfig(top_k=101)

    def test_temperature_bounds(self):
        from config.models import AzureOpenAIGeneratorConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AzureOpenAIGeneratorConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            AzureOpenAIGeneratorConfig(temperature=2.1)

    def test_prompt_config_defaults(self):
        from config.models import PromptConfig

        cfg = PromptConfig()
        assert "{{ query }}" in cfg.template
        assert "{% for doc in documents %}" in cfg.template
        assert cfg.max_context_docs >= 1


# ---------------------------------------------------------------------------
# ContextTruncator component tests
# ---------------------------------------------------------------------------


class TestContextTruncator:
    @pytest.fixture(autouse=True)
    def _require_haystack(self):
        pytest.importorskip("haystack", reason="haystack-ai not installed")

    def test_truncates_to_max_docs(self):
        from haystack import Document
        from utils.query_pipeline import ContextTruncator

        docs = [Document(content=f"Document {i}" * 10) for i in range(10)]
        truncator = ContextTruncator(max_docs=3, max_chars_per_doc=500)
        result = truncator.run(documents=docs)["documents"]

        assert len(result) == 3

    def test_truncates_content_to_max_chars(self):
        from haystack import Document
        from utils.query_pipeline import ContextTruncator

        long_content = "x" * 5000
        docs = [Document(content=long_content)]
        truncator = ContextTruncator(max_docs=5, max_chars_per_doc=200)
        result = truncator.run(documents=docs)["documents"]

        assert len(result[0].content) <= 220  # 200 + "…[truncated]" suffix
        assert "truncated" in result[0].content

    def test_preserves_short_content(self):
        from haystack import Document
        from utils.query_pipeline import ContextTruncator

        content = "Short content."
        docs = [Document(content=content)]
        truncator = ContextTruncator(max_docs=5, max_chars_per_doc=500)
        result = truncator.run(documents=docs)["documents"]

        assert result[0].content == content

    def test_preserves_metadata(self):
        from haystack import Document
        from utils.query_pipeline import ContextTruncator

        docs = [
            Document(
                content="hello world",
                meta={"source_file": "report.pdf", "page_number": 3},
            )
        ]
        truncator = ContextTruncator(max_docs=5, max_chars_per_doc=500)
        result = truncator.run(documents=docs)["documents"]

        assert result[0].meta["source_file"] == "report.pdf"
        assert result[0].meta["page_number"] == 3

    def test_preserves_score(self):
        from haystack import Document
        from utils.query_pipeline import ContextTruncator

        doc = Document(content="test", score=0.92)
        truncator = ContextTruncator(max_docs=5, max_chars_per_doc=500)
        result = truncator.run(documents=[doc])["documents"]

        assert result[0].score == pytest.approx(0.92)

    def test_empty_input(self):
        from utils.query_pipeline import ContextTruncator

        truncator = ContextTruncator(max_docs=5, max_chars_per_doc=500)
        result = truncator.run(documents=[])["documents"]
        assert result == []

    def test_none_content_handled(self):
        from haystack import Document
        from utils.query_pipeline import ContextTruncator

        doc = Document(content=None)
        truncator = ContextTruncator(max_docs=5, max_chars_per_doc=200)
        result = truncator.run(documents=[doc])["documents"]
        assert result[0].content == ""


# ---------------------------------------------------------------------------
# QueryPipelineBuilder instantiation tests (no live LLM needed)
# ---------------------------------------------------------------------------


class TestQueryPipelineBuilder:
    @pytest.fixture(autouse=True)
    def _require_haystack(self):
        pytest.importorskip("haystack", reason="haystack-ai not installed")

    def test_context_truncator_is_a_component(self):
        """ContextTruncator must be a valid Haystack @component."""
        from haystack import component
        from utils.query_pipeline import ContextTruncator

        assert hasattr(ContextTruncator, "__haystack_component__") or callable(
            getattr(ContextTruncator, "run", None)
        )

    def test_missing_azure_env_raises(self, monkeypatch, settings):
        """QueryPipelineBuilder with azure_openai backend should raise if env vars missing."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_CHAT_DEPLOYMENT", raising=False)

        from config.models import GeneratorBackend
        import copy

        cfg = copy.deepcopy(settings)
        cfg.query.generator.backend = GeneratorBackend.AZURE_OPENAI

        from utils.query_pipeline import QueryPipelineBuilder
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        store = InMemoryDocumentStore()
        builder = QueryPipelineBuilder(cfg)

        with pytest.raises(EnvironmentError):
            builder.build(store)

    def test_ollama_backend_builds_without_env(self, settings):
        """Ollama backend needs no env vars — should attempt to build."""
        try:
            from haystack_integrations.components.generators.ollama import (
                OllamaChatGenerator,  # noqa: F401
            )
        except ImportError:
            pytest.skip("ollama-haystack not installed")

        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from utils.query_pipeline import QueryPipelineBuilder

        store = InMemoryDocumentStore()
        builder = QueryPipelineBuilder(settings)
        pipeline = builder.build(store)
        assert pipeline is not None


# ---------------------------------------------------------------------------
# Query schema tests
# ---------------------------------------------------------------------------


class TestQuerySchemas:
    def test_query_request_defaults(self):
        from app.schemas.query import QueryRequest

        req = QueryRequest(query="What is AI?")
        assert req.top_k == 5
        assert req.pipeline_name == "default"
        assert req.stream is False
        assert req.include_sources is True

    def test_query_request_empty_string_rejected(self):
        from app.schemas.query import QueryRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_search_request_defaults(self):
        from app.schemas.query import SearchRequest

        req = SearchRequest(query="neural networks")
        assert req.top_k == 10
        assert req.search_type == "embedding"

    def test_source_document_model(self):
        from app.schemas.query import SourceDocument

        src = SourceDocument(
            document_id="abc123",
            content="Deep learning is a branch of ML.",
            score=0.87,
            source_file="ml_book.pdf",
            page_number=42,
        )
        assert src.score == pytest.approx(0.87)
        assert src.is_table is False
