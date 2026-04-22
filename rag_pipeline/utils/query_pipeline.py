"""
QueryPipelineBuilder
====================

Builds a Haystack 2.x ``Pipeline`` for RAG-style question answering.

Supported pipeline variants (all config-driven)
------------------------------------------------
Standard (embedding):
    TextEmbedder → EmbeddingRetriever → [Reranker] → PromptBuilder → Generator

Hybrid (embedding + BM25):
    TextEmbedder → EmbeddingRetriever ─┐
    BM25Retriever ─────────────────────┤ DocumentJoiner → [Reranker] → PromptBuilder → Generator

Hierarchical (with auto-merging):
    TextEmbedder → EmbeddingRetriever → AutoMergingRetriever → [Reranker] → PromptBuilder → Generator

Reranker options:
    none            — skip reranking
    lost_in_middle  — LostInTheMiddleRanker (fast, no API call)
    llm             — LLMRanker (Haystack 2.26, uses LLM to score docs)

Generator backends:
    azure_openai    — AzureOpenAIChatGenerator
    openai          — OpenAIChatGenerator
    ollama          — OllamaChatGenerator

Streaming is handled at the endpoint layer by passing a streaming callback;
the pipeline itself is stateless.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from config.models import (
    ChunkingStrategy,
    GeneratorBackend,
    QueryConfig,
    RerankerStrategy,
    SearchType,
)
from config.settings import AppSettings
from utils.embedding import EmbeddingFactory

if TYPE_CHECKING:
    from haystack import Pipeline

logger = logging.getLogger(__name__)


class QueryPipelineBuilder:
    """
    Builds and returns a configured Haystack RAG query pipeline.

    Usage::

        from config.settings import get_settings
        from utils.query_pipeline import QueryPipelineBuilder

        settings  = get_settings()
        builder   = QueryPipelineBuilder(settings)
        pipeline  = builder.build(document_store)

        result = pipeline.run({
            "text_embedder": {"text": "What is machine learning?"}
        })
        print(result["answer_builder"]["answers"][0].data)
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.query_cfg: QueryConfig = settings.query
        self.chunking_strategy = settings.chunking.strategy

    def build(self, document_store: Any) -> "Pipeline":
        """
        Build the query pipeline.

        Parameters
        ----------
        document_store:
            The same document store used by the indexing pipeline.
            Pass ``registry.get_indexing().get_component('writer').document_store``
            to share the store.
        """
        from haystack import Pipeline

        search_type = self.query_cfg.retrieval.search_type
        is_hierarchical = (
            self.chunking_strategy == ChunkingStrategy.HIERARCHICAL
            and self.query_cfg.auto_merging.enabled
        )

        logger.info(
            "Building query pipeline | search=%s | reranker=%s | generator=%s | hierarchical=%s",
            search_type.value,
            self.query_cfg.reranker.strategy.value,
            self.query_cfg.generator.backend.value,
            is_hierarchical,
        )

        pipeline = Pipeline()

        # ── 1. Query text embedder ────────────────────────────────────────
        text_embedder = EmbeddingFactory(self.settings.embedding).build_text_embedder()
        pipeline.add_component("text_embedder", text_embedder)

        # ── 2. Retriever(s) ───────────────────────────────────────────────
        top_k = self.query_cfg.retrieval.top_k

        embedding_retriever = self._build_embedding_retriever(document_store, top_k)
        pipeline.add_component("embedding_retriever", embedding_retriever)
        pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")

        if search_type == SearchType.HYBRID:
            bm25_retriever = self._build_bm25_retriever(document_store, top_k)
            pipeline.add_component("bm25_retriever", bm25_retriever)

            joiner = self._build_document_joiner()
            pipeline.add_component("document_joiner", joiner)

            pipeline.connect("embedding_retriever.documents", "document_joiner.documents")
            pipeline.connect("bm25_retriever.documents", "document_joiner.documents")

            last_retrieval_output = "document_joiner.documents"
        else:
            last_retrieval_output = "embedding_retriever.documents"

        # ── 3. Auto-merging (hierarchical only) ──────────────────────────
        if is_hierarchical:
            auto_merger = self._build_auto_merging_retriever(document_store)
            pipeline.add_component("auto_merging_retriever", auto_merger)
            pipeline.connect(last_retrieval_output, "auto_merging_retriever.matched_leaf_documents")
            last_retrieval_output = "auto_merging_retriever.documents"

        # ── 4. Reranker ───────────────────────────────────────────────────
        reranker_strategy = self.query_cfg.reranker.strategy

        if reranker_strategy == RerankerStrategy.LOST_IN_MIDDLE:
            reranker = self._build_lost_in_middle_ranker()
            pipeline.add_component("reranker", reranker)
            pipeline.connect(last_retrieval_output, "reranker.documents")
            last_retrieval_output = "reranker.documents"

        elif reranker_strategy == RerankerStrategy.LLM:
            llm_ranker = self._build_llm_ranker()
            pipeline.add_component("reranker", llm_ranker)
            pipeline.connect(last_retrieval_output, "reranker.documents")
            last_retrieval_output = "reranker.documents"

        # ── 5. Context truncation (custom component) ──────────────────────
        truncator = ContextTruncator(
            max_docs=self.query_cfg.prompt.max_context_docs,
            max_chars_per_doc=self.query_cfg.prompt.max_chars_per_doc,
        )
        pipeline.add_component("context_truncator", truncator)
        pipeline.connect(last_retrieval_output, "context_truncator.documents")

        # ── 6. Prompt builder ─────────────────────────────────────────────
        prompt_builder = self._build_prompt_builder()
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.connect("context_truncator.documents", "prompt_builder.documents")

        # ── 7. LLM generator ─────────────────────────────────────────────
        generator = self._build_generator()
        pipeline.add_component("generator", generator)
        pipeline.connect("prompt_builder.prompt", "generator.messages")

        # ── 8. Answer builder ─────────────────────────────────────────────
        answer_builder = self._build_answer_builder()
        pipeline.add_component("answer_builder", answer_builder)
        pipeline.connect("generator.replies", "answer_builder.replies")
        pipeline.connect("context_truncator.documents", "answer_builder.documents")

        logger.info("Query pipeline built successfully.")
        return pipeline

    # ------------------------------------------------------------------
    # Retriever builders
    # ------------------------------------------------------------------

    def _build_embedding_retriever(self, store: Any, top_k: int) -> Any:
        """Build the vector-search retriever for the given document store type."""
        store_type = type(store).__name__

        retriever_map = {
            "InMemoryDocumentStore": self._in_memory_embedding_retriever,
            "QdrantDocumentStore": self._qdrant_embedding_retriever,
            "WeaviateDocumentStore": self._weaviate_embedding_retriever,
            "OpenSearchDocumentStore": self._opensearch_embedding_retriever,
            "PgvectorDocumentStore": self._pgvector_embedding_retriever,
        }

        builder = retriever_map.get(store_type, self._in_memory_embedding_retriever)
        return builder(store, top_k)

    def _in_memory_embedding_retriever(self, store: Any, top_k: int) -> Any:
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

        filters = self.query_cfg.retrieval.default_filters
        return InMemoryEmbeddingRetriever(
            document_store=store,
            top_k=top_k,
            filters=filters,
        )

    def _qdrant_embedding_retriever(self, store: Any, top_k: int) -> Any:
        from haystack_integrations.components.retrievers.qdrant import (
            QdrantEmbeddingRetriever,
        )

        return QdrantEmbeddingRetriever(document_store=store, top_k=top_k)

    def _weaviate_embedding_retriever(self, store: Any, top_k: int) -> Any:
        from haystack_integrations.components.retrievers.weaviate import (
            WeaviateEmbeddingRetriever,
        )

        return WeaviateEmbeddingRetriever(document_store=store, top_k=top_k)

    def _opensearch_embedding_retriever(self, store: Any, top_k: int) -> Any:
        from haystack_integrations.components.retrievers.opensearch import (
            OpenSearchEmbeddingRetriever,
        )

        return OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k)

    def _pgvector_embedding_retriever(self, store: Any, top_k: int) -> Any:
        from haystack_integrations.components.retrievers.pgvector import (
            PgvectorEmbeddingRetriever,
        )

        return PgvectorEmbeddingRetriever(document_store=store, top_k=top_k)

    def _build_bm25_retriever(self, store: Any, top_k: int) -> Any:
        """Build a BM25 keyword retriever. Currently only InMemory supports this natively."""
        store_type = type(store).__name__
        if store_type == "InMemoryDocumentStore":
            from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

            return InMemoryBM25Retriever(
                document_store=store,
                top_k=top_k,
                filters=self.query_cfg.retrieval.default_filters,
            )
        elif store_type == "OpenSearchDocumentStore":
            from haystack_integrations.components.retrievers.opensearch import (
                OpenSearchBM25Retriever,
            )

            return OpenSearchBM25Retriever(document_store=store, top_k=top_k)
        else:
            logger.warning(
                "BM25 retriever not supported for '%s'; "
                "falling back to embedding-only retrieval.",
                store_type,
            )
            return self._in_memory_embedding_retriever(store, top_k)

    def _build_document_joiner(self) -> Any:
        """Join embedding + BM25 results with Reciprocal Rank Fusion or concatenation."""
        from haystack.components.joiners import DocumentJoiner

        join_mode = self.query_cfg.retrieval.hybrid.join_mode
        weights = [
            self.query_cfg.retrieval.hybrid.embedding_weight,
            self.query_cfg.retrieval.hybrid.bm25_weight,
        ]

        return DocumentJoiner(
            join_mode=join_mode,
            weights=weights,
            top_k=self.query_cfg.retrieval.top_k,
        )

    # ------------------------------------------------------------------
    # Auto-merging retriever
    # ------------------------------------------------------------------

    def _build_auto_merging_retriever(self, store: Any) -> Any:
        from haystack.components.retrievers import AutoMergingRetriever

        return AutoMergingRetriever(
            document_store=store,
            threshold=self.query_cfg.auto_merging.threshold,
        )

    # ------------------------------------------------------------------
    # Rerankers
    # ------------------------------------------------------------------

    def _build_lost_in_middle_ranker(self) -> Any:
        from haystack.components.rankers import LostInTheMiddleRanker

        return LostInTheMiddleRanker(
            top_k=self.query_cfg.retrieval.top_k,
        )

    def _build_llm_ranker(self) -> Any:
        """
        Build LLMRanker (Haystack 2.26).
        Uses the same generator backend as the main generator.
        """
        try:
            from haystack.components.rankers import LLMRanker
        except ImportError:
            logger.warning(
                "LLMRanker not available; falling back to LostInTheMiddleRanker. "
                "Upgrade to haystack-ai>=2.26."
            )
            return self._build_lost_in_middle_ranker()

        ranker_generator = self._build_generator(for_ranker=True)
        return LLMRanker(
            generator=ranker_generator,
            top_k=self.query_cfg.reranker.llm_ranker.top_k,
            meta_fields_to_embed=self.query_cfg.reranker.llm_ranker.meta_fields,
        )

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt_builder(self) -> Any:
        from haystack.components.builders import ChatPromptBuilder
        from haystack.dataclasses import ChatMessage

        system_msg = self.query_cfg.prompt.system_message
        template = self.query_cfg.prompt.template

        # Build a ChatPromptBuilder with system + user messages.
        messages = [
            ChatMessage.from_system(system_msg),
            ChatMessage.from_user(template),
        ]

        return ChatPromptBuilder(
            template=messages,
            required_variables=["documents", "query"],
        )

    # ------------------------------------------------------------------
    # Generator builders
    # ------------------------------------------------------------------

    def _build_generator(self, for_ranker: bool = False) -> Any:
        backend = self.query_cfg.generator.backend
        builders = {
            GeneratorBackend.AZURE_OPENAI: self._azure_openai_generator,
            GeneratorBackend.OPENAI: self._openai_generator,
            GeneratorBackend.OLLAMA: self._ollama_generator,
        }
        builder = builders.get(backend)
        if builder is None:
            raise ValueError(f"Unknown generator backend: '{backend}'")
        return builder(for_ranker=for_ranker)

    def _azure_openai_generator(self, for_ranker: bool = False) -> Any:
        try:
            from haystack.components.generators.chat import AzureOpenAIChatGenerator
        except ImportError:
            raise ImportError(
                "AzureOpenAIChatGenerator not found. "
                "Run: pip install haystack-ai"
            )

        az = self.query_cfg.generator.azure_openai
        self._check_env(az.api_key_env, az.azure_endpoint_env)

        deployment = os.environ.get(az.deployment_name_env)
        if not deployment:
            raise EnvironmentError(
                f"Environment variable '{az.deployment_name_env}' is not set. "
                "Set it to your Azure OpenAI chat deployment name."
            )

        generation_kwargs: dict[str, Any] = {
            "temperature": az.temperature,
            "max_tokens": az.max_tokens,
            "top_p": az.top_p,
        }

        streaming_callback = None
        if az.streaming and not for_ranker:
            streaming_callback = self._default_streaming_callback

        return AzureOpenAIChatGenerator(
            azure_deployment=deployment,
            azure_endpoint=os.environ[az.azure_endpoint_env],
            api_version=az.api_version,
            api_key=os.environ[az.api_key_env],
            generation_kwargs=generation_kwargs,
            streaming_callback=streaming_callback,
        )

    def _openai_generator(self, for_ranker: bool = False) -> Any:
        try:
            from haystack.components.generators.chat import OpenAIChatGenerator
        except ImportError:
            raise ImportError("Run: pip install haystack-ai")

        oa = self.query_cfg.generator.openai
        self._check_env(oa.api_key_env)

        streaming_callback = None
        if oa.streaming and not for_ranker:
            streaming_callback = self._default_streaming_callback

        return OpenAIChatGenerator(
            api_key=os.environ[oa.api_key_env],
            model=oa.model,
            generation_kwargs={
                "temperature": oa.temperature,
                "max_tokens": oa.max_tokens,
                "top_p": oa.top_p,
            },
            streaming_callback=streaming_callback,
        )

    def _ollama_generator(self, for_ranker: bool = False) -> Any:
        try:
            from haystack_integrations.components.generators.ollama import (
                OllamaChatGenerator,
            )
        except ImportError:
            raise ImportError(
                "ollama-haystack not found. Run: pip install ollama-haystack"
            )

        ol = self.query_cfg.generator.ollama

        streaming_callback = None
        if ol.streaming and not for_ranker:
            streaming_callback = self._default_streaming_callback

        return OllamaChatGenerator(
            model=ol.model,
            url=f"{ol.base_url.rstrip('/')}/api/chat",
            generation_kwargs={
                "temperature": ol.temperature,
                "options": {"num_ctx": ol.context_window},
            },
            streaming_callback=streaming_callback,
        )

    # ------------------------------------------------------------------
    # Answer builder
    # ------------------------------------------------------------------

    def _build_answer_builder(self) -> Any:
        from haystack.components.builders import AnswerBuilder

        return AnswerBuilder()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _check_env(*var_names: str) -> None:
        missing = [v for v in var_names if not os.environ.get(v)]
        if missing:
            raise EnvironmentError(
                f"Required environment variable(s) not set: {missing}. "
                "Add them to .env."
            )

    @staticmethod
    def _default_streaming_callback(chunk: Any) -> None:
        """
        Default no-op streaming callback.
        Replace with an SSE-aware callback at the endpoint layer:

            from haystack.dataclasses import StreamingChunk

            def sse_callback(chunk: StreamingChunk) -> None:
                queue.put(chunk.content)

            pipeline.run(..., include_outputs_from={"generator": {"streaming_callback": sse_callback}})
        """
        pass


# ---------------------------------------------------------------------------
# ContextTruncator — custom Haystack component
# ---------------------------------------------------------------------------
# Import guard: these are only available when haystack-ai is installed.
# The class body is only evaluated when actually imported with haystack present.
try:
    from haystack import Document as _HaystackDocument, component as _component

    @_component
    class ContextTruncator:
        """
        Trims the document list to ``max_docs`` and truncates each document's
        content to ``max_chars_per_doc`` characters before prompt injection.

        This prevents prompt overflow on very long chunks and limits API token cost.
        """

        def __init__(self, max_docs: int = 5, max_chars_per_doc: int = 2000) -> None:
            self.max_docs = max_docs
            self.max_chars_per_doc = max_chars_per_doc

        @_component.output_types(documents=list[_HaystackDocument])
        def run(self, documents: list[_HaystackDocument]) -> dict[str, list[_HaystackDocument]]:
            truncated: list[_HaystackDocument] = []
            for doc in documents[: self.max_docs]:
                content = doc.content or ""
                if len(content) > self.max_chars_per_doc:
                    content = content[: self.max_chars_per_doc] + " …[truncated]"
                truncated.append(
                    _HaystackDocument(
                        content=content,
                        meta=doc.meta,
                        id=doc.id,
                        score=doc.score,
                    )
                )
            return {"documents": truncated}

except ImportError:
    # Haystack not installed — provide a stub so the module can be imported
    # without error. Tests will be skipped via pytest.importorskip.
    class ContextTruncator:  # type: ignore[no-redef]
        """Stub used when haystack-ai is not installed."""

        def __init__(self, max_docs: int = 5, max_chars_per_doc: int = 2000) -> None:
            raise ImportError(
                "haystack-ai is required for ContextTruncator. "
                "Run: pip install haystack-ai"
            )
