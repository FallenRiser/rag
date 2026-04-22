"""
EmbeddingFactory
================

Builds Haystack 2.x document and text (query) embedder components from
``EmbeddingConfig``.

Supported providers
-------------------
- azure_openai        Azure OpenAI text-embedding-3-large / ada-002
- openai              OpenAI API
- ollama              Local Ollama server (nomic-embed-text, bge-m3, …)
- sentence_transformers HuggingFace local (BAAI/bge-*, intfloat/e5-*, …)
- cohere              Cohere Embed v3

Each provider exposes two components:
    build_document_embedder()  — used during indexing
    build_text_embedder()      — used during query time
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from config.models import EmbeddingConfig, EmbeddingProvider

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """
    Factory that builds Haystack embedder components from ``EmbeddingConfig``.

    Usage::

        from config.settings import get_settings
        from utils.embedding import EmbeddingFactory

        factory = EmbeddingFactory(get_settings().embedding)
        doc_embedder  = factory.build_document_embedder()
        text_embedder = factory.build_text_embedder()
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_document_embedder(self) -> Any:
        """Return a Haystack document embedder for the configured provider."""
        provider = self.cfg.provider
        logger.info("Building document embedder | provider=%s", provider.value)

        builders = {
            EmbeddingProvider.AZURE_OPENAI: self._azure_doc_embedder,
            EmbeddingProvider.OPENAI: self._openai_doc_embedder,
            EmbeddingProvider.OLLAMA: self._ollama_doc_embedder,
            EmbeddingProvider.SENTENCE_TRANSFORMERS: self._st_doc_embedder,
            EmbeddingProvider.COHERE: self._cohere_doc_embedder,
        }
        return self._dispatch(builders, "document embedder")

    def build_text_embedder(self) -> Any:
        """Return a Haystack text (query) embedder for the configured provider."""
        provider = self.cfg.provider
        logger.info("Building text embedder | provider=%s", provider.value)

        builders = {
            EmbeddingProvider.AZURE_OPENAI: self._azure_text_embedder,
            EmbeddingProvider.OPENAI: self._openai_text_embedder,
            EmbeddingProvider.OLLAMA: self._ollama_text_embedder,
            EmbeddingProvider.SENTENCE_TRANSFORMERS: self._st_text_embedder,
            EmbeddingProvider.COHERE: self._cohere_text_embedder,
        }
        return self._dispatch(builders, "text embedder")

    @property
    def embedding_dim(self) -> int:
        """Return the vector dimension for the active provider."""
        dim_map = {
            EmbeddingProvider.AZURE_OPENAI: self.cfg.azure_openai.dimensions,
            EmbeddingProvider.OPENAI: self.cfg.openai.dimensions,
            EmbeddingProvider.OLLAMA: self.cfg.ollama.dimensions,
            EmbeddingProvider.SENTENCE_TRANSFORMERS: self._st_dim(),
            EmbeddingProvider.COHERE: 1024,  # Cohere Embed v3 default
        }
        return dim_map.get(self.cfg.provider, 1024)

    # ------------------------------------------------------------------
    # Azure OpenAI
    # ------------------------------------------------------------------

    def _azure_doc_embedder(self) -> Any:
        try:
            from haystack.components.embedders import AzureOpenAIDocumentEmbedder
        except ImportError:
            self._missing_package("azure-openai (haystack)", "pip install haystack-ai[azure]")

        az = self.cfg.azure_openai
        self._check_env(az.api_key_env, az.azure_endpoint_env)

        return AzureOpenAIDocumentEmbedder(
            azure_deployment=az.azure_deployment,
            azure_endpoint=self._env(az.azure_endpoint_env),
            api_version=az.api_version,
            api_key=self._env(az.api_key_env),
            dimensions=az.dimensions,
            batch_size=az.batch_size,
            meta_fields_to_embed=["headings", "source_file"],
        )

    def _azure_text_embedder(self) -> Any:
        try:
            from haystack.components.embedders import AzureOpenAITextEmbedder
        except ImportError:
            self._missing_package("azure-openai (haystack)", "pip install haystack-ai[azure]")

        az = self.cfg.azure_openai
        self._check_env(az.api_key_env, az.azure_endpoint_env)

        return AzureOpenAITextEmbedder(
            azure_deployment=az.azure_deployment,
            azure_endpoint=self._env(az.azure_endpoint_env),
            api_version=az.api_version,
            api_key=self._env(az.api_key_env),
            dimensions=az.dimensions,
        )

    # ------------------------------------------------------------------
    # OpenAI (non-Azure)
    # ------------------------------------------------------------------

    def _openai_doc_embedder(self) -> Any:
        try:
            from haystack.components.embedders import OpenAIDocumentEmbedder
        except ImportError:
            self._missing_package("openai", "pip install haystack-ai")

        oa = self.cfg.openai
        self._check_env(oa.api_key_env)

        return OpenAIDocumentEmbedder(
            api_key=self._env(oa.api_key_env),
            model=oa.model,
            dimensions=oa.dimensions,
            batch_size=oa.batch_size,
            meta_fields_to_embed=["headings", "source_file"],
        )

    def _openai_text_embedder(self) -> Any:
        try:
            from haystack.components.embedders import OpenAITextEmbedder
        except ImportError:
            self._missing_package("openai", "pip install haystack-ai")

        oa = self.cfg.openai
        self._check_env(oa.api_key_env)

        return OpenAITextEmbedder(
            api_key=self._env(oa.api_key_env),
            model=oa.model,
            dimensions=oa.dimensions,
        )

    # ------------------------------------------------------------------
    # Ollama (local)
    # ------------------------------------------------------------------

    def _ollama_doc_embedder(self) -> Any:
        try:
            from haystack_integrations.components.embedders.ollama import (
                OllamaDocumentEmbedder,
            )
        except ImportError:
            self._missing_package(
                "ollama-haystack",
                "pip install ollama-haystack",
            )

        ol = self.cfg.ollama
        logger.info(
            "Ollama document embedder | model=%s | url=%s", ol.model, ol.base_url
        )

        return OllamaDocumentEmbedder(
            model=ol.model,
            url=f"{ol.base_url.rstrip('/')}/api/embeddings",
            batch_size=ol.batch_size,
        )

    def _ollama_text_embedder(self) -> Any:
        try:
            from haystack_integrations.components.embedders.ollama import (
                OllamaTextEmbedder,
            )
        except ImportError:
            self._missing_package("ollama-haystack", "pip install ollama-haystack")

        ol = self.cfg.ollama

        return OllamaTextEmbedder(
            model=ol.model,
            url=f"{ol.base_url.rstrip('/')}/api/embeddings",
        )

    # ------------------------------------------------------------------
    # SentenceTransformers (local HuggingFace)
    # ------------------------------------------------------------------

    def _st_doc_embedder(self) -> Any:
        try:
            from haystack.components.embedders import SentenceTransformersDocumentEmbedder
        except ImportError:
            self._missing_package(
                "sentence-transformers",
                "pip install sentence-transformers",
            )

        st = self.cfg.sentence_transformers
        device = self._resolve_device(st.device)
        logger.info(
            "SentenceTransformers document embedder | model=%s | device=%s",
            st.model,
            device,
        )

        kwargs: dict[str, Any] = dict(
            model=st.model,
            device=device,
            batch_size=st.batch_size,
            normalize_embeddings=st.normalize_embeddings,
            meta_fields_to_embed=["headings", "source_file"],
        )
        if st.document_instruction:
            kwargs["prefix"] = st.document_instruction
        if st.cache_dir:
            kwargs["cache_folder"] = str(st.cache_dir)

        return SentenceTransformersDocumentEmbedder(**kwargs)

    def _st_text_embedder(self) -> Any:
        try:
            from haystack.components.embedders import SentenceTransformersTextEmbedder
        except ImportError:
            self._missing_package(
                "sentence-transformers",
                "pip install sentence-transformers",
            )

        st = self.cfg.sentence_transformers
        device = self._resolve_device(st.device)

        kwargs: dict[str, Any] = dict(
            model=st.model,
            device=device,
            normalize_embeddings=st.normalize_embeddings,
        )
        if st.query_instruction:
            kwargs["prefix"] = st.query_instruction
        if st.cache_dir:
            kwargs["cache_folder"] = str(st.cache_dir)

        return SentenceTransformersTextEmbedder(**kwargs)

    # ------------------------------------------------------------------
    # Cohere
    # ------------------------------------------------------------------

    def _cohere_doc_embedder(self) -> Any:
        try:
            from haystack_integrations.components.embedders.cohere import (
                CohereDocumentEmbedder,
            )
        except ImportError:
            self._missing_package(
                "cohere-haystack",
                "pip install cohere-haystack",
            )

        return CohereDocumentEmbedder(
            api_key=self._env("COHERE_API_KEY"),
            model="embed-english-v3.0",
            input_type="search_document",
        )

    def _cohere_text_embedder(self) -> Any:
        try:
            from haystack_integrations.components.embedders.cohere import (
                CohereTextEmbedder,
            )
        except ImportError:
            self._missing_package("cohere-haystack", "pip install cohere-haystack")

        return CohereTextEmbedder(
            api_key=self._env("COHERE_API_KEY"),
            model="embed-english-v3.0",
            input_type="search_query",
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _dispatch(self, builders: dict, label: str) -> Any:
        builder = builders.get(self.cfg.provider)
        if builder is None:
            raise ValueError(
                f"Unsupported embedding provider for {label}: '{self.cfg.provider}'. "
                f"Valid providers: {[p.value for p in EmbeddingProvider]}"
            )
        return builder()

    @staticmethod
    def _env(var_name: str) -> str:
        """Read an env var; raise a clear error if missing."""
        value = os.environ.get(var_name)
        if not value:
            raise EnvironmentError(
                f"Required environment variable '{var_name}' is not set. "
                f"Add it to your .env file or export it in the shell."
            )
        return value

    def _check_env(self, *var_names: str) -> None:
        missing = [v for v in var_names if not os.environ.get(v)]
        if missing:
            raise EnvironmentError(
                f"Required environment variable(s) not set: {missing}. "
                f"Add them to .env."
            )

    @staticmethod
    def _missing_package(pkg: str, install_cmd: str) -> None:
        raise ImportError(
            f"Package '{pkg}' is required but not installed. "
            f"Run: {install_cmd}"
        )

    def _st_dim(self) -> int:
        """
        Infer SentenceTransformers embedding dimension from the model name.
        Falls back to 768 for unknown models.
        """
        known_dims = {
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-m3": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-MiniLM-L12-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "intfloat/e5-large-v2": 1024,
            "intfloat/e5-base-v2": 768,
            "intfloat/multilingual-e5-large": 1024,
        }
        return known_dims.get(self.cfg.sentence_transformers.model, 768)

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
