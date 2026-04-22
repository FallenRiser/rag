"""
StoreFactory
============

Builds Haystack document store instances from ``DocumentStoreConfig``.

Supported backends
------------------
- in_memory    InMemoryDocumentStore (dev / testing)
- qdrant       QdrantDocumentStore
- weaviate     WeaviateDocumentStore
- opensearch   OpenSearchDocumentStore
- pgvector     PgvectorDocumentStore
"""

from __future__ import annotations

import logging
import os
from typing import Any

from config.models import DocumentStoreBackend, DocumentStoreConfig

logger = logging.getLogger(__name__)


class StoreFactory:
    """
    Factory that builds a Haystack document store from ``DocumentStoreConfig``.

    Usage::

        from config.settings import get_settings
        from utils.document_store import StoreFactory

        store = StoreFactory(get_settings().document_store).build()
    """

    def __init__(self, cfg: DocumentStoreConfig) -> None:
        self.cfg = cfg

    def build(self) -> Any:
        backend = self.cfg.backend
        logger.info("Building document store | backend=%s", backend.value)

        builders = {
            DocumentStoreBackend.IN_MEMORY: self._in_memory,
            DocumentStoreBackend.QDRANT: self._qdrant,
            DocumentStoreBackend.WEAVIATE: self._weaviate,
            DocumentStoreBackend.OPENSEARCH: self._opensearch,
            DocumentStoreBackend.PGVECTOR: self._pgvector,
        }

        builder = builders.get(backend)
        if builder is None:
            raise ValueError(f"Unknown document store backend: '{backend}'")

        store = builder()
        logger.info("Document store built: %s", type(store).__name__)
        return store

    # ------------------------------------------------------------------
    # Backend builders
    # ------------------------------------------------------------------

    def _in_memory(self) -> Any:
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        return InMemoryDocumentStore(
            bm25_tokenization_regex=r"(?u)\b\w\w+\b",
        )

    def _qdrant(self) -> Any:
        try:
            from haystack_integrations.document_stores.qdrant import (
                QdrantDocumentStore,
            )
        except ImportError:
            raise ImportError(
                "qdrant-haystack not installed. Run: pip install qdrant-haystack"
            )

        q = self.cfg.qdrant
        api_key = os.environ.get(q.api_key_env) if q.api_key_env else None

        return QdrantDocumentStore(
            url=q.url,
            index=q.collection_name,
            embedding_dim=q.embedding_dim,
            recreate_index=q.recreate_index,
            api_key=api_key,
        )

    def _weaviate(self) -> Any:
        try:
            from haystack_integrations.document_stores.weaviate import (
                WeaviateDocumentStore,
            )
        except ImportError:
            raise ImportError(
                "weaviate-haystack not installed. Run: pip install weaviate-haystack"
            )

        w = self.cfg.weaviate
        return WeaviateDocumentStore(url=w.url)

    def _opensearch(self) -> Any:
        try:
            from haystack_integrations.document_stores.opensearch import (
                OpenSearchDocumentStore,
            )
        except ImportError:
            raise ImportError(
                "opensearch-haystack not installed. Run: pip install opensearch-haystack"
            )

        o = self.cfg.opensearch
        auth = None
        if o.username_env and o.password_env:
            username = os.environ.get(o.username_env, "")
            password = os.environ.get(o.password_env, "")
            if username and password:
                auth = (username, password)

        return OpenSearchDocumentStore(
            hosts=o.hosts,
            index=o.index_name,
            http_auth=auth,
        )

    def _pgvector(self) -> Any:
        try:
            from haystack_integrations.document_stores.pgvector import (
                PgvectorDocumentStore,
            )
        except ImportError:
            raise ImportError(
                "pgvector-haystack not installed. Run: pip install pgvector-haystack"
            )

        pg = self.cfg.pgvector
        conn_str = os.environ.get(pg.connection_string_env)
        if not conn_str:
            raise EnvironmentError(
                f"Environment variable '{pg.connection_string_env}' is not set."
            )

        return PgvectorDocumentStore(
            connection_string=conn_str,
            table_name=pg.table_name,
            embedding_dimension=pg.embedding_dim,
        )
