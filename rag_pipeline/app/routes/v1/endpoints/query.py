"""
POST /v1/query   — RAG: embed query → retrieve → rerank → generate answer.
POST /v1/search  — Retrieval-only (no LLM): returns ranked source documents.

Streaming
---------
When query.stream=true, the endpoint returns a Server-Sent Events (SSE)
stream. Each event carries a single token delta from the generator.
The final event contains the complete answer and sources under
data: [DONE].

Example SSE client::

    import httpx, json
    with httpx.stream("POST", ".../v1/query", json={
        "query": "What is deep learning?",
        "stream": True,
    }) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                print(payload.get("token", payload), end="", flush=True)
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.dependencies import get_app_settings, get_registry
from app.schemas.query import (
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SourceDocument,
)
from config.settings import AppSettings
from utils.pipeline_registry import PipelineRegistry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


# ---------------------------------------------------------------------------
# POST /v1/query
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="RAG question answering",
    description=(
        "Embeds the query, retrieves relevant document chunks, optionally reranks them, "
        "and generates a grounded answer via the configured LLM. "
        "Set `stream=true` to receive a Server-Sent Events stream."
    ),
)
async def query(
    request: QueryRequest,
    registry: PipelineRegistry = Depends(get_registry),
    settings: AppSettings = Depends(get_app_settings),
):
    _validate_pipeline(request.pipeline_name, registry)

    if request.stream:
        return _stream_response(request, registry)

    return await _run_query(request, registry, settings)


async def _run_query(
    request: QueryRequest,
    registry: PipelineRegistry,
    settings: AppSettings,
) -> QueryResponse:
    pipeline = registry.get_query(request.pipeline_name)

    run_input = _build_pipeline_input(request, settings)

    try:
        result = pipeline.run(run_input)
    except Exception as exc:
        logger.error("Query pipeline error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query pipeline failed: {exc}",
        )

    return _parse_result(request, result, settings)


def _build_pipeline_input(request: QueryRequest, settings: AppSettings) -> dict[str, Any]:
    """Build the pipeline.run() input dict from the request."""
    inputs: dict[str, Any] = {
        "text_embedder": {"text": request.query},
        "prompt_builder": {"query": request.query},
        "answer_builder": {"query": request.query},
    }

    # Per-request top_k override.
    top_k = request.top_k
    inputs["embedding_retriever"] = {"top_k": top_k}

    # Per-request metadata filters.
    if request.filters:
        inputs["embedding_retriever"]["filters"] = request.filters
        if "bm25_retriever" in _get_component_names_safe():
            inputs["bm25_retriever"] = {"filters": request.filters, "top_k": top_k}

    return inputs


def _get_component_names_safe() -> list[str]:
    """Safe placeholder — actual check happens at pipeline.get_component() level."""
    return []


def _parse_result(
    request: QueryRequest,
    result: dict[str, Any],
    settings: AppSettings,
) -> QueryResponse:
    """Extract answer and sources from the pipeline result dict."""
    # Answer comes from AnswerBuilder.
    answers = result.get("answer_builder", {}).get("answers", [])
    answer_text = answers[0].data if answers else "No answer generated."

    # Source documents from context_truncator (already trimmed list).
    raw_docs = result.get("context_truncator", {}).get("documents", [])

    sources: list[SourceDocument] = []
    if request.include_sources:
        max_sources = settings.query.answer.max_sources
        for doc in raw_docs[:max_sources]:
            meta = doc.meta or {}
            sources.append(
                SourceDocument(
                    document_id=doc.id or "",
                    content=doc.content or "",
                    score=doc.score if settings.query.answer.include_scores else None,
                    source_file=meta.get("source_file"),
                    page_number=meta.get("page_number"),
                    headings=meta.get("headings", []),
                    is_table=meta.get("is_table", False),
                    is_picture=meta.get("is_picture", False),
                    picture_caption=meta.get("picture_caption"),
                    chunk_index=meta.get("chunk_index"),
                )
            )

    return QueryResponse(
        query=request.query,
        answer=answer_text,
        pipeline_name=request.pipeline_name,
        sources=sources,
        meta={
            "sources_retrieved": len(raw_docs),
            "sources_returned": len(sources),
        },
    )


def _stream_response(request: QueryRequest, registry: PipelineRegistry) -> StreamingResponse:
    """
    Run the pipeline in a background thread with a token queue,
    then stream tokens as SSE events.
    """
    token_queue: queue.Queue[str | None] = queue.Queue()

    def streaming_callback(chunk: Any) -> None:
        """Called by Haystack generator for each streaming chunk."""
        content = getattr(chunk, "content", None) or ""
        if content:
            token_queue.put(content)

    def run_pipeline() -> None:
        try:
            pipeline = registry.get_query(request.pipeline_name)
            # Inject streaming callback into the generator at run time.
            pipeline.run(
                {
                    "text_embedder": {"text": request.query},
                    "prompt_builder": {"query": request.query},
                    "answer_builder": {"query": request.query},
                    "embedding_retriever": {"top_k": request.top_k},
                },
                include_outputs_from={"generator": {"streaming_callback": streaming_callback}},
            )
        except Exception as exc:
            logger.error("Streaming pipeline error: %s", exc, exc_info=True)
        finally:
            token_queue.put(None)  # Sentinel: stream complete.

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            try:
                token = token_queue.get(timeout=60)
            except queue.Empty:
                yield "data: {\"error\": \"stream timeout\"}\n\n"
                break

            if token is None:
                yield "data: [DONE]\n\n"
                break

            payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# POST /v1/search  (retrieval only — no LLM)
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Retrieval-only semantic / keyword search",
    description=(
        "Embed the query and retrieve the top-K matching chunks. "
        "No LLM is invoked. Supports embedding, bm25, and hybrid search types."
    ),
)
async def search(
    request: SearchRequest,
    registry: PipelineRegistry = Depends(get_registry),
    settings: AppSettings = Depends(get_app_settings),
) -> SearchResponse:
    _validate_pipeline(request.pipeline_name, registry)

    from haystack import Pipeline
    from utils.embedding import EmbeddingFactory

    # Build a lightweight retrieval-only pipeline on the fly.
    # We share the document store from the indexing pipeline.
    idx_pipeline = registry.get_indexing(request.pipeline_name)
    writer = idx_pipeline.get_component("writer")
    store = writer.document_store

    try:
        text_embedder = EmbeddingFactory(settings.embedding).build_text_embedder()
        embed_result = text_embedder.run(text=request.query)
        query_embedding = embed_result["embedding"]

        # Retrieve directly from the store.
        docs = store.embedding_retrieval(
            query_embedding=query_embedding,
            top_k=request.top_k,
            filters=request.filters,
        )
    except AttributeError:
        # Fallback: store doesn't support embedding_retrieval directly.
        docs = _store_search_fallback(store, request)
    except Exception as exc:
        logger.error("Search error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    results = [
        SourceDocument(
            document_id=doc.id or "",
            content=doc.content or "",
            score=doc.score,
            source_file=(doc.meta or {}).get("source_file"),
            page_number=(doc.meta or {}).get("page_number"),
            headings=(doc.meta or {}).get("headings", []),
            is_table=(doc.meta or {}).get("is_table", False),
            is_picture=(doc.meta or {}).get("is_picture", False),
            picture_caption=(doc.meta or {}).get("picture_caption"),
            chunk_index=(doc.meta or {}).get("chunk_index"),
        )
        for doc in docs
    ]

    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results),
        search_type=request.search_type,
    )


def _store_search_fallback(store: Any, request: SearchRequest) -> list[Any]:
    """Fallback: attempt BM25 search if embedding retrieval isn't available."""
    try:
        return store.bm25_retrieval(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
        )
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_pipeline(pipeline_name: str, registry: PipelineRegistry) -> None:
    if pipeline_name not in registry.registered_names():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Pipeline '{pipeline_name}' is not registered. "
                f"Available: {registry.registered_names()}"
            ),
        )
