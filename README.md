# RAG Pipeline

Production-grade document extraction and retrieval-augmented generation (RAG) pipeline built on **Haystack 2.26** and **Docling**.

---

## Features

| Layer | Capability |
|---|---|
| **Extraction** | PDF, DOCX, HTML, PPTX, images — all via Docling |
| **OCR** | EasyOCR (default), Tesseract, RapidOCR — auto-fallback on scanned pages |
| **Table extraction** | TableFormer → Markdown tables preserved in chunks |
| **Image captioning** | LLM-based via Azure OpenAI, Ollama (local), or any OpenAI-compatible API |
| **Chunking** | 13 strategies: document-aware, hierarchical, recursive, sentence, word, passage, page, line, character, token, markdown-header, semantic (embedding-based) |
| **Embeddings** | Azure OpenAI, OpenAI, Ollama (local), SentenceTransformers (local) |
| **Vector stores** | InMemory, Qdrant, Weaviate, OpenSearch, PgVector |
| **Retrieval** | Embedding, BM25, or hybrid (RRF) |
| **Reranking** | LostInTheMiddle, LLMRanker (Haystack 2.26) |
| **Generation** | Azure OpenAI, OpenAI, Ollama — streaming SSE supported |
| **Observability** | OpenTelemetry traces per pipeline component, structlog JSON logging |

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/your-org/rag-pipeline
cd rag-pipeline
pip install -e ".[sentence_transformers,dev]"

# 2. Configure
cp .env.example .env
# Edit .env — at minimum set AZURE_OPENAI_* or leave embedding.provider=sentence_transformers

# 3. Run
uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

---

## Configuration

All behaviour is driven by five YAML files in `config/`. Switch strategies by changing a single key:

```yaml
# config/chunking_config.yaml
chunking:
  strategy: "hierarchical"   # change this one line
```

```yaml
# config/embedding_config.yaml
embedding:
  provider: "ollama"         # azure_openai | ollama | sentence_transformers
```

```yaml
# config/query_config.yaml
query:
  retrieval:
    search_type: "hybrid"    # embedding | bm25 | hybrid
  reranker:
    strategy: "llm"          # none | lost_in_middle | llm
  generator:
    backend: "azure_openai"  # azure_openai | openai | ollama
```

Environment variables override YAML values using double-underscore notation:

```bash
CHUNKING__STRATEGY=sentence
EMBEDDING__PROVIDER=sentence_transformers
QUERY__GENERATOR__BACKEND=ollama
```

---

## API

### Ingest documents

```bash
curl -X POST http://localhost:8000/v1/ingest \
  -F "files=@report.pdf" \
  -F "files=@slides.pptx" \
  -F "pipeline_name=default"
# → { "job_id": "...", "status": "pending", "files_received": 2 }

# Poll status
curl http://localhost:8000/v1/ingest/{job_id}
```

### Query (RAG)

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "top_k": 5,
    "include_sources": true
  }'
```

### Query with streaming

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise the report.", "stream": true}'
# Receives Server-Sent Events
```

### Semantic search (no LLM)

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "deep learning", "top_k": 10, "search_type": "hybrid"}'
```

---

## Chunking strategies

| Strategy | Component | Best for |
|---|---|---|
| `document_aware` | Docling `HybridChunker` | Structured docs (papers, reports) |
| `hierarchical` | `HierarchicalDocumentSplitter` | Long docs + auto-merge retrieval |
| `recursive` | `RecursiveDocumentSplitter` | General text, code |
| `sentence` | `DocumentSplitter` | Short QA, chatbots |
| `word` | `DocumentSplitter` | Simple chunking |
| `passage` | `DocumentSplitter` | Paragraph-level |
| `page` | `DocumentSplitter` | Page-scoped retrieval |
| `line` | `DocumentSplitter` | Line-by-line (logs, data) |
| `character` | `RecursiveDocumentSplitter` | Character budgets |
| `token` | `RecursiveDocumentSplitter` | Token-exact budgets |
| `markdown_header` | `MarkdownHeaderSplitter` | Markdown / wiki exports |
| `semantic` | `EmbeddingBasedDocumentSplitter` | Topic-coherent chunks |

---

## Image captioning setup

**Ollama (local, free):**
```bash
ollama pull granite3.2-vision   # or llava, moondream
# Set in docling_config.yaml:
# enrichments.picture_description.backend: "ollama"
# enrichments.picture_description.ollama.model: "granite3.2-vision:latest"
```

**Azure OpenAI (GPT-4o vision):**
```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_VISION_DEPLOYMENT=gpt-4o
# Set in docling_config.yaml:
# enrichments.picture_description.backend: "azure_openai"
```

---

## Docker

```bash
docker compose up -d          # starts API + Qdrant
docker compose logs -f api
```

Switch to Qdrant in `store_config.yaml` and set `QDRANT_API_KEY` for cloud.

---

## Tests

```bash
# Unit tests only (fast, no network)
pytest tests/unit/ -v

# Skip tests needing model weights
pytest -m "not slow" -v

# All tests with coverage
pytest --cov --cov-report=term-missing
```

---

## Project structure

```
config/              ← YAML config files + Pydantic models
utils/
  docling_pipeline.py   ← DoclingPipelineBuilder
  chunking.py           ← ChunkingFactory (13 strategies)
  embedding.py          ← EmbeddingFactory (4 providers)
  document_store.py     ← StoreFactory (5 backends)
  indexing_pipeline.py  ← IndexingPipelineBuilder
  query_pipeline.py     ← QueryPipelineBuilder + ContextTruncator
  pipeline_registry.py  ← Thread-safe lazy pipeline cache
  metadata_enricher.py  ← Custom Haystack @component
  tracing.py            ← OTel + structlog
app/
  main.py               ← FastAPI application factory
  routes/v1/endpoints/  ← ingest, query, health, pipelines
  schemas/              ← Pydantic request/response models
tests/
  unit/                 ← Config, chunking, query unit tests
  integration/          ← Pipeline + API endpoint tests
```
