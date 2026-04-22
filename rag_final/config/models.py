"""
Pydantic v2 configuration models.

Every YAML key has a typed, validated Python counterpart here.
Field defaults mirror the YAML defaults so the system works even if
a YAML file is partially omitted or missing.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OcrEngine(str, Enum):
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    RAPIDOCR = "rapidocr"
    SURYA = "surya"


class TableStructureMode(str, Enum):
    ACCURATE = "accurate"
    FAST = "fast"


class CaptioningBackend(str, Enum):
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    HUGGINGFACE_LOCAL = "huggingface_local"


class ExportType(str, Enum):
    DOC_CHUNKS = "doc_chunks"
    MARKDOWN = "markdown"


class InferenceDevice(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class ChunkingStrategy(str, Enum):
    DOCUMENT_AWARE = "document_aware"
    HIERARCHICAL = "hierarchical"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    WORD = "word"
    PASSAGE = "passage"
    PAGE = "page"
    LINE = "line"
    CHARACTER = "character"
    TOKEN = "token"
    MARKDOWN_HEADER = "markdown_header"
    SEMANTIC = "semantic"


class EmbeddingProvider(str, Enum):
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    OLLAMA = "ollama"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    COHERE = "cohere"


class DocumentStoreBackend(str, Enum):
    IN_MEMORY = "in_memory"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    OPENSEARCH = "opensearch"
    PGVECTOR = "pgvector"


# ---------------------------------------------------------------------------
# Docling OCR config models
# ---------------------------------------------------------------------------


class OcrOptionsConfig(BaseModel):
    lang: list[str] = Field(default=["en"])
    force_full_page_ocr: bool = False
    tesseract_extra_args: list[str] = Field(default_factory=list)
    rapidocr_config_path: Path | None = None


class TableStructureOptionsConfig(BaseModel):
    mode: TableStructureMode = TableStructureMode.ACCURATE
    do_cell_matching: bool = True


class FormatPipelineConfig(BaseModel):
    do_ocr: bool = True
    ocr_engine: OcrEngine = OcrEngine.EASYOCR
    ocr_options: OcrOptionsConfig = Field(default_factory=OcrOptionsConfig)
    do_table_structure: bool = True
    table_structure_options: TableStructureOptionsConfig = Field(
        default_factory=TableStructureOptionsConfig
    )
    generate_page_images: bool = True
    generate_picture_images: bool = True
    images_scale: float = Field(default=2.0, ge=0.5, le=4.0)
    force_full_page_ocr: bool = False


class ImageFormatConfig(BaseModel):
    do_ocr: bool = True
    ocr_engine: OcrEngine = OcrEngine.EASYOCR
    force_full_page_ocr: bool = True


# ---------------------------------------------------------------------------
# Docling enrichment config models
# ---------------------------------------------------------------------------


class OllamaCaptionConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "granite3.2-vision:latest"
    prompt: str = (
        "Describe this image from a technical document concisely and accurately. "
        "Focus on charts, diagrams, tables, equations, or any relevant visual content. "
        "Return plain text only."
    )
    timeout_seconds: int = Field(default=120, ge=5)
    fail_on_error: bool = False


class AzureOpenAICaptionConfig(BaseModel):
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    deployment_name_env: str = "AZURE_OPENAI_VISION_DEPLOYMENT"
    api_version: str = "2024-10-01-preview"
    prompt: str = (
        "Describe this image from a technical document concisely and accurately. "
        "Focus on charts, diagrams, tables, equations, or any relevant visual content. "
        "Return plain text only."
    )
    max_tokens: int = Field(default=512, ge=64, le=4096)
    timeout_seconds: int = Field(default=60, ge=5)
    fail_on_error: bool = True


class OpenAICompatibleCaptionConfig(BaseModel):
    base_url_env: str = "VLM_BASE_URL"
    api_key_env: str = "VLM_API_KEY"
    model: str = "gpt-4o"
    prompt: str = "Describe this image from a technical document concisely and accurately."
    max_tokens: int = Field(default=512, ge=64, le=4096)
    timeout_seconds: int = Field(default=60, ge=5)
    fail_on_error: bool = True


class PictureDescriptionConfig(BaseModel):
    enabled: bool = False
    backend: CaptioningBackend = CaptioningBackend.OLLAMA
    ollama: OllamaCaptionConfig = Field(default_factory=OllamaCaptionConfig)
    azure_openai: AzureOpenAICaptionConfig = Field(default_factory=AzureOpenAICaptionConfig)
    openai_compatible: OpenAICompatibleCaptionConfig = Field(
        default_factory=OpenAICompatibleCaptionConfig
    )


class EnrichmentsConfig(BaseModel):
    picture_description: PictureDescriptionConfig = Field(
        default_factory=PictureDescriptionConfig
    )
    picture_classification: dict[str, Any] = Field(default={"enabled": False})
    code_understanding: dict[str, Any] = Field(default={"enabled": False})
    formula_understanding: dict[str, Any] = Field(default={"enabled": False})

    @property
    def picture_classification_enabled(self) -> bool:
        return bool(self.picture_classification.get("enabled", False))

    @property
    def code_understanding_enabled(self) -> bool:
        return bool(self.code_understanding.get("enabled", False))

    @property
    def formula_understanding_enabled(self) -> bool:
        return bool(self.formula_understanding.get("enabled", False))


class MarkdownExportOptions(BaseModel):
    image_mode: str = "placeholder"
    add_page_break: bool = True


class ExportConfig(BaseModel):
    type: ExportType = ExportType.DOC_CHUNKS
    markdown_options: MarkdownExportOptions = Field(default_factory=MarkdownExportOptions)


class PerformanceConfig(BaseModel):
    num_threads: int = Field(default=0, ge=0)
    device: InferenceDevice = InferenceDevice.AUTO
    batch_size: int = Field(default=4, ge=1, le=64)
    timeout_per_document_seconds: int = Field(default=300, ge=10)


class DoclingConfig(BaseModel):
    pdf: FormatPipelineConfig = Field(default_factory=FormatPipelineConfig)
    docx: FormatPipelineConfig = Field(
        default_factory=lambda: FormatPipelineConfig(do_ocr=False, generate_page_images=False)
    )
    html: FormatPipelineConfig = Field(
        default_factory=lambda: FormatPipelineConfig(do_ocr=False, generate_page_images=False)
    )
    pptx: FormatPipelineConfig = Field(
        default_factory=lambda: FormatPipelineConfig(do_ocr=False)
    )
    image: ImageFormatConfig = Field(default_factory=ImageFormatConfig)
    enrichments: EnrichmentsConfig = Field(default_factory=EnrichmentsConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


# ---------------------------------------------------------------------------
# Chunking config models
# ---------------------------------------------------------------------------


class DocumentAwareConfig(BaseModel):
    tokenizer: str = "BAAI/bge-small-en-v1.5"
    max_tokens: int = Field(default=512, ge=32, le=8192)
    merge_peers: bool = True
    include_headings_in_metadata: bool = True


class HierarchicalConfig(BaseModel):
    block_sizes: list[int] = Field(default=[512, 128, 32])
    split_by: str = "word"
    split_overlap: int = Field(default=0, ge=0)

    @field_validator("block_sizes")
    @classmethod
    def sizes_must_be_descending(cls, v: list[int]) -> list[int]:
        if v != sorted(v, reverse=True):
            raise ValueError("block_sizes must be in descending order, e.g. [512, 128, 32]")
        if len(v) < 2:
            raise ValueError("block_sizes must have at least 2 levels")
        return v


class RecursiveConfig(BaseModel):
    split_length: int = Field(default=512, ge=1)
    split_overlap: int = Field(default=64, ge=0)
    split_unit: str = "word"
    separators: list[str] = Field(default=["\n\n", "sentence", "\n", " "])

    @field_validator("split_unit")
    @classmethod
    def valid_unit(cls, v: str) -> str:
        if v not in {"word", "char", "token"}:
            raise ValueError(f"split_unit must be word|char|token, got '{v}'")
        return v


class SimpleDocSplitterConfig(BaseModel):
    split_length: int = Field(default=512, ge=1)
    split_overlap: int = Field(default=64, ge=0)
    split_threshold: int = Field(default=10, ge=0)


class MarkdownHeaderConfig(BaseModel):
    keep_headers: bool = True
    secondary_split: str | None = None
    split_length: int = Field(default=300, ge=1)
    split_overlap: int = Field(default=0, ge=0)
    split_threshold: int = Field(default=0, ge=0)
    skip_empty_documents: bool = True

    @field_validator("secondary_split")
    @classmethod
    def valid_secondary(cls, v: str | None) -> str | None:
        valid = {None, "word", "passage", "period", "line"}
        if v not in valid:
            raise ValueError(f"secondary_split must be one of {valid}, got '{v}'")
        return v


class SemanticConfig(BaseModel):
    sentences_per_group: int = Field(default=3, ge=1)
    percentile: float = Field(default=0.95, ge=0.0, le=1.0)
    min_length: int = Field(default=50, ge=1)
    max_length: int = Field(default=1024, ge=100)


class CleanerConfig(BaseModel):
    remove_empty_lines: bool = True
    remove_extra_whitespaces: bool = True
    remove_repeated_substrings: bool = False
    min_content_length: int = Field(default=20, ge=0)


class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy = ChunkingStrategy.DOCUMENT_AWARE
    split_length: int = Field(default=512, ge=1)
    split_overlap: int = Field(default=64, ge=0)
    split_threshold: int = Field(default=10, ge=0)

    document_aware: DocumentAwareConfig = Field(default_factory=DocumentAwareConfig)
    hierarchical: HierarchicalConfig = Field(default_factory=HierarchicalConfig)
    recursive: RecursiveConfig = Field(default_factory=RecursiveConfig)
    sentence: SimpleDocSplitterConfig = Field(
        default_factory=lambda: SimpleDocSplitterConfig(split_length=10, split_overlap=2, split_threshold=1)
    )
    word: SimpleDocSplitterConfig = Field(
        default_factory=lambda: SimpleDocSplitterConfig(split_length=300, split_overlap=50)
    )
    passage: SimpleDocSplitterConfig = Field(
        default_factory=lambda: SimpleDocSplitterConfig(split_length=5, split_overlap=1, split_threshold=1)
    )
    page: SimpleDocSplitterConfig = Field(
        default_factory=lambda: SimpleDocSplitterConfig(split_length=1, split_overlap=0, split_threshold=0)
    )
    line: SimpleDocSplitterConfig = Field(
        default_factory=lambda: SimpleDocSplitterConfig(split_length=20, split_overlap=2, split_threshold=1)
    )
    character: RecursiveConfig = Field(
        default_factory=lambda: RecursiveConfig(split_length=1500, split_overlap=200, split_unit="char")
    )
    token: RecursiveConfig = Field(
        default_factory=lambda: RecursiveConfig(split_length=512, split_overlap=64, split_unit="token")
    )
    markdown_header: MarkdownHeaderConfig = Field(default_factory=MarkdownHeaderConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    cleaner: CleanerConfig = Field(default_factory=CleanerConfig)

    @model_validator(mode="after")
    def semantic_needs_embedder_warning(self) -> "ChunkingConfig":
        # Purely informational — actual check happens at pipeline build time.
        return self


# ---------------------------------------------------------------------------
# Embedding config models
# ---------------------------------------------------------------------------


class AzureOpenAIEmbeddingConfig(BaseModel):
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    azure_endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    azure_deployment: str = "text-embedding-3-large"
    api_version: str = "2024-10-01-preview"
    dimensions: int = Field(default=3072, ge=64)
    batch_size: int = Field(default=512, ge=1)
    max_retries: int = Field(default=3, ge=0)


class OpenAIEmbeddingConfig(BaseModel):
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "text-embedding-3-large"
    dimensions: int = Field(default=3072, ge=64)
    batch_size: int = Field(default=512, ge=1)
    max_retries: int = Field(default=3, ge=0)
    organization_env: str | None = None


class OllamaEmbeddingConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    dimensions: int = Field(default=768, ge=64)
    batch_size: int = Field(default=32, ge=1)
    timeout: int = Field(default=120, ge=5)


class SentenceTransformersConfig(BaseModel):
    model: str = "BAAI/bge-large-en-v1.5"
    device: str = "auto"
    batch_size: int = Field(default=32, ge=1)
    normalize_embeddings: bool = True
    cache_dir: Path | None = None
    query_instruction: str = ""
    document_instruction: str = ""


class EmbeddingConfig(BaseModel):
    provider: EmbeddingProvider = EmbeddingProvider.AZURE_OPENAI
    azure_openai: AzureOpenAIEmbeddingConfig = Field(default_factory=AzureOpenAIEmbeddingConfig)
    openai: OpenAIEmbeddingConfig = Field(default_factory=OpenAIEmbeddingConfig)
    ollama: OllamaEmbeddingConfig = Field(default_factory=OllamaEmbeddingConfig)
    sentence_transformers: SentenceTransformersConfig = Field(
        default_factory=SentenceTransformersConfig
    )


# ---------------------------------------------------------------------------
# Document store config models
# ---------------------------------------------------------------------------


class InMemoryStoreConfig(BaseModel):
    bm25_retrieval: bool = True


class QdrantStoreConfig(BaseModel):
    url: str = "http://localhost:6333"
    collection_name: str = "rag_documents"
    embedding_dim: int = Field(default=3072, ge=64)
    recreate_index: bool = False
    api_key_env: str | None = None


class WeaviateStoreConfig(BaseModel):
    url: str = "http://localhost:8080"
    class_name: str = "RagDocument"


class OpenSearchStoreConfig(BaseModel):
    hosts: list[str] = Field(default=["http://localhost:9200"])
    index_name: str = "rag_documents"
    embedding_field: str = "embedding"
    username_env: str | None = None
    password_env: str | None = None


class PgVectorStoreConfig(BaseModel):
    connection_string_env: str = "PGVECTOR_CONNECTION_STRING"
    table_name: str = "haystack_documents"
    embedding_dim: int = Field(default=3072, ge=64)


class DocumentStoreConfig(BaseModel):
    backend: DocumentStoreBackend = DocumentStoreBackend.IN_MEMORY
    in_memory: InMemoryStoreConfig = Field(default_factory=InMemoryStoreConfig)
    qdrant: QdrantStoreConfig = Field(default_factory=QdrantStoreConfig)
    weaviate: WeaviateStoreConfig = Field(default_factory=WeaviateStoreConfig)
    opensearch: OpenSearchStoreConfig = Field(default_factory=OpenSearchStoreConfig)
    pgvector: PgVectorStoreConfig = Field(default_factory=PgVectorStoreConfig)


# ---------------------------------------------------------------------------
# Query / RAG config models
# ---------------------------------------------------------------------------


class SearchType(str, Enum):
    EMBEDDING = "embedding"
    BM25 = "bm25"
    HYBRID = "hybrid"


class HybridSearchConfig(BaseModel):
    embedding_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    join_mode: str = "reciprocal_rank_fusion"

    @field_validator("bm25_weight")
    @classmethod
    def weights_sum_to_one(cls, v: float, info: Any) -> float:
        emb = info.data.get("embedding_weight", 0.7)
        if abs(emb + v - 1.0) > 1e-6:
            raise ValueError(
                f"embedding_weight + bm25_weight must equal 1.0, "
                f"got {emb} + {v} = {emb + v}"
            )
        return v


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=100)
    search_type: SearchType = SearchType.EMBEDDING
    hybrid: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    default_filters: dict[str, Any] | None = None


class AutoMergingConfig(BaseModel):
    enabled: bool = True
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class RerankerStrategy(str, Enum):
    NONE = "none"
    LOST_IN_MIDDLE = "lost_in_middle"
    LLM = "llm"


class LLMRankerConfig(BaseModel):
    top_k: int = Field(default=10, ge=1)
    meta_fields: list[str] = Field(default=["source_file", "page_number", "headings"])


class RerankerConfig(BaseModel):
    strategy: RerankerStrategy = RerankerStrategy.LOST_IN_MIDDLE
    llm_ranker: LLMRankerConfig = Field(default_factory=LLMRankerConfig)


class PromptConfig(BaseModel):
    system_message: str = (
        "You are a precise and factual assistant. Answer the user's question "
        "based strictly on the provided context documents."
    )
    template: str = (
        "Context documents:\n"
        "{% for doc in documents %}\n"
        "---\n"
        "Source: {{ doc.meta.get('source_file', 'unknown') }}\n"
        "{{ doc.content }}\n"
        "{% endfor %}\n"
        "---\n\n"
        "Question: {{ query }}\n\n"
        "Answer:"
    )
    max_context_docs: int = Field(default=5, ge=1)
    max_chars_per_doc: int = Field(default=2000, ge=100)


class GeneratorBackend(str, Enum):
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    OLLAMA = "ollama"


class AzureOpenAIGeneratorConfig(BaseModel):
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    azure_endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    deployment_name_env: str = "AZURE_OPENAI_CHAT_DEPLOYMENT"
    api_version: str = "2024-10-01-preview"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=64)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=0)
    streaming: bool = False


class OpenAIGeneratorConfig(BaseModel):
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=64)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    streaming: bool = False


class OllamaGeneratorConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    context_window: int = Field(default=8192, ge=512)
    streaming: bool = False


class GeneratorConfig(BaseModel):
    backend: GeneratorBackend = GeneratorBackend.AZURE_OPENAI
    azure_openai: AzureOpenAIGeneratorConfig = Field(
        default_factory=AzureOpenAIGeneratorConfig
    )
    openai: OpenAIGeneratorConfig = Field(default_factory=OpenAIGeneratorConfig)
    ollama: OllamaGeneratorConfig = Field(default_factory=OllamaGeneratorConfig)


class AnswerConfig(BaseModel):
    include_sources: bool = True
    max_sources: int = Field(default=5, ge=1)
    include_scores: bool = True


class QueryConfig(BaseModel):
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    auto_merging: AutoMergingConfig = Field(default_factory=AutoMergingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
