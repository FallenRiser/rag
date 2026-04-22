"""
Application settings.

Loads all four YAML config files at startup, validates them via
Pydantic models, and merges environment variable overrides.

Usage:
    from config.settings import get_settings
    settings = get_settings()
    print(settings.docling.pdf.ocr_engine)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.models import (
    ChunkingConfig,
    DoclingConfig,
    DocumentStoreConfig,
    EmbeddingConfig,
    QueryConfig,
)

logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    """
    Top-level application settings.

    YAML files are loaded from `config_dir`. Each file maps to one
    validated Pydantic sub-model. Environment variables override
    specific leaf values using double-underscore notation, e.g.:

        DOCLING__PDF__DO_OCR=false
        CHUNKING__STRATEGY=sentence
        EMBEDDING__PROVIDER=ollama
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        # Allow extra env vars (e.g. AZURE_OPENAI_API_KEY used inside sub-models)
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # App-level settings
    # -----------------------------------------------------------------------
    env: str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO", description="Python logging level")
    config_dir: Path = Field(
        default=Path("config"),
        description="Directory containing the four YAML config files",
    )

    # -----------------------------------------------------------------------
    # OpenTelemetry
    # -----------------------------------------------------------------------
    otel_endpoint: str | None = Field(
        default=None,
        description="OTLP gRPC endpoint, e.g. http://localhost:4317",
    )
    otel_service_name: str = Field(default="rag-pipeline")

    # -----------------------------------------------------------------------
    # Sub-configs loaded from YAML files (populated in model_post_init)
    # -----------------------------------------------------------------------
    docling: DoclingConfig = Field(default_factory=DoclingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    document_store: DocumentStoreConfig = Field(default_factory=DocumentStoreConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)

    @model_validator(mode="after")
    def load_yaml_configs(self) -> "AppSettings":
        """
        Load and validate each YAML file, then merge into the sub-config
        model. YAML values are the base; env-var overrides have already
        been applied by Pydantic before this hook runs.
        """
        yaml_files = {
            "docling": ("docling_config.yaml", DoclingConfig, "docling"),
            "chunking": ("chunking_config.yaml", ChunkingConfig, "chunking"),
            "embedding": ("embedding_config.yaml", EmbeddingConfig, "embedding"),
            "document_store": ("store_config.yaml", DocumentStoreConfig, "document_store"),
            "query": ("query_config.yaml", QueryConfig, "query"),
        }

        for attr_name, (filename, model_cls, yaml_key) in yaml_files.items():
            path = self.config_dir / filename
            if not path.exists():
                logger.warning(
                    "Config file not found: %s. Using model defaults.", path
                )
                continue

            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                if raw is None:
                    logger.warning("Config file is empty: %s", path)
                    continue
                data = raw.get(yaml_key, {})
                parsed = model_cls.model_validate(data)
                # Env var overrides (set by pydantic-settings) take precedence.
                # We only replace the attr if it still holds the default.
                current = getattr(self, attr_name)
                if current == model_cls():
                    object.__setattr__(self, attr_name, parsed)
                else:
                    # Env vars were applied; merge YAML as base, env as override.
                    merged_data = {**data, **current.model_dump(exclude_unset=True)}
                    object.__setattr__(self, attr_name, model_cls.model_validate(merged_data))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load config file %s: %s", path, exc)
                raise RuntimeError(
                    f"Invalid configuration in {path}: {exc}"
                ) from exc

        self._configure_logging()
        return self

    def _configure_logging(self) -> None:
        logging.basicConfig(
            level=self.log_level.upper(),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )


@lru_cache(maxsize=1)
def get_settings(config_dir: str | None = None) -> AppSettings:
    """
    Return a cached AppSettings instance.
    Pass `config_dir` only in tests to point at a fixture config directory.
    """
    kwargs: dict = {}
    if config_dir is not None:
        kwargs["config_dir"] = Path(config_dir)
    return AppSettings(**kwargs)
