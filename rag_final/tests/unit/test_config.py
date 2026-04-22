"""Unit tests for configuration loading and Pydantic model validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


class TestDoclingConfig:
    def test_defaults_are_valid(self):
        from config.models import DoclingConfig

        cfg = DoclingConfig()
        assert cfg.pdf.do_ocr is True
        assert cfg.pdf.do_table_structure is True
        assert cfg.enrichments.picture_description.enabled is False

    def test_ocr_engine_enum_validation(self):
        from config.models import DoclingConfig, OcrEngine

        cfg = DoclingConfig.model_validate(
            {"pdf": {"ocr_engine": "tesseract"}}
        )
        assert cfg.pdf.ocr_engine == OcrEngine.TESSERACT

    def test_invalid_ocr_engine_raises(self):
        from config.models import DoclingConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="ocr_engine"):
            DoclingConfig.model_validate({"pdf": {"ocr_engine": "nonexistent"}})

    def test_images_scale_bounds(self):
        from config.models import DoclingConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DoclingConfig.model_validate({"pdf": {"images_scale": 0.1}})

        with pytest.raises(ValidationError):
            DoclingConfig.model_validate({"pdf": {"images_scale": 10.0}})

    def test_export_type_doc_chunks(self):
        from config.models import DoclingConfig, ExportType

        cfg = DoclingConfig.model_validate({"export": {"type": "doc_chunks"}})
        assert cfg.export.type == ExportType.DOC_CHUNKS

    def test_export_type_markdown(self):
        from config.models import DoclingConfig, ExportType

        cfg = DoclingConfig.model_validate({"export": {"type": "markdown"}})
        assert cfg.export.type == ExportType.MARKDOWN


class TestChunkingConfig:
    def test_all_strategies_parse(self):
        from config.models import ChunkingConfig, ChunkingStrategy

        for strategy in ChunkingStrategy:
            cfg = ChunkingConfig.model_validate({"strategy": strategy.value})
            assert cfg.strategy == strategy

    def test_hierarchical_block_sizes_descending(self):
        from config.models import ChunkingConfig
        from pydantic import ValidationError

        # Must be descending.
        with pytest.raises(ValidationError, match="descending"):
            ChunkingConfig.model_validate(
                {"hierarchical": {"block_sizes": [32, 128, 512]}}
            )

    def test_hierarchical_block_sizes_need_two_levels(self):
        from config.models import ChunkingConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="2 levels"):
            ChunkingConfig.model_validate(
                {"hierarchical": {"block_sizes": [512]}}
            )

    def test_recursive_invalid_split_unit(self):
        from config.models import ChunkingConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="split_unit"):
            ChunkingConfig.model_validate(
                {"recursive": {"split_unit": "banana"}}
            )

    def test_markdown_header_invalid_secondary_split(self):
        from config.models import ChunkingConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="secondary_split"):
            ChunkingConfig.model_validate(
                {"markdown_header": {"secondary_split": "invalid"}}
            )

    def test_split_length_must_be_positive(self):
        from config.models import ChunkingConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChunkingConfig.model_validate({"split_length": -1})

    def test_semantic_percentile_bounds(self):
        from config.models import ChunkingConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChunkingConfig.model_validate({"semantic": {"percentile": 1.5}})

        with pytest.raises(ValidationError):
            ChunkingConfig.model_validate({"semantic": {"percentile": -0.1}})


class TestEmbeddingConfig:
    def test_defaults_use_azure_openai(self):
        from config.models import EmbeddingConfig, EmbeddingProvider

        cfg = EmbeddingConfig()
        assert cfg.provider == EmbeddingProvider.AZURE_OPENAI

    def test_all_providers_parse(self):
        from config.models import EmbeddingConfig, EmbeddingProvider

        for provider in EmbeddingProvider:
            cfg = EmbeddingConfig.model_validate({"provider": provider.value})
            assert cfg.provider == provider

    def test_batch_size_positive(self):
        from config.models import EmbeddingConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EmbeddingConfig.model_validate(
                {"azure_openai": {"batch_size": 0}}
            )


class TestAppSettings:
    def test_loads_yaml_files(self, settings):
        """AppSettings reads all 4 YAML files without errors."""
        from config.models import ChunkingStrategy, EmbeddingProvider

        assert settings.chunking.strategy == ChunkingStrategy.WORD
        assert settings.embedding.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS
        assert settings.document_store.backend.value == "in_memory"

    def test_missing_yaml_uses_defaults(self, tmp_path):
        """A missing config file falls back to Pydantic model defaults."""
        from config.settings import get_settings

        get_settings.cache_clear()
        # Only write two of the four files.
        (tmp_path / "chunking_config.yaml").write_text(
            yaml.dump({"chunking": {"strategy": "sentence"}}),
            encoding="utf-8",
        )
        (tmp_path / "store_config.yaml").write_text(
            yaml.dump({"document_store": {"backend": "in_memory"}}),
            encoding="utf-8",
        )

        settings = get_settings(config_dir=str(tmp_path))
        # Docling config not provided → uses defaults.
        assert settings.docling.pdf.do_ocr is True
        get_settings.cache_clear()

    def test_invalid_yaml_raises_runtime_error(self, tmp_path):
        """Corrupt YAML raises RuntimeError with a helpful message."""
        from config.settings import get_settings

        get_settings.cache_clear()
        (tmp_path / "chunking_config.yaml").write_text(
            "chunking:\n  strategy: not_a_valid_strategy_xyz\n",
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="chunking_config.yaml"):
            get_settings(config_dir=str(tmp_path))
        get_settings.cache_clear()
