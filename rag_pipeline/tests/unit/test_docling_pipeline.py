"""
Unit tests for DoclingPipelineBuilder.

These tests cover config → options translation without requiring the full
Docling model weights to be downloaded. They use monkeypatching to stub
Docling's import-time heavy dependencies.
"""

from __future__ import annotations

import pytest


class TestOcrOptionsBuilder:
    """Test OCR options construction from config."""

    def test_easyocr_options_built(self, settings):
        """EasyOCR options should be constructed from config langs."""
        from config.models import FormatPipelineConfig, OcrEngine

        cfg = FormatPipelineConfig(
            do_ocr=True,
            ocr_engine=OcrEngine.EASYOCR,
        )
        cfg.ocr_options.lang = ["en", "de"]

        try:
            from utils.docling_pipeline import _build_easyocr_options

            opts = _build_easyocr_options(cfg)
            # If docling is installed, verify the object.
            assert opts is not None
        except ImportError:
            pytest.skip("docling not installed")

    def test_tesseract_lang_mapping(self, settings):
        """Tesseract builder must translate 2-letter ISO codes to 3-letter codes."""
        from config.models import FormatPipelineConfig, OcrEngine

        cfg = FormatPipelineConfig(
            do_ocr=True,
            ocr_engine=OcrEngine.TESSERACT,
        )
        cfg.ocr_options.lang = ["en", "de", "fr"]

        try:
            from utils.docling_pipeline import _build_tesseract_options

            opts = _build_tesseract_options(cfg)
            # Tesseract lang string should be '+'-separated 3-letter codes.
            assert "eng" in opts.lang
            assert "deu" in opts.lang
            assert "fra" in opts.lang
        except ImportError:
            pytest.skip("docling not installed")

    def test_tesseract_unknown_lang_passes_through(self, settings):
        """Unknown language codes should be forwarded unchanged."""
        from config.models import FormatPipelineConfig, OcrEngine

        cfg = FormatPipelineConfig(
            do_ocr=True,
            ocr_engine=OcrEngine.TESSERACT,
        )
        cfg.ocr_options.lang = ["xyz_custom"]

        try:
            from utils.docling_pipeline import _build_tesseract_options

            opts = _build_tesseract_options(cfg)
            assert "xyz_custom" in opts.lang
        except ImportError:
            pytest.skip("docling not installed")


class TestCaptioningBackendBuilder:
    """Test image captioning enricher construction."""

    def test_missing_azure_env_vars_raise_when_fail_on_error(
        self, monkeypatch, settings
    ):
        """Azure backend with fail_on_error=True should raise when env vars missing."""
        from config.models import AzureOpenAICaptionConfig

        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_VISION_DEPLOYMENT", raising=False)

        cfg = AzureOpenAICaptionConfig(fail_on_error=True)

        try:
            from utils.docling_pipeline import _build_azure_openai_captioner

            with pytest.raises(EnvironmentError, match="missing"):
                _build_azure_openai_captioner(cfg)
        except ImportError:
            pytest.skip("docling not installed")

    def test_missing_azure_env_vars_return_none_when_not_fail(
        self, monkeypatch, settings
    ):
        """Azure backend with fail_on_error=False should return None when env vars missing."""
        from config.models import AzureOpenAICaptionConfig

        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_VISION_DEPLOYMENT", raising=False)

        cfg = AzureOpenAICaptionConfig(fail_on_error=False)

        try:
            from utils.docling_pipeline import _build_azure_openai_captioner

            result = _build_azure_openai_captioner(cfg)
            assert result is None
        except ImportError:
            pytest.skip("docling not installed")

    def test_openai_compatible_missing_env_returns_none(self, monkeypatch, settings):
        """OpenAI-compatible backend with fail_on_error=False returns None if env missing."""
        from config.models import OpenAICompatibleCaptionConfig

        monkeypatch.delenv("VLM_BASE_URL", raising=False)
        monkeypatch.delenv("VLM_API_KEY", raising=False)

        cfg = OpenAICompatibleCaptionConfig(fail_on_error=False)

        try:
            from utils.docling_pipeline import _build_openai_compatible_captioner

            result = _build_openai_compatible_captioner(cfg)
            assert result is None
        except ImportError:
            pytest.skip("docling not installed")


class TestDeviceResolver:
    def test_cpu_returned_when_specified(self):
        from config.models import InferenceDevice
        from utils.docling_pipeline import _resolve_device

        assert _resolve_device(InferenceDevice.CPU) == "cpu"

    def test_cuda_returned_when_specified(self):
        from config.models import InferenceDevice
        from utils.docling_pipeline import _resolve_device

        assert _resolve_device(InferenceDevice.CUDA) == "cuda"

    def test_auto_returns_valid_device(self):
        from config.models import InferenceDevice
        from utils.docling_pipeline import _resolve_device

        device = _resolve_device(InferenceDevice.AUTO)
        assert device in {"cpu", "cuda", "mps"}


class TestDoclingPipelineBuilderIntegration:
    """Integration-level tests that require docling to be installed."""

    def test_build_returns_docling_converter(self, settings):
        try:
            from docling_haystack.converter import DoclingConverter
        except ImportError:
            pytest.skip("docling-haystack not installed")

        from utils.docling_pipeline import DoclingPipelineBuilder

        builder = DoclingPipelineBuilder(settings.docling)
        converter = builder.build()
        assert isinstance(converter, DoclingConverter)

    def test_enrichment_summary_no_enrichments(self, settings):
        from utils.docling_pipeline import DoclingPipelineBuilder

        builder = DoclingPipelineBuilder(settings.docling)
        summary = builder._enrichment_summary()
        assert summary == "none"

    def test_enrichment_summary_with_captioning(self, settings):
        import copy

        from config.models import CaptioningBackend
        from utils.docling_pipeline import DoclingPipelineBuilder

        cfg = copy.deepcopy(settings.docling)
        cfg.enrichments.picture_description.enabled = True
        cfg.enrichments.picture_description.backend = CaptioningBackend.OLLAMA

        builder = DoclingPipelineBuilder(cfg)
        summary = builder._enrichment_summary()
        assert "caption(ollama)" in summary
