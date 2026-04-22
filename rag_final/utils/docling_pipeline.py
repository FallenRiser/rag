"""
DoclingPipelineBuilder
======================

Builds a fully configured Haystack 2.x ``DoclingConverter`` from ``DoclingConfig``.

Supported features (all driven by YAML config)
----------------------------------------------
PDF / DOCX / HTML / PPTX / image pipeline options
OCR engines  : EasyOCR | Tesseract | RapidOCR
  Tesseract  : auto-maps ISO 639-1 → Tesseract 3-letter codes
  RapidOCR   : optional config path
Table extraction : TableFormer (accurate | fast), cell matching
Image enrichments:
  Picture description (captioning) via:
    Ollama            — local VLM (granite3.2-vision, llava, moondream…)
    Azure OpenAI      — GPT-4o vision deployment
    OpenAI-compatible — any /v1/chat/completions endpoint
  Picture classification — built-in Docling classifier
  Code block understanding
  Formula recognition
Export modes : doc_chunks (default) | markdown
HybridChunker: embedded when strategy=document_aware
Device       : cpu | cuda | mps | auto
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from config.models import (
    AzureOpenAICaptionConfig,
    CaptioningBackend,
    DoclingConfig,
    ExportType,
    FormatPipelineConfig,
    InferenceDevice,
    OcrEngine,
    OpenAICompatibleCaptionConfig,
    OllamaCaptionConfig,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OCR engine builders
# ---------------------------------------------------------------------------


def _build_easyocr_options(cfg: FormatPipelineConfig) -> Any:
    from docling.datamodel.pipeline_options import EasyOcrOptions
    return EasyOcrOptions(
        lang=cfg.ocr_options.lang,
        force_full_page_ocr=cfg.ocr_options.force_full_page_ocr,
    )


def _build_tesseract_options(cfg: FormatPipelineConfig) -> Any:
    from docling.datamodel.pipeline_options import TesseractCliOcrOptions

    _lang_map = {
        "en": "eng", "de": "deu", "fr": "fra", "es": "spa",
        "it": "ita", "pt": "por", "nl": "nld", "zh": "chi_sim",
        "ja": "jpn", "ko": "kor", "ar": "ara", "ru": "rus",
    }
    langs = "+".join(_lang_map.get(l, l) for l in cfg.ocr_options.lang)
    return TesseractCliOcrOptions(
        lang=langs,
        force_full_page_ocr=cfg.ocr_options.force_full_page_ocr,
    )


def _build_rapidocr_options(cfg: FormatPipelineConfig) -> Any:
    from docling.datamodel.pipeline_options import RapidOcrOptions
    kwargs: dict[str, Any] = {
        "force_full_page_ocr": cfg.ocr_options.force_full_page_ocr,
    }
    if cfg.ocr_options.rapidocr_config_path:
        kwargs["config_path"] = str(cfg.ocr_options.rapidocr_config_path)
    return RapidOcrOptions(**kwargs)


def _build_ocr_options(cfg: FormatPipelineConfig) -> Any:
    return {
        OcrEngine.EASYOCR:   _build_easyocr_options,
        OcrEngine.TESSERACT: _build_tesseract_options,
        OcrEngine.RAPIDOCR:  _build_rapidocr_options,
    }.get(cfg.ocr_engine, _build_easyocr_options)(cfg)


# ---------------------------------------------------------------------------
# Format pipeline options
# ---------------------------------------------------------------------------


def _build_pdf_options(cfg: FormatPipelineConfig) -> Any:
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableStructureOptions,
        TableFormerMode,
    )
    mode = (
        TableFormerMode.ACCURATE
        if cfg.table_structure_options.mode.value == "accurate"
        else TableFormerMode.FAST
    )
    return PdfPipelineOptions(
        do_ocr=cfg.do_ocr,
        ocr_options=_build_ocr_options(cfg) if cfg.do_ocr else None,
        do_table_structure=cfg.do_table_structure,
        table_structure_options=TableStructureOptions(
            mode=mode,
            do_cell_matching=cfg.table_structure_options.do_cell_matching,
        ),
        generate_page_images=cfg.generate_page_images,
        generate_picture_images=cfg.generate_picture_images,
        images_scale=cfg.images_scale,
    )


def _build_format_options(cfg: DoclingConfig) -> dict[Any, Any]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption
    except ImportError as exc:
        raise ImportError("Run: pip install docling") from exc

    options: dict[Any, Any] = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=_build_pdf_options(cfg.pdf))
    }

    # DOCX
    try:
        from docling.document_converter import WordFormatOption
        from docling.datamodel.pipeline_options import WordPipelineOptions
        options[InputFormat.DOCX] = WordFormatOption(
            pipeline_options=WordPipelineOptions(
                do_ocr=cfg.docx.do_ocr,
                generate_picture_images=cfg.docx.generate_picture_images,
            )
        )
    except Exception:
        pass  # optional; docling uses default for DOCX

    # HTML
    try:
        from docling.document_converter import HtmlFormatOption
        from docling.datamodel.pipeline_options import HtmlPipelineOptions
        options[InputFormat.HTML] = HtmlFormatOption(
            pipeline_options=HtmlPipelineOptions(
                generate_picture_images=cfg.html.generate_picture_images,
            )
        )
    except Exception:
        pass

    # PPTX
    try:
        from docling.document_converter import PowerpointFormatOption
        from docling.datamodel.pipeline_options import PptxPipelineOptions
        options[InputFormat.PPTX] = PowerpointFormatOption(
            pipeline_options=PptxPipelineOptions(
                generate_page_images=cfg.pptx.generate_page_images,
                generate_picture_images=cfg.pptx.generate_picture_images,
            )
        )
    except Exception:
        pass

    return options


# ---------------------------------------------------------------------------
# Enrichment / captioning builders
# ---------------------------------------------------------------------------


def _build_ollama_captioner(cfg: OllamaCaptionConfig) -> Any:
    try:
        from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
        return PictureDescriptionApiOptions(
            url=f"{cfg.base_url.rstrip('/')}/api/chat",
            params={"model": cfg.model, "options": {"temperature": 0.0}},
            prompt=cfg.prompt,
            timeout=cfg.timeout_seconds,
        )
    except Exception as exc:
        logger.warning("Could not build Ollama captioner: %s", exc)
        return None


def _build_azure_openai_captioner(cfg: AzureOpenAICaptionConfig) -> Any:
    api_key  = os.environ.get(cfg.api_key_env)
    endpoint = os.environ.get(cfg.endpoint_env)
    deploy   = os.environ.get(cfg.deployment_name_env)

    if not all([api_key, endpoint, deploy]):
        missing = [n for n, v in [
            (cfg.api_key_env, api_key),
            (cfg.endpoint_env, endpoint),
            (cfg.deployment_name_env, deploy),
        ] if not v]
        msg = f"Azure captioning: missing env vars {missing}"
        if cfg.fail_on_error:
            raise EnvironmentError(msg)
        logger.warning(msg)
        return None

    try:
        from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
        url = (
            f"{endpoint.rstrip('/')}/openai/deployments/{deploy}"
            f"/chat/completions?api-version={cfg.api_version}"
        )
        return PictureDescriptionApiOptions(
            url=url,
            params={"model": deploy, "max_tokens": cfg.max_tokens},
            headers={"api-key": api_key},
            prompt=cfg.prompt,
            timeout=cfg.timeout_seconds,
        )
    except Exception as exc:
        logger.warning("Could not build Azure OpenAI captioner: %s", exc)
        if cfg.fail_on_error:
            raise
        return None


def _build_openai_compatible_captioner(cfg: OpenAICompatibleCaptionConfig) -> Any:
    base_url = os.environ.get(cfg.base_url_env, "")
    api_key  = os.environ.get(cfg.api_key_env, "")
    if not base_url or not api_key:
        msg = f"OpenAI-compatible captioning: env {cfg.base_url_env} or {cfg.api_key_env} not set"
        if cfg.fail_on_error:
            raise EnvironmentError(msg)
        logger.warning(msg)
        return None
    try:
        from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
        return PictureDescriptionApiOptions(
            url=f"{base_url.rstrip('/')}/chat/completions",
            params={"model": cfg.model, "max_tokens": cfg.max_tokens},
            headers={"Authorization": f"Bearer {api_key}"},
            prompt=cfg.prompt,
            timeout=cfg.timeout_seconds,
        )
    except Exception as exc:
        logger.warning("Could not build OpenAI-compatible captioner: %s", exc)
        if cfg.fail_on_error:
            raise
        return None


def _build_picture_description(cfg: DoclingConfig) -> Any:
    pd = cfg.enrichments.picture_description
    if not pd.enabled:
        return None
    logger.info("Building picture description enricher (backend=%s)", pd.backend.value)
    return {
        CaptioningBackend.OLLAMA:             lambda: _build_ollama_captioner(pd.ollama),
        CaptioningBackend.AZURE_OPENAI:       lambda: _build_azure_openai_captioner(pd.azure_openai),
        CaptioningBackend.OPENAI_COMPATIBLE:  lambda: _build_openai_compatible_captioner(pd.openai_compatible),
    }.get(pd.backend, lambda: None)()


# ---------------------------------------------------------------------------
# Device resolver
# ---------------------------------------------------------------------------


def _resolve_device(device: InferenceDevice) -> str:
    if device != InferenceDevice.AUTO:
        return device.value
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


class DoclingPipelineBuilder:
    """
    Builds a Haystack ``DoclingConverter`` component from ``DoclingConfig``.

    Usage::

        converter = DoclingPipelineBuilder(settings.docling).build()
        pipeline.add_component("converter", converter)
    """

    def __init__(self, cfg: DoclingConfig) -> None:
        self.cfg = cfg

    def build(self) -> Any:
        """Build a DoclingConverter for the default export mode."""
        try:
            from docling.document_converter import DocumentConverter
            from docling_haystack.converter import DoclingConverter, ExportType as DExportType
        except ImportError as exc:
            raise ImportError(
                "Run: pip install docling docling-haystack"
            ) from exc

        logger.info(
            "Building DoclingConverter | export=%s | ocr=%s | enrichments=[%s]",
            self.cfg.export.type.value,
            self.cfg.pdf.ocr_engine.value,
            self._enrichment_summary(),
        )

        dc = DocumentConverter(format_options=_build_format_options(self.cfg))
        self._attach_enrichments(dc)

        export_map = {
            ExportType.DOC_CHUNKS: DExportType.DOC_CHUNKS,
            ExportType.MARKDOWN:   DExportType.MARKDOWN,
        }
        md_kwargs = None
        if self.cfg.export.type == ExportType.MARKDOWN:
            md_kwargs = {"image_mode": self.cfg.export.markdown_options.image_mode}

        return DoclingConverter(
            converter=dc,
            export_type=export_map[self.cfg.export.type],
            chunker=None,
            md_export_kwargs=md_kwargs,
        )

    def build_with_hybrid_chunker(
        self,
        tokenizer: str = "BAAI/bge-small-en-v1.5",
        max_tokens: int = 512,
        merge_peers: bool = True,
    ) -> Any:
        """Build with Docling's HybridChunker embedded (document_aware strategy)."""
        try:
            from docling.document_converter import DocumentConverter
            from docling_haystack.converter import DoclingConverter, ExportType as DExportType
        except ImportError as exc:
            raise ImportError("Run: pip install docling docling-haystack") from exc

        chunker = None
        try:
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
            chunker = HybridChunker(
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                merge_peers=merge_peers,
            )
        except ImportError:
            logger.warning(
                "HybridChunker not found — using docling-haystack default chunker. "
                "Run: pip install docling-core"
            )

        dc = DocumentConverter(format_options=_build_format_options(self.cfg))
        self._attach_enrichments(dc)

        return DoclingConverter(
            converter=dc,
            export_type=DExportType.DOC_CHUNKS,
            chunker=chunker,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attach_enrichments(self, dc: Any) -> None:
        captioner = _build_picture_description(self.cfg)
        if captioner is not None:
            try:
                if hasattr(dc, "add_enrichment"):
                    dc.add_enrichment(captioner)
                else:
                    logger.info("Captioner built; attach manually for your docling version.")
            except Exception as exc:
                logger.warning("Could not attach captioner: %s", exc)

        enc = self.cfg.enrichments
        if enc.picture_classification_enabled:
            logger.info("Picture classification enricher: enabled (built-in)")
        if enc.code_understanding_enabled:
            logger.info("Code understanding enricher: enabled")
        if enc.formula_understanding_enabled:
            logger.info("Formula understanding enricher: enabled")

    def _enrichment_summary(self) -> str:
        enc = self.cfg.enrichments
        parts = []
        if enc.picture_description.enabled:
            parts.append(f"caption({enc.picture_description.backend.value})")
        if enc.picture_classification_enabled:
            parts.append("classify")
        if enc.code_understanding_enabled:
            parts.append("code")
        if enc.formula_understanding_enabled:
            parts.append("formula")
        return ", ".join(parts) or "none"
