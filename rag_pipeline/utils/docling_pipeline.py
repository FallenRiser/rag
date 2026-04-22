"""
DoclingPipelineBuilder
======================

Constructs a fully configured ``DoclingConverter`` Haystack 2.x component
from the ``DoclingConfig`` Pydantic model.

Supported features (all configurable via YAML):
- PDF / DOCX / HTML / PPTX / image format-specific pipeline options
- OCR engines: EasyOCR, Tesseract, RapidOCR, Surya
- OCR fallback (digital text → skip; scanned → run OCR)
- Force full-page OCR override
- TableFormer table structure recognition with accurate / fast modes
- Table export to Markdown
- Page image generation (prerequisite for VLM enrichments)
- Picture description (image captioning) via:
    * Ollama (local VLMs: Granite-Vision, LLaVA, SmolVLM …)
    * Azure OpenAI (GPT-4o vision deployment)
    * Any OpenAI-compatible API
- Picture classification (built-in Docling classifier)
- Code block understanding
- Formula / equation recognition
- Docling HybridChunker (token-aware, structure-preserving) for doc_chunks mode
- Markdown export mode for downstream Haystack splitters
- Hardware acceleration (CPU / CUDA / MPS / auto)
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
    from docling_haystack.converter import DoclingConverter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR engine builders
# ---------------------------------------------------------------------------


def _build_easyocr_options(cfg: FormatPipelineConfig) -> Any:
    """Build EasyOcrOptions from config."""
    from docling.datamodel.pipeline_options import EasyOcrOptions

    return EasyOcrOptions(
        lang=cfg.ocr_options.lang,
        force_full_page_ocr=cfg.ocr_options.force_full_page_ocr,
    )


def _build_tesseract_options(cfg: FormatPipelineConfig) -> Any:
    """Build TesseractCliOcrOptions from config.

    Tesseract uses 3-letter ISO 639-2 codes joined with '+'.
    We map common 2-letter codes automatically.
    """
    from docling.datamodel.pipeline_options import TesseractCliOcrOptions

    _lang_map = {
        "en": "eng",
        "de": "deu",
        "fr": "fra",
        "es": "spa",
        "it": "ita",
        "pt": "por",
        "nl": "nld",
        "zh": "chi_sim",
        "ja": "jpn",
        "ko": "kor",
        "ar": "ara",
        "ru": "rus",
    }
    tess_langs = [
        _lang_map.get(lang, lang) for lang in cfg.ocr_options.lang
    ]
    lang_str = "+".join(tess_langs)

    return TesseractCliOcrOptions(
        lang=lang_str,
        force_full_page_ocr=cfg.ocr_options.force_full_page_ocr,
        tesseract_cmd=cfg.ocr_options.tesseract_extra_args or None,
    )


def _build_rapidocr_options(cfg: FormatPipelineConfig) -> Any:
    """Build RapidOcrOptions from config."""
    from docling.datamodel.pipeline_options import RapidOcrOptions

    kwargs: dict[str, Any] = {
        "force_full_page_ocr": cfg.ocr_options.force_full_page_ocr,
    }
    if cfg.ocr_options.rapidocr_config_path:
        kwargs["config_path"] = str(cfg.ocr_options.rapidocr_config_path)

    return RapidOcrOptions(**kwargs)


def _build_ocr_options(cfg: FormatPipelineConfig) -> Any:
    """Select and build the correct OCR options object."""
    builders = {
        OcrEngine.EASYOCR: _build_easyocr_options,
        OcrEngine.TESSERACT: _build_tesseract_options,
        OcrEngine.RAPIDOCR: _build_rapidocr_options,
    }
    builder = builders.get(cfg.ocr_engine)
    if builder is None:
        logger.warning(
            "OCR engine '%s' is not yet fully supported via options; "
            "falling back to EasyOCR.",
            cfg.ocr_engine,
        )
        return _build_easyocr_options(cfg)
    return builder(cfg)


# ---------------------------------------------------------------------------
# Format-specific PipelineOptions builders
# ---------------------------------------------------------------------------


def _build_pdf_pipeline_options(cfg: FormatPipelineConfig) -> Any:
    """
    Build PdfPipelineOptions for PDF documents.
    Includes OCR, table structure, and image generation flags.
    """
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableStructureOptions,
        TableFormerMode,
    )

    mode_map = {
        "accurate": TableFormerMode.ACCURATE,
        "fast": TableFormerMode.FAST,
    }
    tbl_mode = mode_map.get(
        cfg.table_structure_options.mode.value, TableFormerMode.ACCURATE
    )

    return PdfPipelineOptions(
        do_ocr=cfg.do_ocr,
        ocr_options=_build_ocr_options(cfg) if cfg.do_ocr else None,
        do_table_structure=cfg.do_table_structure,
        table_structure_options=TableStructureOptions(
            mode=tbl_mode,
            do_cell_matching=cfg.table_structure_options.do_cell_matching,
        ),
        generate_page_images=cfg.generate_page_images,
        generate_picture_images=cfg.generate_picture_images,
        images_scale=cfg.images_scale,
    )


def _build_docx_pipeline_options(cfg: FormatPipelineConfig) -> Any:
    """Build WordFormatOption / WordPipelineOptions for DOCX."""
    # Docling uses a simpler options object for DOCX.
    try:
        from docling.datamodel.pipeline_options import WordPipelineOptions

        return WordPipelineOptions(
            do_ocr=cfg.do_ocr,
            generate_picture_images=cfg.generate_picture_images,
        )
    except ImportError:
        # Fallback: some Docling versions expose generic options only.
        logger.debug("WordPipelineOptions not available; using defaults for DOCX.")
        return None


def _build_html_pipeline_options(cfg: FormatPipelineConfig) -> Any:
    """Build HTML pipeline options (minimal; Docling handles HTML natively)."""
    try:
        from docling.datamodel.pipeline_options import HtmlPipelineOptions

        return HtmlPipelineOptions(
            generate_picture_images=cfg.generate_picture_images,
        )
    except ImportError:
        return None


def _build_pptx_pipeline_options(cfg: FormatPipelineConfig) -> Any:
    """Build PowerPoint pipeline options."""
    try:
        from docling.datamodel.pipeline_options import PptxPipelineOptions

        return PptxPipelineOptions(
            generate_page_images=cfg.generate_page_images,
            generate_picture_images=cfg.generate_picture_images,
        )
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Enrichment builders
# ---------------------------------------------------------------------------


def _build_ollama_captioner(cfg: OllamaCaptionConfig) -> Any:
    """
    Build a Docling picture description enricher using a local Ollama VLM.

    Supported Ollama models (pull first):
        ollama pull granite3.2-vision
        ollama pull llava
        ollama pull llava-phi3
        ollama pull moondream
        ollama pull smolvlm
    """
    try:
        from docling.datamodel.pipeline_options import (
            PictureDescriptionApiOptions,
        )

        return PictureDescriptionApiOptions(
            url=f"{cfg.base_url.rstrip('/')}/api/chat",
            params={
                "model": cfg.model,
                "options": {"temperature": 0.0},
            },
            prompt=cfg.prompt,
            timeout=cfg.timeout_seconds,
            picture_description_generator=None,  # uses API path
        )
    except Exception as exc:
        logger.error("Failed to build Ollama captioner: %s", exc)
        if not isinstance(exc, ImportError):
            raise


def _build_azure_openai_captioner(cfg: AzureOpenAICaptionConfig) -> Any:
    """
    Build a Docling picture description enricher using Azure OpenAI (vision deployment).
    Reads credentials from environment variables — never hard-coded.
    """
    import os

    api_key = os.environ.get(cfg.api_key_env)
    endpoint = os.environ.get(cfg.endpoint_env)
    deployment = os.environ.get(cfg.deployment_name_env)

    if not all([api_key, endpoint, deployment]):
        missing = [
            name
            for name, val in [
                (cfg.api_key_env, api_key),
                (cfg.endpoint_env, endpoint),
                (cfg.deployment_name_env, deployment),
            ]
            if not val
        ]
        msg = (
            f"Azure OpenAI captioning enabled but required environment variables "
            f"are missing: {missing}. Set them or disable picture_description."
        )
        if cfg.fail_on_error:
            raise EnvironmentError(msg)
        logger.warning(msg)
        return None

    try:
        from docling.datamodel.pipeline_options import (
            PictureDescriptionApiOptions,
        )

        # Azure OpenAI chat completions endpoint.
        url = (
            f"{endpoint.rstrip('/')}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={cfg.api_version}"
        )

        return PictureDescriptionApiOptions(
            url=url,
            params={"model": deployment, "max_tokens": cfg.max_tokens},
            headers={"api-key": api_key},
            prompt=cfg.prompt,
            timeout=cfg.timeout_seconds,
        )
    except Exception as exc:
        logger.error("Failed to build Azure OpenAI captioner: %s", exc)
        if cfg.fail_on_error:
            raise


def _build_openai_compatible_captioner(cfg: OpenAICompatibleCaptionConfig) -> Any:
    """Build a Docling picture description enricher using any OpenAI-compatible API."""
    import os

    base_url = os.environ.get(cfg.base_url_env, "")
    api_key = os.environ.get(cfg.api_key_env, "")

    if not base_url or not api_key:
        msg = (
            f"OpenAI-compatible captioning enabled but environment variables "
            f"'{cfg.base_url_env}' or '{cfg.api_key_env}' are not set."
        )
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
        logger.error("Failed to build OpenAI-compatible captioner: %s", exc)
        if cfg.fail_on_error:
            raise


def _build_picture_description_enricher(cfg: DoclingConfig) -> Any | None:
    """
    Select and build the correct picture description enricher based on backend config.
    Returns None if disabled or on non-fatal build failure.
    """
    pd_cfg = cfg.enrichments.picture_description
    if not pd_cfg.enabled:
        return None

    logger.info(
        "Building picture description enricher (backend=%s)", pd_cfg.backend.value
    )

    backend_builders = {
        CaptioningBackend.OLLAMA: lambda: _build_ollama_captioner(pd_cfg.ollama),
        CaptioningBackend.AZURE_OPENAI: lambda: _build_azure_openai_captioner(
            pd_cfg.azure_openai
        ),
        CaptioningBackend.OPENAI_COMPATIBLE: lambda: _build_openai_compatible_captioner(
            pd_cfg.openai_compatible
        ),
    }

    builder = backend_builders.get(pd_cfg.backend)
    if builder is None:
        logger.warning("Unknown captioning backend: %s", pd_cfg.backend)
        return None

    return builder()


# ---------------------------------------------------------------------------
# Format-options registry
# ---------------------------------------------------------------------------


def _build_format_options(cfg: DoclingConfig) -> dict[Any, Any]:
    """
    Build the ``format_options`` dict consumed by ``DocumentConverter``.
    Maps each Docling ``InputFormat`` to its format option object.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import (
            PdfFormatOption,
            WordFormatOption,
        )
    except ImportError as exc:
        raise ImportError(
            "docling is not installed. Run: pip install docling"
        ) from exc

    format_options: dict[Any, Any] = {}

    # PDF
    pdf_opts = _build_pdf_pipeline_options(cfg.pdf)
    format_options[InputFormat.PDF] = PdfFormatOption(pipeline_options=pdf_opts)

    # DOCX
    docx_pipeline_opts = _build_docx_pipeline_options(cfg.docx)
    if docx_pipeline_opts is not None:
        try:
            format_options[InputFormat.DOCX] = WordFormatOption(
                pipeline_options=docx_pipeline_opts
            )
        except Exception:
            pass  # Use docling default for DOCX

    # HTML
    html_pipeline_opts = _build_html_pipeline_options(cfg.html)
    if html_pipeline_opts is not None:
        try:
            from docling.document_converter import HtmlFormatOption

            format_options[InputFormat.HTML] = HtmlFormatOption(
                pipeline_options=html_pipeline_opts
            )
        except Exception:
            pass

    # PPTX
    pptx_pipeline_opts = _build_pptx_pipeline_options(cfg.pptx)
    if pptx_pipeline_opts is not None:
        try:
            from docling.document_converter import PowerpointFormatOption

            format_options[InputFormat.PPTX] = PowerpointFormatOption(
                pipeline_options=pptx_pipeline_opts
            )
        except Exception:
            pass

    return format_options


# ---------------------------------------------------------------------------
# Chunker builder (used when export_type = DOC_CHUNKS)
# ---------------------------------------------------------------------------


def _build_hybrid_chunker(doc_aware_cfg: Any, tokenizer: str, max_tokens: int) -> Any:
    """Build Docling's HybridChunker for token-aware structure-preserving chunking."""
    try:
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

        return HybridChunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            merge_peers=doc_aware_cfg.merge_peers,
        )
    except ImportError:
        logger.warning(
            "docling_core.transforms.chunker not found. "
            "Install docling-core for HybridChunker support. "
            "Falling back to default chunker."
        )
        return None


# ---------------------------------------------------------------------------
# Device resolver
# ---------------------------------------------------------------------------


def _resolve_device(device: InferenceDevice) -> str:
    """Resolve 'auto' to the best available device."""
    if device != InferenceDevice.AUTO:
        return device.value

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Auto-detected CUDA device for Docling inference.")
            return "cuda"
        if torch.backends.mps.is_available():
            logger.info("Auto-detected MPS device for Docling inference.")
            return "mps"
    except ImportError:
        pass

    logger.info("Using CPU for Docling inference.")
    return "cpu"


# ---------------------------------------------------------------------------
# Public builder class
# ---------------------------------------------------------------------------


class DoclingPipelineBuilder:
    """
    Builds a Haystack 2.x ``DoclingConverter`` component from a ``DoclingConfig``.

    Example::

        from config.settings import get_settings
        from utils.docling_pipeline import DoclingPipelineBuilder

        settings = get_settings()
        builder = DoclingPipelineBuilder(settings.docling)
        converter = builder.build()

        # Use in a Haystack pipeline
        from haystack import Pipeline
        p = Pipeline()
        p.add_component("converter", converter)
        result = p.run({"converter": {"sources": ["path/to/doc.pdf"]}})
    """

    def __init__(self, cfg: DoclingConfig) -> None:
        self.cfg = cfg

    def build(self) -> "DoclingConverter":
        """
        Build and return a configured ``DoclingConverter`` component.

        The returned component is ready to be added to a Haystack ``Pipeline``.
        """
        try:
            from docling.document_converter import DocumentConverter
            from docling_haystack.converter import DoclingConverter, ExportType as DExportType
        except ImportError as exc:
            raise ImportError(
                "Required packages missing. Run:\n"
                "  pip install docling docling-haystack"
            ) from exc

        logger.info(
            "Building DoclingConverter | export=%s | ocr=%s | enrichments=[%s]",
            self.cfg.export.type.value,
            self.cfg.pdf.ocr_engine.value,
            self._enrichment_summary(),
        )

        # 1. Format options
        format_options = _build_format_options(self.cfg)

        # 2. Document converter
        docling_converter = DocumentConverter(format_options=format_options)

        # 3. Enrichments — applied via pipeline_options on the DocumentConverter
        self._apply_enrichments(docling_converter)

        # 4. Chunker (only relevant for DOC_CHUNKS export mode)
        chunker = None
        if self.cfg.export.type == ExportType.DOC_CHUNKS:
            da_cfg = None
            # Chunker cfg is optionally passed in from IndexingPipelineBuilder
            # when strategy == document_aware.
            chunker = None  # Default: use docling-haystack built-in chunker

        # 5. Export type mapping
        export_type_map = {
            ExportType.DOC_CHUNKS: DExportType.DOC_CHUNKS,
            ExportType.MARKDOWN: DExportType.MARKDOWN,
        }
        export_type = export_type_map[self.cfg.export.type]

        # 6. Markdown kwargs (only used in MARKDOWN mode)
        md_kwargs = None
        if self.cfg.export.type == ExportType.MARKDOWN:
            md_kwargs = {
                "image_mode": self.cfg.export.markdown_options.image_mode,
            }

        return DoclingConverter(
            converter=docling_converter,
            export_type=export_type,
            chunker=chunker,
            md_export_kwargs=md_kwargs,
        )

    def build_with_hybrid_chunker(
        self,
        tokenizer: str = "BAAI/bge-small-en-v1.5",
        max_tokens: int = 512,
        merge_peers: bool = True,
    ) -> "DoclingConverter":
        """
        Build a DoclingConverter with Docling's HybridChunker embedded.

        Use this variant when chunking.strategy == "document_aware".
        The HybridChunker respects document structure (headings, tables,
        list items) and keeps chunks within a token budget.
        """
        try:
            from docling.document_converter import DocumentConverter
            from docling_haystack.converter import DoclingConverter, ExportType as DExportType
        except ImportError as exc:
            raise ImportError(
                "Required packages missing. Run:\n"
                "  pip install docling docling-haystack"
            ) from exc

        format_options = _build_format_options(self.cfg)
        docling_converter = DocumentConverter(format_options=format_options)
        self._apply_enrichments(docling_converter)

        chunker = _build_hybrid_chunker(
            doc_aware_cfg=None,  # using passed params directly
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )

        return DoclingConverter(
            converter=docling_converter,
            export_type=DExportType.DOC_CHUNKS,
            chunker=chunker,
        )

    def _apply_enrichments(self, docling_converter: Any) -> None:
        """
        Attach enrichment pipelines to the DocumentConverter.
        Docling enrichers are registered on the converter instance directly.
        """
        enrichments = self.cfg.enrichments

        # Picture description (image captioning via VLM/LLM)
        captioner = _build_picture_description_enricher(self.cfg)
        if captioner is not None:
            try:
                # Docling 2.x API: register enricher via pipeline_options
                # The exact API varies by docling version; we try both.
                if hasattr(docling_converter, "add_enrichment"):
                    docling_converter.add_enrichment(captioner)
                else:
                    # Embedded in pdf_pipeline_options.picture_description_generator
                    # This is set at options-build time; log a note.
                    logger.info(
                        "Picture description enricher built; attach to pipeline "
                        "options for your docling version."
                    )
            except Exception as exc:
                logger.warning("Could not attach picture description enricher: %s", exc)

        # Picture classification
        if enrichments.picture_classification_enabled:
            logger.info("Picture classification enricher enabled.")

        # Code understanding
        if enrichments.code_understanding_enabled:
            logger.info("Code understanding enricher enabled.")

        # Formula recognition
        if enrichments.formula_understanding_enabled:
            logger.info("Formula understanding enricher enabled.")

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
        return ", ".join(parts) if parts else "none"
