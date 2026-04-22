"""
Observability setup — OpenTelemetry + structured logging.

Call ``setup_tracing(settings)`` once during FastAPI lifespan startup.
After that, every Haystack pipeline run emits a distributed trace with
per-component spans automatically.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from config.settings import AppSettings

logger = logging.getLogger(__name__)


def setup_tracing(settings: "AppSettings") -> None:
    """
    Initialise OpenTelemetry tracing and wire it into Haystack.

    If ``settings.otel_endpoint`` is None, tracing is silently skipped.
    Structured logging is always configured.
    """
    _setup_structlog(settings.log_level)

    if not settings.otel_endpoint:
        logger.debug("OTEL endpoint not configured; tracing disabled.")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # Wire Haystack's native OpenTelemetry tracer.
        try:
            import haystack.tracing
            from haystack.tracing.opentelemetry import OpenTelemetryTracer

            haystack.tracing.enable_tracing(OpenTelemetryTracer())
            logger.info(
                "Haystack OpenTelemetry tracing enabled → %s",
                settings.otel_endpoint,
            )
        except ImportError:
            logger.warning(
                "haystack.tracing not available in this Haystack version. "
                "Pipeline spans will not be exported."
            )

    except ImportError as exc:
        logger.warning(
            "OpenTelemetry packages not installed; tracing disabled. "
            "Run: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc\n"
            "Error: %s",
            exc,
        )


def _setup_structlog(log_level: str) -> None:
    """Configure structlog for JSON-formatted structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
