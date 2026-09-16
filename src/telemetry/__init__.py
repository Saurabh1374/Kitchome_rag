"""
Telemetry module for Kitchome RAG.
Provides Prometheus metrics, OpenTelemetry distributed tracing, and Loki log shipping.
"""

from .metrics import metrics_registry, Counter, Gauge, Histogram
from .tracing import tracer, trace_span, get_current_trace_id, get_current_span_id, bind_trace_context, set_active_span, reset_active_span
from .logging_config import setup_telemetry_logging, LokiAsyncHttpHandler, JsonFormatter

__all__ = [
    "metrics_registry",
    "Counter",
    "Gauge",
    "Histogram",
    "tracer",
    "trace_span",
    "get_current_trace_id",
    "get_current_span_id",
    "bind_trace_context",
    "set_active_span",
    "reset_active_span",
    "setup_telemetry_logging",
    "LokiAsyncHttpHandler",
    "JsonFormatter"
]
