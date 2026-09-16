import os
import sys
import json
import logging
import time
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.main import app
from src.telemetry.metrics import metrics_registry, Counter, Gauge, Histogram
from src.telemetry.tracing import tracer, trace_span, generate_trace_id, generate_span_id
from src.telemetry.logging_config import JsonFormatter, LokiAsyncHttpHandler, setup_telemetry_logging

client = TestClient(app)

def test_prometheus_metrics_endpoint():
    """Validates that GET /metrics returns valid OpenMetrics/Prometheus format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    
    body = response.text
    assert "# HELP kitchome_rag_queries_total" in body
    assert "# TYPE kitchome_rag_queries_total counter" in body
    assert "# HELP kitchome_rag_query_duration_seconds" in body
    assert "# TYPE kitchome_rag_query_duration_seconds histogram" in body
    assert "# HELP kitchome_cursor_idempotency_skips_total" in body
    assert "# HELP kitchome_http_requests_total" in body

def test_telemetry_middleware_injects_trace_headers():
    """Validates that HTTP requests receive X-Trace-Id and W3C traceparent headers."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert "X-Trace-Id" in response.headers
    trace_id = response.headers["X-Trace-Id"]
    assert len(trace_id) == 32 # 128-bit hex string
    
    assert "traceparent" in response.headers
    traceparent = response.headers["traceparent"]
    assert traceparent.startswith(f"00-{trace_id}-")

def test_telemetry_middleware_propagates_incoming_trace_id():
    """Validates that an incoming X-Trace-Id header is preserved across the request."""
    custom_trace_id = "0123456789abcdef0123456789abcdef"
    response = client.get("/healthz", headers={"X-Trace-Id": custom_trace_id})
    assert response.status_code == 200
    assert response.headers["X-Trace-Id"] == custom_trace_id
    assert f"00-{custom_trace_id}-" in response.headers["traceparent"]

def test_json_log_formatter():
    """Validates structured JSON log formatting with trace context."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test structured log message",
        args=(),
        exc_info=None
    )
    record.trace_id = "test_trace_1234567890abcdef"
    record.span_id = "test_span_1234"
    record.tenant_id = "tenant_test"

    output = formatter.format(record)
    parsed = json.loads(output)
    
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"
    assert parsed["message"] == "Test structured log message"
    assert parsed["trace_id"] == "test_trace_1234567890abcdef"
    assert parsed["span_id"] == "test_span_1234"
    assert parsed["tenant_id"] == "tenant_test"
    assert "timestamp" in parsed

def test_loki_async_handler_offline_resilience():
    """
    Validates that LokiAsyncHttpHandler gracefully handles an unreachable remote
    host (192.168.0.117:3100) without crashing, throwing exceptions, or blocking.
    """
    # Create handler pointing to unreachable port
    handler = LokiAsyncHttpHandler(
        loki_url="http://192.168.0.117:3100/loki/api/v1/push",
        batch_size=5,
        flush_interval_seconds=0.1,
        enabled=True
    )
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("test.loki.offline")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    t0 = time.time()
    for i in range(10):
        logger.info(f"Test log message #{i}")
    elapsed = time.time() - t0

    # Emission into queue must be virtually instantaneous (< 100ms for 10 records)
    assert elapsed < 0.1, f"Log emission took too long: {elapsed}s"

    # Wait for background thread to attempt flush
    time.sleep(0.3)
    handler.close()

def test_custom_metrics_collection():
    """Validates Counter, Gauge, and Histogram metrics recording and rendering."""
    c = Counter("test_counter", "Test counter metric", ["status"])
    c.inc(status="ok")
    c.inc(value=2.5, status="ok")
    c.inc(status="error")
    assert c.get(status="ok") == 3.5
    assert c.get(status="error") == 1.0

    lines = c.collect()
    assert any('test_counter{status="ok"} 3.5' in line for line in lines)

    g = Gauge("test_gauge", "Test gauge metric", ["pool"])
    g.set(10.0, pool="default")
    g.inc(2.0, pool="default")
    g.dec(1.0, pool="default")
    assert any('test_gauge{pool="default"} 11.0' in line for line in g.collect())

    h = Histogram("test_hist", "Test histogram metric", buckets=(0.1, 0.5, 1.0))
    h.observe(0.05)
    h.observe(0.25)
    h.observe(0.75)
    h_lines = h.collect()
    assert any("test_hist_bucket" in line for line in h_lines)
    assert any("test_hist_sum" in line for line in h_lines)
    assert any("test_hist_count" in line for line in h_lines)

def test_trace_span_context_manager():
    """Validates trace_span context manager and error handling."""
    with trace_span("unit_test_span", {"test.attr": "value"}) as span:
        assert span.name == "unit_test_span"
        assert span.attributes["test.attr"] == "value"
        assert len(span.trace_id) == 32
        assert len(span.span_id) == 16
        assert span.status == "OK"

    # Test error capturing in span
    with pytest.raises(ValueError):
        with trace_span("error_span") as span:
            raise ValueError("Simulation error inside span")
    assert span.status == "ERROR"
    assert "Simulation error inside span" in span.error_message
