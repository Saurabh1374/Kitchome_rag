import os
import time
import uuid
import json
import threading
import queue
import urllib.request
import urllib.error
from contextvars import ContextVar
from typing import Optional, Dict, Any, List

# Active trace context variable
_active_span_context: ContextVar[Optional["Span"]] = ContextVar("active_span_context", default=None)

def generate_trace_id() -> str:
    """Generates a standard 128-bit hex trace ID (32 chars)."""
    return uuid.uuid4().hex

def generate_span_id() -> str:
    """Generates a standard 64-bit hex span ID (16 chars)."""
    return uuid.uuid4().hex[:16]

class Span:
    """Represents an OpenTelemetry-compatible distributed trace span."""
    def __init__(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.span_id = generate_span_id()
        self.parent_span_id = parent_span_id
        self.trace_id = trace_id or generate_trace_id()
        self.start_time_unix_nano = int(time.time() * 1e9)
        self.end_time_unix_nano: Optional[int] = None
        self.attributes: Dict[str, Any] = attributes or {}
        self.status = "OK"
        self.error_message: Optional[str] = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_error(self, message: str) -> None:
        self.status = "ERROR"
        self.error_message = message

    def finish(self) -> None:
        if self.end_time_unix_nano is None:
            self.end_time_unix_nano = int(time.time() * 1e9)

    def duration_seconds(self) -> float:
        end = self.end_time_unix_nano or int(time.time() * 1e9)
        return (end - self.start_time_unix_nano) / 1e9

    def to_otlp_dict(self) -> Dict[str, Any]:
        """Formats span according to standard OpenTelemetry Protobuf/JSON schema for Tempo."""
        attrs = []
        for k, v in self.attributes.items():
            if isinstance(v, bool):
                val_dict = {"boolValue": v}
            elif isinstance(v, int):
                val_dict = {"intValue": str(v)}
            elif isinstance(v, float):
                val_dict = {"doubleValue": v}
            else:
                val_dict = {"stringValue": str(v)}
            attrs.append({"key": k, "value": val_dict})

        status_dict = {"code": 1 if self.status == "OK" else 2}
        if self.error_message:
            status_dict["message"] = self.error_message

        span_data = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "name": self.name,
            "kind": 1, # SPAN_KIND_INTERNAL
            "startTimeUnixNano": str(self.start_time_unix_nano),
            "endTimeUnixNano": str(self.end_time_unix_nano or self.start_time_unix_nano),
            "attributes": attrs,
            "status": status_dict
        }
        if self.parent_span_id:
            span_data["parentSpanId"] = self.parent_span_id
        return span_data


try:
    import urllib3
    _urllib3_tracing_available = True
except ImportError:
    _urllib3_tracing_available = False

class Tracer:
    """Lightweight OTLP tracer with asynchronous background shipping to Tempo."""
    def __init__(
        self,
        service_name: str = "kitchome-rag",
        otlp_endpoint: Optional[str] = None,
        circuit_cooldown_seconds: float = 30.0,
        enabled: bool = True
    ):
        self.service_name = service_name
        self.enabled = enabled
        self.circuit_cooldown = circuit_cooldown_seconds
        self._circuit_open_until = 0.0

        self._http_pool = None
        if _urllib3_tracing_available:
            self._http_pool = urllib3.PoolManager(
                maxsize=4,
                timeout=urllib3.Timeout(connect=0.2, read=0.2),
                retries=False
            )

        raw_endpoint = otlp_endpoint or os.getenv("TEMPO_OTLP_ENDPOINT", "http://192.168.0.117:4318")
        # Normalize endpoint to HTTP JSON endpoint if needed
        if ":4318" in raw_endpoint and not raw_endpoint.endswith("/v1/traces"):
            self.otlp_url = f"{raw_endpoint.rstrip('/')}/v1/traces"
        elif ":4317" in raw_endpoint:
            # If gRPC port 4317 was specified, fall back to HTTP port 4318 for lightweight standard HTTP export
            self.otlp_url = raw_endpoint.replace(":4317", ":4318").rstrip("/") + "/v1/traces"
        else:
            self.otlp_url = raw_endpoint if "/v1/traces" in raw_endpoint else f"{raw_endpoint.rstrip('/')}/v1/traces"

        self._queue: queue.Queue = queue.Queue(maxsize=2000)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._export_worker, daemon=True, name="TempoTraceExporter")
        if self.enabled:
            self._worker_thread.start()

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        parent = _active_span_context.get()
        effective_trace_id = trace_id or (parent.trace_id if parent else generate_trace_id())
        effective_parent_span_id = parent_span_id or (parent.span_id if parent else None)
        
        span = Span(
            name=name,
            trace_id=effective_trace_id,
            parent_span_id=effective_parent_span_id,
            attributes=attributes
        )
        return span

    def record_span(self, span: Span) -> None:
        """Enqueues finished span for background export to Tempo."""
        span.finish()
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(span)
        except queue.Full:
            # Drop silently on queue overflow to protect application memory
            pass
        except Exception:
            pass

    def _export_worker(self) -> None:
        """Background daemon pushing spans in batches to Tempo OTLP receiver."""
        while not self._stop_event.is_set():
            spans_to_send: List[Span] = []
            try:
                # Wait for at least one span
                span = self._queue.get(timeout=0.5)
                spans_to_send.append(span)
                # Drain up to 50 more spans for batching
                while len(spans_to_send) < 50:
                    try:
                        spans_to_send.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            if spans_to_send:
                self._send_batch_to_tempo(spans_to_send)

    def _send_batch_to_tempo(self, spans: List[Span]) -> None:
        """Constructs OTLP HTTP JSON payload and POSTs to Tempo receiver."""
        # Circuit Breaker Check: if Tempo endpoint is unreachable, skip immediately
        now = time.time()
        if now < self._circuit_open_until:
            return

        payload = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}},
                            {"key": "deployment.environment", "value": {"stringValue": os.getenv("ENVIRONMENT", "development")}}
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "kitchome.tracer", "version": "1.0.0"},
                            "spans": [s.to_otlp_dict() for s in spans]
                        }
                    ]
                }
            ]
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            if self._http_pool:
                resp = self._http_pool.request(
                    "POST",
                    self.otlp_url,
                    body=data,
                    headers={"Content-Type": "application/json"}
                )
                if resp.status in (200, 204):
                    self._circuit_open_until = 0.0
                else:
                    self._circuit_open_until = time.time() + self.circuit_cooldown
            else:
                req = urllib.request.Request(
                    self.otlp_url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=0.2) as response:
                    self._circuit_open_until = 0.0
        except Exception:
            # Offline resilience: Trip circuit breaker for cooldown period so we don't stall threads
            self._circuit_open_until = time.time() + self.circuit_cooldown


# Global Tracer singleton
tracer = Tracer(
    service_name="kitchome-rag",
    otlp_endpoint=os.getenv("TEMPO_OTLP_ENDPOINT", "http://192.168.0.117:4318/v1/traces"),
    enabled=os.getenv("TRACING_ENABLED", "true").lower() in ("true", "1", "yes")
)

class TraceSpanContext:
    """Context manager for tracing blocks of code."""
    def __init__(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ):
        self.name = name
        self.attributes = attributes or {}
        self.trace_id = trace_id
        self.span: Optional[Span] = None
        self.token = None

    def __enter__(self) -> Span:
        self.span = tracer.start_span(self.name, trace_id=self.trace_id, attributes=self.attributes)
        self.token = _active_span_context.set(self.span)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.span:
            self.span.set_error(str(exc_val))
        if self.span:
            tracer.record_span(self.span)
        if self.token:
            _active_span_context.reset(self.token)

def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None) -> TraceSpanContext:
    """Helper function to create a span context manager."""
    return TraceSpanContext(name=name, attributes=attributes, trace_id=trace_id)

def get_current_trace_id() -> Optional[str]:
    """Returns active trace ID or None."""
    span = _active_span_context.get()
    return span.trace_id if span else None

def get_current_span_id() -> Optional[str]:
    """Returns active span ID or None."""
    span = _active_span_context.get()
    return span.span_id if span else None

def set_active_span(span: Optional[Span]):
    """Sets the active span context on the current thread/task."""
    return _active_span_context.set(span)

def reset_active_span(token):
    """Resets the active span context using the returned context token."""
    _active_span_context.reset(token)

def bind_trace_context(
    name: str = "ambient_operation",
    trace_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> Span:
    """
    Creates and immediately activates a trace span on the current thread.
    Guarantees that all subsequent logs and child spans on this thread inherit
    the trace_id and span_id without requiring context manager nesting.
    """
    span = tracer.start_span(name=name, trace_id=trace_id, attributes=attributes)
    _active_span_context.set(span)
    return span

