import os
import sys
import time
import json
import queue
import logging
import threading
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List

from .tracing import get_current_trace_id, get_current_span_id, _active_span_context
from .metrics import metrics_registry

class JsonFormatter(logging.Formatter):
    """Formats log records as structured JSON with trace correlation."""
    def format(self, record: logging.LogRecord) -> str:
        trace_id = getattr(record, "trace_id", None) or get_current_trace_id() or ""
        span_id = getattr(record, "span_id", None) or get_current_span_id() or ""
        tenant_id = getattr(record, "tenant_id", None) or ""
        user_id = getattr(record, "user_id", None) or ""

        # Auto-enrich from ambient span context if available
        span = _active_span_context.get()
        if span:
            if not tenant_id and "tenant_id" in span.attributes:
                tenant_id = str(span.attributes["tenant_id"])
            if not user_id and "user_id" in span.attributes:
                user_id = str(span.attributes["user_id"])

        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": trace_id,
            "span_id": span_id
        }
        if tenant_id:
            log_data["tenant_id"] = tenant_id
        if user_id:
            log_data["user_id"] = user_id
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


try:
    import urllib3
    _urllib3_available = True
except ImportError:
    _urllib3_available = False

class LokiAsyncHttpHandler(logging.Handler):
    """
    Asynchronous, non-blocking HTTP log handler shipping structured logs
    directly to Loki at http://192.168.0.117:3100/loki/api/v1/push.
    Features persistent connection pooling, strict sub-second timeouts,
    and a circuit breaker to prevent application lag if Loki is unreachable.
    """
    def __init__(
        self,
        loki_url: Optional[str] = None,
        service_name: str = "kitchome-rag",
        environment: str = "development",
        batch_size: int = 50,
        flush_interval_seconds: float = 0.5,
        max_queue_size: int = 5000,
        circuit_cooldown_seconds: float = 30.0,
        enabled: bool = True
    ):
        super().__init__()
        self.loki_url = loki_url or os.getenv("LOKI_URL", "http://192.168.0.117:3100/loki/api/v1/push")
        self.service_name = service_name
        self.environment = environment
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self.circuit_cooldown = circuit_cooldown_seconds
        self.enabled = enabled

        self._circuit_open_until = 0.0
        self._http_pool = None
        if _urllib3_available:
            self._http_pool = urllib3.PoolManager(
                maxsize=4,
                timeout=urllib3.Timeout(connect=0.2, read=0.2),
                retries=False
            )

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._shipping_worker, daemon=True, name="LokiLogShipper")
        if self.enabled:
            self._worker.start()

    def emit(self, record: logging.LogRecord) -> None:
        if not self.enabled:
            return
        try:
            formatted_msg = self.format(record)
            timestamp_ns = str(int(record.created * 1e9))
            level = record.levelname
            item = (timestamp_ns, level, formatted_msg)
            self._queue.put_nowait(item)
        except queue.Full:
            metrics_registry.logs_shipped_total.inc(status="dropped_queue_full")
        except Exception:
            # Under no circumstances should log shipping raise into application code
            pass

    def _shipping_worker(self) -> None:
        """Background thread that batches logs and sends them to Loki."""
        while not self._stop_event.is_set():
            batch = []
            try:
                # Wait for at least one log entry
                item = self._queue.get(timeout=self.flush_interval)
                batch.append(item)
                # Drain up to batch_size items
                while len(batch) < self.batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            if batch:
                self._send_to_loki(batch)

    def _send_to_loki(self, batch: List[tuple]) -> None:
        """Constructs Loki streams payload and POSTs to /loki/api/v1/push."""
        # Circuit Breaker Check: if remote endpoint is down, drop batch silently with zero latency
        now = time.time()
        if now < self._circuit_open_until:
            metrics_registry.logs_shipped_total.inc(len(batch), status="circuit_breaker_active")
            return

        # Group entries by level
        grouped: Dict[str, List[List[str]]] = {}
        for ts_ns, level, msg in batch:
            if level not in grouped:
                grouped[level] = []
            grouped[level].append([ts_ns, msg])

        streams = []
        for level, values in grouped.items():
            streams.append({
                "stream": {
                    "service": self.service_name,
                    "job": self.service_name,
                    "environment": self.environment,
                    "level": level
                },
                "values": values
            })

        payload = {"streams": streams}

        try:
            data = json.dumps(payload).encode("utf-8")
            if self._http_pool:
                resp = self._http_pool.request(
                    "POST",
                    self.loki_url,
                    body=data,
                    headers={"Content-Type": "application/json"}
                )
                if resp.status in (200, 204):
                    self._circuit_open_until = 0.0
                    metrics_registry.logs_shipped_total.inc(len(batch), status="success")
                else:
                    metrics_registry.logs_shipped_total.inc(len(batch), status=f"http_{resp.status}")
            else:
                req = urllib.request.Request(
                    self.loki_url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=0.2) as resp:
                    if resp.status in (200, 204):
                        self._circuit_open_until = 0.0
                        metrics_registry.logs_shipped_total.inc(len(batch), status="success")
                    else:
                        metrics_registry.logs_shipped_total.inc(len(batch), status=f"http_{resp.status}")
        except Exception:
            # Offline resilience: Trip circuit breaker for cooldown period so we don't hold threads
            self._circuit_open_until = time.time() + self.circuit_cooldown
            metrics_registry.logs_shipped_total.inc(len(batch), status="failed_network")

    def close(self) -> None:
        self._stop_event.set()
        super().close()


def setup_telemetry_logging(
    level: str = "INFO",
    loki_enabled: Optional[bool] = None,
    log_file_path: Optional[str] = None
) -> logging.Logger:
    """Configures structured JSON logging with Loki shipping, console, and local disk output."""
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z")

    # Clear existing handlers to prevent duplicate lines
    if root_logger.handlers:
        root_logger.handlers.clear()

    # 1. Console / Stdout Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # 2. Local File Handler (logs/kitchome.log)
    file_path = log_file_path or os.path.join(os.getcwd(), "logs", "kitchome.log")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
    except Exception as e:
        sys.stderr.write(f"Warning: Could not initialize local log file {file_path}: {e}\n")

    # 3. Remote Loki Async HTTP Handler
    if loki_enabled is None:
        loki_enabled = os.getenv("LOKI_ENABLED", "true").lower() in ("true", "1", "yes")

    loki_url = os.getenv("LOKI_URL", "http://192.168.0.117:3100/loki/api/v1/push")
    loki_handler = LokiAsyncHttpHandler(
        loki_url=loki_url,
        service_name="kitchome-rag",
        environment=os.getenv("ENVIRONMENT", "development"),
        enabled=loki_enabled
    )
    loki_handler.setFormatter(formatter)
    loki_handler.setLevel(numeric_level)
    root_logger.addHandler(loki_handler)

    return root_logger
