import time
import threading
from typing import Dict, List, Tuple, Optional

class Counter:
    """Thread-safe Prometheus Counter metric."""
    def __init__(self, name: str, description: str, label_names: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, **labels):
        if value < 0:
            raise ValueError("Counter increments must be non-negative.")
        label_tuple = tuple(sorted((k, str(labels.get(k, ""))) for k in self.label_names))
        with self._lock:
            self._values[label_tuple] = self._values.get(label_tuple, 0.0) + value

    def get(self, **labels) -> float:
        label_tuple = tuple(sorted((k, str(labels.get(k, ""))) for k in self.label_names))
        with self._lock:
            return self._values.get(label_tuple, 0.0)

    def collect(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter"
        ]
        with self._lock:
            if not self._values:
                # Emit zero-value if no labels or baseline
                if not self.label_names:
                    lines.append(f"{self.name} 0.0")
            for label_tuple, val in self._values.items():
                if label_tuple:
                    lbl_str = ",".join(f'{k}="{v}"' for k, v in label_tuple)
                    lines.append(f"{self.name}{{{lbl_str}}} {val}")
                else:
                    lines.append(f"{self.name} {val}")
        return lines


class Gauge:
    """Thread-safe Prometheus Gauge metric."""
    def __init__(self, name: str, description: str, label_names: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **labels):
        label_tuple = tuple(sorted((k, str(labels.get(k, ""))) for k in self.label_names))
        with self._lock:
            self._values[label_tuple] = float(value)

    def inc(self, value: float = 1.0, **labels):
        label_tuple = tuple(sorted((k, str(labels.get(k, ""))) for k in self.label_names))
        with self._lock:
            self._values[label_tuple] = self._values.get(label_tuple, 0.0) + value

    def dec(self, value: float = 1.0, **labels):
        label_tuple = tuple(sorted((k, str(labels.get(k, ""))) for k in self.label_names))
        with self._lock:
            self._values[label_tuple] = self._values.get(label_tuple, 0.0) - value

    def collect(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge"
        ]
        with self._lock:
            if not self._values and not self.label_names:
                lines.append(f"{self.name} 0.0")
            for label_tuple, val in self._values.items():
                if label_tuple:
                    lbl_str = ",".join(f'{k}="{v}"' for k, v in label_tuple)
                    lines.append(f"{self.name}{{{lbl_str}}} {val}")
                else:
                    lines.append(f"{self.name} {val}")
        return lines


class Histogram:
    """Thread-safe Prometheus Histogram metric with configurable latency buckets."""
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(self, name: str, description: str, label_names: Optional[List[str]] = None, buckets: Optional[Tuple[float, ...]] = None):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: Dict[Tuple[Tuple[str, str], ...], int] = {}
        self._sums: Dict[Tuple[Tuple[str, str], ...], float] = {}
        self._bucket_counts: Dict[Tuple[Tuple[str, str], ...], Dict[float, int]] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **labels):
        label_tuple = tuple(sorted((k, str(labels.get(k, ""))) for k in self.label_names))
        with self._lock:
            self._counts[label_tuple] = self._counts.get(label_tuple, 0) + 1
            self._sums[label_tuple] = self._sums.get(label_tuple, 0.0) + value

            if label_tuple not in self._bucket_counts:
                self._bucket_counts[label_tuple] = {b: 0 for b in self.buckets}

            for b in self.buckets:
                if value <= b:
                    self._bucket_counts[label_tuple][b] += 1

    def collect(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram"
        ]
        with self._lock:
            for label_tuple in self._counts:
                base_labels = dict(label_tuple)
                cumulative = 0
                for b in self.buckets:
                    cumulative = self._bucket_counts[label_tuple][b]
                    b_lbls = {**base_labels, "le": str(b)}
                    lbl_str = ",".join(f'{k}="{v}"' for k, v in sorted(b_lbls.items()))
                    lines.append(f"{self.name}_bucket{{{lbl_str}}} {cumulative}")

                # +Inf bucket
                inf_lbls = {**base_labels, "le": "+Inf"}
                lbl_str_inf = ",".join(f'{k}="{v}"' for k, v in sorted(inf_lbls.items()))
                lines.append(f"{self.name}_bucket{{{lbl_str_inf}}} {self._counts[label_tuple]}")

                # Sum and count
                lbl_str_base = ",".join(f'{k}="{v}"' for k, v in sorted(base_labels.items()))
                prefix = f"{{{lbl_str_base}}}" if lbl_str_base else ""
                lines.append(f"{self.name}_sum{prefix} {self._sums[label_tuple]}")
                lines.append(f"{self.name}_count{prefix} {self._counts[label_tuple]}")
        return lines


class MetricsRegistry:
    """Central registry for Prometheus metrics."""
    def __init__(self):
        self._metrics = []
        self._lock = threading.Lock()

        # Core Kitchome RAG metrics
        self.rag_queries_total = self.register_counter(
            "kitchome_rag_queries_total",
            "Total number of RAG queries processed",
            ["status", "tenant_id", "clearance_level"]
        )
        self.rag_query_duration_seconds = self.register_histogram(
            "kitchome_rag_query_duration_seconds",
            "Latency of RAG queries in seconds",
            ["status", "tier"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        self.ingestion_jobs_total = self.register_counter(
            "kitchome_ingestion_jobs_total",
            "Total ingestion jobs processed",
            ["status", "domain", "tenant_id"]
        )
        self.ingestion_chunks_total = self.register_counter(
            "kitchome_ingestion_chunks_total",
            "Total chunks generated and indexed during ingestion",
            ["domain"]
        )
        self.cursor_idempotency_skips_total = self.register_counter(
            "kitchome_cursor_idempotency_skips_total",
            "Total ingestion operations skipped via cursor hash idempotency gatekeeper",
            ["domain"]
        )
        self.auth_requests_total = self.register_counter(
            "kitchome_auth_requests_total",
            "Total authentication and authorization attempts",
            ["status", "role"]
        )
        self.vector_store_chunks_total = self.register_gauge(
            "kitchome_vector_store_chunks_total",
            "Estimated chunks stored per vector store namespace",
            ["namespace"]
        )
        self.http_requests_total = self.register_counter(
            "kitchome_http_requests_total",
            "Total HTTP requests to the FastAPI backend",
            ["method", "endpoint", "status_code"]
        )
        self.http_request_duration_seconds = self.register_histogram(
            "kitchome_http_request_duration_seconds",
            "HTTP request latency in seconds",
            ["method", "endpoint"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
        )
        self.logs_shipped_total = self.register_counter(
            "kitchome_telemetry_logs_shipped_total",
            "Total logs shipped to remote Loki instance",
            ["status"]
        )

    def register_counter(self, name: str, description: str, label_names: Optional[List[str]] = None) -> Counter:
        c = Counter(name, description, label_names)
        with self._lock:
            self._metrics.append(c)
        return c

    def register_gauge(self, name: str, description: str, label_names: Optional[List[str]] = None) -> Gauge:
        g = Gauge(name, description, label_names)
        with self._lock:
            self._metrics.append(g)
        return g

    def register_histogram(self, name: str, description: str, label_names: Optional[List[str]] = None, buckets: Optional[Tuple[float, ...]] = None) -> Histogram:
        h = Histogram(name, description, label_names, buckets)
        with self._lock:
            self._metrics.append(h)
        return h

    def generate_metrics_text(self) -> str:
        """Generates standard Prometheus/OpenMetrics text exposition."""
        all_lines = []
        with self._lock:
            metrics_snapshot = list(self._metrics)
        for m in metrics_snapshot:
            all_lines.extend(m.collect())
        all_lines.append("") # Trailing newline
        return "\n".join(all_lines)


# Global registry singleton
metrics_registry = MetricsRegistry()
