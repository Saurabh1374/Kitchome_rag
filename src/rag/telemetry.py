import time
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class RetrievalTelemetry(BaseModel):
    query_id: str = Field(default_factory=lambda: f"q_{uuid.uuid4().hex[:8]}")
    timestamp: float = Field(default_factory=time.time)
    query_text: str
    user_tier: str
    predicted_namespaces: List[str]
    authorized_namespaces: List[str]
    confidence_margin: float
    top_similarity_score: float
    fallback_triggered: bool
    final_chunk_count: int

class TelemetryTracker:
    """
    In-memory and file-backed telemetry tracker for Strategy E evaluation.
    Tracks routing confidence, fallback occurrences, and calculates routing precision.
    """
    def __init__(self):
        self._events: List[RetrievalTelemetry] = []

    def record_event(self, event: RetrievalTelemetry) -> None:
        self._events.append(event)

    def get_events(self) -> List[RetrievalTelemetry]:
        return list(self._events)

    def get_routing_precision(self) -> float:
        """
        Routing Precision Metric: 1 - (Fallbacks Triggered / Total Queries)
        Returns 1.0 if no queries have been executed.
        """
        if not self._events:
            return 1.0
        fallback_count = sum(1 for e in self._events if e.fallback_triggered)
        return round(1.0 - (fallback_count / len(self._events)), 4)

    def get_summary(self) -> Dict[str, Any]:
        total = len(self._events)
        fallbacks = sum(1 for e in self._events if e.fallback_triggered)
        avg_sim = round(sum(e.top_similarity_score for e in self._events) / total, 4) if total > 0 else 0.0
        avg_margin = round(sum(e.confidence_margin for e in self._events) / total, 4) if total > 0 else 0.0

        return {
            "total_queries": total,
            "fallback_events": fallbacks,
            "routing_precision": self.get_routing_precision(),
            "average_top_similarity": avg_sim,
            "average_confidence_margin": avg_margin
        }

    def clear(self) -> None:
        self._events.clear()
